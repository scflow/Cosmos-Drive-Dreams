"""Convert a short single- or multi-camera nuPlan slice to RDS-HQ layout.

This converter intentionally targets the subset consumed by the HD-map renderer:
camera poses/intrinsics, tracked boxes, vector-map layers, and optionally RGB.
It does not require nuPlan lidar blobs or the nuPlan ORM stack.
"""

import json
import math
import os
import pickle
import sqlite3
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import click
import imageio.v2 as imageio
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation, Slerp
from shapely import from_wkb
from shapely.geometry import Polygon
from shapely.ops import unary_union

from utils.wds_utils import write_to_tar


TARGET_POSE_FPS = 30
DEFAULT_TRAFFIC_LIGHT_LWH = (0.35, 0.25, 0.75)
NUPLAN_CAMERA_CHANNELS = (
    "CAM_F0",
    "CAM_L0",
    "CAM_L1",
    "CAM_L2",
    "CAM_B0",
    "CAM_R2",
    "CAM_R1",
    "CAM_R0",
)


def quaternion_matrix(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    """Create a homogeneous transform rotation from a wxyz quaternion."""
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
    return matrix


def interpolate_transform(first: np.ndarray, second: np.ndarray, fraction: float) -> np.ndarray:
    """Interpolate a rigid transform with linear translation and rotation SLERP."""
    result = np.eye(4, dtype=np.float64)
    result[:3, 3] = (1.0 - fraction) * first[:3, 3] + fraction * second[:3, 3]
    rotations = Rotation.from_matrix(np.stack([first[:3, :3], second[:3, :3]]))
    result[:3, :3] = Slerp([0.0, 1.0], rotations)(fraction).as_matrix()
    return result


def upsample_transforms_by_three(source_transforms: Sequence[np.ndarray]) -> List[np.ndarray]:
    """Upsample 10 Hz image poses to 30 Hz while preserving every image pose."""
    if not source_transforms:
        raise ValueError("At least one source pose is required")
    if len(source_transforms) == 1:
        # A final one-frame clip has no following pose from which to estimate
        # motion. Repeating the calibrated pose preserves the source frame and
        # maintains the expected three 30 Hz poses per 10 Hz image.
        return [source_transforms[0].copy() for _ in range(3)]

    result: List[np.ndarray] = []
    for index in range(len(source_transforms) - 1):
        first = source_transforms[index]
        second = source_transforms[index + 1]
        result.extend(
            [
                first.copy(),
                interpolate_transform(first, second, 1.0 / 3.0),
                interpolate_transform(first, second, 2.0 / 3.0),
            ]
        )

    final_pose = source_transforms[-1]
    previous_pose = source_transforms[-2]
    translation_step = (final_pose[:3, 3] - previous_pose[:3, 3]) / 3.0
    result.append(final_pose.copy())
    for multiplier in (1.0, 2.0):
        extrapolated = final_pose.copy()
        extrapolated[:3, 3] += multiplier * translation_step
        result.append(extrapolated)
    return result


def decode_gpkg_geometry(blob: bytes):
    """Decode a GeoPackage geometry blob into a Shapely geometry."""
    if blob[:2] != b"GP":
        raise ValueError("Invalid GeoPackage geometry header")
    flags = blob[3]
    envelope_code = (flags >> 1) & 0b111
    envelope_bytes = {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}.get(envelope_code)
    if envelope_bytes is None:
        raise ValueError(f"Unsupported GeoPackage envelope code: {envelope_code}")
    return from_wkb(blob[8 + envelope_bytes :])


def lonlat_to_utm(lon: np.ndarray, lat: np.ndarray, epsg: int) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorized WGS84 longitude/latitude to UTM conversion.

    nuPlan ego poses use UTM while vector layers in its GeoPackage use EPSG:4326.
    This implementation avoids requiring pyproj for the small conversion utility.
    """
    if not (32601 <= epsg <= 32660 or 32701 <= epsg <= 32760):
        raise ValueError(f"Expected a WGS84 UTM EPSG code, got {epsg}")

    northern = epsg < 32700
    zone = epsg - (32600 if northern else 32700)
    central_meridian = math.radians(zone * 6 - 183)

    a = 6378137.0
    flattening = 1.0 / 298.257223563
    eccentricity_sq = flattening * (2.0 - flattening)
    second_eccentricity_sq = eccentricity_sq / (1.0 - eccentricity_sq)
    scale = 0.9996

    latitude = np.radians(np.asarray(lat, dtype=np.float64))
    longitude = np.radians(np.asarray(lon, dtype=np.float64))
    sin_latitude = np.sin(latitude)
    cos_latitude = np.cos(latitude)
    tan_latitude = np.tan(latitude)

    n = a / np.sqrt(1.0 - eccentricity_sq * sin_latitude**2)
    t = tan_latitude**2
    c = second_eccentricity_sq * cos_latitude**2
    aa = cos_latitude * (longitude - central_meridian)

    e4 = eccentricity_sq**2
    e6 = eccentricity_sq**3
    meridional_arc = a * (
        (1.0 - eccentricity_sq / 4.0 - 3.0 * e4 / 64.0 - 5.0 * e6 / 256.0) * latitude
        - (3.0 * eccentricity_sq / 8.0 + 3.0 * e4 / 32.0 + 45.0 * e6 / 1024.0)
        * np.sin(2.0 * latitude)
        + (15.0 * e4 / 256.0 + 45.0 * e6 / 1024.0) * np.sin(4.0 * latitude)
        - (35.0 * e6 / 3072.0) * np.sin(6.0 * latitude)
    )

    x = scale * n * (
        aa
        + (1.0 - t + c) * aa**3 / 6.0
        + (5.0 - 18.0 * t + t**2 + 72.0 * c - 58.0 * second_eccentricity_sq) * aa**5 / 120.0
    ) + 500000.0
    y = scale * (
        meridional_arc
        + n
        * tan_latitude
        * (
            aa**2 / 2.0
            + (5.0 - t + 9.0 * c + 4.0 * c**2) * aa**4 / 24.0
            + (61.0 - 58.0 * t + t**2 + 600.0 * c - 330.0 * second_eccentricity_sq)
            * aa**6
            / 720.0
        )
    )
    if not northern:
        y += 10_000_000.0
    return x, y


def iter_geometry_coordinates(geometry) -> Iterable[np.ndarray]:
    """Yield coordinate arrays for line and polygon geometry components."""
    geometry_type = geometry.geom_type
    if geometry_type == "LineString":
        yield np.asarray(geometry.coords, dtype=np.float64)[:, :2]
    elif geometry_type == "MultiLineString":
        for item in geometry.geoms:
            yield np.asarray(item.coords, dtype=np.float64)[:, :2]
    elif geometry_type == "Polygon":
        yield np.asarray(geometry.exterior.coords, dtype=np.float64)[:, :2]
    elif geometry_type == "MultiPolygon":
        for item in geometry.geoms:
            yield np.asarray(item.exterior.coords, dtype=np.float64)[:, :2]


def load_map_polylines(
    gpkg_path: Path,
    table: str,
    epsg: int,
    ego_xyz_global: np.ndarray,
    local_origin: np.ndarray,
    radius: float,
    where_clause: str = None,
) -> List[List[List[float]]]:
    """Load, crop, height-annotate, and localize a nuPlan vector-map layer."""
    connection = sqlite3.connect(gpkg_path)
    query = f"SELECT geom FROM {table}"
    if where_clause is not None:
        query += f" WHERE {where_clause}"
    blobs = connection.execute(query).fetchall()
    connection.close()

    trajectory_tree = cKDTree(ego_xyz_global[:, :2])
    trajectory_min = ego_xyz_global[:, :2].min(axis=0) - radius
    trajectory_max = ego_xyz_global[:, :2].max(axis=0) + radius
    polylines: List[List[List[float]]] = []

    for (blob,) in blobs:
        geometry = decode_gpkg_geometry(blob)
        for lonlat in iter_geometry_coordinates(geometry):
            x, y = lonlat_to_utm(lonlat[:, 0], lonlat[:, 1], epsg)
            xy = np.stack([x, y], axis=1)
            bounds_min = xy.min(axis=0)
            bounds_max = xy.max(axis=0)
            if np.any(bounds_max < trajectory_min) or np.any(bounds_min > trajectory_max):
                continue

            _, nearest_indices = trajectory_tree.query(xy)
            z = ego_xyz_global[nearest_indices, 2]
            xyz_local = np.column_stack([xy, z]) - local_origin
            if len(xyz_local) >= 2:
                polylines.append(xyz_local.tolist())

    return polylines


def load_road_boundaries(
    gpkg_path: Path,
    epsg: int,
    ego_xyz_global: np.ndarray,
    local_origin: np.ndarray,
    radius: float,
) -> List[List[List[float]]]:
    """Build road edges from the exterior of merged roadblock/intersection surfaces."""
    connection = sqlite3.connect(gpkg_path)
    blobs = []
    for table in ("lane_groups_polygons", "intersections"):
        blobs.extend(connection.execute(f"SELECT geom FROM {table}").fetchall())
    connection.close()

    trajectory_min = ego_xyz_global[:, :2].min(axis=0) - radius
    trajectory_max = ego_xyz_global[:, :2].max(axis=0) + radius
    road_polygons = []
    for (blob,) in blobs:
        geometry = decode_gpkg_geometry(blob)
        geometries = geometry.geoms if geometry.geom_type == "MultiPolygon" else [geometry]
        for polygon in geometries:
            if polygon.geom_type != "Polygon":
                continue
            lonlat = np.asarray(polygon.exterior.coords, dtype=np.float64)[:, :2]
            x, y = lonlat_to_utm(lonlat[:, 0], lonlat[:, 1], epsg)
            xy = np.column_stack([x, y])
            if np.any(xy.max(axis=0) < trajectory_min) or np.any(xy.min(axis=0) > trajectory_max):
                continue
            road_polygon = Polygon(xy)
            if road_polygon.is_valid and not road_polygon.is_empty:
                road_polygons.append(road_polygon)

    if not road_polygons:
        return []

    trajectory_tree = cKDTree(ego_xyz_global[:, :2])
    merged_boundary = unary_union(road_polygons).boundary
    boundaries = []
    for xy in iter_geometry_coordinates(merged_boundary):
        _, nearest_indices = trajectory_tree.query(xy)
        z = ego_xyz_global[nearest_indices, 2]
        xyz_local = np.column_stack([xy, z]) - local_origin
        if len(xyz_local) >= 2:
            boundaries.append(xyz_local.tolist())
    return boundaries


def _comma_separated_ids(value) -> List[str]:
    """Decode nuPlan's comma-separated map foreign-key columns."""
    if value is None:
        return []
    return [item.strip() for item in str(value).split(",") if item.strip()]


def load_wait_lines(
    gpkg_path: Path,
    epsg: int,
    ego_xyz_global: np.ndarray,
    local_origin: np.ndarray,
    radius: float,
) -> List[List[List[float]]]:
    """Convert stop/yield polygons into their long-axis center lines."""
    connection = sqlite3.connect(gpkg_path)
    rows = connection.execute(
        """
        SELECT geom, stop_polygon_type_fid
        FROM stop_polygons
        WHERE stop_polygon_type_fid IN (1, 2, 4)
        """
    ).fetchall()
    connection.close()

    trajectory_tree = cKDTree(ego_xyz_global[:, :2])
    trajectory_min = ego_xyz_global[:, :2].min(axis=0) - radius
    trajectory_max = ego_xyz_global[:, :2].max(axis=0) + radius
    wait_lines = []
    for blob, _ in rows:
        geometry = decode_gpkg_geometry(blob)
        polygons = geometry.geoms if geometry.geom_type == "MultiPolygon" else [geometry]
        for polygon in polygons:
            if polygon.geom_type != "Polygon":
                continue
            lonlat = np.asarray(polygon.exterior.coords, dtype=np.float64)[:-1, :2]
            if len(lonlat) < 3:
                continue
            x, y = lonlat_to_utm(lonlat[:, 0], lonlat[:, 1], epsg)
            xy = np.column_stack([x, y])
            if np.any(xy.max(axis=0) < trajectory_min) or np.any(xy.min(axis=0) > trajectory_max):
                continue

            center = xy.mean(axis=0)
            covariance = np.cov(xy - center, rowvar=False)
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            long_axis = eigenvectors[:, int(np.argmax(eigenvalues))]
            projections = (xy - center) @ long_axis
            endpoints_xy = np.stack(
                [
                    center + long_axis * projections.min(),
                    center + long_axis * projections.max(),
                ]
            )
            _, nearest_indices = trajectory_tree.query(endpoints_xy)
            endpoints_z = ego_xyz_global[nearest_indices, 2]
            endpoints_local = np.column_stack([endpoints_xy, endpoints_z]) - local_origin
            wait_lines.append(endpoints_local.tolist())
    return wait_lines


def build_oriented_cuboid(center: np.ndarray, yaw: float, lwh: Sequence[float]) -> np.ndarray:
    """Build eight cuboid vertices in the ordering expected by RDS rendering."""
    length, width, height = np.asarray(lwh, dtype=np.float64)
    vertices = np.asarray(
        [
            [length / 2, width / 2, height / 2],
            [length / 2, -width / 2, height / 2],
            [-length / 2, -width / 2, height / 2],
            [-length / 2, width / 2, height / 2],
            [length / 2, width / 2, -height / 2],
            [length / 2, -width / 2, -height / 2],
            [-length / 2, -width / 2, -height / 2],
            [-length / 2, width / 2, -height / 2],
        ]
    )
    rotation = Rotation.from_euler("z", yaw).as_matrix()
    return vertices @ rotation.T + center


def load_traffic_light_heads(
    gpkg_path: Path,
    epsg: int,
    ego_xyz_global: np.ndarray,
    local_origin: np.ndarray,
    radius: float,
    traffic_light_lwh: Sequence[float],
):
    """Load physical signal heads and their controlled lane connectors."""
    connection = sqlite3.connect(gpkg_path)
    connection.row_factory = sqlite3.Row

    stop_to_heads = {
        str(row["fid"]): _comma_separated_ids(row["traffic_light_fids"])
        for row in connection.execute("SELECT fid, traffic_light_fids FROM stop_polygons")
    }
    head_to_connectors: Dict[str, set] = {}
    for row in connection.execute(
        "SELECT fid, traffic_light_stop_line_fids FROM lane_connectors"
    ):
        connector_id = str(row["fid"])
        for stop_id in _comma_separated_ids(row["traffic_light_stop_line_fids"]):
            for head_id in stop_to_heads.get(stop_id, []):
                head_to_connectors.setdefault(head_id, set()).add(connector_id)

    trajectory_tree = cKDTree(ego_xyz_global[:, :2])
    trajectory_min = ego_xyz_global[:, :2].min(axis=0) - radius
    trajectory_max = ego_xyz_global[:, :2].max(axis=0) + radius
    heads = []
    for row in connection.execute(
        """
        SELECT fid, geom, ori_mean_yaw, light_face_type_fid
        FROM traffic_lights
        """
    ):
        geometry = decode_gpkg_geometry(row["geom"])
        if geometry.geom_type != "Point":
            continue
        coordinates = list(geometry.coords[0])
        x, y = lonlat_to_utm(
            np.asarray([coordinates[0]]), np.asarray([coordinates[1]]), epsg
        )
        xy = np.asarray([x[0], y[0]])
        if np.any(xy < trajectory_min) or np.any(xy > trajectory_max):
            continue
        distance, nearest_index = trajectory_tree.query(xy)
        if distance > radius:
            continue

        height_above_road = float(coordinates[2]) if len(coordinates) >= 3 else 4.5
        center_global = np.asarray(
            [xy[0], xy[1], ego_xyz_global[nearest_index, 2] + height_above_road]
        )
        center_local = center_global - local_origin
        feature_id = str(row["fid"])
        heads.append(
            {
                "feature_id": feature_id,
                "connector_ids": sorted(head_to_connectors.get(feature_id, set())),
                "light_face_type_fid": int(row["light_face_type_fid"]),
                "vertices": build_oriented_cuboid(
                    center_local,
                    float(row["ori_mean_yaw"]),
                    traffic_light_lwh,
                ).tolist(),
            }
        )
    connection.close()
    heads.sort(key=lambda item: int(item["feature_id"]))
    return heads


def traffic_light_samples(clip_id: str, heads, statuses_by_lidar, selected_tokens, pose_count):
    """Create RDS traffic-light position and dense per-pose status samples."""
    labels = []
    traffic_light_states = {}
    for index, head in enumerate(heads):
        feature_id = head["feature_id"]
        labels.append(
            {
                "labelData": {
                    "shape3d": {
                        "cuboid3d": {"vertices": head["vertices"]},
                        "attributes": [{"name": "feature_id", "text": feature_id}],
                    }
                }
            }
        )

        image_states = []
        for lidar_token in selected_tokens:
            frame_statuses = statuses_by_lidar.get(lidar_token, {})
            states = {
                frame_statuses[connector_id]
                for connector_id in head["connector_ids"]
                if connector_id in frame_statuses
            }
            image_states.append(next(iter(states)) if len(states) == 1 else "unknown")
        dense_states = [state for state in image_states for _ in range(3)][:pose_count]
        if len(dense_states) < pose_count:
            dense_states.extend([dense_states[-1] if dense_states else "unknown"] * (pose_count - len(dense_states)))
        traffic_light_states[str(index)] = {
            "feature_id": feature_id,
            "state": dense_states,
        }

    if not labels:
        labels = [{"labelData": {"emptyLabel": True}}]
    return (
        {"__key__": clip_id, "traffic_lights.json": {"labels": labels}},
        {
            "__key__": clip_id,
            "aggregated_states.json": {"traffic_light_states": traffic_light_states},
        },
    )


def map_sample(clip_id: str, layer_name: str, vertices: Sequence[Sequence[Sequence[float]]], polygon: bool):
    geometry_name = "surface" if polygon else "polyline3d"
    labels = [
        {"labelData": {"shape3d": {geometry_name: {"vertices": item}}}}
        for item in vertices
    ]
    return {"__key__": clip_id, f"{layer_name}.json": {"labels": labels}}


def laneline_sample(clip_id: str, typed_vertices):
    """Create RDS lane-marking labels with color and dash-style attributes."""
    labels = []
    for vertices, color, style in typed_vertices:
        for item in vertices:
            labels.append(
                {
                    "labelData": {
                        "shape3d": {
                            "polyline3d": {"vertices": item},
                            "attributes": [
                                {"name": "colors", "enumsList": {"enumsList": [color]}},
                                {"name": "styles", "enumsList": {"enumsList": [style]}},
                            ],
                        }
                    }
                }
            )
    return {"__key__": clip_id, "lanelines.json": {"labels": labels}}


def category_to_rds(category: str) -> str:
    return {
        "vehicle": "Car",
        "pedestrian": "Pedestrian",
        "bicycle": "Cyclist",
    }.get(category, "Others")


def convert_clip(
    db_path: Path,
    sensor_root: Path,
    map_root: Path,
    output_root: Path,
    camera_channels: Sequence[str],
    start_index: int,
    frame_count: int,
    map_radius: float,
    write_rgb: bool,
    traffic_light_lwh: Sequence[float] = DEFAULT_TRAFFIC_LIGHT_LWH,
) -> str:
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    if not camera_channels:
        raise ValueError("At least one camera channel is required")

    camera_rows = {}
    for channel in camera_channels:
        camera_row = connection.execute(
            "SELECT * FROM camera WHERE channel = ?", (channel,)
        ).fetchone()
        if camera_row is None:
            raise ValueError(f"Camera {channel} is not present in {db_path}")
        camera_rows[channel] = camera_row

    reference_channel = camera_channels[0]
    reference_image_rows = connection.execute(
        """
        SELECT image.timestamp, image.filename_jpg,
               ego_pose.x, ego_pose.y, ego_pose.z,
               ego_pose.qw, ego_pose.qx, ego_pose.qy, ego_pose.qz,
               ego_pose.epsg
        FROM image
        JOIN ego_pose ON image.ego_pose_token = ego_pose.token
        WHERE image.camera_token = ?
        ORDER BY image.timestamp
        LIMIT ? OFFSET ?
        """,
        (camera_rows[reference_channel]["token"], frame_count, start_index),
    ).fetchall()
    if len(reference_image_rows) != frame_count:
        raise ValueError(
            f"Requested {frame_count} {reference_channel} images but found {len(reference_image_rows)}"
        )

    sanitized_log_name = db_path.stem.replace(".", "_").replace("-", "_")
    camera_suffix = "8cam" if tuple(camera_channels) == NUPLAN_CAMERA_CHANNELS else "_".join(
        channel.lower() for channel in camera_channels
    )
    clip_id = f"{sanitized_log_name}_{start_index:06d}_{start_index + frame_count - 1:06d}_{camera_suffix}"
    timestamps = np.asarray([row["timestamp"] for row in reference_image_rows], dtype=np.int64)
    ego_xyz_global = np.asarray(
        [[row["x"], row["y"], row["z"]] for row in reference_image_rows], dtype=np.float64
    )
    local_origin = ego_xyz_global[0].copy()
    epsg = int(reference_image_rows[0]["epsg"])
    pose_count = frame_count * TARGET_POSE_FPS // 10
    pose_sample = {"__key__": clip_id}
    intrinsic_sample = {"__key__": clip_id}
    opencv_intrinsic_sample = {"__key__": clip_id}
    selected_images_by_camera = {}
    synchronization_by_camera = {}

    for channel in camera_channels:
        camera_row = camera_rows[channel]
        candidate_rows = connection.execute(
            """
            SELECT image.timestamp, image.filename_jpg,
                   ego_pose.x, ego_pose.y, ego_pose.z,
                   ego_pose.qw, ego_pose.qx, ego_pose.qy, ego_pose.qz,
                   ego_pose.epsg
            FROM image
            JOIN ego_pose ON image.ego_pose_token = ego_pose.token
            WHERE image.camera_token = ? AND image.timestamp BETWEEN ? AND ?
            ORDER BY image.timestamp
            """,
            (camera_row["token"], int(timestamps[0] - 100_000), int(timestamps[-1] + 100_000)),
        ).fetchall()
        if not candidate_rows:
            raise ValueError(f"No {channel} images overlap the selected time range")

        candidate_timestamps = np.asarray([row["timestamp"] for row in candidate_rows], dtype=np.int64)
        nearest_indices = np.abs(candidate_timestamps[:, None] - timestamps[None, :]).argmin(axis=0)
        selected_rows = [candidate_rows[index] for index in nearest_indices]
        selected_images_by_camera[channel] = selected_rows
        synchronization_offsets = candidate_timestamps[nearest_indices] - timestamps
        synchronization_by_camera[channel] = {
            "minimum_offset_micros": int(synchronization_offsets.min()),
            "maximum_offset_micros": int(synchronization_offsets.max()),
            "maximum_absolute_offset_micros": int(np.abs(synchronization_offsets).max()),
            "mean_absolute_offset_micros": float(np.abs(synchronization_offsets).mean()),
        }
        print(
            f"{channel}: max synchronization offset "
            f"{synchronization_by_camera[channel]['maximum_absolute_offset_micros']} us"
        )

        camera_to_ego = quaternion_matrix(*pickle.loads(camera_row["rotation"]))
        camera_to_ego[:3, 3] = np.asarray(pickle.loads(camera_row["translation"]), dtype=np.float64)
        camera_poses_at_images = []
        for row in selected_rows:
            ego_to_global = quaternion_matrix(row["qw"], row["qx"], row["qy"], row["qz"])
            ego_to_global[:3, 3] = [row["x"], row["y"], row["z"]]
            camera_to_local = ego_to_global @ camera_to_ego
            camera_to_local[:3, 3] -= local_origin
            camera_poses_at_images.append(camera_to_local)

        camera_poses = upsample_transforms_by_three(camera_poses_at_images)
        if len(camera_poses) != pose_count:
            raise AssertionError(f"Unexpected pose count for {channel}: {len(camera_poses)}")
        rds_camera_name = channel.lower()
        for index, pose in enumerate(camera_poses):
            pose_sample[f"{index:06d}.pose.{rds_camera_name}.npy"] = pose

        intrinsic = np.asarray(pickle.loads(camera_row["intrinsic"]), dtype=np.float64)
        intrinsic_sample[f"pinhole_intrinsic.{rds_camera_name}.npy"] = np.asarray(
            [
                intrinsic[0, 0],
                intrinsic[1, 1],
                intrinsic[0, 2],
                intrinsic[1, 2],
                camera_row["width"],
                camera_row["height"],
            ],
            dtype=np.float64,
        )
        distortion = np.asarray(pickle.loads(camera_row["distortion"]), dtype=np.float64)
        if distortion.shape != (5,):
            raise ValueError(
                f"Expected five OpenCV distortion coefficients for {channel}, got {distortion}"
            )
        opencv_intrinsic_sample[f"opencv_intrinsic.{rds_camera_name}.npy"] = np.asarray(
            [
                intrinsic[0, 0],
                intrinsic[1, 1],
                intrinsic[0, 2],
                intrinsic[1, 2],
                camera_row["width"],
                camera_row["height"],
                *distortion,
            ],
            dtype=np.float64,
        )

    write_to_tar(pose_sample, output_root / "pose" / f"{clip_id}.tar")
    write_to_tar(intrinsic_sample, output_root / "pinhole_intrinsic" / f"{clip_id}.tar")
    write_to_tar(
        opencv_intrinsic_sample,
        output_root / "opencv_intrinsic" / f"{clip_id}.tar",
    )

    lidar_rows = connection.execute(
        "SELECT token, timestamp FROM lidar_pc WHERE timestamp BETWEEN ? AND ? ORDER BY timestamp",
        (int(timestamps[0] - 100_000), int(timestamps[-1] + 100_000)),
    ).fetchall()
    lidar_timestamps = np.asarray([row["timestamp"] for row in lidar_rows], dtype=np.int64)
    lidar_tokens = [row["token"] for row in lidar_rows]
    nearest_lidar_indices = np.abs(lidar_timestamps[:, None] - timestamps[None, :]).argmin(axis=0)
    selected_tokens = [lidar_tokens[index] for index in nearest_lidar_indices]

    placeholders = ",".join("?" for _ in selected_tokens)
    box_rows = connection.execute(
        f"""
        SELECT lidar_box.lidar_pc_token, lidar_box.track_token,
               lidar_box.x, lidar_box.y, lidar_box.z,
               lidar_box.length, lidar_box.width, lidar_box.height,
               lidar_box.yaw, lidar_box.vx, lidar_box.vy, lidar_box.vz,
               category.name AS category_name
        FROM lidar_box
        JOIN track ON lidar_box.track_token = track.token
        JOIN category ON track.category_token = category.token
        WHERE lidar_box.lidar_pc_token IN ({placeholders})
        """,
        selected_tokens,
    ).fetchall()
    boxes_by_lidar = {token: [] for token in selected_tokens}
    for row in box_rows:
        boxes_by_lidar[row["lidar_pc_token"]].append(row)

    bbox_sample = {"__key__": clip_id}
    for image_index, lidar_token in enumerate(selected_tokens):
        frame_boxes = {}
        for row in boxes_by_lidar[lidar_token]:
            object_to_local = np.eye(4, dtype=np.float64)
            object_to_local[:3, :3] = Rotation.from_euler("z", row["yaw"]).as_matrix()
            object_to_local[:3, 3] = np.asarray([row["x"], row["y"], row["z"]]) - local_origin
            speed = float(np.linalg.norm([row["vx"], row["vy"], row["vz"]]))
            track_id = bytes(row["track_token"]).hex()
            frame_boxes[track_id] = {
                "object_to_world": object_to_local.tolist(),
                "object_lwh": [row["length"], row["width"], row["height"]],
                "object_is_moving": speed > 0.2,
                "object_type": category_to_rds(row["category_name"]),
            }
        bbox_sample[f"{image_index * 3:06d}.all_object_info.json"] = frame_boxes
    write_to_tar(bbox_sample, output_root / "all_object_info" / f"{clip_id}.tar")

    traffic_status_rows = connection.execute(
        f"""
        SELECT lidar_pc_token, lane_connector_id, status
        FROM traffic_light_status
        WHERE lidar_pc_token IN ({placeholders})
        """,
        selected_tokens,
    ).fetchall()
    statuses_by_lidar = {token: {} for token in selected_tokens}
    for row in traffic_status_rows:
        statuses_by_lidar[row["lidar_pc_token"]][str(row["lane_connector_id"])] = str(
            row["status"]
        ).lower()

    log_row = connection.execute("SELECT location, map_version FROM log LIMIT 1").fetchone()
    connection.close()
    map_name = log_row["map_version"] or log_row["location"]
    map_versions = sorted((map_root / map_name).glob("*/map.gpkg"))
    if not map_versions:
        raise FileNotFoundError(f"No map.gpkg found for {map_name} under {map_root}")
    gpkg_path = map_versions[-1]

    # nuPlan baseline_paths are routing/lane centerlines. Waymo exports the
    # equivalent `lane` layer but does not render it in the default condition,
    # so only physical painted boundaries belong in RDS `lanelines` here.
    boundary_type_specs = (
        (0, "WHITE", "LONG_DASHED_SINGLE"),
        (1, "WHITE", "SOLID_SINGLE"),
        (2, "WHITE", "SOLID_SINGLE"),
        (4, "YELLOW", "SOLID_SINGLE"),
        (5, "YELLOW", "LONG_DASHED_SINGLE"),
        (6, "WHITE", "SOLID_GROUP"),
        (7, "YELLOW", "SOLID_GROUP"),
    )
    typed_lanelines = []
    for boundary_type, color, style in boundary_type_specs:
        vertices = load_map_polylines(
            gpkg_path,
            "boundaries",
            epsg,
            ego_xyz_global,
            local_origin,
            map_radius,
            where_clause=f"boundary_type_fid = {boundary_type}",
        )
        if vertices:
            typed_lanelines.append((vertices, color, style))
    write_to_tar(
        laneline_sample(clip_id, typed_lanelines),
        output_root / "3d_lanelines" / f"{clip_id}.tar",
    )
    print(f"lanelines: {sum(len(item[0]) for item in typed_lanelines)} geometries")

    road_boundaries = load_road_boundaries(
        gpkg_path, epsg, ego_xyz_global, local_origin, map_radius
    )
    write_to_tar(
        map_sample(clip_id, "road_boundaries", road_boundaries, False),
        output_root / "3d_road_boundaries" / f"{clip_id}.tar",
    )
    print(f"road_boundaries: {len(road_boundaries)} geometries")

    crosswalks = load_map_polylines(
        gpkg_path, "crosswalks", epsg, ego_xyz_global, local_origin, map_radius
    )
    write_to_tar(
        map_sample(clip_id, "crosswalks", crosswalks, True),
        output_root / "3d_crosswalks" / f"{clip_id}.tar",
    )
    print(f"crosswalks: {len(crosswalks)} geometries")

    wait_lines = load_wait_lines(
        gpkg_path, epsg, ego_xyz_global, local_origin, map_radius
    )
    write_to_tar(
        map_sample(clip_id, "wait_lines", wait_lines, False),
        output_root / "3d_wait_lines" / f"{clip_id}.tar",
    )
    print(f"wait_lines: {len(wait_lines)} geometries")

    traffic_light_heads = load_traffic_light_heads(
        gpkg_path,
        epsg,
        ego_xyz_global,
        local_origin,
        map_radius,
        traffic_light_lwh,
    )
    traffic_light_position_sample, traffic_light_status_sample = traffic_light_samples(
        clip_id,
        traffic_light_heads,
        statuses_by_lidar,
        selected_tokens,
        pose_count,
    )
    write_to_tar(
        traffic_light_position_sample,
        output_root / "3d_traffic_lights" / f"{clip_id}.tar",
    )
    write_to_tar(
        traffic_light_status_sample,
        output_root / "3d_traffic_lights_status" / f"{clip_id}.tar",
    )
    associated_head_count = sum(bool(head["connector_ids"]) for head in traffic_light_heads)
    print(
        f"traffic_lights: {len(traffic_light_heads)} physical heads, "
        f"{associated_head_count} associated with lane connectors"
    )

    if write_rgb:
        for channel in camera_channels:
            rds_camera_name = channel.lower()
            video_path = output_root / f"pinhole_{rds_camera_name}" / f"{clip_id}.mp4"
            video_path.parent.mkdir(parents=True, exist_ok=True)
            writer = imageio.get_writer(
                video_path,
                fps=10,
                macro_block_size=None,
                ffmpeg_params=["-threads", os.environ.get("RDS_FFMPEG_THREADS", "2")],
            )
            for row in selected_images_by_camera[channel]:
                image_path = sensor_root / row["filename_jpg"]
                if not image_path.exists():
                    writer.close()
                    raise FileNotFoundError(image_path)
                writer.append_data(imageio.imread(image_path))
            writer.close()
            print(f"Saved {video_path}")

    manifest = {
        "clip_id": clip_id,
        "source_db": str(db_path),
        "reference_camera": reference_channel,
        "cameras": list(camera_channels),
        "image_frames": frame_count,
        "pose_frames": pose_count,
        "camera_model": "opencv_brown_conrady",
        "synchronization_by_camera": synchronization_by_camera,
        "image_timestamps_by_camera": {
            channel.lower(): [int(row["timestamp"]) for row in rows]
            for channel, rows in selected_images_by_camera.items()
        },
        "local_origin_global": local_origin.tolist(),
        "epsg": epsg,
        "map": str(gpkg_path),
        "map_layer_counts": {
            "lanelines": sum(len(item[0]) for item in typed_lanelines),
            "road_boundaries": len(road_boundaries),
            "crosswalks": len(crosswalks),
            "wait_lines": len(wait_lines),
            "traffic_lights": len(traffic_light_heads),
            "traffic_lights_with_connector": associated_head_count,
        },
        "traffic_light_lwh": list(map(float, traffic_light_lwh)),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / f"{clip_id}.manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return clip_id


@click.command()
@click.option("--db_path", type=click.Path(path_type=Path, exists=True), required=True)
@click.option("--sensor_root", type=click.Path(path_type=Path, exists=True), required=True)
@click.option("--map_root", type=click.Path(path_type=Path, exists=True), required=True)
@click.option("--output_root", type=click.Path(path_type=Path), required=True)
@click.option("--camera_channel", default="CAM_F0", show_default=True)
@click.option("--all_cameras", is_flag=True, help="Convert all eight nuPlan cameras")
@click.option("--start_index", type=int, default=0, show_default=True)
@click.option("--frame_count", type=int, default=200, show_default=True)
@click.option("--map_radius", type=float, default=100.0, show_default=True)
@click.option(
    "--traffic_light_lwh",
    type=float,
    nargs=3,
    default=DEFAULT_TRAFFIC_LIGHT_LWH,
    show_default=True,
    help="Canonical nuPlan signal-head cuboid length, width, height in meters",
)
@click.option("--write_rgb/--no_write_rgb", default=True, show_default=True)
def main(
    db_path,
    sensor_root,
    map_root,
    output_root,
    camera_channel,
    all_cameras,
    start_index,
    frame_count,
    map_radius,
    traffic_light_lwh,
    write_rgb,
):
    """Convert one short nuPlan camera slice for an RDS-HQ smoke test."""
    clip_id = convert_clip(
        db_path=db_path,
        sensor_root=sensor_root,
        map_root=map_root,
        output_root=output_root,
        camera_channels=NUPLAN_CAMERA_CHANNELS if all_cameras else (camera_channel,),
        start_index=start_index,
        frame_count=frame_count,
        map_radius=map_radius,
        write_rgb=write_rgb,
        traffic_light_lwh=traffic_light_lwh,
    )
    print(f"Converted clip: {clip_id}")


if __name__ == "__main__":
    main()
