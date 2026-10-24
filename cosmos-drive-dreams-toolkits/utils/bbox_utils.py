# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import numpy as np
from tqdm import tqdm
import json
from pathlib import Path
from utils.minimap_utils import cuboid3d_to_polyline
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from pycg import Isometry
from utils.graphics_utils import (
    ALL_FACE_INDICES,
    BACK_FACE_INDICES,
    BoundingBox2D,
    EDGE_INDICES,
    FRONT_FACE_INDICES,
    LineSegment2D,
    LineSegment2DPerVertex,
    TriangleList2D,
    TriangleList2DPerVertex,
    get_remaining_face_indices,
)

OBJECT_CLASSES = json.load(open(Path(__file__).parent.parent / 'config' /'hdmap_color_config.json'))['bbox'].keys()

def simplify_type_in_object_info(object_info):
    if object_info['object_type'] not in OBJECT_CLASSES:
        # labels from category v1
        if object_info['object_type'] == "Bus":
            object_info['object_type'] = "Truck"
        # labels from category v1
        elif object_info['object_type'] == 'Vehicle':
            object_info['object_type'] = "Car"

        # labels from category v2
        elif object_info['object_type'] == "Heavy_truck" or \
            object_info['object_type'] == "Train_or_tram_car" or \
            object_info['object_type'] == "Trolley_bus" or \
            object_info['object_type'] == "Trailer":
            object_info['object_type'] = "Truck"
        # labels from category v2
        elif object_info['object_type'] == 'Automobile' or \
                object_info['object_type'] == 'Other_vehicle':
            object_info['object_type'] = "Car"
        # labels from category v2
        elif object_info['object_type'] == 'Person':
            object_info['object_type'] = "Pedestrian"
        # labels from category v2
        elif object_info['object_type'] == 'Rider':
            object_info['object_type'] = "Cyclist"
        else:
            object_info['object_type'] = "Others"

    return object_info

def create_bbox_projection(all_object_info, camera_poses, valid_frame_ids, camera_model):
    """
    Create a projection of bounding boxes on the minimap.
    Args:
        all_object_info: dict, containing all object info
        camera_poses: np.ndarray, shape (N, 4, 4), dtype=np.float32, camera to world transformation matrix
        camera_model: CameraModel, camera model
        valid_frame_ids: list[int], valid frame ids
        draw_heading: bool, whether to draw heading on the bounding boxes
        diff_color: bool, whether to use different colors for dynamic and static objects

    Returns:
        np.ndarray, shape (N, H, W, 3), dtype=np.uint8, projected bounding boxes on canvas
    """
    CLASS_COLORS = json.load(open(Path(__file__).parent.parent / 'config' /'hdmap_color_config.json'))['bbox']
    bbox_projections = []

    for i in valid_frame_ids:
        current_object_info = all_object_info[f"{i:06d}.all_object_info.json"]

        polylines_cars = []
        polylines_trucks = []
        polylines_pedestrians = []
        polylines_cyclists = []
        polylines_others = []

        # sort tracking ids. avoid jittering when drawing bbox.
        tracking_ids = list(current_object_info.keys())
        tracking_ids.sort()

        for tracking_id in tracking_ids:
            object_info = current_object_info[tracking_id]
            object_info = simplify_type_in_object_info(object_info)

            object_to_world = np.array(object_info['object_to_world'])
            object_lwh = np.array(object_info['object_lwh'])
            cuboid_eight_vertices = build_cuboid_bounding_box(object_lwh[0], object_lwh[1], object_lwh[2], object_to_world)
            polyline = cuboid3d_to_polyline(cuboid_eight_vertices)

            # draw by the object type
            if object_info['object_type'] == "Car":
                polylines_cars.append(polyline)
            elif object_info['object_type'] == "Truck":
                polylines_trucks.append(polyline)
            elif object_info['object_type'] == "Pedestrian":
                polylines_pedestrians.append(polyline)
            elif object_info['object_type'] == "Cyclist":
                polylines_cyclists.append(polyline)
            else:
                polylines_others.append(polyline)

        cars_bbox_projection = camera_model.draw_line_depth(camera_poses[i], polylines_cars, radius=5, colors=np.array(CLASS_COLORS["Car"]))
        trucks_bbox_projection = camera_model.draw_line_depth(camera_poses[i], polylines_trucks, radius=5, colors=np.array(CLASS_COLORS["Truck"]))
        pedestrians_bbox_projection = camera_model.draw_line_depth(camera_poses[i], polylines_pedestrians, radius=5, colors=np.array(CLASS_COLORS["Pedestrian"]))
        cyclists_bbox_projection = camera_model.draw_line_depth(camera_poses[i], polylines_cyclists, radius=5, colors=np.array(CLASS_COLORS["Cyclist"]))
        others_bbox_projection = camera_model.draw_line_depth(camera_poses[i], polylines_others, radius=5, colors=np.array(CLASS_COLORS["Others"]))

        # combine the dynamic and static bbox projection
        bbox_projection = np.maximum.reduce([cars_bbox_projection, trucks_bbox_projection, pedestrians_bbox_projection, cyclists_bbox_projection, others_bbox_projection])
        bbox_projections.append(bbox_projection)

    return np.concatenate(bbox_projections, axis=0)


def interpolate_pose(prev_pose, next_pose, t):
    """
    new pose = (1 - t) * prev_pose + t * next_pose.
    - linear interpolation for translation
    - slerp interpolation for rotation

    Args:
        prev_pose: np.ndarray, shape (4, 4), dtype=np.float32, previous pose
        next_pose: np.ndarray, shape (4, 4), dtype=np.float32, next pose
        t: float, interpolation factor

    Returns:
        np.ndarray, shape (4, 4), dtype=np.float32, interpolated pose

    Note:
        if input is list, also return list.
    """
    input_is_list = isinstance(prev_pose, list)
    prev_pose = np.array(prev_pose)
    next_pose = np.array(next_pose)

    prev_translation = prev_pose[:3, 3]
    next_translation = next_pose[:3, 3]
    translation = (1 - t) * prev_translation + t * next_translation

    prev_rotation = R.from_matrix(prev_pose[:3, :3])
    next_rotation = R.from_matrix(next_pose[:3, :3])
    
    times = [0, 1]
    rotations = R.from_quat([prev_rotation.as_quat(), next_rotation.as_quat()])
    rotation = Slerp(times, rotations)(t)

    new_pose = np.eye(4)
    new_pose[:3, :3] = rotation.as_matrix()
    new_pose[:3, 3] = translation

    if input_is_list:
        return new_pose.tolist()
    else:
        return new_pose
    

def interpolate_bbox(all_object_info, valid_frame_ids):
    """
    Interpolate bbox from 10Hz to 30Hz.
    Args:
        all_object_info: dict, containing all object info. Keys will be 
            {frame_id.06d}.all_object_info.json, where frame_id has a interval of 3.
            For example, 000000.all_object_info.json, 000003.all_object_info.json, 000006.all_object_info.json, etc.

            For one key, the value is a dict, containing all object info for that frame.
            "000000.all_object_info.json": {
                "1": {
                    "object_to_world": 4x4 matrix,
                    "object_lwh": 3-length array,
                    "object_is_moving": bool,
                    "object_type": str,
                },
                "2": {
                    ...
                },  
            }

            Here "1" is the tracking id, and the value is the object info for that frame.

        valid_frame_ids: list[int], valid frame ids

    Returns:
        dict, containing interpolated object info
    """
    interpolated_all_object_info = {}

    for frame_id in valid_frame_ids:
        # no need to interpolate
        if f"{frame_id:06d}.all_object_info.json" in all_object_info:
            interpolated_all_object_info[f"{frame_id:06d}.all_object_info.json"] = \
                all_object_info[f"{frame_id:06d}.all_object_info.json"]
        else:
            # find the nearest frame with object info
            prev_frame_id = frame_id
            next_frame_id = frame_id

            while f"{prev_frame_id:06d}.all_object_info.json" not in all_object_info and prev_frame_id >= 0:
                prev_frame_id -= 1
            while f"{next_frame_id:06d}.all_object_info.json" not in all_object_info and next_frame_id <= max(valid_frame_ids):
                next_frame_id += 1

            # usually prev_frame_id can be found. If next_frame_id is out of range, we just duplicate prev_frame_id
            if next_frame_id > max(valid_frame_ids):
                interpolated_all_object_info[f"{frame_id:06d}.all_object_info.json"] = \
                    interpolated_all_object_info[f"{prev_frame_id:06d}.all_object_info.json"]
                continue

            # interpolate the object info from the previous and next frame
            prev_object_info = all_object_info[f"{prev_frame_id:06d}.all_object_info.json"]
            next_object_info = all_object_info[f"{next_frame_id:06d}.all_object_info.json"]
            
            # tracking ids in the previous and next frame
            prev_tracking_ids = set(prev_object_info.keys())
            next_tracking_ids = set(next_object_info.keys())

            # common tracking ids in the previous and next frame
            common_tracking_ids = prev_tracking_ids & next_tracking_ids

            t = (frame_id - prev_frame_id) / (next_frame_id - prev_frame_id)

            interpolated_object_info = {}
            # interpolate the object info from the previous and next frame
            for tracking_id in common_tracking_ids:
                prev_pose = np.array(prev_object_info[tracking_id]['object_to_world'])
                next_pose = np.array(next_object_info[tracking_id]['object_to_world'])
                interpolated_pose = interpolate_pose(prev_pose, next_pose, t)
                interpolated_object_info[tracking_id] = {}
                interpolated_object_info[tracking_id]['object_to_world'] = interpolated_pose.tolist()

                prev_lwh = np.array(prev_object_info[tracking_id]['object_lwh'])
                next_lwh = np.array(next_object_info[tracking_id]['object_lwh'])
                interpolated_lwh = (1 - t) * prev_lwh + t * next_lwh
                interpolated_object_info[tracking_id]['object_lwh'] = interpolated_lwh.tolist()

                interpolated_object_info[tracking_id]['object_is_moving'] = prev_object_info[tracking_id]['object_is_moving']
                interpolated_object_info[tracking_id]['object_type'] = prev_object_info[tracking_id]['object_type']

            interpolated_all_object_info[f"{frame_id:06d}.all_object_info.json"] = interpolated_object_info

    return interpolated_all_object_info


def fix_static_objects(all_object_info):
    ############## 1. fix object_lwh ##############
    # record the lwh of static objects
    static_tracking_id_to_lwhs = {}

    for frame_id, object_info_dict in all_object_info.items():
        if frame_id.startswith('__'):
            continue

        for tracking_id, object_info in object_info_dict.items():
            if not object_info['object_is_moving']:
                if tracking_id not in static_tracking_id_to_lwhs:
                    static_tracking_id_to_lwhs[tracking_id] = []
                static_tracking_id_to_lwhs[tracking_id].append(object_info['object_lwh'])

    static_tracking_id_to_mean_lwh = {}

    for tracking_id, lwhs in static_tracking_id_to_lwhs.items():
        static_tracking_id_to_mean_lwh[tracking_id] = np.mean(lwhs, axis=0)

    # update the lwh of static objects
    for frame_id, object_info_dict in all_object_info.items():
        if frame_id.startswith('__'):
            continue

        for tracking_id, object_info in object_info_dict.items():
            if not object_info['object_is_moving']:
                object_info['object_lwh'] = static_tracking_id_to_mean_lwh[tracking_id].tolist()

    ############## 2. fix object_to_world ##############
    # record the object_to_world of static objects
    static_tracking_id_to_tfms = {}
    static_tracking_id_to_headings = {}
    for frame_id, object_info_dict in all_object_info.items():
        if frame_id.startswith('__'):
            continue

        for tracking_id, object_info in object_info_dict.items():
            if not object_info['object_is_moving']:
                if tracking_id not in static_tracking_id_to_tfms:
                    static_tracking_id_to_tfms[tracking_id] = []
                    static_tracking_id_to_headings[tracking_id] = []

                static_tracking_id_to_tfms[tracking_id].append(np.array(object_info['object_to_world']))
                static_tracking_id_to_headings[tracking_id].append(object_tfm_to_heading(np.array(object_info['object_to_world'])))

    # compute mean heading of static objects (used to remove outlier)
    static_tracking_id_to_mean_heading = {}
    for tracking_id, headings in static_tracking_id_to_headings.items():
        static_tracking_id_to_mean_heading[tracking_id] = np.mean(headings, axis=0)
        static_tracking_id_to_mean_heading[tracking_id] /= np.linalg.norm(static_tracking_id_to_mean_heading[tracking_id])

    # remove outlier
    threshold = 0.7
    static_tracking_id_to_tfms_remove_outlier = {}
    for tracking_id, tfms in static_tracking_id_to_tfms.items():
        for tfm in tfms:
            heading = object_tfm_to_heading(tfm)
            if np.dot(heading, static_tracking_id_to_mean_heading[tracking_id]) > threshold:
                if tracking_id not in static_tracking_id_to_tfms_remove_outlier:
                    static_tracking_id_to_tfms_remove_outlier[tracking_id] = []

                tfm_orthogoal = Isometry.from_matrix(tfm, ortho=True).matrix
                static_tracking_id_to_tfms_remove_outlier[tracking_id].append(tfm_orthogoal)

    # get the mean tfm of static objects (separate translation and rotation, use quaternion mean for rotation)
    static_tracking_id_to_mean_tfm = {}
    for tracking_id, tfms in static_tracking_id_to_tfms_remove_outlier.items():
        translation_mean = np.mean([tfm[:3, 3] for tfm in tfms], axis=0)

        front_dir_mean = np.mean([tfm[:3, 0] for tfm in tfms], axis=0)
        front_dir_mean /= np.linalg.norm(front_dir_mean)
        up_dir_mean = np.mean([tfm[:3, 2] for tfm in tfms], axis=0)
        up_dir_mean /= np.linalg.norm(up_dir_mean)

        left_dir_mean = np.cross(up_dir_mean, front_dir_mean)
        left_dir_mean /= np.linalg.norm(left_dir_mean)
        rotation_mean = np.stack([front_dir_mean, left_dir_mean, up_dir_mean], axis=1)
        
        static_tracking_id_to_mean_tfm[tracking_id] = np.eye(4)
        static_tracking_id_to_mean_tfm[tracking_id][:3, 3] = translation_mean
        static_tracking_id_to_mean_tfm[tracking_id][:3, :3] = rotation_mean

    # update the object_to_world of static objects
    for frame_id, object_info_dict in all_object_info.items():
        if frame_id.startswith('__'):
            continue

        for tracking_id, object_info in object_info_dict.items():
            if not object_info['object_is_moving'] and tracking_id in static_tracking_id_to_mean_tfm:
                object_info['object_to_world'] = static_tracking_id_to_mean_tfm[tracking_id].tolist()

    return all_object_info

def build_cuboid_bounding_box(dimXMeters, dimYMeters, dimZMeters, cuboid_transform=np.eye(4)):
    """
    Args
        dimXMeters, dimYMeters, dimZMeters: float, the dimensions of the cuboid
        cuboid_transform: 4x4 numpy array, the transformation matrix from the cuboid coordinate to the other coordinate

        z
        ^
        |   y
        | / 
        |/
        o----------> x  (heading)

           3 ---------------- 0
          /|                 /|
         / |                / |
        2 ---------------- 1  |
        |  |               |  |
        |  7 ------------- |- 4
        | /                | /
        6 ---------------- 5 
        
    Returns
        8x3 numpy array: the 8 vertices of the cuboid
    """
    # Build the cuboid bounding box
    cuboid = np.array([
        [dimXMeters / 2, dimYMeters / 2, dimZMeters / 2],
        [dimXMeters / 2, -dimYMeters / 2, dimZMeters / 2],
        [-dimXMeters / 2, -dimYMeters / 2, dimZMeters / 2],
        [-dimXMeters / 2, dimYMeters / 2, dimZMeters / 2],
        [dimXMeters / 2, dimYMeters / 2, -dimZMeters / 2],
        [dimXMeters / 2, -dimYMeters / 2, -dimZMeters / 2],
        [-dimXMeters / 2, -dimYMeters / 2, -dimZMeters / 2],
        [-dimXMeters / 2, dimYMeters / 2, -dimZMeters / 2]
    ])
    cuboid = np.hstack([cuboid, np.ones((8, 1))]) # [8, 4]
    cuboid = np.dot(cuboid_transform, cuboid.T).T 
    return cuboid[:, :3]


def object_tfm_to_heading(tfm):
    """
    Args:
        tfm: 4x4 numpy array, the transformation matrix
    Returns:
        heading_vector: [3,] numpy array, the heading of the object
    """
    if isinstance(tfm, list):
        tfm = np.array(tfm)
        
    heading_vector = tfm[:3, 0]
    heading_vector = heading_vector / np.linalg.norm(heading_vector)
    return heading_vector


def bbox_face_indices(fill_face: str, fill_face_style: str):
    """Return the colored and dark triangle sets used by upstream PR #46."""
    if fill_face == 'front' and fill_face_style == 'solid':
        solid_face_indices = FRONT_FACE_INDICES
        black_face_indices = get_remaining_face_indices(ALL_FACE_INDICES, FRONT_FACE_INDICES)
    elif fill_face == 'back' and fill_face_style == 'solid':
        solid_face_indices = BACK_FACE_INDICES
        black_face_indices = get_remaining_face_indices(ALL_FACE_INDICES, BACK_FACE_INDICES)
    elif fill_face == 'front_and_back' and fill_face_style == 'solid':
        solid_face_indices = FRONT_FACE_INDICES + BACK_FACE_INDICES
        black_face_indices = get_remaining_face_indices(ALL_FACE_INDICES, solid_face_indices)
    elif fill_face == 'all' and fill_face_style == 'solid':
        solid_face_indices = ALL_FACE_INDICES
        black_face_indices = None
    elif fill_face == 'none' and fill_face_style == 'solid':
        solid_face_indices = None
        black_face_indices = ALL_FACE_INDICES
    elif fill_face_style == 'diagonal':
        solid_face_indices = None
        black_face_indices = ALL_FACE_INDICES
    else:
        raise ValueError(
            f"Invalid fill_face_style x fill_face combination: {fill_face_style} x {fill_face}"
        )
    return solid_face_indices, black_face_indices


def lerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    return a + t * (b - a)


def clip_polygon_to_z_near(pts: list[np.ndarray], cols: list[np.ndarray], z_near: float):
    """Sutherland-Hodgman clipping for the camera half-space z >= z_near."""
    def inside(point):
        return float(point[2]) >= z_near

    out_pts, out_cols = [], []
    if not pts:
        return out_pts, out_cols
    for index in range(len(pts)):
        p1, c1 = pts[index], cols[index]
        p2, c2 = pts[(index + 1) % len(pts)], cols[(index + 1) % len(pts)]
        inside1, inside2 = inside(p1), inside(p2)
        if inside1 and inside2:
            out_pts.append(p2)
            out_cols.append(c2)
        elif inside1 and not inside2:
            denominator = float(p2[2] - p1[2])
            if abs(denominator) > 1e-9:
                t = float((z_near - p1[2]) / denominator)
                out_pts.append(lerp(p1, p2, t))
                out_cols.append(lerp(c1, c2, t))
        elif not inside1 and inside2:
            denominator = float(p2[2] - p1[2])
            if abs(denominator) > 1e-9:
                t = float((z_near - p1[2]) / denominator)
                out_pts.append(lerp(p1, p2, t))
                out_cols.append(lerp(c1, c2, t))
            out_pts.append(p2)
            out_cols.append(c2)
    return out_pts, out_cols


def triangulate_fan(pts: list[np.ndarray], cols: list[np.ndarray]):
    """Triangulate a convex clipped polygon and preserve per-vertex colors."""
    if len(pts) < 3:
        return []
    return [
        (
            np.stack([pts[0], pts[index], pts[index + 1]], axis=0),
            np.stack([cols[0], cols[index], cols[index + 1]], axis=0),
        )
        for index in range(1, len(pts) - 1)
    ]


def clip_triangle_to_z_near(tri_pts: np.ndarray, tri_cols: np.ndarray, z_near: float):
    pts, cols = clip_polygon_to_z_near(list(tri_pts), list(tri_cols), z_near=z_near)
    return triangulate_fan(pts, cols)


def create_bbox_geometry_objects_for_frame(
    current_object_info,
    current_camera_pose,
    camera_model,
    fill_face='front_and_back',
    fill_face_style='diagonal',
    object_type_to_per_vertex_color=None,
    line_width=9,
    edge_color=None,
):
    """
    Build BoundingBox2D geometry objects for a single frame.
    Args:
        current_object_info: dict, containing all object info for the current frame
        current_camera_pose: np.ndarray, shape (4, 4), dtype=np.float32, camera pose
        camera_model: CameraModel, camera model
        fill_face: str, which faces to fill
        fill_face_style: str, style of face filling
        object_type_to_per_vertex_color: dict, mapping from object type to per-vertex color
        color_version: str, version key for bbox colors from config_color_bbox.json
        line_width: int, line width for rendering
        edge_color: np.ndarray, shape (3,), dtype=np.float32, optional edge color

    Returns:
        list[TriangleList2D | LineSegment2D], clipped geometry objects
    """
    # Prepare per-vertex colors if not supplied
    if edge_color is not None:
        edge_color = np.array(edge_color) / 255.0

    geometry_objects = []
    world_to_camera = np.linalg.inv(current_camera_pose)
    tracking_ids = list(current_object_info.keys())
    tracking_ids.sort()

    for tracking_id in tracking_ids:
        object_info = current_object_info[tracking_id]
        object_info = simplify_type_in_object_info(object_info)

        object_to_world = np.array(object_info['object_to_world'])
        object_lwh = np.array(object_info['object_lwh'])
        cuboid_eight_vertices = build_cuboid_bounding_box(
            object_lwh[0], object_lwh[1], object_lwh[2], object_to_world
        ) # (8, 3)
        object_type = object_info['object_type']
        if object_type_to_per_vertex_color is None:
            raise ValueError("object_type_to_per_vertex_color must be provided")
        if object_type not in object_type_to_per_vertex_color:
            object_type = 'Others'
        per_corner_color = object_type_to_per_vertex_color[object_type].astype(np.float32)
        corners_cam = camera_model.transform_points_np(
            np.asarray(cuboid_eight_vertices, dtype=np.float32),
            world_to_camera.astype(np.float32),
        )
        depths = corners_cam[:, 2]
        z_near = 5e-2

        # Upstream PR #46: intact boxes keep the original BoundingBox2D path.
        if np.all(depths >= z_near):
            xy = camera_model.ray2pixel_np(corners_cam)
            xy_and_depth = np.hstack([xy, depths.reshape(-1, 1)]).astype(np.float32)
            geometry_objects.append(
                BoundingBox2D(
                    xy_and_depth=xy_and_depth,
                    base_color_or_per_vertex_color=per_corner_color,
                    fill_face=fill_face,
                    fill_face_style=fill_face_style,
                    line_width=line_width,
                    edge_color=edge_color,
                )
            )
            continue

        # A box with no vertex in front of the near plane is invisible.
        if np.all(depths < z_near):
            continue

        # Crossing boxes use the PR's camera-space face clipping path.
        solid_face_indices, black_face_indices = bbox_face_indices(
            fill_face, fill_face_style
        )
        solid_tris_xyzd = []
        solid_tris_col = []
        black_tris_xyzd = []

        if solid_face_indices is not None:
            for i0, i1, i2 in solid_face_indices:
                tri_pts = corners_cam[[i0, i1, i2]]
                tri_cols = per_corner_color[[i0, i1, i2]]
                for clipped_points, clipped_colors in clip_triangle_to_z_near(
                    tri_pts, tri_cols, z_near=z_near
                ):
                    xy = camera_model.ray2pixel_np(clipped_points)
                    solid_tris_xyzd.append(
                        np.concatenate([xy, clipped_points[:, 2:3]], axis=1)
                    )
                    solid_tris_col.append(clipped_colors)

        if black_face_indices is not None:
            for i0, i1, i2 in black_face_indices:
                tri_pts = corners_cam[[i0, i1, i2]]
                tri_cols = np.ones((3, 3), dtype=np.float32) * 0.25
                for clipped_points, _ in clip_triangle_to_z_near(
                    tri_pts, tri_cols, z_near=z_near
                ):
                    xy = camera_model.ray2pixel_np(clipped_points)
                    black_tris_xyzd.append(
                        np.concatenate([xy, clipped_points[:, 2:3]], axis=1)
                    )

        if black_tris_xyzd:
            geometry_objects.append(
                TriangleList2D(
                    np.asarray(black_tris_xyzd, dtype=np.float32),
                    base_color=np.array([0.25, 0.25, 0.25], dtype=np.float32),
                )
            )
        if solid_tris_xyzd:
            geometry_objects.append(
                TriangleList2DPerVertex(
                    np.asarray(solid_tris_xyzd, dtype=np.float32),
                    np.asarray(solid_tris_col, dtype=np.float32),
                )
            )

        # The PR clips cuboid edges independently and interpolates their colors.
        segments = []
        segment_colors = []
        for index0, index1 in EDGE_INDICES:
            p0, p1 = corners_cam[index0], corners_cam[index1]
            c0, c1 = per_corner_color[index0], per_corner_color[index1]
            z0, z1 = float(p0[2]), float(p1[2])
            if z0 < z_near and z1 < z_near:
                continue
            if (z0 < z_near) != (z1 < z_near):
                denominator = z1 - z0
                if abs(denominator) < 1e-9:
                    if z0 >= z_near:
                        p1, c1 = p0, c0
                    else:
                        p0, c0 = p1, c1
                else:
                    t = float(np.clip((z_near - z0) / denominator, 0.0, 1.0))
                    intersection = lerp(p0, p1, t)
                    intersection_color = lerp(c0, c1, t)
                    if z0 < z_near:
                        p0, c0 = intersection, intersection_color
                    else:
                        p1, c1 = intersection, intersection_color
            xy = camera_model.ray2pixel_np(np.stack([p0, p1], axis=0))
            segments.append(
                [[xy[0, 0], xy[0, 1], float(p0[2])],
                 [xy[1, 0], xy[1, 1], float(p1[2])]]
            )
            if edge_color is None:
                segment_colors.append([c0, c1])

        if segments:
            segments = np.asarray(segments, dtype=np.float32)
            if edge_color is not None:
                geometry_objects.append(
                    LineSegment2D(segments, base_color=edge_color.astype(np.float32), line_width=line_width)
                )
            else:
                geometry_objects.append(
                    LineSegment2DPerVertex(
                        segments,
                        np.asarray(segment_colors, dtype=np.float32),
                        line_width=line_width,
                    )
                )

    return geometry_objects


def build_per_vertex_color_map_for_world_scenario():
    """
    Returns:
        dict[str, np.ndarray(8,3)]: mapping from class name to per-vertex RGB
    """
    gradient_class_colors = json.load(open(Path(__file__).parent.parent / 'config' /'world_scenario_color_config.json'))['bbox']
    object_type_to_per_vertex_color = {}
    for object_type, colors in gradient_class_colors.items():
        per_vertex_color = np.zeros((8, 3))
        per_vertex_color[[0,1,4,5]] = np.array(colors[0]) / 255.0
        per_vertex_color[[2,3,6,7]] = np.array(colors[1]) / 255.0
        object_type_to_per_vertex_color[object_type] = per_vertex_color
    return object_type_to_per_vertex_color
