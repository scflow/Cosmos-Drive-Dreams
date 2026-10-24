"""Backfill recoverable nuPlan World Scenario layers in existing RDS clips."""

import json
import sqlite3
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import click
import numpy as np
from tqdm import tqdm

from convert_nuplan_to_rds_hq import (
    DEFAULT_TRAFFIC_LIGHT_LWH,
    load_map_polylines,
    load_traffic_light_heads,
    load_wait_lines,
    map_sample,
    traffic_light_samples,
)
from utils.wds_utils import write_to_tar


LAYERS_VERSION = "nuplan-world-scenario-v2"


def _load_clip_source(manifest):
    """Recover source ego poses and nearest lidar tokens from a clip manifest."""
    db_path = Path(manifest["source_db"])
    reference_camera = manifest["reference_camera"]
    timestamps = manifest["image_timestamps_by_camera"][reference_camera.lower()]
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    camera_token = connection.execute(
        "SELECT token FROM camera WHERE channel = ?", (reference_camera,)
    ).fetchone()["token"]

    placeholders = ",".join("?" for _ in timestamps)
    rows = connection.execute(
        f"""
        SELECT image.timestamp, ego_pose.x, ego_pose.y, ego_pose.z, ego_pose.epsg
        FROM image
        JOIN ego_pose ON image.ego_pose_token = ego_pose.token
        WHERE image.camera_token = ? AND image.timestamp IN ({placeholders})
        """,
        [camera_token, *timestamps],
    ).fetchall()
    row_by_timestamp = {int(row["timestamp"]): row for row in rows}
    ordered_rows = [row_by_timestamp[int(timestamp)] for timestamp in timestamps]
    ego_xyz_global = np.asarray(
        [[row["x"], row["y"], row["z"]] for row in ordered_rows], dtype=np.float64
    )

    lidar_rows = connection.execute(
        """
        SELECT token, timestamp FROM lidar_pc
        WHERE timestamp BETWEEN ? AND ? ORDER BY timestamp
        """,
        (int(timestamps[0] - 100_000), int(timestamps[-1] + 100_000)),
    ).fetchall()
    lidar_timestamps = np.asarray([row["timestamp"] for row in lidar_rows], dtype=np.int64)
    nearest_indices = np.abs(
        lidar_timestamps[:, None] - np.asarray(timestamps, dtype=np.int64)[None, :]
    ).argmin(axis=0)
    selected_tokens = [lidar_rows[index]["token"] for index in nearest_indices]

    token_placeholders = ",".join("?" for _ in selected_tokens)
    status_rows = connection.execute(
        f"""
        SELECT lidar_pc_token, lane_connector_id, status
        FROM traffic_light_status
        WHERE lidar_pc_token IN ({token_placeholders})
        """,
        selected_tokens,
    ).fetchall()
    statuses_by_lidar = {token: {} for token in selected_tokens}
    for row in status_rows:
        statuses_by_lidar[row["lidar_pc_token"]][str(row["lane_connector_id"])] = str(
            row["status"]
        ).lower()
    connection.close()
    return ego_xyz_global, int(ordered_rows[0]["epsg"]), selected_tokens, statuses_by_lidar


def update_clip(manifest_path, rds_root, map_radius, traffic_light_lwh, force):
    """Update one manifest and its four recoverable World Scenario layer tars."""
    manifest_path = Path(manifest_path)
    rds_root = Path(rds_root)
    try:
        manifest = json.loads(manifest_path.read_text())
        if not force and manifest.get("world_scenario_layers_version") == LAYERS_VERSION:
            return {"clip_id": manifest["clip_id"], "status": "skipped"}

        clip_id = manifest["clip_id"]
        ego_xyz_global, epsg, selected_tokens, statuses_by_lidar = _load_clip_source(manifest)
        local_origin = np.asarray(manifest["local_origin_global"], dtype=np.float64)
        gpkg_path = Path(manifest["map"])

        crosswalks = load_map_polylines(
            gpkg_path, "crosswalks", epsg, ego_xyz_global, local_origin, map_radius
        )
        wait_lines = load_wait_lines(
            gpkg_path, epsg, ego_xyz_global, local_origin, map_radius
        )
        heads = load_traffic_light_heads(
            gpkg_path,
            epsg,
            ego_xyz_global,
            local_origin,
            map_radius,
            traffic_light_lwh,
        )
        position_sample, status_sample = traffic_light_samples(
            clip_id,
            heads,
            statuses_by_lidar,
            selected_tokens,
            int(manifest["pose_frames"]),
        )

        write_to_tar(
            map_sample(clip_id, "crosswalks", crosswalks, True),
            rds_root / "3d_crosswalks" / f"{clip_id}.tar",
        )
        write_to_tar(
            map_sample(clip_id, "wait_lines", wait_lines, False),
            rds_root / "3d_wait_lines" / f"{clip_id}.tar",
        )
        write_to_tar(
            position_sample,
            rds_root / "3d_traffic_lights" / f"{clip_id}.tar",
        )
        write_to_tar(
            status_sample,
            rds_root / "3d_traffic_lights_status" / f"{clip_id}.tar",
        )

        counts = dict(manifest.get("map_layer_counts", {}))
        counts.update(
            {
                "crosswalks": len(crosswalks),
                "wait_lines": len(wait_lines),
                "traffic_lights": len(heads),
                "traffic_lights_with_connector": sum(
                    bool(head["connector_ids"]) for head in heads
                ),
            }
        )
        manifest["map_layer_counts"] = counts
        manifest["traffic_light_lwh"] = list(map(float, traffic_light_lwh))
        manifest["world_scenario_layers_version"] = LAYERS_VERSION
        temporary = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
        temporary.write_text(json.dumps(manifest, indent=2) + "\n")
        temporary.replace(manifest_path)
        return {"clip_id": clip_id, "status": "updated", "counts": counts}
    except Exception as error:
        return {
            "clip_id": manifest_path.stem,
            "status": "failed",
            "error": repr(error),
            "traceback": traceback.format_exc(),
        }


@click.command()
@click.option("--rds_root", type=click.Path(path_type=Path, exists=True), required=True)
@click.option("--clip_id", default=None, help="Update one clip; default scans all manifests")
@click.option("--map_radius", type=float, default=100.0, show_default=True)
@click.option(
    "--traffic_light_lwh",
    type=float,
    nargs=3,
    default=DEFAULT_TRAFFIC_LIGHT_LWH,
    show_default=True,
)
@click.option("--workers", type=click.IntRange(min=1), default=8, show_default=True)
@click.option("--force/--resume", default=False, show_default=True)
@click.option("--dry_run", is_flag=True)
def main(rds_root, clip_id, map_radius, traffic_light_lwh, workers, force, dry_run):
    """Add crosswalk, wait-line, and traffic-light layers without rewriting RGB."""
    manifests = (
        [rds_root / f"{clip_id}.manifest.json"]
        if clip_id
        else sorted(rds_root.glob("*.manifest.json"))
    )
    missing = [path for path in manifests if not path.exists()]
    if missing:
        raise click.ClickException(f"Manifest does not exist: {missing[0]}")
    click.echo(f"Planned {len(manifests)} clip(s), workers={workers}, force={force}")
    if dry_run:
        return

    results = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                update_clip,
                path,
                rds_root,
                map_radius,
                traffic_light_lwh,
                force,
            ): path
            for path in manifests
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="RDS map layers", unit="clip"):
            results.append(future.result())

    failed = [result for result in results if result["status"] == "failed"]
    summary = {
        "layers_version": LAYERS_VERSION,
        "total": len(results),
        "updated": sum(result["status"] == "updated" for result in results),
        "skipped": sum(result["status"] == "skipped" for result in results),
        "failed": len(failed),
        "results": results,
    }
    summary_path = rds_root / "world_scenario_layer_update_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    click.echo(json.dumps({key: value for key, value in summary.items() if key != "results"}, indent=2))
    click.echo(f"Summary: {summary_path}")
    if failed:
        raise click.ClickException(f"{len(failed)} clip(s) failed")


if __name__ == "__main__":
    main()
