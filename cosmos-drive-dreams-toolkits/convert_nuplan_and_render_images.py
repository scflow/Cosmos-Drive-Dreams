"""Convert a nuPlan clip to RDS and export aligned visualization images.

For every selected camera/frame this script saves World Scenario and an
untitled 2x2 grid arranged as:

    RGB              RDS HD map
    World Scenario   RGB + RDS HD map
"""

import copy
import json
import os
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import click
import cv2
import numpy as np

# The pipeline calls the rendering functions directly in one process.
os.environ["USE_RAY"] = "False"

from convert_nuplan_to_rds_hq import NUPLAN_CAMERA_CHANNELS, convert_clip
from render_from_rds_hq import render_sample_hdmap, render_sample_world_scenario


TOOLKIT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = TOOLKIT_ROOT / "config" / "dataset_nuplan_8cam.json"
BEIJING_TIMEZONE = ZoneInfo("Asia/Shanghai")


def sanitize_name(value: str) -> str:
    """Make a location/camera component safe and stable in a filename."""
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", value.strip()).strip("_").lower()


def timestamp_name(timestamp_micros: int) -> str:
    """Format a Unix-microsecond timestamp as filesystem-safe Beijing time."""
    dt_utc = datetime.fromtimestamp(timestamp_micros / 1_000_000, tz=timezone.utc)
    return dt_utc.astimezone(BEIJING_TIMEZONE).strftime("%Y%m%dT%H%M%S_%f_BJT")


def beijing_date_name(timestamp_micros: int) -> str:
    """Return the Beijing calendar date used to group exported images."""
    dt_utc = datetime.fromtimestamp(timestamp_micros / 1_000_000, tz=timezone.utc)
    return dt_utc.astimezone(BEIJING_TIMEZONE).strftime("%Y%m%d")


def open_video(video_path: Path):
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    return capture


def resize_like(image: np.ndarray, reference: np.ndarray) -> np.ndarray:
    if image.shape[:2] == reference.shape[:2]:
        return image
    return cv2.resize(image, (reference.shape[1], reference.shape[0]), interpolation=cv2.INTER_LANCZOS4)


def fuse_rgb_and_rds(rgb: np.ndarray, hdmap: np.ndarray) -> np.ndarray:
    """Alpha-composite non-black RDS pixels over RGB."""
    hdmap = resize_like(hdmap, rgb)
    strength = hdmap.max(axis=2).astype(np.float32)
    alpha = np.clip((strength - 8.0) / 24.0, 0.0, 1.0) * 0.85
    alpha = alpha[..., None]
    return np.clip(rgb * (1.0 - alpha) + hdmap * alpha, 0, 255).astype(np.uint8)


def save_visualizations(
    rds_root: Path,
    render_root: Path,
    image_root: Path,
    clip_id: str,
    location: str,
    cameras,
    timestamps_by_camera,
    camera_model: str,
    save_every: int,
    max_saved_frames: int,
):
    image_root.mkdir(parents=True, exist_ok=True)
    for camera_name in cameras:
        rgb_video = rds_root / f"pinhole_{camera_name}" / f"{clip_id}.mp4"
        hdmap_video = render_root / "hdmap" / f"{camera_model}_{camera_name}" / f"{clip_id}_0.mp4"
        world_video = (
            render_root
            / "world_scenario"
            / f"{camera_model}_{camera_name}"
            / f"{clip_id}_0.mp4"
        )
        timestamps = timestamps_by_camera[camera_name]
        captures = {
            "rgb": open_video(rgb_video),
            "hdmap": open_video(hdmap_video),
            "world": open_video(world_video),
        }
        saved = 0
        try:
            for frame_index, timestamp in enumerate(timestamps):
                decoded = {}
                for name, capture in captures.items():
                    ok, decoded[name] = capture.read()
                    if not ok:
                        raise IndexError(
                            f"Could not read aligned frame {frame_index} from {name} video "
                            f"for {camera_name}"
                        )
                if frame_index % save_every != 0:
                    continue
                if max_saved_frames > 0 and saved >= max_saved_frames:
                    break

                rgb = decoded["rgb"]
                hdmap = resize_like(decoded["hdmap"], rgb)
                world = resize_like(decoded["world"], rgb)
                fusion = fuse_rgb_and_rds(rgb, hdmap)
                four_grid = np.vstack(
                    [np.hstack([rgb, hdmap]), np.hstack([world, fusion])]
                )

                stem = "_".join(
                    [
                        sanitize_name(location),
                        timestamp_name(timestamp),
                        sanitize_name(camera_name),
                    ]
                )
                # RGB, RDS HD map, and fusion remain visible in the four-grid,
                # but are not duplicated as standalone JPEGs.
                outputs = {
                    "world_scenario": world,
                    "four_grid": four_grid,
                }
                frame_root = (
                    image_root
                    / sanitize_name(location)
                    / beijing_date_name(timestamp)
                    / sanitize_name(camera_name)
                )
                for suffix, image in outputs.items():
                    output_dir = frame_root / suffix
                    output_dir.mkdir(parents=True, exist_ok=True)
                    output_path = output_dir / f"{stem}_{suffix}.jpg"
                    if not cv2.imwrite(str(output_path), image, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                        raise RuntimeError(f"Failed to save image: {output_path}")
                saved += 1
        finally:
            for capture in captures.values():
                capture.release()
        print(f"Saved {saved} visualization frame(s) for {camera_name} to {image_root}")


def remove_rendered_intermediates(
    render_root: Path,
    clip_id: str,
    cameras,
    camera_model: str,
):
    """Remove per-clip render videos after the requested JPEGs are saved."""
    removed = 0
    for render_name in ("hdmap", "world_scenario"):
        for camera_name in cameras:
            camera_dir = render_root / render_name / f"{camera_model}_{camera_name}"
            for video_path in camera_dir.glob(f"{clip_id}_*.mp4"):
                video_path.unlink()
                removed += 1
    print(f"Removed {removed} rendered intermediate video(s) for {clip_id}")


def run_pipeline(
    db_path,
    sensor_root,
    map_root,
    output_root,
    config_path,
    camera_channel,
    all_cameras,
    start_index,
    frame_count,
    map_radius,
    camera_model,
    save_every,
    max_saved_frames,
    render_images=True,
):
    """Run one nuPlan clip through conversion and optional visualization."""
    db_path = Path(db_path)
    sensor_root = Path(sensor_root)
    map_root = Path(map_root)
    output_root = Path(output_root)
    config_path = Path(config_path)
    output_root.mkdir(parents=True, exist_ok=True)
    rds_root = output_root / "rds"
    render_root = output_root / "rendered"
    image_root = output_root / "images"
    camera_channels = NUPLAN_CAMERA_CHANNELS if all_cameras else (camera_channel.upper(),)

    clip_id = convert_clip(
        db_path=db_path,
        sensor_root=sensor_root,
        map_root=map_root,
        output_root=rds_root,
        camera_channels=camera_channels,
        start_index=start_index,
        frame_count=frame_count,
        map_radius=map_radius,
        write_rgb=True,
    )

    result = {
        "clip_id": clip_id,
        "db_path": str(db_path),
        "start_index": start_index,
        "frame_count": frame_count,
        "rds_root": str(rds_root),
    }
    if not render_images:
        print(f"RDS data: {rds_root}")
        return result

    settings = copy.deepcopy(json.loads(config_path.read_text()))
    settings["CAMERAS"] = [channel.lower() for channel in camera_channels]
    settings["NOT_POST_TRAINING"].update(
        {
            "TARGET_RENDER_FPS": 10,
            "TARGET_CHUNK_FRAME": frame_count,
            "OVERLAP_FRAME": 0,
            "MAX_CHUNK": 1,
        }
    )
    resize_resolution = tuple(settings["NOT_POST_TRAINING"]["RESIZE_RESOLUTION"])
    cosmos_resolution = tuple(settings["COSMOS_RESOLUTION"])

    render_sample_hdmap(
        str(rds_root), str(render_root), clip_id, settings, camera_model,
        False, None, resize_resolution, cosmos_resolution
    )
    render_sample_world_scenario(
        str(rds_root), str(render_root), clip_id, settings, camera_model,
        False, None, resize_resolution, cosmos_resolution
    )

    manifest = json.loads((rds_root / f"{clip_id}.manifest.json").read_text())
    connection = sqlite3.connect(db_path)
    log_row = connection.execute("SELECT location, map_version FROM log LIMIT 1").fetchone()
    connection.close()
    location = log_row[0] or log_row[1] or "unknown_location"
    save_visualizations(
        rds_root=rds_root,
        render_root=render_root,
        image_root=image_root,
        clip_id=clip_id,
        location=location,
        cameras=settings["CAMERAS"],
        timestamps_by_camera=manifest["image_timestamps_by_camera"],
        camera_model=camera_model,
        save_every=save_every,
        max_saved_frames=max_saved_frames,
    )
    remove_rendered_intermediates(
        render_root=render_root,
        clip_id=clip_id,
        cameras=settings["CAMERAS"],
        camera_model=camera_model,
    )
    print(f"RDS data: {rds_root}")
    print(f"Images: {image_root}")
    result.update(
        {
            "image_root": str(image_root),
        }
    )
    return result


@click.command()
@click.option("--db_path", type=click.Path(path_type=Path, exists=True), required=True)
@click.option("--sensor_root", type=click.Path(path_type=Path, exists=True), required=True)
@click.option("--map_root", type=click.Path(path_type=Path, exists=True), required=True)
@click.option("--output_root", type=click.Path(path_type=Path), required=True)
@click.option("--config_path", type=click.Path(path_type=Path, exists=True), default=DEFAULT_CONFIG, show_default=True)
@click.option("--camera_channel", default="CAM_F0", show_default=True)
@click.option("--all_cameras", is_flag=True, help="Process all eight nuPlan cameras")
@click.option("--start_index", type=click.IntRange(min=0), default=0, show_default=True)
@click.option("--frame_count", type=click.IntRange(min=1), default=200, show_default=True)
@click.option("--map_radius", type=float, default=100.0, show_default=True)
@click.option("--camera_model", type=click.Choice(["opencv", "pinhole"]), default="opencv", show_default=True)
@click.option("--save_every", type=click.IntRange(min=1), default=1, show_default=True)
@click.option("--max_saved_frames", type=click.IntRange(min=0), default=0, show_default=True,
              help="Maximum images per camera; 0 saves every selected frame")
@click.option("--render_images/--convert_only", default=True, show_default=True)
def main(**kwargs):
    """Run nuPlan -> RDS -> rendered images as one reproducible pipeline."""
    run_pipeline(**kwargs)


if __name__ == "__main__":
    main()
