#!/usr/bin/env python3
"""Restartable Waymo TFRecord -> per-clip RDS and multiview conversion."""

from __future__ import annotations

import argparse
import concurrent.futures
import fcntl
import json
import os
import queue
import shutil
import subprocess
import sys
import types
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLKIT_ROOT = REPO_ROOT / "cosmos-drive-dreams-toolkits"
DEFAULT_INPUT_ROOT = REPO_ROOT / "data" / "waymo"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "waymo"
DEFAULT_WAYMO_PYTHON = Path(os.environ.get("WAYMO_PYTHON", sys.executable))
DEFAULT_RENDER_PYTHON = Path(os.environ.get("RENDER_PYTHON", sys.executable))
CAMERAS = ("front", "front_left", "front_right", "side_left", "side_right")
RDS_TAR_FOLDERS = (
    "3d_lanes",
    "3d_lanelines",
    "3d_road_boundaries",
    "3d_crosswalks",
    "pinhole_intrinsic",
    "pose",
    "vehicle_pose",
    "timestamp",
    "all_object_info",
    "lidar_raw",
)
MULTIVIEW_FOLDERS = ("videos", "hdmap", "lidar")
PREVIEW_VERSION = "rgb-screen-v2\n"


def clip_id_from_path(path: Path) -> str:
    return path.stem.removeprefix("segment-").removesuffix("_with_camera_labels")


def clip_root(output_root: Path, clip_id: str) -> Path:
    return output_root / clip_id


def nonempty(path: Path, minimum_size: int = 1) -> bool:
    return path.is_file() and path.stat().st_size >= minimum_size


def rds_outputs_valid(package_root: Path, clip_id: str) -> bool:
    root = package_root / "rds"
    tar_ok = all(nonempty(root / folder / f"{clip_id}.tar", 1024) for folder in RDS_TAR_FOLDERS)
    video_ok = all(nonempty(root / f"pinhole_{camera}" / f"{clip_id}.mp4", 1024) for camera in CAMERAS)
    return tar_ok and video_ok


def multiview_outputs_valid(package_root: Path, clip_id: str) -> bool:
    root = package_root / "multiview"
    filename = f"{clip_id}_0.mp4"
    return all(
        nonempty(root / modality / f"pinhole_{camera}" / filename, 1024)
        for modality in MULTIVIEW_FOLDERS
        for camera in CAMERAS
    )


def preview_outputs_valid(package_root: Path) -> bool:
    return all(nonempty(package_root / "preview" / f"pinhole_{camera}_four_grid.mp4", 1024) for camera in CAMERAS)


def write_marker(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("ok\n", encoding="utf-8")


def acquire_batch_lock(output_root: Path):
    """Prevent two batch drivers from writing the same package concurrently."""
    lock_path = output_root / "_batch" / "batch.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_file = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        lock_file.seek(0)
        owner = lock_file.read().strip() or "unknown"
        lock_file.close()
        raise RuntimeError(f"Another batch process is already running (PID {owner})") from error
    lock_file.seek(0)
    lock_file.truncate()
    lock_file.write(f"{os.getpid()}\n")
    lock_file.flush()
    return lock_file


def load_patched_waymo_frame_utils() -> None:
    """Patch Waymo 1.6.x bytearray/protobuf-4 incompatibility in memory."""
    import waymo_open_dataset.utils as waymo_utils
    from waymo_open_dataset.utils import frame_utils as installed_frame_utils

    source_path = Path(installed_frame_utils.__file__)
    source = source_path.read_text(encoding="utf-8").replace("bytearray(", "bytes(")
    module_name = "waymo_open_dataset.utils.frame_utils"
    patched = types.ModuleType(module_name)
    patched.__file__ = str(source_path)
    exec(compile(source, str(source_path), "exec"), patched.__dict__)
    sys.modules[module_name] = patched
    waymo_utils.frame_utils = patched


def run_rds_worker(tfrecord: Path, output_root: Path) -> None:
    clip_id = clip_id_from_path(tfrecord)
    package_root = clip_root(output_root, clip_id)
    rds_root = package_root / "rds"
    marker = package_root / ".state" / "rds.done"
    rds_root.mkdir(parents=True, exist_ok=True)
    if rds_outputs_valid(package_root, clip_id):
        write_marker(marker)
        print(f"SKIP {clip_id}: RDS outputs validated")
        return

    load_patched_waymo_frame_utils()
    sys.path.insert(0, str(TOOLKIT_ROOT))
    os.chdir(TOOLKIT_ROOT)
    import tensorflow as tf
    from convert_waymo_to_rds_hq import (
        convert_waymo_bbox,
        convert_waymo_hdmap,
        convert_waymo_image,
        convert_waymo_intrinsics,
        convert_waymo_lidar,
        convert_waymo_pose,
        convert_waymo_timestamp,
    )

    dataset = tf.data.TFRecordDataset(str(tfrecord), compression_type="")
    convert_waymo_hdmap(rds_root, clip_id, dataset)
    convert_waymo_intrinsics(rds_root, clip_id, dataset)
    convert_waymo_pose(rds_root, clip_id, dataset)
    convert_waymo_timestamp(rds_root, clip_id, dataset)
    convert_waymo_bbox(rds_root, clip_id, dataset)
    convert_waymo_image(rds_root, clip_id, dataset, single_camera=False)
    convert_waymo_lidar(rds_root, clip_id, dataset)

    if not rds_outputs_valid(package_root, clip_id):
        raise RuntimeError(f"RDS output validation failed for {clip_id}")
    write_marker(marker)
    print(f"DONE {clip_id}")


def run_logged(command: list[str], log_path: Path, env: dict[str, str], cwd: Path, append: bool = False) -> bool:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a" if append else "w", encoding="utf-8") as log_file:
        process = subprocess.run(
            command,
            cwd=cwd,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
    return process.returncode == 0


def create_four_grid(package_root: Path, clip_id: str, camera: str, log_path: Path) -> bool:
    """Create RGB/LiDAR/HDMap/fused 2x2 video for one camera."""
    multiview = package_root / "multiview"
    filename = f"{clip_id}_0.mp4"
    rgb = multiview / "videos" / f"pinhole_{camera}" / filename
    lidar = multiview / "lidar" / f"pinhole_{camera}" / filename
    hdmap = multiview / "hdmap" / f"pinhole_{camera}" / filename
    output = package_root / "preview" / f"pinhole_{camera}_four_grid.mp4"
    temporary_output = output.with_name(f".{output.stem}.tmp.mp4")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output.unlink(missing_ok=True)

    filter_graph = (
        "[0:v]setpts=PTS-STARTPTS,split=2[rgb_tile][rgb_fuse_raw];"
        "[1:v]setpts=PTS-STARTPTS,split=2[lidar_tile][lidar_fuse_raw];"
        "[2:v]setpts=PTS-STARTPTS,split=2[map_tile][map_fuse_raw];"
        "[rgb_fuse_raw]format=gbrp[rgb_fuse];"
        "[lidar_fuse_raw]format=gbrp[lidar_fuse];"
        "[map_fuse_raw]format=gbrp[map_fuse];"
        "[rgb_fuse][lidar_fuse]blend=all_mode=screen:all_opacity=0.60[rgb_lidar];"
        "[rgb_lidar][map_fuse]blend=all_mode=screen:all_opacity=0.75[fused];"
        "[rgb_tile]scale=640:352,drawtext=text=RGB:x=18:y=18:fontsize=28:fontcolor=white:box=1:boxcolor=black@0.55[q0];"
        "[lidar_tile]scale=640:352,drawtext=text=LIDAR:x=18:y=18:fontsize=28:fontcolor=white:box=1:boxcolor=black@0.55[q1];"
        "[map_tile]scale=640:352,drawtext=text=HDMAP:x=18:y=18:fontsize=28:fontcolor=white:box=1:boxcolor=black@0.55[q2];"
        "[fused]scale=640:352,drawtext=text=FUSION:x=18:y=18:fontsize=28:fontcolor=white:box=1:boxcolor=black@0.55[q3];"
        "[q0][q1][q2][q3]xstack=inputs=4:layout=0_0|640_0|0_352|640_352[v]"
    )
    command = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(rgb),
        "-i",
        str(lidar),
        "-i",
        str(hdmap),
        "-filter_complex",
        filter_graph,
        "-map",
        "[v]",
        "-r",
        "10",
        "-c:v",
        "libx264",
        "-crf",
        "18",
        "-preset",
        "slow",
        "-pix_fmt",
        "yuv420p",
        str(temporary_output),
    ]
    if not run_logged(command, log_path, os.environ.copy(), REPO_ROOT, append=True):
        return False
    if not nonempty(temporary_output, 1024):
        return False
    os.replace(temporary_output, output)
    return True


def create_all_previews(package_root: Path, clip_id: str, log_path: Path) -> bool:
    marker = package_root / ".state" / "preview.done"
    current_version = marker.read_text(encoding="utf-8") if marker.is_file() else ""
    if preview_outputs_valid(package_root) and current_version == PREVIEW_VERSION:
        return True
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("in-progress\n", encoding="utf-8")
    ok = True
    for camera in CAMERAS:
        output = package_root / "preview" / f"pinhole_{camera}_four_grid.mp4"
        ok = create_four_grid(package_root, clip_id, camera, log_path) and ok
    if ok and preview_outputs_valid(package_root):
        marker.write_text(PREVIEW_VERSION, encoding="utf-8")
        return True
    return False


def run_rds_stage(args: argparse.Namespace, tfrecords: list[Path]) -> list[dict[str, str]]:
    failures: list[dict[str, str]] = []

    def task(path: Path) -> tuple[Path, bool, str]:
        clip_id = clip_id_from_path(path)
        package_root = clip_root(args.output_root, clip_id)
        if rds_outputs_valid(package_root, clip_id):
            write_marker(package_root / ".state" / "rds.done")
            return path, True, "already complete"
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ""
        env["TF_CPP_MIN_LOG_LEVEL"] = "2"
        command = [
            str(args.waymo_python),
            str(Path(__file__).resolve()),
            "--_rds-worker",
            str(path),
            "--output-root",
            str(args.output_root),
        ]
        log_path = package_root / "logs" / "rds.log"
        return path, run_logged(command, log_path, env, REPO_ROOT), str(log_path)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.rds_workers) as executor:
        futures = {executor.submit(task, path): path for path in tfrecords}
        for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
            path = futures[future]
            try:
                _, ok, detail = future.result()
            except Exception as error:
                ok, detail = False, repr(error)
            status = "OK" if ok else "FAIL"
            print(f"[RDS {index}/{len(tfrecords)}] {status} {clip_id_from_path(path)}", flush=True)
            if not ok:
                failures.append({"stage": "rds", "file": str(path), "detail": detail})
    return failures


def run_multiview_stage(args: argparse.Namespace, tfrecords: list[Path], render: bool = True) -> list[dict[str, str]]:
    failures: list[dict[str, str]] = []
    validator = rds_outputs_valid if render else multiview_outputs_valid
    ready = [
        path
        for path in tfrecords
        if validator(clip_root(args.output_root, clip_id_from_path(path)), clip_id_from_path(path))
    ]
    gpu_pool: queue.Queue[int] = queue.Queue()
    for gpu_id in range(args.gpu_count):
        gpu_pool.put(gpu_id)

    def task(path: Path) -> tuple[Path, bool, str]:
        clip_id = clip_id_from_path(path)
        package_root = clip_root(args.output_root, clip_id)
        multiview_root = package_root / "multiview"
        log_path = package_root / "logs" / "multiview.log"

        if render and not multiview_outputs_valid(package_root, clip_id):
            gpu_id = gpu_pool.get()
            env = os.environ.copy()
            env["USE_RAY"] = "False"
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            command = [
                str(args.render_python),
                str(TOOLKIT_ROOT / "render_from_rds_hq.py"),
                "-d",
                "waymo_mv_short",
                "-i",
                str(package_root / "rds"),
                "-o",
                str(multiview_root),
                "-c",
                "pinhole",
                "-p",
                "True",
                "-cj",
                clip_id,
                "-s",
                "world_scenario",
            ]
            try:
                render_ok = run_logged(command, log_path, env, TOOLKIT_ROOT)
            finally:
                gpu_pool.put(gpu_id)
            if not render_ok or not multiview_outputs_valid(package_root, clip_id):
                return path, False, str(log_path)

        if not multiview_outputs_valid(package_root, clip_id):
            return path, False, "multiview inputs missing"
        write_marker(package_root / ".state" / "multiview.done")
        preview_ok = create_all_previews(package_root, clip_id, log_path)
        return path, preview_ok, str(log_path)

    worker_count = min(args.render_workers, args.gpu_count) if render else args.preview_workers
    with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {executor.submit(task, path): path for path in ready}
        for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
            path = futures[future]
            try:
                _, ok, detail = future.result()
            except Exception as error:
                ok, detail = False, repr(error)
            label = "MV" if render else "PREVIEW"
            status = "OK" if ok else "FAIL"
            print(f"[{label} {index}/{len(ready)}] {status} {clip_id_from_path(path)}", flush=True)
            if not ok:
                failures.append({"stage": label.lower(), "file": str(path), "detail": detail})
    return failures


def move_file(source: Path, destination: Path) -> None:
    if not source.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.stat().st_size == source.stat().st_size:
            source.unlink()
            return
        destination.unlink()
    shutil.move(str(source), str(destination))


def migrate_old_layout(output_root: Path, tfrecords: list[Path]) -> None:
    """Move outputs from the former modality-first layout into per-clip folders."""
    for index, tfrecord in enumerate(tfrecords, 1):
        clip_id = clip_id_from_path(tfrecord)
        package_root = clip_root(output_root, clip_id)
        for folder in RDS_TAR_FOLDERS:
            move_file(output_root / "rds" / folder / f"{clip_id}.tar", package_root / "rds" / folder / f"{clip_id}.tar")
        for camera in CAMERAS:
            move_file(
                output_root / "rds" / f"pinhole_{camera}" / f"{clip_id}.mp4",
                package_root / "rds" / f"pinhole_{camera}" / f"{clip_id}.mp4",
            )
        for modality in MULTIVIEW_FOLDERS:
            for camera in CAMERAS:
                filename = f"{clip_id}_0.mp4"
                move_file(
                    output_root / "multiview" / modality / f"pinhole_{camera}" / filename,
                    package_root / "multiview" / modality / f"pinhole_{camera}" / filename,
                )
        move_file(output_root / "logs" / "rds" / f"{clip_id}.log", package_root / "logs" / "rds.log")
        move_file(output_root / "logs" / "multiview" / f"{clip_id}.log", package_root / "logs" / "multiview.log")
        move_file(output_root / ".state" / "rds" / f"{clip_id}.done", package_root / ".state" / "rds.done")
        move_file(
            output_root / ".state" / "multiview" / f"{clip_id}.done",
            package_root / ".state" / "multiview.done",
        )
        if rds_outputs_valid(package_root, clip_id):
            write_marker(package_root / ".state" / "rds.done")
        if multiview_outputs_valid(package_root, clip_id):
            write_marker(package_root / ".state" / "multiview.done")
        print(f"[MIGRATE {index}/{len(tfrecords)}] {clip_id}", flush=True)

    for old_root in (output_root / "rds", output_root / "multiview", output_root / "logs", output_root / ".state"):
        if old_root.exists():
            for directory in sorted((p for p in old_root.rglob("*") if p.is_dir()), reverse=True):
                try:
                    directory.rmdir()
                except OSError:
                    pass
            try:
                old_root.rmdir()
            except OSError:
                pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--stage", choices=("all", "rds", "multiview", "preview", "migrate"), default="all")
    parser.add_argument("--rds-workers", type=int, default=12)
    parser.add_argument("--render-workers", type=int, default=4)
    parser.add_argument("--preview-workers", type=int, default=4)
    parser.add_argument("--gpu-count", type=int, default=4)
    parser.add_argument("--waymo-python", type=Path, default=DEFAULT_WAYMO_PYTHON)
    parser.add_argument("--render-python", type=Path, default=DEFAULT_RENDER_PYTHON)
    parser.add_argument("--_rds-worker", type=Path, help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.input_root = args.input_root.resolve()
    args.output_root = args.output_root.resolve()
    if args._rds_worker is not None:
        run_rds_worker(args._rds_worker.resolve(), args.output_root)
        return 0

    tfrecords = sorted(args.input_root.glob("*.tfrecord"))
    if not tfrecords:
        raise FileNotFoundError(f"No TFRecord files found in {args.input_root}")
    if min(args.rds_workers, args.render_workers, args.preview_workers, args.gpu_count) < 1:
        raise ValueError("Worker and GPU counts must be positive")

    args.output_root.mkdir(parents=True, exist_ok=True)
    _batch_lock = acquire_batch_lock(args.output_root)
    print(f"Found {len(tfrecords)} TFRecords in {args.input_root}")
    print(f"Per-clip output root: {args.output_root}")
    if args.stage == "migrate":
        migrate_old_layout(args.output_root, tfrecords)
        return 0

    failures: list[dict[str, str]] = []
    if args.stage in ("all", "rds"):
        failures.extend(run_rds_stage(args, tfrecords))
    if args.stage in ("all", "multiview"):
        failures.extend(run_multiview_stage(args, tfrecords, render=True))
    if args.stage == "preview":
        failures.extend(run_multiview_stage(args, tfrecords, render=False))

    report_dir = args.output_root / "_batch"
    report_dir.mkdir(parents=True, exist_ok=True)
    failure_path = report_dir / "failures.json"
    failure_path.write_text(json.dumps(failures, indent=2), encoding="utf-8")
    print(f"Finished with {len(failures)} failures; report: {failure_path}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
