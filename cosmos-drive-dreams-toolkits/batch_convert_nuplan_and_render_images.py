"""Parallel whole-dataset nuPlan -> RDS conversion and visualization."""

import hashlib
import json
import logging
import multiprocessing
import os
import sqlite3
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import click
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm


TOOLKIT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = TOOLKIT_ROOT / "config" / "dataset_nuplan_8cam.json"
PIPELINE_VERSION = "nuplan-world-scenario-v3-pr46"
DEFAULT_WORKERS = max(1, min(24, (os.cpu_count() or 4) // 4))
ALL_CAMERAS = (
    "CAM_F0",
    "CAM_L0",
    "CAM_L1",
    "CAM_L2",
    "CAM_B0",
    "CAM_R2",
    "CAM_R1",
    "CAM_R0",
)


def configure_batch_logger(output_root: Path):
    """Log batch-level events to both the terminal and a persistent file."""
    logger = logging.getLogger("nuplan_batch")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(output_root / "batch.log")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger


def get_camera_frame_count(db_path: Path, camera_channel: str) -> int:
    connection = sqlite3.connect(db_path)
    try:
        row = connection.execute(
            """
            SELECT COUNT(*)
            FROM image
            JOIN camera ON image.camera_token = camera.token
            WHERE camera.channel = ?
            """,
            (camera_channel,),
        ).fetchone()
        return int(row[0])
    finally:
        connection.close()


def discover_databases(db_root: Path):
    candidates = [db_root] if db_root.is_file() else sorted(db_root.rglob("*.db"))
    # The mini/trainval trees can contain the same log. Clip IDs are based on
    # the DB stem, so process each log only once.
    unique = {}
    duplicates = []
    for db_path in candidates:
        if db_path.stem in unique:
            duplicates.append(db_path)
        else:
            unique[db_path.stem] = db_path
    return list(unique.values()), duplicates


def task_id(db_path: Path, start_index: int, frame_count: int) -> str:
    digest = hashlib.sha1(str(db_path).encode("utf-8")).hexdigest()[:8]
    return f"{db_path.stem.replace('.', '_')}_{digest}_{start_index:06d}_{frame_count:06d}"


def write_json_atomic(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def run_task(task):
    """Worker entry point; imports the rendering stack after thread limits."""
    threads = str(task["threads_per_worker"])
    os.environ["OMP_NUM_THREADS"] = threads
    os.environ["MKL_NUM_THREADS"] = threads
    os.environ["OPENBLAS_NUM_THREADS"] = threads
    os.environ["NUMEXPR_NUM_THREADS"] = threads
    os.environ["RDS_FFMPEG_THREADS"] = threads
    os.environ["USE_RAY"] = "False"

    from convert_nuplan_and_render_images import run_pipeline

    try:
        import cv2
        import torch

        cv2.setNumThreads(task["threads_per_worker"])
        torch.set_num_threads(task["threads_per_worker"])
    except (ImportError, RuntimeError):
        pass

    status_path = Path(task["status_path"])
    log_path = Path(task["log_path"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    status = {
        "task_id": task["task_id"],
        "db_path": task["db_path"],
        "start_index": task["start_index"],
        "frame_count": task["frame_count"],
        "task_options": {
            "pipeline_version": PIPELINE_VERSION,
            "camera_channel": task["camera_channel"],
            "all_cameras": task["all_cameras"],
            "camera_model": task["camera_model"],
            "config_path": task["config_path"],
            "render_images": task["render_images"],
            "save_every": task["save_every"],
            "max_saved_frames": task["max_saved_frames"],
        },
        "status": "running",
    }
    write_json_atomic(status_path, status)

    try:
        with log_path.open("w", buffering=1) as log_file, redirect_stdout(log_file), redirect_stderr(log_file):
            result = run_pipeline(
                db_path=task["db_path"],
                sensor_root=task["sensor_root"],
                map_root=task["map_root"],
                output_root=task["output_root"],
                config_path=task["config_path"],
                camera_channel=task["camera_channel"],
                all_cameras=task["all_cameras"],
                start_index=task["start_index"],
                frame_count=task["frame_count"],
                map_radius=task["map_radius"],
                camera_model=task["camera_model"],
                save_every=task["save_every"],
                max_saved_frames=task["max_saved_frames"],
                render_images=task["render_images"],
            )
        status.update(
            {
                "status": "success",
                "elapsed_seconds": round(time.time() - started, 3),
                "result": result,
                "log_path": str(log_path),
            }
        )
    except Exception as error:
        with log_path.open("a") as log_file:
            log_file.write("\n" + traceback.format_exc())
        status.update(
            {
                "status": "failed",
                "elapsed_seconds": round(time.time() - started, 3),
                "error": repr(error),
                "log_path": str(log_path),
            }
        )
    write_json_atomic(status_path, status)
    return status


def build_tasks(
    databases,
    output_root,
    sensor_root,
    map_root,
    config_path,
    camera_channel,
    all_cameras,
    clip_frames,
    map_radius,
    camera_model,
    render_images,
    threads_per_worker,
):
    tasks = []
    planning_errors = []
    source_frame_count = 0
    status_root = output_root / "status"
    log_root = output_root / "logs"
    for db_path in databases:
        try:
            frame_count = get_camera_frame_count(db_path, camera_channel)
        except Exception as error:
            planning_errors.append({"db_path": str(db_path), "error": repr(error)})
            continue
        source_frame_count += frame_count
        for start_index in range(0, frame_count, clip_frames):
            current_frames = min(clip_frames, frame_count - start_index)
            current_task_id = task_id(db_path, start_index, current_frames)
            tasks.append(
                {
                    "task_id": current_task_id,
                    "db_path": str(db_path),
                    "sensor_root": str(sensor_root),
                    "map_root": str(map_root),
                    "output_root": str(output_root),
                    "config_path": str(config_path),
                    "camera_channel": camera_channel,
                    "all_cameras": all_cameras,
                    "start_index": start_index,
                    "frame_count": current_frames,
                    "map_radius": map_radius,
                    "camera_model": camera_model,
                    # Whole-dataset mode is deliberately lossless at the
                    # source-camera rate: every frame gets every image type.
                    "save_every": 1,
                    "max_saved_frames": 0,
                    "render_images": render_images,
                    "threads_per_worker": threads_per_worker,
                    "status_path": str(status_root / f"{current_task_id}.json"),
                    "log_path": str(log_root / f"{current_task_id}.log"),
                }
            )
    return tasks, planning_errors, source_frame_count


@click.command()
@click.option("--db_root", type=click.Path(path_type=Path, exists=True), required=True,
              help="A nuPlan DB file or a directory recursively containing .db files")
@click.option("--sensor_root", type=click.Path(path_type=Path, exists=True), required=True)
@click.option("--map_root", type=click.Path(path_type=Path, exists=True), required=True)
@click.option("--output_root", type=click.Path(path_type=Path), required=True)
@click.option("--config_path", type=click.Path(path_type=Path, exists=True), default=DEFAULT_CONFIG, show_default=True)
@click.option("--camera_channel", default="CAM_F0", show_default=True,
              help="Reference camera and the only camera when --single_camera is used")
@click.option("--all_cameras/--single_camera", default=True, show_default=True)
@click.option("--clip_frames", type=click.IntRange(min=1), default=200, show_default=True)
@click.option("--map_radius", type=float, default=100.0, show_default=True)
@click.option("--camera_model", type=click.Choice(["opencv", "pinhole"]), default="opencv", show_default=True)
@click.option("--render_images/--convert_only", default=True, show_default=True)
@click.option("--workers", type=click.IntRange(min=1),
              default=DEFAULT_WORKERS, show_default=True)
@click.option("--threads_per_worker", type=click.IntRange(min=1), default=2, show_default=True)
@click.option("--max_in_flight", type=click.IntRange(min=0), default=0, show_default=True,
              help="Queued/running tasks; 0 uses twice the worker count")
@click.option("--resume/--no_resume", default=True, show_default=True)
@click.option("--max_databases", type=click.IntRange(min=0), default=0, show_default=True,
              help="Debug limit; 0 scans every database")
@click.option("--max_tasks", type=click.IntRange(min=0), default=0, show_default=True,
              help="Debug limit; 0 runs every planned clip")
@click.option("--dry_run", is_flag=True, help="Only enumerate and report tasks")
def main(
    db_root,
    sensor_root,
    map_root,
    output_root,
    config_path,
    camera_channel,
    all_cameras,
    clip_frames,
    map_radius,
    camera_model,
    render_images,
    workers,
    threads_per_worker,
    max_in_flight,
    resume,
    max_databases,
    max_tasks,
    dry_run,
):
    """Convert every nuPlan DB into restartable RDS clips in parallel."""
    output_root.mkdir(parents=True, exist_ok=True)
    logger = configure_batch_logger(output_root)
    camera_channel = camera_channel.upper()
    if camera_channel not in ALL_CAMERAS:
        raise click.BadParameter(f"Unknown nuPlan camera: {camera_channel}", param_hint="camera_channel")
    if max_in_flight == 0:
        max_in_flight = workers * 2
    databases, duplicates = discover_databases(db_root)
    if max_databases > 0:
        databases = databases[:max_databases]
    tasks, planning_errors, source_frame_count = build_tasks(
        databases=databases,
        output_root=output_root,
        sensor_root=sensor_root,
        map_root=map_root,
        config_path=config_path,
        camera_channel=camera_channel,
        all_cameras=all_cameras,
        clip_frames=clip_frames,
        map_radius=map_radius,
        camera_model=camera_model,
        render_images=render_images,
        threads_per_worker=threads_per_worker,
    )
    if max_tasks > 0:
        tasks = tasks[:max_tasks]
    planned_frame_count = sum(task["frame_count"] for task in tasks)

    skipped = []
    pending = []
    for task in tasks:
        status_path = Path(task["status_path"])
        previous = json.loads(status_path.read_text()) if status_path.exists() else None
        expected_options = {
            "pipeline_version": PIPELINE_VERSION,
            "camera_channel": task["camera_channel"],
            "all_cameras": task["all_cameras"],
            "camera_model": task["camera_model"],
            "config_path": task["config_path"],
            "render_images": task["render_images"],
            "save_every": task["save_every"],
            "max_saved_frames": task["max_saved_frames"],
        }
        previous_options = previous.get("task_options", {}) if previous else {}
        # A rendered result also satisfies convert-only, but an RDS-only result
        # must not suppress a later rendering pass.
        options_match = all(
            previous_options.get(key) == value
            for key, value in expected_options.items()
            if key != "render_images"
        ) and (
            previous_options.get("render_images") is True
            or previous_options.get("render_images") == expected_options["render_images"]
        )
        if resume and previous and previous.get("status") == "success" and options_match:
            skipped.append(task["task_id"])
        else:
            pending.append(task)

    plan = {
        "database_count": len(databases),
        "duplicate_database_count": len(duplicates),
        "planned_task_count": len(tasks),
        "pending_task_count": len(pending),
        "resume_skipped_count": len(skipped),
        "planning_errors": planning_errors,
        "source_frame_count": source_frame_count,
        "planned_frame_count": planned_frame_count,
        "frame_coverage_complete": (
            max_tasks == 0
            and not planning_errors
            and planned_frame_count == source_frame_count
        ),
        "workers": workers,
        "threads_per_worker": threads_per_worker,
        "max_in_flight": max_in_flight,
        "all_cameras": all_cameras,
        "render_images": render_images,
        "export_all_frames": True,
        "clip_frames": clip_frames,
        "pipeline_version": PIPELINE_VERSION,
    }
    write_json_atomic(output_root / "batch_plan.json", plan)
    print(json.dumps(plan, indent=2))
    logger.info(
        "Planned %d clips covering %d/%d source frames; %d pending, %d resume-skipped",
        len(tasks),
        planned_frame_count,
        source_frame_count,
        len(pending),
        len(skipped),
    )
    if duplicates:
        logger.info("Ignored %d duplicate database paths", len(duplicates))
    if planning_errors:
        logger.warning("Could not plan %d database(s); see batch_plan.json", len(planning_errors))
    if dry_run or not pending:
        logger.info("No workers started (%s)", "dry run" if dry_run else "nothing pending")
        return

    results = []
    success_count = 0
    failed_count = 0
    progress_path = output_root / "batch_progress.json"
    write_json_atomic(
        progress_path,
        {
            "total": len(pending),
            "completed": 0,
            "running_or_queued": len(pending),
            "in_flight": min(len(pending), max_in_flight),
            "success": 0,
            "failed": 0,
        },
    )
    logger.info(
        "Starting %d worker process(es), %d thread(s) per worker",
        workers,
        threads_per_worker,
    )
    spawn_context = multiprocessing.get_context("spawn")
    with logging_redirect_tqdm(), ProcessPoolExecutor(
        max_workers=workers, mp_context=spawn_context
    ) as executor:
        task_iterator = iter(pending)
        future_to_task = {}

        def fill_submission_window():
            while len(future_to_task) < max_in_flight:
                try:
                    task = next(task_iterator)
                except StopIteration:
                    break
                future_to_task[executor.submit(run_task, task)] = task

        fill_submission_window()
        with tqdm(
            total=len(pending),
            desc="nuPlan clips",
            unit="clip",
            dynamic_ncols=True,
        ) as progress_bar:
            completed = 0
            while future_to_task:
                done, _ = wait(tuple(future_to_task), return_when=FIRST_COMPLETED)
                for future in done:
                    completed += 1
                    task = future_to_task.pop(future)
                    try:
                        result = future.result()
                    except Exception as error:
                        result = {
                            "task_id": task["task_id"],
                            "status": "worker_crashed",
                            "error": repr(error),
                        }
                    results.append(result)
                    if result["status"] == "success":
                        success_count += 1
                    else:
                        failed_count += 1
                    progress_bar.set_postfix(
                        success=success_count,
                        failed=failed_count,
                        refresh=False,
                    )
                    progress_bar.update(1)
                    logger.info(
                        "[%d/%d] %s: %s",
                        completed,
                        len(pending),
                        result["status"],
                        result["task_id"],
                    )
                    write_json_atomic(
                        progress_path,
                        {
                            "total": len(pending),
                            "completed": completed,
                            "running_or_queued": len(pending) - completed,
                            "in_flight": len(future_to_task),
                            "success": success_count,
                            "failed": failed_count,
                            "last_task_id": result["task_id"],
                            "last_status": result["status"],
                        },
                    )
                fill_submission_window()

    summary = {
        **plan,
        "success_count": sum(result["status"] == "success" for result in results),
        "failed_count": sum(result["status"] != "success" for result in results),
        "results": results,
    }
    write_json_atomic(output_root / "batch_summary.json", summary)
    logger.info(
        "Batch finished: %d success, %d failed; summary=%s",
        summary["success_count"],
        summary["failed_count"],
        output_root / "batch_summary.json",
    )
    if summary["failed_count"]:
        raise click.ClickException(
            f"{summary['failed_count']} task(s) failed; see batch_summary.json and logs/"
        )


if __name__ == "__main__":
    main()
