import os
import time
import queue
import json
import logging
import shutil
import threading
from typing import Optional
from PIL import Image

# Import dependencies
from scripts.config import resolve_paths, TEMP_OUTPUT_DIR
from scripts.utils import get_unique_filename
from scripts.qc import run_technical_qc
from scripts.upscaler import run_upscale

logger = logging.getLogger("AdobeStockUpscaler.Processor")

class FileItem:
    def __init__(self, file_id: str, name: str, size: int, width: int, height: int, temp_path: str):
        self.id = file_id
        self.name = name
        self.size = size
        self.width = width
        self.height = height
        self.megapixels = round((width * height) / 1_000_000.0, 2)
        self.temp_path = temp_path
        self.status = "ready"  # ready, uploading, uploaded, queued, processing, completed, failed, cancelled
        self.output_width = 0
        self.output_height = 0
        self.output_megapixels = 0.0
        self.qc = {
            "passed": True,
            "hard_failures": [],
            "warnings": [],
            "checks": {
                "resolution": "pass",
                "format": "pass",
                "integrity": "pass",
                "aspect_ratio": "pass",
                "transparency": "pass",
                "size": "pass"
            }
        }
        self.error_stage = ""
        self.error_reason = ""
        self.error_details = ""
        self.processing_seconds = 0.0

files_state = {}
task_queue = queue.Queue()
cancel_requested = False
gpu_inference_lock = threading.Lock()
metrics_lock = threading.Lock()

progress_metrics = {
    "total": 0,
    "completed": 0,
    "failed": 0,
    "processing": False,
    "current_file": "",
    "current_file_id": "",
    "percentage": 0,
    "eta_seconds": None,
    "processing_speed": 0.0
}

processing_durations = []

def get_progress_metrics():
    with metrics_lock:
        return dict(progress_metrics)

def get_files_state():
    return files_state

def get_task_queue():
    return task_queue

def set_cancel_requested(val: bool):
    global cancel_requested
    cancel_requested = val

def retry_failed_items() -> int:
    """
    Re-enqueues all failed file items for reprocessing.
    """
    failed_ids = [fid for fid, item in files_state.items() if item.status == "failed"]
    if not failed_ids:
        return 0

    for fid in failed_ids:
        files_state[fid].status = "ready"
        files_state[fid].error_stage = ""
        files_state[fid].error_reason = ""
        files_state[fid].error_details = ""

    return len(failed_ids)

def process_single_file(
    file_id: str,
    scale_factor: int = 4,
    output_format: str = "jpg",
    jpeg_quality: int = 95,
    model: str = "RealESRGAN_x4plus",
    target_width: Optional[int] = None,
    target_height: Optional[int] = None
) -> dict:
    """
    Process a single image item through Real-ESRGAN and Technical QC.
    Protected with thread-safe GPU locking and accurate metric tracking.
    """
    file_item: Optional[FileItem] = files_state.get(file_id)
    if not file_item:
        logger.error(f"File ID '{file_id}' not found in state.")
        return {
            "status": "failed",
            "error_stage": "Initialization",
            "error_reason": "File ID not found",
            "error_details": f"File item '{file_id}' not registered in backend state."
        }

    file_item.status = "processing"
    with metrics_lock:
        progress_metrics["processing"] = True
        progress_metrics["current_file"] = file_item.name
        progress_metrics["current_file_id"] = file_item.id

    paths = resolve_paths()
    active_scale = scale_factor
    if target_width and target_height and file_item.width > 0 and file_item.height > 0:
        scale_w = target_width / file_item.width
        scale_h = target_height / file_item.height
        active_scale = max(scale_w, scale_h)
        active_scale = max(active_scale, 1.0)

    active_scale = round(active_scale, 2)
    temp_output_path = os.path.join(TEMP_OUTPUT_DIR, f"{file_id}_out.{output_format}")
    img_start_time = time.time()

    # GPU / Inference execution protected by lock
    with gpu_inference_lock:
        logger.info(f"[Inference Start] Processing '{file_item.name}' (ID: {file_id}, Scale: {active_scale}x, Model: {model})")
        success, stage, reason, details = run_upscale(
            input_path=file_item.temp_path,
            output_path=temp_output_path,
            scale=active_scale,
            model_name=model,
            ext=output_format,
            quality=jpeg_quality
        )

    img_duration = round(time.time() - img_start_time, 2)
    file_item.processing_seconds = img_duration

    if success and os.path.exists(temp_output_path):
        # Execute Technical Quality Control
        qc_results = run_technical_qc(
            temp_output_path,
            file_item.width,
            file_item.height,
            output_format
        )
        file_item.qc = qc_results
        file_item.output_width = qc_results["output"]["width"]
        file_item.output_height = qc_results["output"]["height"]
        file_item.output_megapixels = qc_results["megapixels"]

        if not qc_results["passed"]:
            file_item.status = "failed"
            file_item.error_stage = "Quality Control (QC)"
            file_item.error_reason = f"QC Failed ({', '.join(qc_results['hard_failures'])})"
            file_item.error_details = json.dumps(qc_results, indent=2)
            dest_path = os.path.join(paths["failed"], file_item.name)
            shutil.move(temp_output_path, dest_path)

            with metrics_lock:
                progress_metrics["failed"] += 1
        else:
            file_item.status = "completed"
            file_item.error_stage = ""
            file_item.error_reason = ""
            file_item.error_details = ""
            unique_name = get_unique_filename(paths["output"], file_item.name, output_format)
            dest_path = os.path.join(paths["output"], unique_name)
            file_item.name = unique_name
            shutil.move(temp_output_path, dest_path)

            with metrics_lock:
                progress_metrics["completed"] += 1
    else:
        file_item.status = "failed"
        file_item.error_stage = stage or "Real-ESRGAN"
        file_item.error_reason = reason or "AI Inference Error"
        file_item.error_details = details or "Execution terminated unexpectedly"

        with metrics_lock:
            progress_metrics["failed"] += 1

    # Update processing performance metrics
    with metrics_lock:
        processing_durations.append(img_duration)
        avg_speed = sum(processing_durations) / len(processing_durations)
        progress_metrics["processing_speed"] = round(avg_speed, 2)
        total_items = progress_metrics.get("total", 0)
        done_items = progress_metrics["completed"] + progress_metrics["failed"]
        if total_items > 0:
            remaining = max(0, total_items - done_items)
            progress_metrics["eta_seconds"] = int(remaining * avg_speed)
            progress_metrics["percentage"] = int((done_items / total_items) * 100)

        # Check if all currently tracked are done
        if total_items > 0 and done_items >= total_items:
            progress_metrics["processing"] = False
            progress_metrics["current_file"] = ""
            progress_metrics["current_file_id"] = ""

    # Append to run log
    try:
        log_filename = f"upscale_run_{time.strftime('%Y-%m-%d')}.json"
        log_path = os.path.join(paths["logs"], log_filename)
        log_item = {
            "filename": file_item.name,
            "file_id": file_item.id,
            "input": f"{file_item.width}x{file_item.height}",
            "output": f"{file_item.output_width}x{file_item.output_height}",
            "megapixels": file_item.output_megapixels,
            "processing_seconds": file_item.processing_seconds,
            "model": model,
            "scale": active_scale,
            "status": file_item.status,
            "qc": file_item.qc,
            "timestamp": time.strftime('%Y-%m-%dT%H:%M:%S')
        }
        if file_item.error_reason:
            log_item["error"] = {
                "stage": file_item.error_stage,
                "reason": file_item.error_reason,
                "details": file_item.error_details
            }
        
        current_logs = []
        if os.path.exists(log_path):
            try:
                with open(log_path, "r", encoding="utf-8") as lf:
                    current_logs = json.load(lf)
            except Exception:
                current_logs = []
        current_logs.append(log_item)
        with open(log_path, "w", encoding="utf-8") as lf:
            json.dump(current_logs, lf, indent=2)
    except Exception as log_err:
        logger.warning(f"Failed logging file run: {str(log_err)}")

    return {
        "status": file_item.status,
        "file_id": file_item.id,
        "name": file_item.name,
        "processing_seconds": file_item.processing_seconds,
        "output": {
            "width": file_item.output_width,
            "height": file_item.output_height,
            "megapixels": file_item.output_megapixels
        } if file_item.status == "completed" else None,
        "qc": file_item.qc,
        "error_stage": file_item.error_stage,
        "error_reason": file_item.error_reason,
        "error_details": file_item.error_details
    }

def batch_worker():
    """
    Background batch processor queue worker.
    """
    global cancel_requested

    while True:
        try:
            task = task_queue.get()
            if task is None:
                break

            file_ids = task["file_ids"]
            scale_factor = task["scale_factor"]
            output_format = task["output_format"]
            jpeg_quality = task["jpeg_quality"]
            model = task["model"]
            target_width = task["target_width"]
            target_height = task["target_height"]

            logger.info(f"Worker picked up batch containing {len(file_ids)} tasks.")

            cancel_requested = False
            with metrics_lock:
                progress_metrics["processing"] = True
                progress_metrics["total"] = len(file_ids)
                progress_metrics["completed"] = 0
                progress_metrics["failed"] = 0
                progress_metrics["percentage"] = 0
                progress_metrics["eta_seconds"] = None

            for idx, file_id in enumerate(file_ids):
                if cancel_requested:
                    logger.info("Batch run cancelled by user.")
                    for rem_id in file_ids[idx:]:
                        if rem_id in files_state:
                            files_state[rem_id].status = "cancelled"
                    break

                process_single_file(
                    file_id=file_id,
                    scale_factor=scale_factor,
                    output_format=output_format,
                    jpeg_quality=jpeg_quality,
                    model=model,
                    target_width=target_width,
                    target_height=target_height
                )

            with metrics_lock:
                progress_metrics["processing"] = False
                progress_metrics["current_file"] = ""
                progress_metrics["current_file_id"] = ""

            task_queue.task_done()
        except Exception as e:
            logger.error(f"Worker loop exception: {str(e)}")
            with metrics_lock:
                progress_metrics["processing"] = False

worker_thread = threading.Thread(target=batch_worker, daemon=True)
worker_thread.start()
