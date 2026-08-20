"""
scripts/batch_processor.py — Adobe Stock AI Studio

Sequential batch processing pipeline:
  Upload ALL → User clicks Start → Process ONE BY ONE:
  Image N → Upscale → Save → Ollama Analysis → Metadata → Next Image

Protected with:
- T4 VRAM lock (one GPU op at a time)
- Thread-safe state management
- Per-stage error recovery (upscale fail ≠ metadata fail)
- Sequential filename assignment: stock_image_up1.jpg, stock_image_up2.jpg, ...
"""

import csv
import io
import json
import logging
import os
import queue
import shutil
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from scripts.config import (
    CSV_COLUMNS,
    CSV_FILENAME,
    JSON_FILENAME,
    MAX_KEYWORDS,
    TEMP_OUTPUT_DIR,
    resolve_paths,
)
from scripts.qc import run_technical_qc
from scripts.upscaler import run_upscale
from scripts.utils import get_unique_output_filename

logger = logging.getLogger("AdobeStockStudio.Processor")

# ──────────────────────────────────────────────
# FileItem — per-image state
# ──────────────────────────────────────────────
class FileItem:
    def __init__(
        self,
        file_id: str,
        original_name: str,
        size: int,
        width: int,
        height: int,
        temp_path: str,
        upload_index: int,
    ):
        self.id = file_id
        self.original_name = original_name  # user's original filename
        self.output_name = ""               # stock_image_upN.ext — assigned at processing start
        self.size = size
        self.width = width
        self.height = height
        self.megapixels = round((width * height) / 1_000_000.0, 2)
        self.temp_path = temp_path
        self.upload_index = upload_index    # Order uploaded

        # Status: uploading→uploaded→queued→upscaling→analyzing→completed|failed
        self.status = "uploaded"

        # Output
        self.output_path = ""
        self.output_width = 0
        self.output_height = 0
        self.output_megapixels = 0.0

        # Per-stage status
        self.upscale_status = "pending"   # pending | running | done | failed
        self.upscale_progress = 0         # 0-100
        self.metadata_status = "pending"  # pending | running | done | failed
        self.metadata_progress = 0        # 0-100

        # QC
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
                "size": "pass",
            },
        }

        # Metadata (populated after Ollama analysis)
        self.metadata = {
            "title": "",
            "keywords": [],
            "category": 22,
            "releases": "",
        }

        # Error details per stage
        self.error_stage = ""    # "upscale" | "metadata"
        self.error_reason = ""
        self.error_details = ""

        self.processing_seconds = 0.0
        self.completed_at = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "original_name": self.original_name,
            "output_name": self.output_name,
            "size": self.size,
            "width": self.width,
            "height": self.height,
            "megapixels": self.megapixels,
            "status": self.status,
            "output_path": self.output_path,
            "output_width": self.output_width,
            "output_height": self.output_height,
            "output_megapixels": self.output_megapixels,
            "upscale_status": self.upscale_status,
            "upscale_progress": self.upscale_progress,
            "metadata_status": self.metadata_status,
            "metadata_progress": self.metadata_progress,
            "qc": self.qc,
            "metadata": self.metadata,
            "error_stage": self.error_stage,
            "error_reason": self.error_reason,
            "error_details": self.error_details,
            "processing_seconds": self.processing_seconds,
            "completed_at": self.completed_at,
            "upload_index": self.upload_index,
        }


# ──────────────────────────────────────────────
# Global shared state
# ──────────────────────────────────────────────
files_state: dict[str, FileItem] = {}
task_queue: queue.Queue = queue.Queue()

# Sequential output filename counter (thread-safe)
_output_counter_lock = threading.Lock()
_output_counter = 1

# Processing control
cancel_requested = False
gpu_inference_lock = threading.Lock()
metrics_lock = threading.Lock()
_worker_thread: Optional[threading.Thread] = None
_worker_running = False

# Master metadata store {output_name: {title, keywords, category, releases}}
metadata_store: dict[str, dict] = {}
metadata_lock = threading.Lock()

# Progress metrics
progress_metrics = {
    "total": 0,
    "uploaded": 0,
    "queued": 0,
    "processing_count": 0,
    "completed": 0,
    "failed": 0,
    "processing": False,
    "current_file": "",
    "current_file_id": "",
    "current_upscale_progress": 0,
    "current_metadata_progress": 0,
    "percentage": 0,
    "eta_seconds": None,
    "processing_speed": 0.0,
    "processing_index": 0,
}

processing_durations: list[float] = []

# SSE event queue for real-time push
sse_event_queue: queue.Queue = queue.Queue(maxsize=500)

# Log store
log_entries: list[dict] = []
log_lock = threading.Lock()


# ──────────────────────────────────────────────
# Logging helpers
# ──────────────────────────────────────────────
def _log(level: str, message: str, extra: dict = None):
    """Dual: Python logger + in-memory log store + SSE push."""
    entry = {
        "ts": datetime.now().strftime("%H:%M:%S"),
        "level": level.upper(),
        "message": message,
    }
    with log_lock:
        log_entries.append(entry)
        if len(log_entries) > 2000:
            log_entries.pop(0)

    # Push as SSE event
    _push_event("log", entry)

    level_map = {
        "INFO": logger.info,
        "SUCCESS": logger.info,
        "WARNING": logger.warning,
        "ERROR": logger.error,
    }
    level_map.get(level.upper(), logger.info)(message)


def get_logs() -> list[dict]:
    with log_lock:
        return list(log_entries)


def clear_logs():
    with log_lock:
        log_entries.clear()


# ──────────────────────────────────────────────
# SSE push
# ──────────────────────────────────────────────
def _push_event(event_type: str, data: dict):
    try:
        sse_event_queue.put_nowait({"type": event_type, "data": data})
    except queue.Full:
        pass  # Drop oldest-ish if full


def _push_progress():
    """Push current progress snapshot to SSE."""
    with metrics_lock:
        snap = dict(progress_metrics)
    _push_event("progress", snap)


# ──────────────────────────────────────────────
# State accessors
# ──────────────────────────────────────────────
def get_progress_metrics() -> dict:
    with metrics_lock:
        return dict(progress_metrics)


def get_files_state() -> dict:
    return {fid: item.to_dict() for fid, item in files_state.items()}


def set_cancel_requested(val: bool):
    global cancel_requested
    cancel_requested = val
    if val:
        _log("WARNING", "Cancellation requested. Current image will finish.")


# ──────────────────────────────────────────────
# Output filename assignment
# ──────────────────────────────────────────────
def _assign_output_filename(file_item: FileItem, output_dir: str) -> str:
    """
    Assign a unique stock_image_upN.ext filename.
    Collision-safe sequential counter.
    """
    global _output_counter
    ext = Path(file_item.original_name).suffix.lstrip(".").lower() or "jpg"
    if ext not in ("jpg", "jpeg", "png", "webp"):
        ext = "jpg"
    if ext == "jpeg":
        ext = "jpg"

    with _output_counter_lock:
        _, idx = get_unique_output_filename(output_dir, _output_counter, ext)
        _output_counter = idx + 1
        name = f"stock_image_up{idx}.{ext}"

    file_item.output_name = name
    return name


# ──────────────────────────────────────────────
# Metadata file writers
# ──────────────────────────────────────────────
def _write_master_metadata():
    """Write both AdobeStock_Metadata.json and AdobeStock_Metadata.csv."""
    paths = resolve_paths()
    metadata_dir = paths["metadata"]
    os.makedirs(metadata_dir, exist_ok=True)

    with metadata_lock:
        store_snapshot = dict(metadata_store)

    # JSON
    json_path = os.path.join(metadata_dir, JSON_FILENAME)
    try:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(store_snapshot, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Failed to write JSON: {e}")

    # CSV (UTF-8-SIG for Excel compatibility)
    csv_path = os.path.join(metadata_dir, CSV_FILENAME)
    try:
        with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
            writer.writerow(CSV_COLUMNS)
            for filename, meta in store_snapshot.items():
                keywords_str = ", ".join(meta.get("keywords", [])[:MAX_KEYWORDS])
                writer.writerow([
                    filename,
                    meta.get("title", ""),
                    keywords_str,
                    meta.get("category", 22),
                    meta.get("releases", ""),
                ])
    except Exception as e:
        logger.error(f"Failed to write CSV: {e}")

    return json_path, csv_path


def update_metadata_entry(filename: str, title: str, keywords: list,
                           category: int, releases: str):
    """Update a single metadata entry and rewrite master files."""
    with metadata_lock:
        metadata_store[filename] = {
            "title": title[:200],
            "keywords": keywords[:MAX_KEYWORDS],
            "category": category,
            "releases": releases,
        }
    # Also update the FileItem
    for item in files_state.values():
        if item.output_name == filename:
            item.metadata = metadata_store[filename]
            break
    _write_master_metadata()
    _log("SUCCESS", f"Metadata updated for {filename}")


def get_master_metadata() -> dict:
    with metadata_lock:
        return dict(metadata_store)


def get_csv_path() -> str:
    paths = resolve_paths()
    return os.path.join(paths["metadata"], CSV_FILENAME)


def get_json_path() -> str:
    paths = resolve_paths()
    return os.path.join(paths["metadata"], JSON_FILENAME)


# ──────────────────────────────────────────────
# Retry helpers
# ──────────────────────────────────────────────
def retry_failed_items() -> int:
    """Re-enqueue all failed items (both stages)."""
    failed_ids = [
        fid for fid, item in files_state.items()
        if item.status == "failed"
    ]
    for fid in failed_ids:
        item = files_state[fid]
        item.status = "queued"
        item.error_stage = ""
        item.error_reason = ""
        item.error_details = ""
        item.upscale_status = "pending"
        item.metadata_status = "pending"
        task_queue.put(fid)
    _log("INFO", f"Re-enqueued {len(failed_ids)} failed items.")
    return len(failed_ids)


def retry_metadata_only(file_id: str) -> bool:
    """
    Retry ONLY the metadata stage for an image that upscaled OK
    but whose metadata generation failed.
    """
    item = files_state.get(file_id)
    if not item:
        return False
    if item.upscale_status != "done":
        _log("WARNING", f"Cannot retry metadata: upscale not completed for {file_id}")
        return False
    if not item.output_path or not os.path.exists(item.output_path):
        _log("ERROR", f"Cannot retry metadata: output file missing for {item.output_name}")
        return False

    item.status = "analyzing"
    item.metadata_status = "pending"
    item.error_stage = ""
    item.error_reason = ""

    def _do_retry():
        _run_metadata_stage(item)

    t = threading.Thread(target=_do_retry, daemon=True)
    t.start()
    return True


# ──────────────────────────────────────────────
# Processing stages
# ──────────────────────────────────────────────
def _run_upscale_stage(
    item: FileItem,
    scale: float,
    output_format: str,
    jpeg_quality: int,
    model: str,
    output_dir: str,
) -> bool:
    """Run Real-ESRGAN upscaling for one image."""
    item.status = "upscaling"
    item.upscale_status = "running"
    item.upscale_progress = 0

    _push_event("image_status", item.to_dict())
    _log("INFO", f"Upscaling {item.output_name} (original: {item.original_name})")

    ext = Path(item.original_name).suffix.lstrip(".").lower() or output_format
    if ext == "jpeg":
        ext = "jpg"
    if ext not in ("jpg", "png", "webp"):
        ext = output_format

    output_path = os.path.join(output_dir, item.output_name)

    def _progress_cb(pct: int):
        item.upscale_progress = pct
        _push_event("upscale_progress", {"file_id": item.id, "progress": pct})

    with gpu_inference_lock:
        success = run_upscale(
            input_path=item.temp_path,
            output_path=output_path,
            scale=scale,
            model_name=model,
            ext=ext,
            quality=jpeg_quality,
            progress_callback=_progress_cb,
        )

    if success and os.path.exists(output_path):
        item.output_path = output_path
        item.upscale_status = "done"
        item.upscale_progress = 100

        # Read output dimensions
        try:
            from PIL import Image as PILImage
            with PILImage.open(output_path) as img:
                item.output_width, item.output_height = img.size
                item.output_megapixels = round(
                    (item.output_width * item.output_height) / 1_000_000.0, 2
                )
        except Exception:
            pass

        _log("SUCCESS", f"Upscale complete: {item.output_name} ({item.output_megapixels} MP)")
        return True
    else:
        item.upscale_status = "failed"
        item.upscale_progress = 0
        item.error_stage = "upscale"
        item.error_reason = "Real-ESRGAN returned no output file"
        _log("ERROR", f"Upscale FAILED for {item.output_name}: {item.error_reason}")
        return False


def _run_qc_stage(item: FileItem, output_format: str):
    """Run QC on upscaled image."""
    if not item.output_path:
        return
    try:
        qc_result = run_technical_qc(
            output_path=item.output_path,
            original_w=item.width,
            original_h=item.height,
            req_format=output_format,
        )
        item.qc = qc_result
        if qc_result.get("hard_failures"):
            _log("WARNING", f"QC warnings for {item.output_name}: {qc_result['hard_failures']}")
        else:
            _log("INFO", f"QC passed for {item.output_name} ({item.output_megapixels} MP)")
    except Exception as e:
        _log("WARNING", f"QC error for {item.output_name}: {e}")


def _run_metadata_stage(item: FileItem):
    """Run Ollama vision analysis and generate Adobe Stock metadata."""
    item.status = "analyzing"
    item.metadata_status = "running"
    item.metadata_progress = 0

    _push_event("image_status", item.to_dict())
    _log("INFO", f"Ollama analysis started: {item.output_name}")

    try:
        from scripts.ollama_vision import analyze_image, get_ollama_status

        ollama_st = get_ollama_status()
        if not ollama_st["ready"]:
            item.metadata_status = "failed"
            item.error_stage = "metadata"
            item.error_reason = f"Ollama not ready: {ollama_st['error']}"
            _log("ERROR", f"Metadata FAILED for {item.output_name}: {item.error_reason}")
            return False

        item.metadata_progress = 20
        _push_event("metadata_progress", {"file_id": item.id, "progress": 20})

        meta = analyze_image(item.output_path)

        item.metadata_progress = 80
        _push_event("metadata_progress", {"file_id": item.id, "progress": 80})

        if meta.get("error"):
            item.metadata_status = "failed"
            item.error_stage = "metadata"
            item.error_reason = "Ollama analysis error"
            item.error_details = meta["error"]
            _log("ERROR", f"Metadata FAILED for {item.output_name}: {meta['error']}")
            return False

        # Store metadata on item
        item.metadata = {
            "title": meta.get("title", ""),
            "keywords": meta.get("keywords", [])[:MAX_KEYWORDS],
            "category": meta.get("category", 22),
            "releases": "",  # Always empty default
        }

        # Accumulate into master store
        with metadata_lock:
            metadata_store[item.output_name] = item.metadata

        # Write master files
        _write_master_metadata()

        item.metadata_status = "done"
        item.metadata_progress = 100
        _push_event("metadata_progress", {"file_id": item.id, "progress": 100})
        _log("SUCCESS", f"Metadata generated for {item.output_name}: \"{meta.get('title', '')[:60]}...\"")
        return True

    except Exception as e:
        item.metadata_status = "failed"
        item.error_stage = "metadata"
        item.error_reason = str(e)
        _log("ERROR", f"Metadata exception for {item.output_name}: {e}")
        return False


def _copy_to_drive(item: FileItem):
    """Copy completed output to Google Drive persistent storage."""
    try:
        paths = resolve_paths()
        dst = os.path.join(paths["output"], item.output_name)
        if item.output_path and os.path.exists(item.output_path):
            shutil.copy2(item.output_path, dst)
            _log("INFO", f"Saved to Drive: {dst}")
    except Exception as e:
        _log("WARNING", f"Drive copy failed for {item.output_name}: {e}")


# ──────────────────────────────────────────────
# Single-image pipeline
# ──────────────────────────────────────────────
def process_single_file(
    file_id: str,
    scale_factor: int = 4,
    output_format: str = "jpg",
    jpeg_quality: int = 95,
    model: str = "RealESRGAN_x4plus",
) -> dict:
    """
    Full sequential pipeline for one image:
    1. Assign standardized output filename
    2. Upscale
    3. QC
    4. Drive copy (temp)
    5. Ollama metadata
    6. Update master files
    7. Mark completed or failed

    Even if metadata fails, queue CONTINUES.
    """
    item: Optional[FileItem] = files_state.get(file_id)
    if not item:
        logger.error(f"FileItem {file_id} not found in state")
        return {"status": "error", "reason": "File not found"}

    paths = resolve_paths()
    output_dir = paths["output"]
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(TEMP_OUTPUT_DIR, exist_ok=True)

    start_time = time.time()

    # ── Assign output filename ─────────────────────────────────────
    if not item.output_name:
        _assign_output_filename(item, output_dir)

    _log("INFO", f"Processing {item.output_name} (original: {item.original_name})")

    # ── Stage 1: Upscale ──────────────────────────────────────────
    upscale_ok = _run_upscale_stage(
        item=item,
        scale=float(scale_factor),
        output_format=output_format,
        jpeg_quality=jpeg_quality,
        model=model,
        output_dir=output_dir,
    )

    if not upscale_ok:
        item.status = "failed"
        item.processing_seconds = round(time.time() - start_time, 1)
        _push_event("image_status", item.to_dict())
        return {"status": "failed", "stage": "upscale"}

    # ── Stage 2: QC ───────────────────────────────────────────────
    _run_qc_stage(item, output_format)

    # ── Stage 3: Drive persistence ────────────────────────────────
    _copy_to_drive(item)

    # ── Stage 4: Metadata ─────────────────────────────────────────
    meta_ok = _run_metadata_stage(item)

    # ── Finalize ──────────────────────────────────────────────────
    item.processing_seconds = round(time.time() - start_time, 1)
    item.completed_at = datetime.now().isoformat()

    if upscale_ok:
        # Completed even if metadata failed (metadata can be retried)
        item.status = "completed"
    else:
        item.status = "failed"

    _push_event("image_complete", item.to_dict())
    _log(
        "SUCCESS" if item.status == "completed" else "ERROR",
        f"{'✓' if item.status == 'completed' else '✗'} {item.output_name} "
        f"({'metadata failed' if not meta_ok else 'all done'}, "
        f"{item.processing_seconds}s)"
    )

    return {"status": item.status}


# ──────────────────────────────────────────────
# Background worker
# ──────────────────────────────────────────────
def _worker(
    scale_factor: int,
    output_format: str,
    jpeg_quality: int,
    model: str,
):
    """
    Sequential queue worker.
    Processes ONE image at a time. Never concurrent.
    """
    global cancel_requested, _worker_running

    _worker_running = True
    _log("INFO", "Processing queue started.")
    _push_event("queue_started", {})

    queued_ids = []
    while not task_queue.empty():
        try:
            fid = task_queue.get_nowait()
            queued_ids.append(fid)
        except queue.Empty:
            break

    total = len(queued_ids)
    completed = 0
    failed = 0

    with metrics_lock:
        progress_metrics["processing"] = True
        progress_metrics["total"] = total

    for i, file_id in enumerate(queued_ids):
        if cancel_requested:
            _log("WARNING", "Processing cancelled by user.")
            break

        item = files_state.get(file_id)
        if not item:
            continue

        item.status = "queued"

        with metrics_lock:
            progress_metrics["current_file"] = item.original_name
            progress_metrics["current_file_id"] = file_id
            progress_metrics["processing_index"] = i + 1
            progress_metrics["queued"] = total - i - 1
            progress_metrics["processing_count"] = 1

        _push_progress()

        result = process_single_file(
            file_id=file_id,
            scale_factor=scale_factor,
            output_format=output_format,
            jpeg_quality=jpeg_quality,
            model=model,
        )

        if result.get("status") == "completed":
            completed += 1
            processing_durations.append(files_state[file_id].processing_seconds)
        else:
            failed += 1

        with metrics_lock:
            progress_metrics["completed"] = completed
            progress_metrics["failed"] = failed
            progress_metrics["percentage"] = round((completed + failed) / total * 100)

            # ETA calculation
            if processing_durations:
                avg_speed = sum(processing_durations) / len(processing_durations)
                remaining = total - (completed + failed)
                progress_metrics["eta_seconds"] = avg_speed * remaining
                progress_metrics["processing_speed"] = avg_speed

        _push_progress()

    # Done
    with metrics_lock:
        progress_metrics["processing"] = False
        progress_metrics["current_file"] = ""
        progress_metrics["current_file_id"] = ""
        progress_metrics["processing_count"] = 0

    _worker_running = False
    _push_event("queue_complete", {
        "total": total,
        "completed": completed,
        "failed": failed,
    })
    _log("SUCCESS", f"Queue complete. {completed}/{total} done, {failed} failed.")


def start_processing(
    file_ids: list[str],
    scale_factor: int = 4,
    output_format: str = "jpg",
    jpeg_quality: int = 95,
    model: str = "RealESRGAN_x4plus",
) -> dict:
    """
    Queue all file_ids and start sequential worker.
    Only call after ALL images are uploaded.
    """
    global _worker_thread, _worker_running, cancel_requested

    if _worker_running:
        return {"status": "error", "message": "Processing already running"}

    cancel_requested = False

    # Reset output counter
    global _output_counter
    with _output_counter_lock:
        # Start from current highest + 1 (safe across runs)
        paths = resolve_paths()
        output_dir = paths["output"]
        os.makedirs(output_dir, exist_ok=True)
        existing = [
            f for f in os.listdir(output_dir)
            if f.startswith("stock_image_up")
        ]
        if existing:
            nums = []
            for name in existing:
                try:
                    n = int(name.replace("stock_image_up", "").split(".")[0])
                    nums.append(n)
                except Exception:
                    pass
            _output_counter = max(nums) + 1 if nums else 1
        else:
            _output_counter = 1

    # Enqueue
    for fid in file_ids:
        if fid in files_state:
            files_state[fid].status = "queued"
            task_queue.put(fid)

    with metrics_lock:
        progress_metrics["total"] = len(file_ids)
        progress_metrics["completed"] = 0
        progress_metrics["failed"] = 0
        progress_metrics["queued"] = len(file_ids)
        progress_metrics["processing_count"] = 0
        progress_metrics["percentage"] = 0
        progress_metrics["eta_seconds"] = None
        progress_metrics["processing_speed"] = 0.0

    _log("INFO", f"Starting processing queue: {len(file_ids)} images")

    _worker_thread = threading.Thread(
        target=_worker,
        args=(scale_factor, output_format, jpeg_quality, model),
        daemon=True,
    )
    _worker_thread.start()

    return {"status": "started", "queued": len(file_ids)}


def is_processing() -> bool:
    return _worker_running
