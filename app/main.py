"""
app/main.py — Adobe Stock AI Studio
FastAPI backend with:
- Upload-first workflow
- Sequential processing queue
- SSE real-time events
- Ollama vision integration
- Master metadata (JSON + CSV)
- Built-in assistant chatbot
- ZIP / CSV / JSON export
- Google Drive persistence
"""

import asyncio
import json
import logging
import os
import re
import shutil
import sys
import time
import uuid
import zipfile
from datetime import datetime
from typing import AsyncGenerator, List, Optional

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel

# ── Backward-compat fix for basicsr / torchvision ─────────────────────────
try:
    import torchvision.transforms.functional as _F
    sys.modules["torchvision.transforms.functional_tensor"] = _F
except Exception:
    pass

# ── Local modules ──────────────────────────────────────────────────────────
from scripts.batch_processor import (
    FileItem,
    clear_logs,
    files_state,
    get_csv_path,
    get_files_state,
    get_json_path,
    get_logs,
    get_master_metadata,
    get_progress_metrics,
    is_processing,
    retry_failed_items,
    retry_metadata_only,
    set_cancel_requested,
    sse_event_queue,
    start_processing,
    update_metadata_entry,
)
from scripts.config import (
    CSV_FILENAME,
    DEFAULT_FORMAT,
    DEFAULT_SCALE,
    JPEG_QUALITY,
    JSON_FILENAME,
    MAX_UPLOAD_SIZE_MB,
    MODEL_NAME,
    TEMP_INPUT_DIR,
    ensure_dirs,
    resolve_paths,
)
from scripts.ollama_vision import get_ollama_status, initialize_ollama
from scripts.utils import get_system_resources

# ──────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("AdobeStockStudio")

# ──────────────────────────────────────────────────────────────────────────
app = FastAPI(title="Adobe Stock AI Studio API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure all directories exist at startup
ensure_dirs()

# Global: total upload target (set by client before uploading)
_expected_upload_count: int = 0
_upload_index: int = 0
_upload_lock = __import__("threading").Lock()


# ──────────────────────────────────────────────────────────────────────────
# Static file serving (frontend)
# ──────────────────────────────────────────────────────────────────────────
_frontend_dir = os.path.join(os.path.dirname(__file__))

@app.on_event("startup")
async def startup_event():
    logger.info("Adobe Stock AI Studio starting up...")
    ensure_dirs()
    # Initialize Ollama in background thread
    import threading
    t = threading.Thread(target=_ollama_init_worker, daemon=True)
    t.start()


def _ollama_init_worker():
    logger.info("Initializing Ollama in background...")
    ok = initialize_ollama()
    if ok:
        logger.info("Ollama initialization complete.")
    else:
        logger.warning("Ollama initialization failed. Metadata generation unavailable.")


# ──────────────────────────────────────────────────────────────────────────
# Pydantic models
# ──────────────────────────────────────────────────────────────────────────
class StartProcessRequest(BaseModel):
    file_ids: List[str]
    upscale_factor: int = DEFAULT_SCALE
    output_format: str = DEFAULT_FORMAT
    jpeg_quality: int = JPEG_QUALITY
    model: str = MODEL_NAME


class MetadataUpdateRequest(BaseModel):
    title: str
    keywords: List[str]
    category: int
    releases: str = ""


class AssistantRequest(BaseModel):
    message: str


class ExpectedCountRequest(BaseModel):
    count: int


# ──────────────────────────────────────────────────────────────────────────
# Health
# ──────────────────────────────────────────────────────────────────────────
@app.get("/api/health")
async def health():
    try:
        resources = get_system_resources()
        ollama = get_ollama_status()
        drive_path = "/content/drive/MyDrive"
        drive_mounted = os.path.exists(drive_path)

        return {
            "status": "ok",
            "gpu": resources.get("gpu", False),
            "gpu_name": resources.get("gpu_name", "None"),
            "ram_usage": resources.get("ram_usage", {}),
            "vram_usage": resources.get("vram_usage", {}),
            "ollama_ready": ollama.get("ready", False),
            "ollama_model": ollama.get("model", ""),
            "ollama_error": ollama.get("error", ""),
            "ollama_error_code": ollama.get("error_code", ""),
            "vision_tested": ollama.get("vision_tested", False),
            "drive_mounted": drive_mounted,
            "fastapi_status": "ok",
        }
    except Exception as e:
        logger.error(f"Health check error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ──────────────────────────────────────────────────────────────────────────
# Upload workflow
# ──────────────────────────────────────────────────────────────────────────
@app.post("/api/upload/init")
async def init_upload(req: ExpectedCountRequest):
    """Tell the server how many files the user is about to upload."""
    global _expected_upload_count, _upload_index
    with _upload_lock:
        _expected_upload_count = req.count
        _upload_index = 0
    return {"status": "ok", "expected": req.count}


@app.post("/api/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    """
    Upload images to local storage.
    Returns file metadata for each uploaded file.
    Processing is NOT started here.
    """
    global _upload_index

    uploaded = []
    os.makedirs(TEMP_INPUT_DIR, exist_ok=True)
    ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".webp"}

    for file in files:
        filename = file.filename or "image.jpg"
        _, ext = os.path.splitext(filename.lower())

        if ext not in ALLOWED_EXT:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported format: {filename}. Allowed: JPG, JPEG, PNG, WEBP",
            )

        # Check size
        content = await file.read()
        size_mb = len(content) / (1024 * 1024)
        if size_mb > MAX_UPLOAD_SIZE_MB:
            raise HTTPException(
                status_code=413,
                detail=f"File too large: {filename} ({size_mb:.1f} MB > {MAX_UPLOAD_SIZE_MB} MB)",
            )

        file_id = str(uuid.uuid4())[:8]
        sanitized = re.sub(r"[^a-zA-Z0-9_.\-]", "_", filename)
        temp_path = os.path.join(TEMP_INPUT_DIR, f"{file_id}_{sanitized}")

        # Write to disk
        with open(temp_path, "wb") as f:
            f.write(content)

        # Read image dimensions
        try:
            with Image.open(temp_path) as img:
                width, height = img.size
        except Exception:
            os.remove(temp_path)
            raise HTTPException(status_code=400, detail=f"Cannot open image: {filename}")

        with _upload_lock:
            _upload_index += 1
            idx = _upload_index

        item = FileItem(
            file_id=file_id,
            original_name=filename,
            size=len(content),
            width=width,
            height=height,
            temp_path=temp_path,
            upload_index=idx,
        )
        item.status = "uploaded"
        files_state[file_id] = item

        from scripts.batch_processor import _log, _push_event
        _log("INFO", f"Uploaded ({idx}/{_expected_upload_count}): {filename} ({width}×{height})")
        _push_event("upload_progress", {
            "file_id": file_id,
            "filename": filename,
            "uploaded": idx,
            "total": _expected_upload_count,
        })

        uploaded.append({
            "file_id": file_id,
            "filename": filename,
            "width": width,
            "height": height,
            "size": len(content),
            "megapixels": round((width * height) / 1_000_000, 2),
            "upload_index": idx,
        })

    return {"uploaded": uploaded, "count": len(uploaded)}


# ──────────────────────────────────────────────────────────────────────────
# Processing
# ──────────────────────────────────────────────────────────────────────────
@app.post("/api/start")
async def start_batch(req: StartProcessRequest):
    """
    Start sequential processing after ALL images are uploaded.
    Client is responsible for ensuring upload is complete before calling.
    """
    if is_processing():
        raise HTTPException(status_code=409, detail="Processing already in progress")

    if not req.file_ids:
        raise HTTPException(status_code=400, detail="No file IDs provided")

    # Validate all file IDs exist
    missing = [fid for fid in req.file_ids if fid not in files_state]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown file IDs: {missing[:5]}",
        )

    result = start_processing(
        file_ids=req.file_ids,
        scale_factor=req.upscale_factor,
        output_format=req.output_format,
        jpeg_quality=req.jpeg_quality,
        model=req.model,
    )
    return result


@app.post("/api/cancel")
async def cancel_processing():
    set_cancel_requested(True)
    return {"status": "cancel_requested"}


@app.get("/api/status")
async def get_status():
    return {
        "progress": get_progress_metrics(),
        "files": get_files_state(),
        "ollama": get_ollama_status(),
    }


@app.get("/api/files")
async def get_files():
    return {"files": get_files_state()}


@app.get("/api/files/{file_id}")
async def get_file(file_id: str):
    item = files_state.get(file_id)
    if not item:
        raise HTTPException(status_code=404, detail="File not found")
    return item.to_dict()


# ──────────────────────────────────────────────────────────────────────────
# Retry
# ──────────────────────────────────────────────────────────────────────────
@app.post("/api/retry")
async def retry_all():
    """Re-enqueue all failed items (full pipeline)."""
    if is_processing():
        raise HTTPException(status_code=409, detail="Processing already running")
    count = retry_failed_items()
    if count == 0:
        return {"status": "ok", "message": "No failed items to retry"}

    # Collect re-queued IDs and restart
    fids = [
        fid for fid, item in files_state.items()
        if item.status == "queued"
    ]
    result = start_processing(file_ids=fids)
    return {"status": "restarted", "queued": count}


@app.post("/api/retry_metadata/{file_id}")
async def retry_metadata(file_id: str):
    """Retry only the metadata stage for a file whose upscale succeeded."""
    ok = retry_metadata_only(file_id)
    if not ok:
        raise HTTPException(
            status_code=400,
            detail="Cannot retry metadata: file not found, upscale not done, or output missing",
        )
    return {"status": "retrying_metadata", "file_id": file_id}


# ──────────────────────────────────────────────────────────────────────────
# Metadata
# ──────────────────────────────────────────────────────────────────────────
@app.get("/api/metadata")
async def get_metadata():
    return {"metadata": get_master_metadata()}


@app.patch("/api/metadata/{file_id}")
async def update_metadata(file_id: str, req: MetadataUpdateRequest):
    """Update metadata for a completed file. Rewrites master JSON + CSV."""
    item = files_state.get(file_id)
    if not item:
        raise HTTPException(status_code=404, detail="File not found")
    if not item.output_name:
        raise HTTPException(status_code=400, detail="File has no output name (not processed yet)")

    update_metadata_entry(
        filename=item.output_name,
        title=req.title,
        keywords=req.keywords,
        category=req.category,
        releases=req.releases,
    )
    return {"status": "updated", "filename": item.output_name}


# ──────────────────────────────────────────────────────────────────────────
# Logs
# ──────────────────────────────────────────────────────────────────────────
@app.get("/api/logs")
async def get_log_entries():
    return {"logs": get_logs()}


@app.delete("/api/logs")
async def delete_logs():
    clear_logs()
    return {"status": "cleared"}


@app.get("/api/logs/download")
async def download_logs():
    entries = get_logs()
    lines = [f"[{e['ts']}] {e['level']} — {e['message']}" for e in entries]
    content = "\n".join(lines)
    return StreamingResponse(
        iter([content]),
        media_type="text/plain",
        headers={"Content-Disposition": f"attachment; filename=studio_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"},
    )


# ──────────────────────────────────────────────────────────────────────────
# Server-Sent Events (SSE)
# ──────────────────────────────────────────────────────────────────────────
@app.get("/api/events")
async def sse_stream():
    """
    Real-time SSE endpoint.
    Pushes: upload_progress, progress, image_status, upscale_progress,
            metadata_progress, image_complete, log, queue_started, queue_complete
    """
    async def event_generator() -> AsyncGenerator[str, None]:
        # Send initial state
        yield _sse_format("connected", {"message": "Adobe Stock AI Studio connected"})
        yield _sse_format("status", {
            "progress": get_progress_metrics(),
            "ollama": get_ollama_status(),
        })

        while True:
            try:
                # Non-blocking check of event queue
                try:
                    event = sse_event_queue.get_nowait()
                    yield _sse_format(event["type"], event["data"])
                except __import__("queue").Empty:
                    # Keep-alive heartbeat every 3 seconds
                    yield ": heartbeat\n\n"
                    await asyncio.sleep(3)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"SSE generator error: {e}")
                break

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


def _sse_format(event_type: str, data: dict) -> str:
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"


# ──────────────────────────────────────────────────────────────────────────
# Built-in Assistant (deterministic, no external AI)
# ──────────────────────────────────────────────────────────────────────────
@app.post("/api/assistant")
async def assistant(req: AssistantRequest):
    """
    Local deterministic assistant powered by application state.
    No external AI API required.
    """
    answer = _assistant_respond(req.message)
    return {"answer": answer}


def _assistant_respond(message: str) -> str:
    msg = message.lower().strip()
    metrics = get_progress_metrics()
    files = files_state
    ollama = get_ollama_status()

    total = metrics.get("total", 0)
    completed = metrics.get("completed", 0)
    failed = metrics.get("failed", 0)
    queued = metrics.get("queued", 0)
    processing = metrics.get("processing", False)

    uploaded_count = sum(1 for f in files.values() if f.status in (
        "uploaded", "queued", "upscaling", "analyzing", "completed", "failed"
    ))

    # ── How many complete? ─────────────────────────────────────────
    if any(p in msg for p in ["how many", "complete", "done", "finished", "processed"]):
        if "fail" in msg:
            failed_items = [f for f in files.values() if f.status == "failed"]
            if not failed_items:
                return "No images have failed."
            names = ", ".join(f.original_name for f in failed_items[:5])
            extra = f" (showing first 5)" if len(failed_items) > 5 else ""
            return f"{len(failed_items)} image(s) failed{extra}: {names}"
        if "upload" in msg:
            return f"{uploaded_count} image(s) have been uploaded."
        return f"{completed} of {total} images are complete. {failed} failed, {queued} queued."

    # ── Status / progress ─────────────────────────────────────────
    if any(p in msg for p in ["status", "progress", "running", "processing"]):
        if processing:
            current = metrics.get("current_file", "unknown")
            pct = metrics.get("percentage", 0)
            eta = metrics.get("eta_seconds")
            eta_str = f" ETA: {int(eta//60)}m {int(eta%60)}s" if eta else ""
            return f"Processing is running. Current: {current}. Progress: {pct}%{eta_str}."
        return f"Processing is not running. {completed}/{total} complete."

    # ── Why did X fail? ───────────────────────────────────────────
    if any(p in msg for p in ["why", "fail", "error", "reason"]):
        # Try to extract image number
        nums = re.findall(r'\d+', msg)
        failed_items = [f for f in files.values() if f.status == "failed"]
        if not failed_items:
            return "No images have failed."
        if nums:
            target_idx = int(nums[0])
            # Try by upload index
            match = next((f for f in failed_items if f.upload_index == target_idx), None)
            if not match and target_idx <= len(failed_items):
                match = list(failed_items)[target_idx - 1]
            if match:
                stage = match.error_stage or "unknown stage"
                reason = match.error_reason or "unknown reason"
                detail = f" Details: {match.error_details[:100]}" if match.error_details else ""
                return (
                    f"Image '{match.original_name}' failed at {stage} stage. "
                    f"Reason: {reason}.{detail}"
                )
        # Generic: list all failed
        parts = []
        for f in failed_items[:5]:
            parts.append(f"'{f.original_name}' ({f.error_stage}: {f.error_reason})")
        return "Failed images: " + "; ".join(parts)

    # ── How many left? ────────────────────────────────────────────
    if any(p in msg for p in ["left", "remain", "queue"]):
        remaining = total - completed - failed
        return f"{remaining} image(s) remaining. {queued} queued, {completed} completed, {failed} failed."

    # ── Ollama status ─────────────────────────────────────────────
    if any(p in msg for p in ["ollama", "ai", "vision", "model", "connected"]):
        if ollama["ready"]:
            return f"Ollama is connected and ready. Active model: {ollama['model']}."
        err = ollama.get("error", "unknown error")
        return f"Ollama is NOT ready. Error: {err}"

    # ── CSV / JSON ready? ─────────────────────────────────────────
    if any(p in msg for p in ["csv", "json", "metadata", "export", "ready"]):
        meta = get_master_metadata()
        if meta:
            csv_p = get_csv_path()
            csv_ready = os.path.exists(csv_p)
            return (
                f"Metadata is available for {len(meta)} image(s). "
                f"CSV {'is ready' if csv_ready else 'not yet written'}. "
                f"Use the Export buttons to download."
            )
        return "No metadata generated yet. Process some images first."

    # ── GPU / T4 ──────────────────────────────────────────────────
    if any(p in msg for p in ["gpu", "t4", "vram", "cuda"]):
        resources = get_system_resources()
        if resources["gpu"]:
            vram = resources["vram_usage"]
            return (
                f"GPU detected: {resources['gpu_name']}. "
                f"VRAM: {vram['used']:.1f} GB used / {vram['total']:.1f} GB total."
            )
        return "No GPU detected. Running on CPU (upscaling will be slow)."

    # ── Which images failed? ───────────────────────────────────────
    if any(p in msg for p in ["which", "list failed", "list error"]):
        failed_items = [f for f in files.values() if f.status == "failed"]
        if not failed_items:
            return "No images have failed."
        names = [f"{f.upload_index}. {f.original_name} ({f.error_stage})" for f in failed_items]
        return "Failed images:\n" + "\n".join(names)

    # ── Upload complete? ──────────────────────────────────────────
    if "upload" in msg:
        return f"{uploaded_count} of {_expected_upload_count} images uploaded."

    # ── Help / default ────────────────────────────────────────────
    return (
        "I can answer questions about: image counts, failures (including why), "
        "processing status, Ollama/GPU status, CSV/JSON readiness, and upload progress. "
        f"Current: {completed}/{total} complete, {failed} failed."
    )


# ──────────────────────────────────────────────────────────────────────────
# Export
# ──────────────────────────────────────────────────────────────────────────
@app.get("/api/export/csv")
async def export_csv():
    csv_path = get_csv_path()
    if not os.path.exists(csv_path):
        raise HTTPException(status_code=404, detail="CSV not yet generated. Process images first.")
    return FileResponse(
        csv_path,
        media_type="text/csv",
        filename=CSV_FILENAME,
    )


@app.get("/api/export/json")
async def export_json():
    json_path = get_json_path()
    if not os.path.exists(json_path):
        raise HTTPException(status_code=404, detail="JSON not yet generated.")
    return FileResponse(
        json_path,
        media_type="application/json",
        filename=JSON_FILENAME,
    )


@app.get("/api/export/zip")
async def export_zip():
    """Create and return a ZIP of all successfully upscaled images."""
    paths = resolve_paths()
    output_dir = paths["output"]

    if not os.path.exists(output_dir):
        raise HTTPException(status_code=404, detail="No output directory found.")

    images = [
        f for f in os.listdir(output_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
        and f.startswith("stock_image_up")
    ]

    if not images:
        raise HTTPException(status_code=404, detail="No upscaled images found.")

    zip_name = f"AdobeStock_Upscaled_{datetime.now().strftime('%Y-%m-%d')}.zip"
    paths_arch = paths["archives"]
    os.makedirs(paths_arch, exist_ok=True)
    zip_path = os.path.join(paths_arch, zip_name)

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for img_name in images:
            img_path = os.path.join(output_dir, img_name)
            zf.write(img_path, img_name)

        # Include metadata files if available
        csv_path = get_csv_path()
        json_path = get_json_path()
        if os.path.exists(csv_path):
            zf.write(csv_path, CSV_FILENAME)
        if os.path.exists(json_path):
            zf.write(json_path, JSON_FILENAME)

    return FileResponse(
        zip_path,
        media_type="application/zip",
        filename=zip_name,
    )


# ──────────────────────────────────────────────────────────────────────────
# Image thumbnail serving
# ──────────────────────────────────────────────────────────────────────────
@app.get("/api/image/upload/{file_id}")
async def serve_upload_image(file_id: str):
    item = files_state.get(file_id)
    if not item or not os.path.exists(item.temp_path):
        raise HTTPException(status_code=404, detail="Upload image not found")
    ext = os.path.splitext(item.temp_path)[1].lower()
    media_type = "image/jpeg" if ext in (".jpg", ".jpeg") else (
        "image/png" if ext == ".png" else "image/webp"
    )
    return FileResponse(item.temp_path, media_type=media_type)


@app.get("/api/image/output/{file_id}")
async def serve_output_image(file_id: str):
    item = files_state.get(file_id)
    if not item or not item.output_path or not os.path.exists(item.output_path):
        raise HTTPException(status_code=404, detail="Output image not found")
    ext = os.path.splitext(item.output_path)[1].lower()
    media_type = "image/jpeg" if ext in (".jpg", ".jpeg") else (
        "image/png" if ext == ".png" else "image/webp"
    )
    return FileResponse(item.output_path, media_type=media_type)


# ──────────────────────────────────────────────────────────────────────────
# Frontend — serve index.html
# ──────────────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
@app.get("/{full_path:path}", response_class=HTMLResponse)
async def serve_frontend(full_path: str = ""):
    # Don't intercept API routes
    if full_path.startswith("api/"):
        raise HTTPException(status_code=404)
    index_path = os.path.join(os.path.dirname(__file__), "index.html")
    if not os.path.exists(index_path):
        return HTMLResponse("<h1>Frontend not found</h1>", status_code=404)
    with open(index_path, "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())
