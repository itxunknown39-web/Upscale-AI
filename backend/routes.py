import os
import re
import uuid
import time
import zipfile
import shutil
import logging
import asyncio
from typing import List, Optional
from pydantic import BaseModel
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from PIL import Image

# Import script modules
from scripts.config import resolve_paths, TEMP_INPUT_DIR, MAX_UPLOAD_SIZE_MB
from scripts.utils import get_system_resources
from scripts.upscaler import get_active_subprocess
from scripts.batch_processor import (
    FileItem,
    files_state,
    task_queue,
    get_progress_metrics,
    set_cancel_requested,
    retry_failed_items,
    process_single_file
)

logger = logging.getLogger("AdobeStockUpscaler.Routes")
router = APIRouter()

class ProcessRequest(BaseModel):
    file_ids: List[str]
    upscale_factor: int = 4
    output_format: str = "jpg"
    jpeg_quality: int = 95
    model: str = "RealESRGAN_x4plus"
    target_width: Optional[int] = None
    target_height: Optional[int] = None

class SingleProcessRequest(BaseModel):
    file_id: str
    upscale_factor: int = 4
    output_format: str = "jpg"
    jpeg_quality: int = 95
    model: str = "RealESRGAN_x4plus"
    target_width: Optional[int] = None
    target_height: Optional[int] = None

@router.get("/api/health")
async def get_health():
    try:
        metrics = get_system_resources()
        return {
            "status": "ok",
            "gpu": metrics["gpu"],
            "gpu_name": metrics["gpu_name"],
            "ram_usage": metrics["ram_usage"],
            "vram_usage": metrics["vram_usage"]
        }
    except Exception as e:
        logger.error(f"Health check exception: {str(e)}")
        raise HTTPException(status_code=500, detail="Health check system monitoring failure")

@router.post("/api/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    """
    Streams uploaded files directly to disk, validates dimensions, and returns metadata IDs.
    """
    uploaded_files_list = []
    os.makedirs(TEMP_INPUT_DIR, exist_ok=True)

    for file in files:
        filename = file.filename or "uploaded_image.jpg"
        _, file_ext = os.path.splitext(filename.lower())
        if file_ext not in [".jpg", ".jpeg", ".png"]:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported format '{filename}'. Allowed extensions: .jpg, .jpeg, .png"
            )

        file_id = str(uuid.uuid4())[:8]
        sanitized_name = re.sub(r'[^a-zA-Z0-9_.-]', '_', filename)
        temp_path = os.path.join(TEMP_INPUT_DIR, f"{file_id}_{sanitized_name}")

        # Stream file chunks directly to disk with 1MB I/O buffer
        try:
            with open(temp_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer, length=1024 * 1024)
        except Exception as e:
            logger.error(f"Failed writing temp upload: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to stream upload: {filename} ({str(e)})")

        # Validate size bounds
        file_size = os.path.getsize(temp_path)
        if file_size > MAX_UPLOAD_SIZE_MB * 1024 * 1024:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise HTTPException(
                status_code=413,
                detail=f"File '{filename}' ({file_size / (1024*1024):.2f} MB) exceeds server size limit of {MAX_UPLOAD_SIZE_MB}MB."
            )

        # Inspect resolution metadata
        try:
            with Image.open(temp_path) as img:
                width, height = img.size
        except Exception:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise HTTPException(status_code=400, detail=f"Invalid or corrupted image: {filename}")

        file_item = FileItem(file_id, sanitized_name, file_size, width, height, temp_path)
        files_state[file_id] = file_item

        uploaded_files_list.append({
            "id": file_item.id,
            "name": file_item.name,
            "width": file_item.width,
            "height": file_item.height,
            "megapixels": file_item.megapixels,
            "size": file_item.size
        })

    return {"files": uploaded_files_list}

@router.post("/api/scan-drive-input")
async def scan_drive_input():
    """
    Scans the Google Drive input folder (AdobeStockUpscaler/input) directly
    and adds all found images to the batch queue instantly without needing slow browser uploads.
    """
    paths = resolve_paths()
    input_dir = paths.get("input")
    if not input_dir or not os.path.exists(input_dir):
        return {"files": [], "message": f"Input folder not found: {input_dir}", "folder": input_dir or ""}

    imported_files = []
    os.makedirs(TEMP_INPUT_DIR, exist_ok=True)
    allowed_exts = [".jpg", ".jpeg", ".png"]

    try:
        filenames = os.listdir(input_dir)
    except Exception as e:
        return {"files": [], "message": f"Cannot read input folder: {str(e)}", "folder": input_dir}

    for fname in filenames:
        _, ext = os.path.splitext(fname.lower())
        if ext in allowed_exts:
            src_path = os.path.join(input_dir, fname)
            if not os.path.isfile(src_path):
                continue

            file_id = str(uuid.uuid4())[:8]
            sanitized_name = re.sub(r'[^a-zA-Z0-9_.-]', '_', fname)
            temp_path = os.path.join(TEMP_INPUT_DIR, f"{file_id}_{sanitized_name}")

            try:
                shutil.copy2(src_path, temp_path)
                file_size = os.path.getsize(temp_path)
                with Image.open(temp_path) as img:
                    width, height = img.size

                file_item = FileItem(file_id, sanitized_name, file_size, width, height, temp_path)
                files_state[file_id] = file_item
                imported_files.append({
                    "id": file_item.id,
                    "name": file_item.name,
                    "width": file_item.width,
                    "height": file_item.height,
                    "megapixels": file_item.megapixels,
                    "size": file_item.size
                })
            except Exception as e:
                logger.error(f"Error importing {fname}: {str(e)}")

    return {"files": imported_files, "folder": input_dir, "count": len(imported_files)}


@router.post("/api/process-file")
async def process_single_file_endpoint(req: SingleProcessRequest):
    """
    Direct asynchronous pipeline endpoint for single file AI upscaling and Technical QC.
    Used by the concurrent worker-pool orchestration.
    """
    if req.file_id not in files_state:
        raise HTTPException(status_code=404, detail=f"File ID '{req.file_id}' not found on server.")

    # Execute in threadpool so FastAPI async event loop never blocks
    result = await asyncio.to_thread(
        process_single_file,
        file_id=req.file_id,
        scale_factor=req.upscale_factor,
        output_format=req.output_format,
        jpeg_quality=req.jpeg_quality,
        model=req.model,
        target_width=req.target_width,
        target_height=req.target_height
    )

    return result

@router.post("/api/process")
async def process_batch(req: ProcessRequest):
    """
    Batch queue submission endpoint.
    """
    valid_ids = [fid for fid in req.file_ids if fid in files_state]
    if not valid_ids:
        raise HTTPException(status_code=400, detail="No valid file IDs provided for processing.")

    for fid in valid_ids:
        files_state[fid].status = "queued"

    task_queue.put({
        "file_ids": valid_ids,
        "scale_factor": req.upscale_factor,
        "output_format": req.output_format,
        "jpeg_quality": req.jpeg_quality,
        "model": req.model,
        "target_width": req.target_width,
        "target_height": req.target_height
    })

    return {"status": "queued", "files_count": len(valid_ids)}

@router.get("/api/progress")
async def get_progress():
    return get_progress_metrics()

@router.get("/api/files")
async def get_files():
    sorted_files = []
    for f in files_state.values():
        sorted_files.append({
            "id": f.id,
            "name": f.name,
            "size": f.size,
            "width": f.width,
            "height": f.height,
            "megapixels": f.megapixels,
            "status": f.status,
            "output_width": f.output_width,
            "output_height": f.output_height,
            "output_megapixels": f.output_megapixels,
            "qc": f.qc,
            "error_stage": f.error_stage,
            "error_reason": f.error_reason,
            "error_details": f.error_details,
            "processing_seconds": f.processing_seconds
        })
    return sorted_files

@router.post("/api/retry-failed")
async def retry_failed():
    count = retry_failed_items()
    return {"status": "ready", "retried_count": count}

@router.get("/api/download/{file_id}")
async def download_image(file_id: str):
    paths = resolve_paths()
    filename = file_id
    if file_id in files_state:
        filename = files_state[file_id].name

    file_path = os.path.join(paths["output"], filename)
    if not os.path.exists(file_path):
        file_path = os.path.join(paths["failed"], filename)
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Requested image file was not found on disk.")

    return FileResponse(file_path, filename=filename)

@router.get("/api/download-all")
async def download_all_zip():
    paths = resolve_paths()
    completed_files = [f for f in files_state.values() if f.status == "completed"]

    if not completed_files:
        raise HTTPException(status_code=400, detail="No completed outputs available to package.")

    zip_filename = f"AdobeStock_Upscaled_{time.strftime('%Y-%m-%d')}.zip"
    zip_filepath = os.path.join(paths["archives"], zip_filename)

    try:
        with zipfile.ZipFile(zip_filepath, "w", zipfile.ZIP_DEFLATED) as zf:
            for file_item in completed_files:
                actual_path = os.path.join(paths["output"], file_item.name)
                if os.path.exists(actual_path):
                    zf.write(actual_path, arcname=file_item.name)

        return FileResponse(zip_filepath, filename=zip_filename, media_type="application/zip")
    except Exception as e:
        logger.error(f"Failed constructing ZIP archive: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed compiling ZIP package.")

@router.post("/api/cancel")
async def cancel_batch():
    set_cancel_requested(True)
    proc = get_active_subprocess()
    if proc:
        try:
            proc.terminate()
            for _ in range(20):
                if proc.poll() is not None:
                    break
                time.sleep(0.1)
            if proc.poll() is None:
                proc.kill()
        except Exception as e:
            logger.error(f"Process termination exception: {str(e)}")

    return {"status": "cancel_initiated"}
