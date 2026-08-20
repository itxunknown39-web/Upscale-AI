import os
import sys

# Ensure torchvision functional_tensor backward compatibility for basicsr
try:
    import torchvision.transforms.functional as F
    sys.modules['torchvision.transforms.functional_tensor'] = F
except Exception:
    pass

import re
import uuid
import time
import queue
import logging
import zipfile
import shutil
import threading
import subprocess
from typing import List, Optional
from pydantic import BaseModel
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

# Initialize logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("AdobeStockUpscaler")

# Create FastAPI app
app = FastAPI(title="Adobe Stock AI Upscaler API", version="1.0.0")

# Enable CORS for local testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------------------------------
# Directory configuration
# ----------------------------------------------------
DRIVE_MOUNT_PARENT = "/content/drive/MyDrive"
DRIVE_PROJECT_PATH = os.path.join(DRIVE_MOUNT_PARENT, "AdobeStockUpscaler")

# Determine paths dynamically based on Google Drive mount status
def resolve_paths():
    if os.path.exists(DRIVE_MOUNT_PARENT):
        logger.info("Google Drive detected. Enforcing persistent storage.")
        return {
            "output": os.path.join(DRIVE_PROJECT_PATH, "output"),
            "failed": os.path.join(DRIVE_PROJECT_PATH, "failed"),
            "logs": os.path.join(DRIVE_PROJECT_PATH, "logs"),
            "archives": os.path.join(DRIVE_PROJECT_PATH, "archives")
        }
    else:
        logger.warning("Google Drive NOT detected. Using local storage fallback.")
        local_base = "./AdobeStockUpscaler"
        return {
            "output": os.path.join(local_base, "output"),
            "failed": os.path.join(local_base, "failed"),
            "logs": os.path.join(local_base, "logs"),
            "archives": os.path.join(local_base, "archives")
        }

paths = resolve_paths()
TEMP_INPUT_DIR = "./AdobeStockUpscaler/temp_input"
TEMP_OUTPUT_DIR = "./AdobeStockUpscaler/temp_output"

# Ensure all directories exist
for p in list(paths.values()) + [TEMP_INPUT_DIR, TEMP_OUTPUT_DIR]:
    os.makedirs(p, exist_ok=True)

# ----------------------------------------------------
# Global State Management
# ----------------------------------------------------
class FileItem:
    def __init__(self, file_id: str, name: str, size: int, width: int, height: int, temp_path: str):
        self.id = file_id
        self.name = name
        self.size = size
        self.width = width
        self.height = height
        self.megapixels = (width * height) / 1_000_000.0
        self.temp_path = temp_path
        self.status = "ready"  # ready, queued, processing, completed, failed, cancelled
        self.output_width = 0
        self.output_height = 0
        self.output_megapixels = 0.0
        self.qc = {
            "resolution": "pass",
            "format": "pass",
            "integrity": "pass",
            "aspect_ratio": "pass",
            "transparency": "pass",
            "size": "pass"
        }
        self.error_message = ""
        self.processing_seconds = 0.0

# In-memory storage for tracking batch files
files_state = {}
# Thread-safe queue for worker
task_queue = queue.Queue()
# Cancellation flag
cancel_requested = False
# Reference to active upscaling subprocess
active_subprocess = None
# Active batch lock
batch_lock = threading.Lock()

# Progress metrics
progress_metrics = {
    "total": 0,
    "completed": 0,
    "failed": 0,
    "processing": False,
    "current_file": "",
    "current_file_id": "",
    "percentage": 0,
    "eta_seconds": None,
    "processing_speed": 0.0,  # Average seconds per image
}

# ----------------------------------------------------
# Real-ESRGAN Execution Fallback & Subprocess Isolation
# ----------------------------------------------------
REAL_ESRGAN_PATH = "inference_realesrgan.py"
is_realesrgan_available = os.path.exists(REAL_ESRGAN_PATH)

if not is_realesrgan_available:
    logger.warning("Real-ESRGAN CLI (inference_realesrgan.py) not found. Fallback to mock PIL resizing engine enabled.")

def get_unique_filename(directory: str, filename: str, ext: str) -> str:
    base, _ = os.path.splitext(filename)
    out_name = f"{base}_upscaled.{ext}"
    out_path = os.path.join(directory, out_name)
    if not os.path.exists(out_path):
        return out_name

    counter = 1
    while True:
        out_name = f"{base}_upscaled_{counter:02d}.{ext}"
        out_path = os.path.join(directory, out_name)
        if not os.path.exists(out_path):
            return out_name
        counter += 1

def run_upscale(
    input_path: str,
    output_path: str,
    scale: float,
    model_name: str,
    ext: str,
    quality: int
) -> bool:
    global active_subprocess
    
    if not is_realesrgan_available:
        # Local mock implementation for testing without GPU/PyTorch
        logger.info(f"Mocking upscaling for {input_path} (scale={scale}, model={model_name})")
        time.sleep(2.5) # Simulate processing delay
        try:
            img = Image.open(input_path)
            w, h = img.size
            new_w = int(w * scale)
            new_h = int(h * scale)
            out_img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            
            # Save mock output
            if ext.lower() == "jpg" or ext.lower() == "jpeg":
                out_img.convert("RGB").save(output_path, "JPEG", quality=quality)
            else:
                out_img.save(output_path, "PNG")
            return True
        except Exception as e:
            logger.error(f"Mock upscaling failed: {str(e)}")
            return False

    # Execute official Real-ESRGAN CLI as a separate process
    # This prevents VRAM crash from tearing down FastAPI server
    cmd = [
        "python", REAL_ESRGAN_PATH,
        "-n", model_name,
        "-i", input_path,
        "-o", TEMP_OUTPUT_DIR,
        "-s", str(scale),
        "--ext", ext,
        "--tile", "400",
        "--tile_pad", "10",
        "--pre_pad", "10"
    ]
    
    # Enable FP16 half precision on GPU
    try:
        import torch
        if torch.cuda.is_available():
            cmd.append("--half")
    except Exception:
        pass

    logger.info(f"Running Real-ESRGAN CLI: {' '.join(cmd)}")
    
    try:
        # Run process and monitor
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        active_subprocess = process
        stdout, stderr = process.communicate()
        
        # Reset subprocess reference
        active_subprocess = None
        
        if process.returncode != 0:
            logger.error(f"Real-ESRGAN CLI failed with exit code {process.returncode}: {stderr}")
            return False
            
        # Real-ESRGAN output is written to TEMP_OUTPUT_DIR with suffix '_out'
        # e.g., if input is temp_input/abc.jpg -> TEMP_OUTPUT_DIR/abc_out.jpg
        base_name = os.path.basename(input_path)
        name_no_ext, _ = os.path.splitext(base_name)
        expected_output_name = f"{name_no_ext}_out.{ext}"
        expected_path = os.path.join(TEMP_OUTPUT_DIR, expected_output_name)
        
        if os.path.exists(expected_path):
            shutil.move(expected_path, output_path)
            return True
        else:
            logger.error(f"Expected output file not found: {expected_path}")
            return False
            
    except Exception as e:
        logger.error(f"Process execution error: {str(e)}")
        active_subprocess = None
        return False

# ----------------------------------------------------
# Technical Quality Control (QC)
# ----------------------------------------------------
def run_technical_qc(
    output_path: str,
    original_w: int,
    original_h: int,
    req_format: str
) -> dict:
    qc = {
        "resolution": "pass",
        "format": "pass",
        "integrity": "pass",
        "aspect_ratio": "pass",
        "transparency": "pass",
        "size": "pass"
    }
    
    try:
        # Check Integrity & load image
        img = Image.open(output_path)
        img.verify()
        
        # Re-open after verify() (Pillow requires re-opening to read details after verify)
        img = Image.open(output_path)
        out_w, out_h = img.size
        
        # Check Megapixels (Min 4MP)
        mp = (out_w * out_h) / 1_000_000.0
        if mp < 4.0:
            qc["resolution"] = "warn"
            
        # Check Aspect Ratio preservation
        orig_ratio = original_w / original_h
        out_ratio = out_w / out_h
        if abs(orig_ratio - out_ratio) > 0.01:
            qc["aspect_ratio"] = "fail"
            
        # Check file format
        expected_mime = "JPEG" if req_format.lower() in ["jpg", "jpeg"] else "PNG"
        if img.format != expected_mime:
            qc["format"] = "fail"
            
        # Check transparency preservation for PNG
        if req_format.lower() == "png":
            if "A" not in img.mode and img.mode != "RGBA" and img.mode != "LA":
                # Only warning because input might not have had transparency
                qc["transparency"] = "warn"
                
        # Check extreme size limits (>50MB for JPG, >100MB for PNG)
        file_size = os.path.getsize(output_path) / (1024 * 1024) # MB
        if req_format.lower() in ["jpg", "jpeg"] and file_size > 50.0:
            qc["size"] = "warn"
        elif req_format.lower() == "png" and file_size > 100.0:
            qc["size"] = "warn"
            
    except Exception as e:
        logger.error(f"QC Integrity check failed for {output_path}: {str(e)}")
        qc["integrity"] = "fail"
        
    return qc

# ----------------------------------------------------
# Queue Batch Manager Worker Thread
# ----------------------------------------------------
def batch_worker():
    global cancel_requested, active_subprocess
    
    while True:
        try:
            # Block until a batch task is available
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
            
            logger.info(f"Starting batch process for {len(file_ids)} files.")
            
            with batch_lock:
                cancel_requested = False
                progress_metrics["processing"] = True
                progress_metrics["total"] = len(file_ids)
                progress_metrics["completed"] = 0
                progress_metrics["failed"] = 0
                progress_metrics["percentage"] = 0
                progress_metrics["eta_seconds"] = None
                
                # Dynamic paths check (re-evaluate paths in case Google Drive is mounted later)
                global paths
                paths = resolve_paths()
                
                # Start logging session
                batch_log_data = []
                log_name = f"processing_{time.strftime('%Y-%m-%d_%H-%M-%S')}.json"
                log_filepath = os.path.join(paths["logs"], log_name)
                
                start_time = time.time()
                processing_durations = []
                
                for idx, file_id in enumerate(file_ids):
                    if cancel_requested:
                        logger.info("Batch cancelled by user request.")
                        # Mark remainder as cancelled
                        for rem_id in file_ids[idx:]:
                            if rem_id in files_state:
                                files_state[rem_id].status = "cancelled"
                        break
                        
                    file_item: FileItem = files_state.get(file_id)
                    if not file_item:
                        continue
                        
                    file_item.status = "processing"
                    progress_metrics["current_file"] = file_item.name
                    progress_metrics["current_file_id"] = file_item.id
                    
                    # Calculate scale factor for custom target resolution
                    active_scale = scale_factor
                    if target_width and target_height:
                        scale_w = target_width / file_item.width
                        scale_h = target_height / file_item.height
                        active_scale = max(scale_w, scale_h)
                        # Avoid downscaling
                        active_scale = max(active_scale, 1.0)
                    
                    # Clean up trailing floats
                    active_scale = round(active_scale, 2)
                    
                    # Target output paths
                    temp_output_filepath = os.path.join(TEMP_OUTPUT_DIR, f"{file_id}.{output_format}")
                    
                    img_start = time.time()
                    
                    # Perform Real-ESRGAN Upscaling
                    success = run_upscale(
                        input_path=file_item.temp_path,
                        output_path=temp_output_filepath,
                        scale=active_scale,
                        model_name=model,
                        ext=output_format,
                        quality=jpeg_quality
                    )
                    
                    img_duration = time.time() - img_start
                    file_item.processing_seconds = round(img_duration, 2)
                    
                    if success and os.path.exists(temp_output_filepath):
                        # Run Technical Quality Control (QC)
                        qc_results = run_technical_qc(
                            temp_output_filepath,
                            file_item.width,
                            file_item.height,
                            output_format
                        )
                        
                        file_item.qc = qc_results
                        
                        # Populate actual output sizes
                        try:
                            out_img = Image.open(temp_output_filepath)
                            file_item.output_width, file_item.output_height = out_img.size
                            file_item.output_megapixels = (file_item.output_width * file_item.output_height) / 1_000_000.0
                        except Exception:
                            pass
                            
                        # Move to persistent directories based on QC criteria
                        if "fail" in qc_results.values():
                            file_item.status = "failed"
                            file_item.error_message = "Technical QC Check failed (Integrity/Aspect Ratio drift)"
                            dest_path = os.path.join(paths["failed"], file_item.name)
                            shutil.move(temp_output_filepath, dest_path)
                            progress_metrics["failed"] += 1
                        else:
                            file_item.status = "completed"
                            # Rename using unique non-overwriting rule
                            unique_name = get_unique_filename(paths["output"], file_item.name, output_format)
                            dest_path = os.path.join(paths["output"], unique_name)
                            # Overwrite the original name inside state to serve downloaded file correctly
                            file_item.name = unique_name
                            shutil.move(temp_output_filepath, dest_path)
                            progress_metrics["completed"] += 1
                    else:
                        file_item.status = "failed"
                        file_item.error_message = "AI upscaling inference crashed or aborted"
                        progress_metrics["failed"] += 1
                        
                    # Calculate speed metric
                    processing_durations.append(img_duration)
                    avg_speed = sum(processing_durations) / len(processing_durations)
                    progress_metrics["processing_speed"] = avg_speed
                    
                    # Calculate ETA
                    remaining = len(file_ids) - (idx + 1)
                    progress_metrics["eta_seconds"] = int(remaining * avg_speed)
                    progress_metrics["percentage"] = int(((idx + 1) / len(file_ids)) * 100)
                    
                    # Log to batch run list
                    log_item = {
                        "filename": file_item.name,
                        "input": f"{file_item.width}x{file_item.height}",
                        "output": f"{file_item.output_width}x{file_item.output_height}",
                        "processing_seconds": file_item.processing_seconds,
                        "status": file_item.status,
                        "model": model,
                        "scale": active_scale,
                        "qc": file_item.qc
                    }
                    if file_item.error_message:
                        log_item["error"] = file_item.error_message
                    batch_log_data.append(log_item)
                    
                    # Incremental log save to safeguard against Colab disconnects
                    try:
                        import json
                        with open(log_filepath, "w", encoding="utf-8") as f:
                            json.dump(batch_log_data, f, indent=2)
                    except Exception as e:
                        logger.error(f"Failed to save batch log item: {str(e)}")
                        
                # Finished batch cleanup
                progress_metrics["processing"] = False
                progress_metrics["current_file"] = ""
                progress_metrics["current_file_id"] = ""
                logger.info(f"Finished batch job. Completed: {progress_metrics['completed']}, Failed: {progress_metrics['failed']}")
                
            task_queue.task_done()
        except Exception as e:
            logger.error(f"Worker thread error: {str(e)}")
            progress_metrics["processing"] = False

# Start background worker thread
worker_thread = threading.Thread(target=batch_worker, daemon=True)
worker_thread.start()

# ----------------------------------------------------
# API Request Models
# ----------------------------------------------------
class ProcessRequest(BaseModel):
    file_ids: List[str]
    upscale_factor: int
    output_format: str
    jpeg_quality: int
    model: str
    target_width: Optional[int] = None
    target_height: Optional[int] = None

# ----------------------------------------------------
# HTTP Endpoints
# ----------------------------------------------------

@app.get("/api/health")
async def get_health():
    # Detect GPU via torch
    gpu_available = False
    gpu_name = "None"
    vram_info = {"free": 0.0, "total": 0.0}
    
    try:
        import torch
        gpu_available = torch.cuda.is_available()
        if gpu_available:
            gpu_name = torch.cuda.get_device_name(0)
            free_b, total_b = torch.cuda.mem_get_info(0)
            vram_info["free"] = free_b / (1024**3)
            vram_info["total"] = total_b / (1024**3)
    except Exception:
        pass

    # CPU RAM details
    ram_info = {"used": 0.0, "total": 0.0}
    try:
        import psutil
        vm = psutil.virtual_memory()
        ram_info["used"] = (vm.total - vm.available) / (1024**3)
        ram_info["total"] = vm.total / (1024**3)
    except Exception:
        pass

    return {
        "status": "ok",
        "gpu": gpu_available,
        "gpu_name": gpu_name,
        "ram_usage": {
            "used": ram_info["used"],
            "total": ram_info["total"]
        },
        "vram_usage": {
            "used": vram_info["total"] - vram_info["free"] if gpu_available else 0.0,
            "total": vram_info["total"]
        }
    }

@app.post("/api/upload")
async def upload_files(files: List[UploadFile] = File(...)):
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
        # Sanitize filename
        sanitized_name = re.sub(r'[^a-zA-Z0-9_.-]', '_', filename)
        temp_path = os.path.join(TEMP_INPUT_DIR, f"{file_id}_{sanitized_name}")
        
        # Save temp file
        try:
            with open(temp_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer, length=1024 * 1024)
        except Exception as e:
            logger.error(f"Failed writing temp upload: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to stream upload: {filename} ({str(e)})")
            
        # Validate size bounds
        file_size = os.path.getsize(temp_path)
        if file_size > 100 * 1024 * 1024:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise HTTPException(
                status_code=413,
                detail=f"File '{filename}' ({file_size / (1024*1024):.2f} MB) exceeds server size limit of 100MB."
            )

        # Get dimensions
        try:
            with Image.open(temp_path) as img:
                width, height = img.size
        except Exception:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise HTTPException(status_code=400, detail=f"Uploaded file is not a valid image: {filename}")
            
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

@app.post("/api/process")
async def process_batch(req: ProcessRequest):
    if progress_metrics["processing"]:
        raise HTTPException(status_code=400, detail="Another batch job is already running.")
        
    # Filter valid uploaded files
    valid_ids = [fid for fid in req.file_ids if fid in files_state]
    if not valid_ids:
        raise HTTPException(status_code=400, detail="No valid file IDs provided.")
        
    # Queue up the task
    task_queue.put({
        "file_ids": valid_ids,
        "scale_factor": req.upscale_factor,
        "output_format": req.output_format,
        "jpeg_quality": req.jpeg_quality,
        "model": req.model,
        "target_width": req.target_width,
        "target_height": req.target_height
    })
    
    # Mark files as queued
    for fid in valid_ids:
        files_state[fid].status = "queued"
        
    return {"status": "queued", "files_count": len(valid_ids)}

@app.get("/api/progress")
async def get_progress():
    return progress_metrics

@app.get("/api/files")
async def get_files():
    # Return file list sorted by creation/upload time
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
            "error": f.error_message,
            "processing_seconds": f.processing_seconds
        })
    return sorted_files

@app.get("/api/download/{filename}")
async def download_image(filename: str):
    # Dynamic paths check
    global paths
    paths = resolve_paths()
    
    file_path = os.path.join(paths["output"], filename)
    if not os.path.exists(file_path):
        # Fallback check failed directory
        file_path = os.path.join(paths["failed"], filename)
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Processed file not found")
            
    return FileResponse(file_path, filename=filename)

@app.get("/api/download-all")
async def download_all_zip():
    # Dynamic paths check
    global paths
    paths = resolve_paths()
    
    completed_files = [f for f in files_state.values() if f.status == "completed"]
    if not completed_files:
        raise HTTPException(status_code=400, detail="No completed files available to package.")
        
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
        raise HTTPException(status_code=500, detail=f"Failed to create ZIP: {str(e)}")

@app.post("/api/cancel")
async def cancel_batch():
    global cancel_requested, active_subprocess
    
    if not progress_metrics["processing"]:
        return {"status": "no_active_batch"}
        
    cancel_requested = True
    
    # Terminate active subprocess (Real-ESRGAN runner) if running
    if active_subprocess:
        try:
            logger.info("Terminating active Real-ESRGAN subprocess...")
            active_subprocess.terminate()
            # Wait up to 3 seconds for subprocess to exit cleanly
            for _ in range(30):
                if active_subprocess.poll() is not None:
                    break
                time.sleep(0.1)
            # Kill if still alive
            if active_subprocess.poll() is None:
                logger.warning("Subprocess did not exit. Sending SIGKILL.")
                active_subprocess.kill()
        except Exception as e:
            logger.error(f"Error terminating subprocess: {str(e)}")
            
    return {"status": "cancel_initiated"}

# Serves index.html at root
@app.get("/", response_class=HTMLResponse)
async def read_index():
    index_path = "./app/index.html"
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return f.read()
    else:
        return HTMLResponse(content="<h3>Index.html not found! Please check development folder structure.</h3>", status_code=404)

if __name__ == "__main__":
    import uvicorn
    # Start web server on port 8000
    uvicorn.run(app, host="127.0.0.1", port=8000)
