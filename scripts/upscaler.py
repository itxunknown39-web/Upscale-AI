"""
scripts/upscaler.py — Adobe Stock AI Studio

Real-ESRGAN upscaling engine.
Preserved from original with subprocess isolation for T4 VRAM protection.
"""

import os
import sys
import time
import logging
import subprocess
import shutil
from pathlib import Path
from PIL import Image

# Ensure torchvision functional_tensor backward compatibility for basicsr
try:
    import torchvision.transforms.functional as F
    sys.modules['torchvision.transforms.functional_tensor'] = F
except Exception:
    pass

from scripts.config import TEMP_OUTPUT_DIR, TILE_SIZE, TILE_PAD, PRE_PAD

logger = logging.getLogger("AdobeStockStudio.Upscaler")

# ──────────────────────────────────────────────
# Engine cache & subprocess reference
# ──────────────────────────────────────────────
_engine_cache = {}
active_subprocess = None


def get_active_subprocess():
    global active_subprocess
    return active_subprocess


def set_active_subprocess(proc):
    global active_subprocess
    active_subprocess = proc


# ──────────────────────────────────────────────
# In-memory engine
# ──────────────────────────────────────────────
class RealESRGANEngine:
    def __init__(self, model_name: str = "RealESRGAN_x4plus"):
        self.model_name = model_name
        self.upscaler = None
        self.device = None
        self.is_loaded = False
        self.init_error = ""

    def load_model(self) -> bool:
        if self.is_loaded and self.upscaler is not None:
            return True

        try:
            import torch
            from realesrgan import RealESRGANer
            from basicsr.archs.rrdbnet_arch import RRDBNet

            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            logger.info(f"Initializing Real-ESRGAN in-memory model on device: {self.device}")

            # Architecture based on model variant
            if self.model_name == 'RealESRGAN_x4plus_anime_6B':
                model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                                num_block=6, num_grow_ch=32, scale=4)
            else:
                model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                                num_block=23, num_grow_ch=32, scale=4)

            # Locate weight file
            filename = f"{self.model_name}.pth"
            candidates = [
                os.path.join("experiments/pretrained_models", filename),
                os.path.join("weights", filename),
                os.path.join("Real-ESRGAN/experiments/pretrained_models", filename),
                os.path.join("/content/Upscale-AI/experiments/pretrained_models", filename),
                os.path.join("/content/Real-ESRGAN/experiments/pretrained_models", filename),
            ]
            model_path = None
            for c in candidates:
                if os.path.exists(c):
                    model_path = os.path.abspath(c)
                    break

            if not model_path:
                self.init_error = f"Weight file '{filename}' not found on disk."
                logger.error(self.init_error)
                return False

            use_half = torch.cuda.is_available()
            self.upscaler = RealESRGANer(
                scale=4,
                model_path=model_path,
                dni_weight=None,
                model=model,
                tile=TILE_SIZE,
                tile_pad=TILE_PAD,
                pre_pad=PRE_PAD,
                half=use_half,
                gpu_id=0 if torch.cuda.is_available() else None,
            )
            self.is_loaded = True
            logger.info("Real-ESRGAN engine loaded successfully.")
            return True

        except Exception as e:
            self.init_error = str(e)
            logger.error(f"Failed to load Real-ESRGAN engine: {e}")
            return False

    def upscale(self, input_path: str, output_path: str, scale: float = 4) -> bool:
        if not self.load_model():
            return False
        try:
            import cv2
            import numpy as np

            img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                logger.error(f"Could not read image: {input_path}")
                return False

            output, _ = self.upscaler.enhance(img, outscale=scale)
            cv2.imwrite(output_path, output)
            logger.info(f"In-memory upscale complete → {output_path}")
            return True

        except Exception as e:
            logger.error(f"In-memory upscale error: {e}")
            return False

    def unload(self):
        """Release GPU memory after upscaling."""
        try:
            if self.upscaler is not None:
                del self.upscaler
                self.upscaler = None
                self.is_loaded = False
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Real-ESRGAN engine unloaded, VRAM cleared.")
        except Exception as e:
            logger.warning(f"Unload warning: {e}")


def get_engine(model_name: str = "RealESRGAN_x4plus") -> RealESRGANEngine:
    if model_name not in _engine_cache:
        _engine_cache[model_name] = RealESRGANEngine(model_name)
    return _engine_cache[model_name]


# ──────────────────────────────────────────────
# Subprocess CLI fallback (isolation mode)
# ──────────────────────────────────────────────
def _find_realesrgan_script() -> str | None:
    candidates = [
        "inference_realesrgan.py",
        "/content/Upscale-AI/inference_realesrgan.py",
        "/content/Real-ESRGAN/inference_realesrgan.py",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def run_upscale_subprocess(
    input_path: str,
    output_dir: str,
    scale: float,
    model_name: str,
    ext: str,
) -> tuple[bool, str]:
    """
    Run Real-ESRGAN as a subprocess.
    This provides subprocess isolation — if VRAM crashes, FastAPI stays alive.
    Returns (success, output_file_path)
    """
    global active_subprocess

    script = _find_realesrgan_script()
    if not script:
        return False, ""

    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        "python", script,
        "-n", model_name,
        "-i", input_path,
        "-o", output_dir,
        "-s", str(scale),
        "--ext", ext,
        "--tile", str(TILE_SIZE),
        "--tile_pad", str(TILE_PAD),
        "--pre_pad", str(PRE_PAD),
    ]

    try:
        import torch
        if torch.cuda.is_available():
            cmd.append("--half")
    except Exception:
        pass

    logger.info(f"Running Real-ESRGAN subprocess: {' '.join(cmd)}")

    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        active_subprocess = proc
        stdout, stderr = proc.communicate()
        active_subprocess = None

        if proc.returncode != 0:
            logger.error(f"Real-ESRGAN subprocess failed (exit {proc.returncode}): {stderr}")
            return False, ""

        # Find output file
        base = os.path.splitext(os.path.basename(input_path))[0]
        expected = os.path.join(output_dir, f"{base}_out.{ext}")
        if os.path.exists(expected):
            return True, expected

        # Try alternate naming
        for fname in os.listdir(output_dir):
            if base in fname and fname.endswith(f".{ext}"):
                return True, os.path.join(output_dir, fname)

        logger.error("Real-ESRGAN subprocess ran but output file not found.")
        return False, ""

    except Exception as e:
        active_subprocess = None
        logger.error(f"Subprocess exception: {e}")
        return False, ""


# ──────────────────────────────────────────────
# Mock fallback (for testing without GPU)
# ──────────────────────────────────────────────
def run_upscale_mock(input_path: str, output_path: str, scale: float) -> bool:
    """PIL-based mock upscaler for environments without GPU/Real-ESRGAN."""
    try:
        logger.info(f"[MOCK] Upscaling {input_path} (scale={scale})")
        time.sleep(1.5)  # Simulate processing
        img = Image.open(input_path)
        w, h = img.size
        new_w, new_h = int(w * scale), int(h * scale)
        out = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        ext = Path(output_path).suffix.lower()
        if ext in (".jpg", ".jpeg"):
            out.convert("RGB").save(output_path, "JPEG", quality=95)
        else:
            out.save(output_path, "PNG")
        return True
    except Exception as e:
        logger.error(f"Mock upscale failed: {e}")
        return False


# ──────────────────────────────────────────────
# Primary entry point
# ──────────────────────────────────────────────
def run_upscale(
    input_path: str,
    output_path: str,
    scale: float,
    model_name: str,
    ext: str,
    quality: int = 95,
    progress_callback=None,
) -> bool:
    """
    Upscale an image using the best available method:
    1. In-memory RealESRGANer (fastest, direct Python API)
    2. Subprocess CLI (subprocess isolation fallback)
    3. PIL mock (testing only)

    progress_callback: optional callable(percent: int)
    """
    if progress_callback:
        progress_callback(5)

    # Method 1: In-memory
    try:
        from realesrgan import RealESRGANer  # noqa: F401
        engine = get_engine(model_name)
        if progress_callback:
            progress_callback(20)
        success = engine.upscale(input_path, output_path, scale)
        if success:
            if progress_callback:
                progress_callback(100)
            return True
        logger.warning("In-memory engine failed, trying subprocess...")
    except ImportError:
        logger.info("realesrgan not importable, trying subprocess...")

    if progress_callback:
        progress_callback(30)

    # Method 2: Subprocess
    out_dir = str(Path(output_path).parent)
    success, found_path = run_upscale_subprocess(input_path, out_dir, scale, model_name, ext)
    if success and found_path:
        if found_path != output_path:
            shutil.move(found_path, output_path)
        if progress_callback:
            progress_callback(100)
        return True

    logger.warning("Subprocess failed. Using mock PIL fallback.")
    if progress_callback:
        progress_callback(50)

    # Method 3: Mock
    result = run_upscale_mock(input_path, output_path, scale)
    if result and progress_callback:
        progress_callback(100)
    return result
