import os
import sys
import time
import logging
import subprocess
import shutil
from PIL import Image

# Ensure torchvision functional_tensor backward compatibility for basicsr
try:
    import torchvision.transforms.functional as F
    sys.modules['torchvision.transforms.functional_tensor'] = F
except Exception:
    pass

# Import centralized configuration
from scripts.config import TEMP_OUTPUT_DIR, TILE_SIZE, TILE_PAD, PRE_PAD

logger = logging.getLogger("AdobeStockUpscaler.Upscaler")

# Cache global in-memory engine instances
_engine_cache = {}
active_subprocess = None

def get_active_subprocess():
    global active_subprocess
    return active_subprocess

def set_active_subprocess(proc):
    global active_subprocess
    active_subprocess = proc

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

            # Define architecture according to model variant
            if self.model_name == 'RealESRGAN_x4plus_anime_6B':
                model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
            else:
                model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)

            # Locate weight file
            filename = f"{self.model_name}.pth"
            model_path = None
            candidates = [
                os.path.join("experiments/pretrained_models", filename),
                os.path.join("weights", filename),
                os.path.join("Real-ESRGAN/experiments/pretrained_models", filename),
                os.path.join("/content/Upscale-AI/experiments/pretrained_models", filename)
            ]
            for c in candidates:
                if os.path.exists(c):
                    model_path = os.path.abspath(c)
                    break

            if not model_path:
                self.init_error = f"Pretrained weight file '{filename}' not found on disk."
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
                gpu_id=0 if torch.cuda.is_available() else None
            )
            self.is_loaded = True
            logger.info(f"Real-ESRGAN model '{self.model_name}' successfully loaded into VRAM/RAM!")
            return True
        except Exception as e:
            self.init_error = f"In-memory Real-ESRGAN init error: {str(e)}"
            logger.error(self.init_error)
            return False

    def enhance(self, input_path: str, output_path: str, scale: float, ext: str, quality: int) -> tuple[bool, str, str, str]:
        """
        Enhances image using persistent in-memory model.
        Returns: (success: bool, error_stage: str, error_reason: str, error_details: str)
        """
        if not self.is_loaded:
            if not self.load_model():
                return False, "Model Loading", "Weight File or Module Missing", self.init_error

        try:
            import cv2
            import torch
            img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                return False, "Image Reading", "Corrupted / Unreadable File", f"cv2.imread failed to decode {input_path}"

            output, _ = self.upscaler.enhance(img, outscale=scale)

            # Format export
            if ext.lower() in ["jpg", "jpeg"]:
                cv2.imwrite(output_path, output, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
            else:
                cv2.imwrite(output_path, output)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return True, "", "", ""
        except Exception as e:
            err_details = f"In-memory enhancement exception: {str(e)}"
            logger.error(err_details)
            return False, "Real-ESRGAN", "Inference Execution Exception", err_details

def get_engine(model_name: str = "RealESRGAN_x4plus") -> RealESRGANEngine:
    global _engine_cache
    if model_name not in _engine_cache:
        _engine_cache[model_name] = RealESRGANEngine(model_name)
    return _engine_cache[model_name]

def find_realesrgan_cli() -> str:
    candidates = [
        "inference_realesrgan.py",
        "Real-ESRGAN/inference_realesrgan.py",
        "../Real-ESRGAN/inference_realesrgan.py",
        "/content/Upscale-AI/inference_realesrgan.py",
        "/content/Upscale-AI/Real-ESRGAN/inference_realesrgan.py"
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return os.path.abspath(candidate)
    return ""

def run_upscale(
    input_path: str,
    output_path: str,
    scale: float,
    model_name: str,
    ext: str,
    quality: int
) -> tuple[bool, str, str, str]:
    """
    Primary upscaling interface.
    Tries fast in-memory RealESRGANEngine first.
    Falls back to Subprocess CLI runner, and finally to PIL Lanczos.
    Returns: (success: bool, error_stage: str, error_reason: str, error_details: str)
    """
    global active_subprocess

    # 1. Primary: Fast Persistent In-Memory Inference
    try:
        engine = get_engine(model_name)
        if engine.load_model():
            success, stage, reason, details = engine.enhance(input_path, output_path, scale, ext, quality)
            if success:
                return True, "", "", ""
            else:
                logger.warning(f"In-memory enhancement failed ({reason}). Falling back to Subprocess CLI...")
    except Exception as e:
        logger.warning(f"In-memory engine bypass: {str(e)}")

    # 2. Secondary: Subprocess CLI Execution via sys.executable
    cli_path = find_realesrgan_cli()
    python_bin = sys.executable or "python"

    if cli_path:
        env = os.environ.copy()
        cli_dir = os.path.dirname(cli_path)
        current_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{cli_dir}:{os.path.dirname(cli_dir)}:{current_pythonpath}"

        cmd = [
            python_bin, cli_path,
            "-n", model_name,
            "-i", input_path,
            "-o", TEMP_OUTPUT_DIR,
            "-s", str(scale),
            "--ext", ext,
            "--tile", str(TILE_SIZE),
            "--tile_pad", str(TILE_PAD),
            "--pre_pad", str(PRE_PAD)
        ]

        # Weight file location
        filename = f"{model_name}.pth"
        for w_cand in [os.path.join("experiments/pretrained_models", filename), os.path.join("weights", filename)]:
            if os.path.exists(w_cand):
                cmd.extend(["--model_path", os.path.abspath(w_cand)])
                break

        try:
            import torch
            if torch.cuda.is_available():
                cmd.append("--half")
        except Exception:
            pass

        logger.info(f"Subprocess CLI command: {' '.join(cmd)}")

        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env
            )
            active_subprocess = process
            stdout, stderr = process.communicate()
            active_subprocess = None

            if process.returncode != 0:
                err_msg = stderr.strip() or stdout.strip() or f"Subprocess exited with code {process.returncode}"
                return False, "Real-ESRGAN CLI", f"CLI Error Code {process.returncode}", err_msg

            base_name = os.path.basename(input_path)
            name_no_ext, _ = os.path.splitext(base_name)
            candidate_filenames = [
                f"{name_no_ext}_out.{ext}",
                f"{name_no_ext}.{ext}",
                f"{name_no_ext}_out.jpg",
                f"{name_no_ext}_out.png"
            ]

            found_file = None
            for c_name in candidate_filenames:
                c_path = os.path.join(TEMP_OUTPUT_DIR, c_name)
                if os.path.exists(c_path):
                    found_file = c_path
                    break

            if not found_file and os.path.exists(TEMP_OUTPUT_DIR):
                for f in os.listdir(TEMP_OUTPUT_DIR):
                    if name_no_ext in f:
                        found_file = os.path.join(TEMP_OUTPUT_DIR, f)
                        break

            if found_file and os.path.exists(found_file):
                shutil.move(found_file, output_path)
                return True, "", "", ""
            else:
                return False, "Output Move", "Output File Missing", f"Expected output image not found in {TEMP_OUTPUT_DIR}"
        except Exception as e:
            active_subprocess = None
            return False, "Real-ESRGAN CLI", "Subprocess Execution Exception", str(e)

    # 3. Tertiary: Local Development Fallback (Pillow Lanczos)
    logger.warning("Real-ESRGAN engine & CLI absent. Running PIL Lanczos fallback.")
    time.sleep(1.0)
    try:
        with Image.open(input_path) as img:
            orig_w, orig_h = img.size
            new_w = int(orig_w * scale)
            new_h = int(orig_h * scale)
            out_img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            if ext.lower() in ["jpg", "jpeg"]:
                out_img.convert("RGB").save(output_path, "JPEG", quality=quality)
            else:
                out_img.save(output_path, "PNG")
        return True, "", "", ""
    except Exception as e:
        return False, "PIL Resampling", "Lanczos Fallback Exception", str(e)
