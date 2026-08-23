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

import math
import cv2
import numpy as np

# Ensure basicsr shim and RRDBNet are available
from scripts.rrdbnet import RRDBNet, register_basicsr_shim
register_basicsr_shim()

from scripts.model_manager import ensure_model_weights, find_model_weights

# Import centralized configuration
from scripts.config import TEMP_OUTPUT_DIR, TILE_SIZE, TILE_PAD, PRE_PAD, MAX_OUTPUT_DIMENSION

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

def tile_process(model, img_tensor, tile_size=400, tile_pad=10, scale=4):
    """
    Pure PyTorch tiled super-resolution inference to prevent VRAM allocation overflows.
    """
    batch, channel, height, width = img_tensor.shape
    output_height = height * scale
    output_width = width * scale
    output_shape = (batch, channel, output_height, output_width)

    output = img_tensor.new_zeros(output_shape)
    tiles_x = math.ceil(width / tile_size)
    tiles_y = math.ceil(height / tile_size)

    for y in range(tiles_y):
        for x in range(tiles_x):
            ofs_x = x * tile_size
            ofs_y = y * tile_size
            input_start_x = ofs_x
            input_end_x = min(ofs_x + tile_size, width)
            input_start_y = ofs_y
            input_end_y = min(ofs_y + tile_size, height)

            pad_start_x = max(input_start_x - tile_pad, 0)
            pad_end_x = min(input_end_x + tile_pad, width)
            pad_start_y = max(input_start_y - tile_pad, 0)
            pad_end_y = min(input_end_y + tile_pad, height)

            input_tile = img_tensor[:, :, pad_start_y:pad_end_y, pad_start_x:pad_end_x]
            with torch.no_grad():
                output_tile = model(input_tile)

            out_pad_top = (input_start_y - pad_start_y) * scale
            out_pad_bot = out_pad_top + (input_end_y - input_start_y) * scale
            out_pad_left = (input_start_x - pad_start_x) * scale
            out_pad_right = out_pad_left + (input_end_x - input_start_x) * scale

            dest_top = input_start_y * scale
            dest_bot = input_end_y * scale
            dest_left = input_start_x * scale
            dest_right = input_end_x * scale

            output[:, :, dest_top:dest_bot, dest_left:dest_right] = output_tile[:, :, out_pad_top:out_pad_bot, out_pad_left:out_pad_right]

    return output

class RealESRGANEngine:
    def __init__(self, model_name: str = "RealESRGAN_x4plus"):
        self.model_name = model_name
        self.model = None
        self.upscaler = None
        self.device = None
        self.is_loaded = False
        self.init_error = ""

    def load_model(self) -> bool:
        if self.is_loaded and (self.model is not None or self.upscaler is not None):
            return True

        try:
            import torch
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            logger.info(f"Initializing Real-ESRGAN pure PyTorch model on device: {self.device}")

            # Define architecture according to model variant
            if self.model_name == 'RealESRGAN_x4plus_anime_6B':
                net = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
            else:
                net = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)

            # Locate or auto-download weight file cleanly without basicsr
            success, model_path, err_msg = ensure_model_weights(self.model_name, auto_download=True)
            if not success or not model_path:
                self.init_error = f"Model weights '{self.model_name}.pth' not found. Please place weights in 'weights/' folder. ({err_msg})"
                logger.error(self.init_error)
                return False

            # Load weights dictionary directly with torch
            loadnet = torch.load(model_path, map_location=torch.device('cpu'))
            if 'params_ema' in loadnet:
                state_dict = loadnet['params_ema']
            elif 'params' in loadnet:
                state_dict = loadnet['params']
            else:
                state_dict = loadnet

            net.load_state_dict(state_dict, strict=True)
            net.eval()
            net = net.to(self.device)

            if torch.cuda.is_available():
                net = net.half()

            self.model = net
            self.is_loaded = True
            logger.info(f"Real-ESRGAN pure PyTorch model '{self.model_name}' successfully loaded into VRAM/RAM (FP16: {torch.cuda.is_available()})!")
            return True
        except Exception as e:
            self.init_error = f"In-memory Real-ESRGAN init error: {str(e)}"
            logger.error(self.init_error)
            return False

    def enhance(self, input_path: str, output_path: str, scale: float, ext: str, quality: int) -> tuple[bool, str, str, str]:
        """
        Enhances image using persistent in-memory pure PyTorch model.
        """
        if not self.is_loaded:
            if not self.load_model():
                return False, "Model Loading", "Weight File Missing or Model Failure", self.init_error

        try:
            import torch
            img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                return False, "Image Reading", "Corrupted / Unreadable File", f"cv2.imread failed to decode {input_path}"

            # If RGBA, separate alpha
            has_alpha = len(img.shape) == 3 and img.shape[2] == 4
            if has_alpha:
                alpha = img[:, :, 3]
                img_rgb = img[:, :, :3]
            else:
                alpha = None
                img_rgb = img if len(img.shape) == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            # Convert BGR uint8 -> RGB float32 tensor
            img_float = img_rgb.astype(np.float32) / 255.0
            img_t = torch.from_numpy(np.transpose(img_float[:, :, [2, 1, 0]], (2, 0, 1))).float()
            img_t = img_t.unsqueeze(0).to(self.device)

            if torch.cuda.is_available():
                img_t = img_t.half()

            # Execute tile-based super-resolution
            output_t = tile_process(self.model, img_t, tile_size=TILE_SIZE, tile_pad=TILE_PAD, scale=4)

            # Convert back to uint8 BGR numpy array
            output_np = output_t.data.squeeze().float().cpu().clamp_(0, 1).numpy()
            output = np.transpose(output_np[[2, 1, 0], :, :], (1, 2, 0))
            output = (output * 255.0).round().astype(np.uint8)

            # Resize if active scale is different from native 4x
            if scale != 4.0 and scale > 0:
                h, w = img_rgb.shape[:2]
                target_w = int(w * scale)
                target_h = int(h * scale)
                output = cv2.resize(output, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)

            # Restore alpha channel if needed
            if has_alpha and ext.lower() == "png":
                h_out, w_out = output.shape[:2]
                alpha_resized = cv2.resize(alpha, (w_out, h_out), interpolation=cv2.INTER_LANCZOS4)
                output = cv2.merge([output[:, :, 0], output[:, :, 1], output[:, :, 2], alpha_resized])

            # Format export
            if ext.lower() in ["jpg", "jpeg"]:
                if len(output.shape) == 3 and output.shape[2] == 4:
                    output = cv2.cvtColor(output, cv2.COLOR_BGRA2BGR)
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
