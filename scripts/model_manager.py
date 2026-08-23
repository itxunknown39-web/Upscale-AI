import os
import sys
import urllib.request
import logging

logger = logging.getLogger("AdobeStockUpscaler.ModelManager")

MODEL_URLS = {
    "RealESRGAN_x4plus.pth": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
    "RealESRGAN_x4plus_anime_6B.pth": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth"
}

WEIGHT_DIRECTORIES = [
    "weights",
    "experiments/pretrained_models",
    "Real-ESRGAN/experiments/pretrained_models",
    "/content/Upscale-AI/weights",
    "/content/Upscale-AI/experiments/pretrained_models"
]

def find_model_weights(model_name: str) -> str:
    """
    Searches known local directories for model weight file.
    Returns absolute path if found, or empty string if missing.
    """
    filename = f"{model_name}.pth" if not model_name.endswith(".pth") else model_name
    for directory in WEIGHT_DIRECTORIES:
        cand = os.path.join(directory, filename)
        if os.path.exists(cand) and os.path.getsize(cand) > 1024 * 1024:
            return os.path.abspath(cand)
    return ""

def ensure_model_weights(model_name: str, auto_download: bool = True) -> tuple[bool, str, str]:
    """
    Ensures model weights exist locally.
    Returns: (success: bool, model_path: str, error_message: str)
    """
    filename = f"{model_name}.pth" if not model_name.endswith(".pth") else model_name
    existing_path = find_model_weights(filename)
    if existing_path:
        return True, existing_path, ""

    if not auto_download:
        return False, "", f"Model weights '{filename}' not found in local weights directories."

    url = MODEL_URLS.get(filename)
    if not url:
        return False, "", f"Unknown model '{model_name}'. No download source available."

    target_dir = os.path.join("weights")
    os.makedirs(target_dir, exist_ok=True)
    target_path = os.path.abspath(os.path.join(target_dir, filename))

    logger.info(f"Downloading model weights for '{model_name}' from {url} to {target_path}...")
    try:
        urllib.request.urlretrieve(url, target_path)
        if os.path.exists(target_path) and os.path.getsize(target_path) > 1024 * 1024:
            logger.info(f"✓ Model weights downloaded successfully: {target_path}")
            return True, target_path, ""
        else:
            return False, "", f"Downloaded weight file for '{model_name}' is corrupted or empty."
    except Exception as e:
        err_msg = f"Failed downloading weights for '{model_name}': {str(e)}"
        logger.error(err_msg)
        return False, "", err_msg

def load_file_from_url(url: str, model_dir: str = None, progress: bool = True, file_name: str = None) -> str:
    """
    Drop-in replacement for basicsr.utils.download_util.load_file_from_url
    using standard urllib without external basicsr dependency.
    """
    if model_dir is None:
        model_dir = "weights"
    os.makedirs(model_dir, exist_ok=True)

    if file_name is None:
        file_name = url.split('/')[-1]

    destination = os.path.abspath(os.path.join(model_dir, file_name))
    if os.path.exists(destination) and os.path.getsize(destination) > 1024 * 1024:
        return destination

    logger.info(f"Downloading {url} to {destination}...")
    urllib.request.urlretrieve(url, destination)
    return destination
