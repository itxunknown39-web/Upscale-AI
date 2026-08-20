"""
scripts/utils.py — Adobe Stock AI Studio

System resource utilities.
Preserved from original.
"""

import os
import psutil

try:
    import torch
except ImportError:
    torch = None


def get_system_resources() -> dict:
    """
    Returns system RAM and GPU VRAM metrics.
    """
    gpu_available = False
    gpu_name = "None"
    vram_info = {"free": 0.0, "total": 0.0, "used": 0.0}

    if torch and torch.cuda.is_available():
        gpu_available = True
        try:
            gpu_name = torch.cuda.get_device_name(0)
            free_b, total_b = torch.cuda.mem_get_info(0)
            vram_info["free"] = round(free_b / (1024 ** 3), 2)
            vram_info["total"] = round(total_b / (1024 ** 3), 2)
            vram_info["used"] = round((total_b - free_b) / (1024 ** 3), 2)
        except Exception:
            pass

    ram = psutil.virtual_memory()
    ram_info = {
        "used": round(ram.used / (1024 ** 3), 2),
        "total": round(ram.total / (1024 ** 3), 2),
        "percent": ram.percent,
    }

    return {
        "gpu": gpu_available,
        "gpu_name": gpu_name,
        "ram_usage": ram_info,
        "vram_usage": vram_info,
    }


def get_unique_output_filename(directory: str, index: int, ext: str) -> str:
    """
    Generate standardized Adobe Stock output filename.
    Format: stock_image_up{N}.{ext}
    Collision-safe: increments index if file already exists.
    """
    while True:
        name = f"stock_image_up{index}.{ext}"
        path = os.path.join(directory, name)
        if not os.path.exists(path):
            return name, index
        index += 1


def format_eta(seconds: float | None) -> str:
    if seconds is None:
        return "--"
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    minutes = seconds // 60
    secs = seconds % 60
    return f"{minutes}m {secs}s"
