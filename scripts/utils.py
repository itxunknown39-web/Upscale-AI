import os
import psutil

# Safe imports for torch
try:
    import torch
except ImportError:
    torch = None

def get_system_resources():
    """
    Retrieves system RAM and target T4 GPU memory usage metrics.
    """
    gpu_available = False
    gpu_name = "None"
    vram_info = {"free": 0.0, "total": 0.0}

    # Verify PyTorch / CUDA
    if torch and torch.cuda.is_available():
        gpu_available = True
        try:
            gpu_name = torch.cuda.get_device_name(0)
            free_b, total_b = torch.cuda.mem_get_info(0)
            vram_info["free"] = free_b / (1024**3)  # GB
            vram_info["total"] = total_b / (1024**3)  # GB
        except Exception:
            pass

    # RAM Info
    ram_info = {"used": 0.0, "total": 0.0}
    try:
        vm = psutil.virtual_memory()
        ram_info["used"] = (vm.total - vm.available) / (1024**3)
        ram_info["total"] = vm.total / (1024**3)
    except Exception:
        pass

    return {
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

def get_unique_filename(directory: str, filename: str, ext: str) -> str:
    """
    Calculates a unique, non-overwriting filename in the destination directory.
    Original files are never overwritten.
    """
    base, _ = os.path.splitext(filename)
    # Sanitize suffix
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
