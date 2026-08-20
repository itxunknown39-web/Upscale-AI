import os

# Centralized application settings
DEFAULT_SCALE = 4
DEFAULT_TARGET_WIDTH = 3840
DEFAULT_TARGET_HEIGHT = 2160
JPEG_QUALITY = 95
DEFAULT_FORMAT = "jpg"
MODEL_NAME = "RealESRGAN_x4plus"
MAX_UPLOAD_SIZE_MB = 100
MAX_CONCURRENT_UPLOADS = 4

# Tiling & VRAM optimization parameters
TILE_SIZE = 400
TILE_PAD = 10
PRE_PAD = 10

# Directory paths configuration
DRIVE_MOUNT_PARENT = "/content/drive/MyDrive"
DRIVE_PROJECT_PATH = os.path.join(DRIVE_MOUNT_PARENT, "AdobeStockUpscaler")

def resolve_paths():
    """
    Dynamically maps folders to Google Drive if mounted,
    falling back to local storage if Drive is not mounted.
    """
    if os.path.exists(DRIVE_MOUNT_PARENT):
        return {
            "output": os.path.join(DRIVE_PROJECT_PATH, "output"),
            "failed": os.path.join(DRIVE_PROJECT_PATH, "failed"),
            "logs": os.path.join(DRIVE_PROJECT_PATH, "logs"),
            "archives": os.path.join(DRIVE_PROJECT_PATH, "archives")
        }
    else:
        local_base = "./AdobeStockUpscaler"
        return {
            "output": os.path.join(local_base, "output"),
            "failed": os.path.join(local_base, "failed"),
            "logs": os.path.join(local_base, "logs"),
            "archives": os.path.join(local_base, "archives")
        }

# Working directories
TEMP_INPUT_DIR = "./AdobeStockUpscaler/temp_input"
TEMP_OUTPUT_DIR = "./AdobeStockUpscaler/temp_output"

# Ensure runtime directories are created
paths = resolve_paths()
for p in list(paths.values()) + [TEMP_INPUT_DIR, TEMP_OUTPUT_DIR]:
    os.makedirs(p, exist_ok=True)
