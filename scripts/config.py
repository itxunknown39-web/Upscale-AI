import os

# Centralized application settings
DEFAULT_SCALE = 2
DEFAULT_TARGET_WIDTH = 3840
DEFAULT_TARGET_HEIGHT = 2160
MAX_OUTPUT_DIMENSION = 8192
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
DRIVE_MOUNT_PARENT = os.getenv("DRIVE_MOUNT_PARENT", "/content/drive/MyDrive")
DRIVE_FOLDER_NAME = os.getenv("DRIVE_FOLDER_NAME", "AdobeStockUpscaler")
CUSTOM_DRIVE_PATH = os.getenv("DRIVE_PATH", os.getenv("GOOGLE_DRIVE_PATH", ""))

def resolve_paths():
    """
    Dynamically maps folders to Google Drive if mounted or configured,
    falling back to local storage if Drive is not mounted.
    """
    if CUSTOM_DRIVE_PATH and os.path.exists(CUSTOM_DRIVE_PATH):
        base_path = CUSTOM_DRIVE_PATH
    elif os.path.exists(DRIVE_MOUNT_PARENT):
        base_path = os.path.join(DRIVE_MOUNT_PARENT, DRIVE_FOLDER_NAME)
    else:
        base_path = f"./{DRIVE_FOLDER_NAME}"

    return {
        "input": os.path.join(base_path, "input"),
        "output": os.path.join(base_path, "output"),
        "failed": os.path.join(base_path, "failed"),
        "logs": os.path.join(base_path, "logs"),
        "archives": os.path.join(base_path, "archives")
    }


# Working directories
TEMP_INPUT_DIR = f"./{DRIVE_FOLDER_NAME}/temp_input"
TEMP_OUTPUT_DIR = f"./{DRIVE_FOLDER_NAME}/temp_output"

# Ensure runtime directories are created
paths = resolve_paths()
for p in list(paths.values()) + [TEMP_INPUT_DIR, TEMP_OUTPUT_DIR]:
    os.makedirs(p, exist_ok=True)

