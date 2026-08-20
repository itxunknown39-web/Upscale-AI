import os

# ──────────────────────────────────────────────
# Adobe Stock AI Studio — Centralized Configuration
# ──────────────────────────────────────────────

# App identity
APP_NAME = "Adobe Stock AI Studio"
APP_VERSION = "2.0.0"

# Default upscaling settings
DEFAULT_SCALE = 4
DEFAULT_TARGET_WIDTH = 3840
DEFAULT_TARGET_HEIGHT = 2160
JPEG_QUALITY = 95
DEFAULT_FORMAT = "jpg"
MODEL_NAME = "RealESRGAN_x4plus"
MAX_UPLOAD_SIZE_MB = 100
MAX_CONCURRENT_UPLOADS = 4

# Tiling & VRAM optimization parameters (protects T4 from OOM)
TILE_SIZE = 400
TILE_PAD = 10
PRE_PAD = 10

# ──────────────────────────────────────────────
# Ollama Configuration
# ──────────────────────────────────────────────
OLLAMA_HOST = "http://127.0.0.1:11434"
OLLAMA_TIMEOUT = 120  # seconds per inference call
# Preferred vision models in order (first available will be used)
OLLAMA_VISION_MODELS = ["llava:13b", "llava:7b", "llava", "bakllava", "moondream", "llava:latest"]
OLLAMA_MODEL_PULL_TIMEOUT = 900  # 15 min max pull time

# ──────────────────────────────────────────────
# Metadata / CSV output settings
# ──────────────────────────────────────────────
MAX_KEYWORDS = 49
MAX_TITLE_LENGTH = 200
CSV_FILENAME = "AdobeStock_Metadata.csv"
JSON_FILENAME = "AdobeStock_Metadata.json"
CSV_COLUMNS = ["Filename", "Title", "Keywords", "Category", "Releases"]

# ──────────────────────────────────────────────
# Adobe Stock numeric category map
# ──────────────────────────────────────────────
ADOBE_CATEGORY_MAP = {
    "animals": 1,
    "buildings": 2,
    "business": 3,
    "drinks": 4,
    "environment": 5,
    "states_of_mind": 6,
    "food": 7,
    "graphic_resources": 8,
    "hobbies_and_leisure": 9,
    "industry": 10,
    "landscape": 11,
    "lifestyle": 12,
    "people": 13,
    "plants_and_flowers": 14,
    "culture_and_religion": 15,
    "science": 16,
    "social_issues": 17,
    "sports": 18,
    "technology": 19,
    "transport": 20,
    "travel": 21,
    "abstract": 22,
}

# ──────────────────────────────────────────────
# Directory paths
# ──────────────────────────────────────────────
DRIVE_MOUNT_PARENT = "/content/drive/MyDrive"
DRIVE_PROJECT_PATH = os.path.join(DRIVE_MOUNT_PARENT, "AdobeStockStudio")

LOCAL_BASE = "./AdobeStockStudio"

TEMP_INPUT_DIR = os.path.join(LOCAL_BASE, "uploads")
TEMP_OUTPUT_DIR = os.path.join(LOCAL_BASE, "temp_output")


def resolve_paths():
    """
    Dynamically maps folders to Google Drive if mounted,
    falling back to local storage if Drive is not mounted.
    """
    if os.path.exists(DRIVE_MOUNT_PARENT):
        base = DRIVE_PROJECT_PATH
    else:
        base = LOCAL_BASE

    return {
        "uploads": os.path.join(base, "uploads"),
        "output": os.path.join(base, "output"),
        "metadata": os.path.join(base, "metadata"),
        "failed": os.path.join(base, "failed"),
        "logs": os.path.join(base, "logs"),
        "archives": os.path.join(base, "archives"),
    }


def ensure_dirs():
    """Create all required directories."""
    paths = resolve_paths()
    for p in paths.values():
        os.makedirs(p, exist_ok=True)
    os.makedirs(TEMP_INPUT_DIR, exist_ok=True)
    os.makedirs(TEMP_OUTPUT_DIR, exist_ok=True)
    return paths
