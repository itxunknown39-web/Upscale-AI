"""
scripts/ollama_vision.py — Adobe Stock AI Studio

Local Ollama vision analysis for Adobe Stock metadata generation.
Runs entirely on-device via http://127.0.0.1:11434.
No external AI API required.
"""

import base64
import json
import logging
import re
import time
from pathlib import Path
from typing import Optional

import httpx

from scripts.config import (
    OLLAMA_HOST,
    OLLAMA_TIMEOUT,
    OLLAMA_VISION_MODELS,
    OLLAMA_MODEL_PULL_TIMEOUT,
    ADOBE_CATEGORY_MAP,
    MAX_KEYWORDS,
    MAX_TITLE_LENGTH,
)

logger = logging.getLogger("AdobeStockStudio.OllamaVision")

# ──────────────────────────────────────────────
# Global state
# ──────────────────────────────────────────────
_active_model: Optional[str] = None
_ollama_ready: bool = False
_ollama_error: str = ""


def get_ollama_status() -> dict:
    return {
        "ready": _ollama_ready,
        "model": _active_model,
        "error": _ollama_error,
    }


# ──────────────────────────────────────────────
# Connectivity check
# ──────────────────────────────────────────────
def check_ollama_running(timeout: float = 5.0) -> bool:
    """Ping Ollama server."""
    try:
        r = httpx.get(f"{OLLAMA_HOST}/api/tags", timeout=timeout)
        return r.status_code == 200
    except Exception:
        return False


def list_installed_models() -> list[str]:
    """Return list of model names currently pulled in Ollama."""
    try:
        r = httpx.get(f"{OLLAMA_HOST}/api/tags", timeout=10)
        if r.status_code == 200:
            data = r.json()
            return [m["name"] for m in data.get("models", [])]
    except Exception:
        pass
    return []


# ──────────────────────────────────────────────
# Model selection & pull
# ──────────────────────────────────────────────
def detect_vision_model() -> Optional[str]:
    """
    From the list of preferred vision models, return the first one
    that is already installed in Ollama.
    """
    installed = list_installed_models()
    logger.info(f"Installed Ollama models: {installed}")

    for preferred in OLLAMA_VISION_MODELS:
        # Match full name or base name
        for installed_name in installed:
            if preferred.split(":")[0] in installed_name:
                logger.info(f"Found vision model: {installed_name}")
                return installed_name

    return None


def pull_model(model_name: str) -> bool:
    """
    Pull a model from Ollama library. Streams progress.
    Returns True on success.
    """
    logger.info(f"Pulling Ollama model: {model_name} (this may take several minutes)...")
    try:
        with httpx.stream(
            "POST",
            f"{OLLAMA_HOST}/api/pull",
            json={"name": model_name},
            timeout=OLLAMA_MODEL_PULL_TIMEOUT,
        ) as resp:
            for line in resp.iter_lines():
                if line:
                    try:
                        obj = json.loads(line)
                        status = obj.get("status", "")
                        if "total" in obj and "completed" in obj:
                            pct = (obj["completed"] / obj["total"]) * 100
                            logger.info(f"  Pull {status}: {pct:.1f}%")
                        else:
                            logger.info(f"  Pull: {status}")
                    except Exception:
                        pass
        # Verify the model is now available
        installed = list_installed_models()
        base = model_name.split(":")[0]
        for name in installed:
            if base in name:
                logger.info(f"Model pull completed: {name}")
                return True
        logger.error(f"Model {model_name} not found after pull.")
        return False
    except Exception as e:
        logger.error(f"Model pull failed: {e}")
        return False


# ──────────────────────────────────────────────
# Inference test
# ──────────────────────────────────────────────
def test_vision_inference(model_name: str, test_image_path: Optional[str] = None) -> bool:
    """
    Verify the model can actually process an image.
    Uses a tiny 1x1 white pixel if no test image is provided.
    """
    import io
    from PIL import Image as PILImage

    try:
        if test_image_path and Path(test_image_path).exists():
            with open(test_image_path, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode()
        else:
            # Create minimal test image
            buf = io.BytesIO()
            PILImage.new("RGB", (64, 64), color=(128, 128, 128)).save(buf, format="JPEG")
            img_b64 = base64.b64encode(buf.getvalue()).decode()

        payload = {
            "model": model_name,
            "prompt": "Describe this image in one sentence.",
            "images": [img_b64],
            "stream": False,
        }

        r = httpx.post(
            f"{OLLAMA_HOST}/api/generate",
            json=payload,
            timeout=60,
        )
        if r.status_code == 200:
            result = r.json()
            response_text = result.get("response", "")
            logger.info(f"Vision test succeeded. Response: {response_text[:80]}")
            return bool(response_text)
        else:
            logger.error(f"Vision test HTTP {r.status_code}: {r.text[:200]}")
            return False
    except Exception as e:
        logger.error(f"Vision inference test failed: {e}")
        return False


# ──────────────────────────────────────────────
# Startup initializer
# ──────────────────────────────────────────────
def initialize_ollama() -> bool:
    """
    Full Ollama readiness check:
    1. Check runtime config (set by notebook pre-validation) — fast path
    2. Check Ollama is running
    3. Detect installed vision model
    4. Pull preferred model if none installed
    5. Run inference test
    6. Mark ready
    """
    global _active_model, _ollama_ready, _ollama_error

    # ── Fast path: notebook already validated Ollama ──────────────────────
    # The Colab notebook saves .runtime_config.json after Cell 7 passes
    # its own inference test. Skip full re-initialization if found.
    import json as _json
    _runtime_paths = [
        "/content/studio/.runtime_config.json",
        "/content/Upscale-AI/.runtime_config.json",
        ".runtime_config.json",
    ]
    for _rp in _runtime_paths:
        if Path(_rp).exists():
            try:
                with open(_rp) as _f:
                    _cfg = _json.load(_f)
                if _cfg.get("ollama_ready") and _cfg.get("ollama_model"):
                    _active_model = _cfg["ollama_model"]
                    _ollama_ready = True
                    _ollama_error = ""
                    logger.info(f"Ollama ready (from runtime config): {_active_model}")
                    return True
            except Exception as _e:
                logger.warning(f"Could not read runtime config {_rp}: {_e}")
    # ── Full initialization ───────────────────────────────────────────────
    logger.info("Initializing Ollama vision system...")

    # 1. Check server
    if not check_ollama_running():
        _ollama_error = "Ollama server not reachable at http://127.0.0.1:11434"
        logger.error(_ollama_error)
        _ollama_ready = False
        return False

    logger.info("Ollama server is running.")

    # 2. Detect installed vision model
    model = detect_vision_model()

    # 3. Pull if none found
    if model is None:
        target = OLLAMA_VISION_MODELS[-1]  # Use smallest/last as default
        logger.info(f"No vision model found. Attempting to pull: {target}")
        if not pull_model(target):
            _ollama_error = f"Failed to pull vision model: {target}"
            logger.error(_ollama_error)
            _ollama_ready = False
            return False
        model = detect_vision_model()

    if model is None:
        _ollama_error = "No vision-capable model available."
        logger.error(_ollama_error)
        _ollama_ready = False
        return False

    # 4. Inference test
    logger.info(f"Running vision inference test with model: {model}")
    if not test_vision_inference(model):
        _ollama_error = f"Inference test failed for model: {model}"
        logger.error(_ollama_error)
        _ollama_ready = False
        return False

    # 5. Mark ready
    _active_model = model
    _ollama_ready = True
    _ollama_error = ""
    logger.info(f"Ollama vision ready. Active model: {model}")
    return True


# ──────────────────────────────────────────────
# Metadata generation prompt
# ──────────────────────────────────────────────
METADATA_PROMPT = """You are an Adobe Stock metadata expert. Analyze this image carefully and provide metadata ONLY.

STRICT OUTPUT FORMAT — return a valid JSON object and nothing else:
{
  "title": "<natural title ≤200 chars, primary subject first, commercially useful, no hype words>",
  "keywords": ["<keyword1>", "<keyword2>", ... up to 49 keywords ordered by importance],
  "category": <single integer from 1-22 based on dominant content>
}

TITLE RULES:
- Maximum 200 characters
- Start with primary subject
- Natural and commercially accurate
- Forbidden words: beautiful, amazing, stunning, perfect, best, premium, high quality, breathtaking

KEYWORD RULES:
- Maximum 49 keywords
- Order: primary intent → secondary concepts → visible objects → style/color (only if visible)
- No duplicates
- No hallucinated objects, locations, professions, demographics, or brands not visible in image
- Only describe what is actually visible

CATEGORY (use numeric value):
1=Animals, 2=Buildings/Architecture, 3=Business, 4=Drinks, 5=Environment/Nature, 
6=States of Mind/Feelings, 7=Food, 8=Graphic Resources, 9=Hobbies/Leisure, 
10=Industry, 11=Landscape, 12=Lifestyle, 13=People, 14=Plants/Flowers, 
15=Culture/Religion, 16=Science, 17=Social Issues, 18=Sports, 
19=Technology, 20=Transport, 21=Travel, 22=Abstract/Backgrounds

Respond with ONLY valid JSON. No explanations. No markdown. No code blocks."""


# ──────────────────────────────────────────────
# Core analysis function
# ──────────────────────────────────────────────
def analyze_image(image_path: str) -> dict:
    """
    Send upscaled image to Ollama for vision analysis.
    Returns: {title, keywords, category, releases, error}

    IMPORTANT: Always analyzes the actual image file.
    Never infers content from filename.
    """
    global _active_model, _ollama_ready

    result = {
        "title": "",
        "keywords": [],
        "category": 22,
        "releases": "",
        "error": "",
    }

    if not _ollama_ready or not _active_model:
        result["error"] = "Ollama not initialized"
        return result

    if not Path(image_path).exists():
        result["error"] = f"Image file not found: {image_path}"
        return result

    # Encode image
    try:
        with open(image_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode()
    except Exception as e:
        result["error"] = f"Failed to read image: {e}"
        return result

    # Call Ollama
    try:
        payload = {
            "model": _active_model,
            "prompt": METADATA_PROMPT,
            "images": [img_b64],
            "stream": False,
            "options": {
                "temperature": 0.1,  # Low temperature for consistent structured output
                "num_predict": 1024,
            },
        }

        r = httpx.post(
            f"{OLLAMA_HOST}/api/generate",
            json=payload,
            timeout=OLLAMA_TIMEOUT,
        )

        if r.status_code != 200:
            result["error"] = f"Ollama HTTP {r.status_code}: {r.text[:200]}"
            return result

        response_text = r.json().get("response", "")

    except Exception as e:
        result["error"] = f"Ollama request failed: {e}"
        return result

    # Parse JSON from response
    parsed = _parse_metadata_response(response_text)
    if "error" in parsed:
        result["error"] = parsed["error"]
        return result

    result.update(parsed)
    result["releases"] = ""  # Always default empty
    return result


def _parse_metadata_response(text: str) -> dict:
    """
    Extract and validate JSON metadata from Ollama response.
    Handles cases where model wraps JSON in markdown code blocks.
    """
    # Strip markdown code blocks if present
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()

    # Find JSON object (first { ... })
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {"error": f"No JSON object found in Ollama response: {text[:200]}"}

    json_str = match.group(0)

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        return {"error": f"JSON parse error: {e}. Response: {json_str[:200]}"}

    # Validate and sanitize title
    title = str(data.get("title", "")).strip()
    if not title:
        return {"error": "Empty title in Ollama response"}
    title = _clean_title(title)

    # Validate and sanitize keywords
    raw_keywords = data.get("keywords", [])
    if isinstance(raw_keywords, str):
        raw_keywords = [k.strip() for k in raw_keywords.split(",")]
    keywords = _clean_keywords(raw_keywords)

    # Validate category
    category = data.get("category", 22)
    try:
        category = int(category)
        if category < 1 or category > 22:
            category = 22
    except (TypeError, ValueError):
        category = 22

    return {
        "title": title[:MAX_TITLE_LENGTH],
        "keywords": keywords[:MAX_KEYWORDS],
        "category": category,
    }


# ──────────────────────────────────────────────
# Sanitization helpers
# ──────────────────────────────────────────────
BANNED_TITLE_WORDS = {
    "beautiful", "amazing", "stunning", "perfect", "best", "premium",
    "high quality", "breathtaking", "gorgeous", "incredible", "wonderful",
    "fantastic", "excellent", "superb", "extraordinary",
}


def _clean_title(title: str) -> str:
    """Remove banned hype words from title."""
    # Replace banned phrases (case-insensitive)
    for word in sorted(BANNED_TITLE_WORDS, key=len, reverse=True):
        pattern = re.compile(re.escape(word), re.IGNORECASE)
        title = pattern.sub("", title)
    # Clean up double spaces
    title = re.sub(r"\s+", " ", title).strip()
    # Remove leading comma or dash artifacts
    title = re.sub(r"^[,\-\s]+", "", title).strip()
    return title


def _clean_keywords(keywords: list) -> list:
    """
    Deduplicate, lowercase, strip, validate keywords.
    Returns ordered list of max MAX_KEYWORDS unique keywords.
    """
    seen = set()
    cleaned = []
    for kw in keywords:
        if not isinstance(kw, str):
            continue
        kw = kw.strip().lower()
        # Remove surrounding quotes
        kw = kw.strip('"\'')
        if not kw:
            continue
        if kw in seen:
            continue
        seen.add(kw)
        cleaned.append(kw)
        if len(cleaned) >= MAX_KEYWORDS:
            break
    return cleaned
