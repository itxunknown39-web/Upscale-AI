"""
scripts/ollama_vision.py — Adobe Stock AI Studio

Robust Centralized Ollama Vision Client & Metadata Pipeline:
- Centralized OllamaVisionClient
- Dual endpoint support: /api/chat (primary) + /api/generate (fallback)
- Comprehensive response parsing (message.content & response)
- Empty response detection, diagnosis, logging, and automatic model fallback
- Structured JSON extraction with correction retry prompt & heuristic fallback
- Real patterned image generation for vision validation
- Full error codes & safe logging (no secrets exposed)
- Bounded retries (maximum 3 attempts)
- Memory management & timeout safety
"""

import base64
import io
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
from PIL import Image, ImageDraw

from scripts.config import (
    ADOBE_CATEGORY_MAP,
    MAX_KEYWORDS,
    MAX_TITLE_LENGTH,
    OLLAMA_HOST,
    OLLAMA_JSON_MAX_RETRIES,
    OLLAMA_MAX_RETRIES,
    OLLAMA_MODEL_PULL_TIMEOUT,
    OLLAMA_TIMEOUT,
    OLLAMA_VISION_MODEL,
    OLLAMA_VISION_MODELS,
)

logger = logging.getLogger("AdobeStockStudio.OllamaVision")

# ──────────────────────────────────────────────
# Error Codes
# ──────────────────────────────────────────────
class OllamaErrorCode:
    SERVER_UNREACHABLE = "OLLAMA_SERVER_UNREACHABLE"
    MODEL_NOT_FOUND = "OLLAMA_MODEL_NOT_FOUND"
    MODEL_PULL_FAILED = "OLLAMA_MODEL_PULL_FAILED"
    EMPTY_RESPONSE = "OLLAMA_EMPTY_RESPONSE"
    TIMEOUT = "OLLAMA_TIMEOUT"
    CONNECTION_ERROR = "OLLAMA_CONNECTION_ERROR"
    HTTP_ERROR = "OLLAMA_HTTP_ERROR"
    JSON_INVALID = "OLLAMA_JSON_INVALID"
    VISION_TEST_FAILED = "OLLAMA_VISION_TEST_FAILED"
    IMAGE_READ_ERROR = "OLLAMA_IMAGE_READ_ERROR"


# ──────────────────────────────────────────────
# Global State
# ──────────────────────────────────────────────
_active_model: Optional[str] = None
_ollama_ready: bool = False
_ollama_error: str = ""
_vision_tested: bool = False
_last_error_code: str = ""


def get_ollama_status() -> dict:
    """Return complete status snapshot for API & UI."""
    return {
        "ready": _ollama_ready,
        "model": _active_model,
        "error": _ollama_error,
        "vision_tested": _vision_tested,
        "error_code": _last_error_code,
        "host": OLLAMA_HOST,
    }


def set_ollama_status(ready: bool, model: Optional[str] = None, error: str = "", code: str = ""):
    global _active_model, _ollama_ready, _ollama_error, _vision_tested, _last_error_code
    _ollama_ready = ready
    if model is not None:
        _active_model = model
    _ollama_error = error
    _last_error_code = code
    if ready:
        _vision_tested = True


# ──────────────────────────────────────────────
# Helper: Patterned Test Image Generation
# ──────────────────────────────────────────────
def create_patterned_test_image_bytes(width: int = 256, height: int = 256) -> bytes:
    """
    Generate a 256x256 RGB image with geometric shapes, gradients, and contrasting colors.
    This guarantees vision token activation in vision encoders (Moondream SigLIP, LLaVA CLIP).
    """
    img = Image.new("RGB", (width, height), color=(30, 60, 120))
    draw = ImageDraw.Draw(img)
    # Draw geometric features
    draw.rectangle([20, 20, 100, 100], fill=(220, 80, 40), outline=(255, 255, 255))
    draw.ellipse([120, 50, 220, 150], fill=(40, 180, 90), outline=(255, 255, 255))
    draw.polygon([(128, 160), (60, 230), (196, 230)], fill=(240, 200, 30))
    draw.line([(0, 0), (width, height)], fill=(255, 255, 255), width=3)
    
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


# ──────────────────────────────────────────────
# Centralized Ollama Vision Client
# ──────────────────────────────────────────────
class OllamaVisionClient:
    """
    Unified client for Ollama Vision API calls.
    Handles /api/chat & /api/generate, bounded retries, empty response recovery,
    model fallback, and robust image base64 preparation.
    """

    def __init__(self, host: str = OLLAMA_HOST, timeout: int = OLLAMA_TIMEOUT):
        self.host = host.rstrip("/")
        self.timeout = timeout

    def check_connection(self) -> Tuple[bool, str]:
        """Ping Ollama server."""
        try:
            r = httpx.get(f"{self.host}/api/tags", timeout=5.0)
            if r.status_code == 200:
                return True, "Ollama server is reachable."
            return False, f"Ollama returned HTTP {r.status_code}"
        except httpx.ConnectError:
            return False, f"Connection refused to {self.host}. Is Ollama running?"
        except httpx.TimeoutException:
            return False, f"Connection timeout to {self.host}"
        except Exception as e:
            return False, f"Connection error: {e}"

    def list_installed_models(self) -> List[str]:
        """Return list of model tag names installed locally."""
        try:
            r = httpx.get(f"{self.host}/api/tags", timeout=10.0)
            if r.status_code == 200:
                data = r.json()
                return [m.get("name", "") for m in data.get("models", []) if m.get("name")]
        except Exception as e:
            logger.warning(f"Failed to list Ollama models: {e}")
        return []

    def pull_model(self, model_name: str) -> bool:
        """
        Pull a model from the Ollama registry with progress logging.
        Returns True on success.
        """
        logger.info(f"Pulling model: {model_name} (timeout {OLLAMA_MODEL_PULL_TIMEOUT}s)...")
        try:
            with httpx.stream(
                "POST",
                f"{self.host}/api/pull",
                json={"name": model_name},
                timeout=OLLAMA_MODEL_PULL_TIMEOUT,
            ) as resp:
                if resp.status_code != 200:
                    logger.error(f"Pull request failed with HTTP {resp.status_code}")
                    return False
                last_pct = -1
                for line in resp.iter_lines():
                    if line:
                        try:
                            obj = json.loads(line)
                            status = obj.get("status", "")
                            if "total" in obj and "completed" in obj and obj["total"] > 0:
                                pct = int((obj["completed"] / obj["total"]) * 100)
                                if pct // 10 != last_pct // 10:
                                    logger.info(f"  Pull {model_name}: {pct}% ({status})")
                                    last_pct = pct
                            elif status:
                                logger.info(f"  Pull {model_name}: {status}")
                        except Exception:
                            pass

            # Verify model is present in tags
            installed = self.list_installed_models()
            base = model_name.split(":")[0].lower()
            for inst in installed:
                if base in inst.lower():
                    logger.info(f"Model pull successful: {inst}")
                    return True
            return False
        except Exception as e:
            logger.error(f"Pull failed for {model_name}: {e}")
            return False

    @staticmethod
    def prepare_image_b64(image_input: Any) -> Tuple[Optional[str], Optional[str]]:
        """
        Convert file path, bytes, or PIL Image into clean Base64 JPEG string (max 1024px).
        Returns (base64_string, error_message).
        """
        try:
            if isinstance(image_input, (str, Path)):
                path = Path(image_input)
                if not path.exists():
                    return None, f"File not found: {path}"
                img = Image.open(path)
            elif isinstance(image_input, bytes):
                img = Image.open(io.BytesIO(image_input))
            elif isinstance(image_input, Image.Image):
                img = image_input
            else:
                return None, f"Unsupported image input type: {type(image_input)}"

            # Ensure RGB
            if img.mode != "RGB":
                img = img.convert("RGB")

            # Resize if dimensions exceed 1024 to protect memory & speed
            max_dim = 1024
            if max(img.size) > max_dim:
                img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)

            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=90)
            b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            return b64, None
        except Exception as e:
            return None, f"Failed to encode image: {e}"

    def query_vision(
        self,
        model_name: str,
        prompt: str,
        image_input: Any,
        num_predict: int = 1024,
        temperature: float = 0.2,
    ) -> Dict[str, Any]:
        """
        Unified vision request:
        1. Encodes image
        2. Tries /api/chat endpoint with messages & images payload (preferred for vision)
        3. If /api/chat fails or returns empty, tries /api/generate endpoint
        4. Validates output is non-empty
        Returns: {"success": bool, "text": str, "error": str, "code": str, "raw": dict}
        """
        img_b64, err = self.prepare_image_b64(image_input)
        if err:
            return {
                "success": False,
                "text": "",
                "error": err,
                "code": OllamaErrorCode.IMAGE_READ_ERROR,
                "raw": {},
            }

        # ── Endpoint 1: /api/chat (Primary & recommended for multimodal) ──────
        chat_payload = {
            "model": model_name,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [img_b64],
                }
            ],
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": num_predict,
            },
        }

        try:
            r = httpx.post(
                f"{self.host}/api/chat",
                json=chat_payload,
                timeout=self.timeout,
            )
            if r.status_code == 200:
                res_json = r.json()
                msg = res_json.get("message", {})
                content = str(msg.get("content", "")).strip()
                if content:
                    return {
                        "success": True,
                        "text": content,
                        "error": "",
                        "code": "",
                        "raw": res_json,
                    }
                else:
                    logger.warning(
                        f"[/api/chat] Empty content returned for model {model_name}. "
                        f"done={res_json.get('done')}, done_reason={res_json.get('done_reason')}"
                    )
            else:
                logger.warning(f"[/api/chat] HTTP {r.status_code} from Ollama: {r.text[:200]}")
        except httpx.TimeoutException:
            logger.warning(f"[/api/chat] Timeout ({self.timeout}s) querying {model_name}")
        except Exception as e:
            logger.warning(f"[/api/chat] Error querying {model_name}: {e}")

        # ── Endpoint 2: /api/generate (Fallback) ─────────────────────────────
        gen_payload = {
            "model": model_name,
            "prompt": prompt,
            "images": [img_b64],
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": num_predict,
            },
        }

        try:
            r = httpx.post(
                f"{self.host}/api/generate",
                json=gen_payload,
                timeout=self.timeout,
            )
            if r.status_code == 200:
                res_json = r.json()
                resp_text = str(res_json.get("response", "")).strip()
                if resp_text:
                    return {
                        "success": True,
                        "text": resp_text,
                        "error": "",
                        "code": "",
                        "raw": res_json,
                    }
                else:
                    logger.error(
                        f"[/api/generate] Empty response for model {model_name}. "
                        f"done={res_json.get('done')}, done_reason={res_json.get('done_reason')}"
                    )
                    return {
                        "success": False,
                        "text": "",
                        "error": f"Model {model_name} returned an empty response (done_reason={res_json.get('done_reason')}).",
                        "code": OllamaErrorCode.EMPTY_RESPONSE,
                        "raw": res_json,
                    }
            elif r.status_code == 404:
                return {
                    "success": False,
                    "text": "",
                    "error": f"Model {model_name} not found in Ollama (HTTP 404).",
                    "code": OllamaErrorCode.MODEL_NOT_FOUND,
                    "raw": {},
                }
            else:
                return {
                    "success": False,
                    "text": "",
                    "error": f"Ollama HTTP error {r.status_code}: {r.text[:200]}",
                    "code": OllamaErrorCode.HTTP_ERROR,
                    "raw": {},
                }
        except httpx.TimeoutException:
            return {
                "success": False,
                "text": "",
                "error": f"Ollama vision inference timed out after {self.timeout}s",
                "code": OllamaErrorCode.TIMEOUT,
                "raw": {},
            }
        except Exception as e:
            return {
                "success": False,
                "text": "",
                "error": f"Ollama request error: {e}",
                "code": OllamaErrorCode.CONNECTION_ERROR,
                "raw": {},
            }


# Global client instance
ollama_client = OllamaVisionClient()


# ──────────────────────────────────────────────
# Part 3: Vision Model Validation
# ──────────────────────────────────────────────
def validate_vision_model(
    model_name: Optional[str] = None,
    test_image: Optional[Any] = None,
    allow_fallback: bool = True,
) -> Tuple[bool, str, Optional[str]]:
    """
    Comprehensive 7-point validation check:
    1. Ollama server reachable
    2. Model exists (or pull candidates)
    3. Model loaded
    4. Image input supplied & encoded
    5. Vision inference executed
    6. Non-empty text returned
    7. Output verified

    If requested model fails and allow_fallback=True, iterates candidate models.
    Returns: (is_valid, status_message, active_model_name)
    """
    logger.info("Starting Ollama vision model validation...")

    # 1. Reachability
    ok, msg = ollama_client.check_connection()
    if not ok:
        set_ollama_status(False, None, msg, OllamaErrorCode.SERVER_UNREACHABLE)
        logger.error(f"[ERROR] {msg}")
        return False, msg, None

    # Prepare candidate list
    candidates = []
    if model_name:
        candidates.append(model_name)
    for m in OLLAMA_VISION_MODELS:
        if m not in candidates:
            candidates.append(m)

    installed = ollama_client.list_installed_models()
    logger.info(f"Currently installed Ollama models: {installed or 'none'}")

    # Prepare patterned test image bytes if not given
    if test_image is None:
        test_image = create_patterned_test_image_bytes(256, 256)

    test_prompt = "Describe the colors, shapes, and objects in this image in one or two clear sentences."

    for candidate in candidates:
        logger.info(f"Validating vision model candidate: '{candidate}'...")

        # If not installed, pull it
        base = candidate.split(":")[0].lower()
        is_installed = any(base in inst.lower() for inst in installed)
        if not is_installed:
            logger.info(f"Model '{candidate}' not installed. Attempting pull...")
            pull_ok = ollama_client.pull_model(candidate)
            if not pull_ok:
                logger.warning(f"Could not pull candidate '{candidate}'. Trying next...")
                continue
            installed = ollama_client.list_installed_models()

        # Find exact installed tag name
        resolved_name = candidate
        for inst in installed:
            if base in inst.lower():
                resolved_name = inst
                break

        # Run actual image vision test with bounded retry
        test_passed = False
        for attempt in range(1, OLLAMA_MAX_RETRIES + 1):
            logger.info(f"Running vision inference test (Attempt {attempt}/{OLLAMA_MAX_RETRIES}) with '{resolved_name}'...")
            res = ollama_client.query_vision(
                model_name=resolved_name,
                prompt=test_prompt,
                image_input=test_image,
                num_predict=200,
                temperature=0.2,
            )
            if res["success"] and res["text"].strip():
                sample = res["text"].strip().replace("\n", " ")[:100]
                logger.info(f"✓ Vision test PASSED on '{resolved_name}': \"{sample}...\"")
                test_passed = True
                break
            else:
                logger.warning(
                    f"Vision test attempt {attempt} failed on '{resolved_name}': "
                    f"Code={res['code']}, Error={res['error']}"
                )
                time.sleep(1.0)

        if test_passed:
            set_ollama_status(True, resolved_name, "", "")
            # Persist runtime config for fast startup
            _save_runtime_config(resolved_name)
            return True, f"Vision model '{resolved_name}' validated successfully.", resolved_name

        if not allow_fallback:
            break

    # All candidates failed
    err_msg = "No vision model passed the image inference test. Please check Ollama server logs."
    set_ollama_status(False, None, err_msg, OllamaErrorCode.VISION_TEST_FAILED)
    logger.error(f"[ERROR] {err_msg}")
    return False, err_msg, None


def _save_runtime_config(model_name: str):
    """Save runtime config so backend & subsequent calls fast-path."""
    config_paths = [
        "/content/studio/.runtime_config.json",
        "/content/Upscale-AI/.runtime_config.json",
        ".runtime_config.json",
    ]
    for cp in config_paths:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(cp)), exist_ok=True)
            with open(cp, "w", encoding="utf-8") as f:
                json.dump({"ollama_ready": True, "ollama_model": model_name, "vision_tested": True}, f)
        except Exception:
            pass


def initialize_ollama() -> bool:
    """
    Fast startup initializer:
    Checks runtime config fast-path, else executes validate_vision_model.
    """
    global _active_model, _ollama_ready, _ollama_error

    # Fast path check
    for cp in ["/content/studio/.runtime_config.json", "/content/Upscale-AI/.runtime_config.json", ".runtime_config.json"]:
        if Path(cp).exists():
            try:
                with open(cp, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                if cfg.get("ollama_ready") and cfg.get("ollama_model"):
                    # Verify Ollama is still reachable
                    if ollama_client.check_connection()[0]:
                        _active_model = cfg["ollama_model"]
                        _ollama_ready = True
                        _ollama_error = ""
                        logger.info(f"Ollama ready from runtime config: {_active_model}")
                        return True
            except Exception:
                pass

    # Run full validation
    ok, msg, model = validate_vision_model()
    return ok


# ──────────────────────────────────────────────
# Metadata Generation & Extraction
# ──────────────────────────────────────────────
METADATA_PROMPT = """You are an Adobe Stock metadata specialist. Analyze this image and generate commercial stock metadata.

Output a SINGLE valid JSON object and nothing else:
{
  "title": "<factual subject-first title ≤200 characters, no forbidden hype words>",
  "keywords": ["<kw1>", "<kw2>", ... 25 to 49 specific commercial keywords ordered by importance],
  "category": <category integer from 1 to 22>
}

RULES:
- Title must be concise, descriptive, subject-first.
- Do NOT use forbidden words: beautiful, amazing, stunning, perfect, premium, high quality, breathtaking, gorgeous.
- Keywords must be lowercase, comma-separated in JSON array, no duplicates, max 49 keywords.
- Category map: 1=Animals, 2=Buildings, 3=Business, 4=Drinks, 5=Environment, 6=States of Mind, 7=Food, 8=Graphic Resources, 9=Hobbies, 10=Industry, 11=Landscape, 12=Lifestyle, 13=People, 14=Plants/Flowers, 15=Culture, 16=Science, 17=Social Issues, 18=Sports, 19=Technology, 20=Transport, 21=Travel, 22=Abstract/Backgrounds.
"""

CORRECTION_PROMPT = """The previous output was not valid JSON. Please fix it and return ONLY a valid JSON object:
{
  "title": "<subject-first title ≤200 chars>",
  "keywords": ["keyword1", "keyword2", ... up to 49 keywords],
  "category": 22
}
"""


def analyze_image(image_path: str) -> dict:
    """
    Perform full vision analysis on the actual image file to generate
    Adobe Stock metadata (title, keywords, category, releases).
    
    Robustness guarantees:
    - Bounded retries (up to 3 attempts)
    - JSON parsing + correction prompt retry
    - Heuristic extraction fallback if JSON fails
    - Clean failure return without throwing unhandled exceptions
    """
    global _active_model, _ollama_ready

    result = {
        "title": "",
        "keywords": [],
        "category": 22,
        "releases": "",
        "error": "",
        "error_code": "",
    }

    if not _ollama_ready or not _active_model:
        # Try self-initializing
        ok = initialize_ollama()
        if not ok:
            result["error"] = _ollama_error or "Ollama vision model is not ready."
            result["error_code"] = _last_error_code or OllamaErrorCode.VISION_TEST_FAILED
            return result

    if not Path(image_path).exists():
        result["error"] = f"Image file not found: {image_path}"
        result["error_code"] = OllamaErrorCode.IMAGE_READ_ERROR
        return result

    # Multi-attempt inference loop
    for attempt in range(1, OLLAMA_MAX_RETRIES + 1):
        prompt = METADATA_PROMPT if attempt == 1 else (METADATA_PROMPT + "\n\n" + CORRECTION_PROMPT)
        logger.info(f"Analyzing '{os.path.basename(image_path)}' with '{_active_model}' (Attempt {attempt}/{OLLAMA_MAX_RETRIES})...")

        res = ollama_client.query_vision(
            model_name=_active_model,
            prompt=prompt,
            image_input=image_path,
            num_predict=1024,
            temperature=0.1,
        )

        if not res["success"] or not res["text"].strip():
            logger.warning(f"Inference attempt {attempt} failed: Code={res['code']}, Error={res['error']}")
            if attempt == OLLAMA_MAX_RETRIES:
                result["error"] = res["error"] or "Ollama returned empty response after retries."
                result["error_code"] = res["code"] or OllamaErrorCode.EMPTY_RESPONSE
                return result
            time.sleep(1.5)
            continue

        # Parse JSON metadata
        parsed = _parse_metadata_response(res["text"])
        if "error" in parsed:
            logger.warning(f"JSON parse error on attempt {attempt}: {parsed['error']}")
            if attempt == OLLAMA_MAX_RETRIES:
                # Heuristic fallback to salvage title & keywords
                heuristic = _heuristic_metadata_extract(res["text"], os.path.basename(image_path))
                if heuristic:
                    logger.info("Heuristic metadata extraction succeeded as fallback.")
                    result.update(heuristic)
                    return result
                result["error"] = parsed["error"]
                result["error_code"] = OllamaErrorCode.JSON_INVALID
                return result
            time.sleep(1.0)
            continue

        # Success!
        result.update(parsed)
        result["releases"] = ""
        result["error"] = ""
        result["error_code"] = ""
        return result

    result["error"] = "Failed to generate metadata."
    result["error_code"] = OllamaErrorCode.EMPTY_RESPONSE
    return result


def _parse_metadata_response(text: str) -> dict:
    """
    Extract and validate JSON metadata from raw model output.
    Handles markdown code fences, embedded JSON, and field validation.
    """
    clean_text = text.strip()
    # Strip markdown block if enclosed
    clean_text = re.sub(r"^```(?:json)?\s*", "", clean_text, flags=re.IGNORECASE)
    clean_text = re.sub(r"\s*```$", "", clean_text)
    clean_text = clean_text.strip()

    # Find JSON structure { ... }
    match = re.search(r"\{[\s\S]*\}", clean_text)
    if not match:
        return {"error": f"No JSON object found in response: {clean_text[:150]}"}

    json_str = match.group(0)

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        return {"error": f"JSON decode error: {e}"}

    # Validate title
    title = str(data.get("title", "")).strip()
    if not title:
        return {"error": "JSON missing required 'title' field."}
    title = _clean_title(title)

    # Validate keywords
    raw_kws = data.get("keywords", [])
    if isinstance(raw_kws, str):
        raw_kws = [k.strip() for k in raw_kws.split(",")]
    keywords = _clean_keywords(raw_kws)
    if not keywords:
        keywords = ["stock", "graphic", "image", "concept"]

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


def _heuristic_metadata_extract(text: str, fallback_subject: str) -> Optional[dict]:
    """Fallback extractor if model output did not adhere to strict JSON."""
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if not lines:
        return None

    # Use first informative line as title
    title = lines[0]
    title = re.sub(r"^[\*#\-\d\.\:\s]+", "", title).strip()
    title = _clean_title(title)
    if len(title) < 5:
        title = f"Stock graphic image of {fallback_subject}"

    # Extract comma-separated words for keywords
    all_words = re.findall(r"\b[a-zA-Z]{3,20}\b", text.lower())
    stop_words = {"the", "and", "for", "with", "this", "that", "image", "photo", "json", "title", "keywords", "category"}
    filtered_kws = [w for w in all_words if w not in stop_words]
    keywords = _clean_keywords(filtered_kws)

    return {
        "title": title[:MAX_TITLE_LENGTH],
        "keywords": keywords[:MAX_KEYWORDS],
        "category": 22,
    }


# ──────────────────────────────────────────────
# Sanitization & Cleaning Helpers
# ──────────────────────────────────────────────
BANNED_TITLE_WORDS = {
    "beautiful", "amazing", "stunning", "perfect", "best", "premium",
    "high quality", "breathtaking", "gorgeous", "incredible", "wonderful",
    "fantastic", "excellent", "superb", "extraordinary", "ultra",
}


def _clean_title(title: str) -> str:
    """Strip banned hype words and normalize punctuation/spaces."""
    # Remove quotes
    title = re.sub(r'^["\']|["\']$', "", title.strip())
    for word in sorted(BANNED_TITLE_WORDS, key=len, reverse=True):
        pattern = re.compile(rf"\b{re.escape(word)}\b", re.IGNORECASE)
        title = pattern.sub("", title)
    title = re.sub(r"\s+", " ", title).strip()
    title = re.sub(r"^[,\-\:\s]+", "", title).strip()
    return title


def _clean_keywords(keywords: list) -> list:
    """Deduplicate, lowercase, strip, and sanitize keywords."""
    seen = set()
    cleaned = []
    for kw in keywords:
        if not isinstance(kw, str):
            continue
        kw = kw.strip().lower()
        # Remove quotes, punctuation, numbers
        kw = re.sub(r"[^\w\s\-]", "", kw).strip()
        if not kw or len(kw) < 2 or kw in seen:
            continue
        seen.add(kw)
        cleaned.append(kw)
        if len(cleaned) >= MAX_KEYWORDS:
            break
    return cleaned
