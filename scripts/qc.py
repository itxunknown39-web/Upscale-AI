"""
scripts/qc.py — Adobe Stock AI Studio

Technical Quality Control for upscaled images.
Preserved from original with 6 verification checks.
"""

import os
import logging
from PIL import Image

logger = logging.getLogger("AdobeStockStudio.QC")

MIN_MEGAPIXELS = 4.0          # Adobe Stock minimum
MAX_FILE_SIZE_MB = 45.0        # Adobe Stock upload limit
MAX_ASPECT_RATIO_DRIFT = 0.02  # 2% tolerance


def run_technical_qc(
    output_path: str,
    original_w: int,
    original_h: int,
    req_format: str,
) -> dict:
    """
    Runs 6 technical QC checks on the upscaled output file.

    Returns:
    {
      "passed": bool,
      "hard_failures": list[str],
      "warnings": list[str],
      "input": {"width": int, "height": int},
      "output": {"width": int, "height": int},
      "megapixels": float,
      "checks": {
          "resolution": "pass"|"fail"|"warn",
          "format": "pass"|"fail",
          "integrity": "pass"|"fail",
          "aspect_ratio": "pass"|"warn",
          "transparency": "pass"|"warn",
          "size": "pass"|"warn"|"fail",
      }
    }
    """
    result = {
        "passed": True,
        "hard_failures": [],
        "warnings": [],
        "input": {"width": original_w, "height": original_h},
        "output": {"width": 0, "height": 0},
        "megapixels": 0.0,
        "checks": {
            "resolution": "pass",
            "format": "pass",
            "integrity": "pass",
            "aspect_ratio": "pass",
            "transparency": "pass",
            "size": "pass",
        }
    }

    try:
        # ── 1. File Integrity ──────────────────────────────────────
        try:
            with Image.open(output_path) as img:
                img.verify()
        except Exception as e:
            result["checks"]["integrity"] = "fail"
            result["hard_failures"].append(f"File integrity check failed: {e}")
            result["passed"] = False
            return result  # Can't proceed without valid file

        # ── 2. Open & measure ─────────────────────────────────────
        with Image.open(output_path) as img:
            out_w, out_h = img.size
            img_format = img.format or ""
            has_alpha = img.mode in ("RGBA", "LA", "PA")
            mode = img.mode

        result["output"]["width"] = out_w
        result["output"]["height"] = out_h
        megapixels = round((out_w * out_h) / 1_000_000.0, 2)
        result["megapixels"] = megapixels

        # ── 3. Resolution ──────────────────────────────────────────
        if megapixels < MIN_MEGAPIXELS:
            result["checks"]["resolution"] = "fail"
            result["hard_failures"].append(
                f"Resolution too low: {megapixels:.2f} MP (minimum {MIN_MEGAPIXELS} MP required)"
            )
            result["passed"] = False
        else:
            result["checks"]["resolution"] = "pass"

        # ── 4. Format ──────────────────────────────────────────────
        expected_formats = {"jpg": {"JPEG"}, "jpeg": {"JPEG"}, "png": {"PNG"}, "webp": {"WEBP"}}
        req_fmt_lower = req_format.lower()
        allowed = expected_formats.get(req_fmt_lower, {img_format})
        if img_format not in allowed:
            result["checks"]["format"] = "fail"
            result["hard_failures"].append(
                f"Format mismatch: expected {req_format.upper()}, got {img_format}"
            )
            result["passed"] = False

        # ── 5. Aspect Ratio Drift ──────────────────────────────────
        if original_w > 0 and original_h > 0:
            orig_ratio = original_w / original_h
            out_ratio = out_w / out_h
            drift = abs(orig_ratio - out_ratio) / orig_ratio
            if drift > MAX_ASPECT_RATIO_DRIFT:
                result["checks"]["aspect_ratio"] = "warn"
                result["warnings"].append(
                    f"Aspect ratio drift detected: {drift*100:.2f}% (>{MAX_ASPECT_RATIO_DRIFT*100}% threshold)"
                )

        # ── 6. Transparency ────────────────────────────────────────
        if has_alpha and req_fmt_lower in ("jpg", "jpeg"):
            result["checks"]["transparency"] = "warn"
            result["warnings"].append("Source has alpha channel but output format is JPEG (transparency lost)")

        # ── 7. File Size ───────────────────────────────────────────
        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        if file_size_mb > MAX_FILE_SIZE_MB:
            result["checks"]["size"] = "warn"
            result["warnings"].append(
                f"File size {file_size_mb:.1f} MB exceeds Adobe Stock limit of {MAX_FILE_SIZE_MB} MB"
            )

    except Exception as e:
        result["checks"]["integrity"] = "fail"
        result["hard_failures"].append(f"QC exception: {e}")
        result["passed"] = False

    return result
