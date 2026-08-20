import os
import logging
from PIL import Image

logger = logging.getLogger("AdobeStockUpscaler.QC")

def run_technical_qc(
    output_path: str,
    original_w: int,
    original_h: int,
    req_format: str
) -> dict:
    """
    Runs technical quality control checks on the upscaled output file.
    Returns structured result dictionary:
    {
      "passed": bool,
      "hard_failures": list[str],
      "warnings": list[str],
      "input": {"width": int, "height": int},
      "output": {"width": int, "height": int},
      "megapixels": float,
      "checks": dict
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
            "size": "pass"
        }
    }

    try:
        # 1. File Integrity Check
        with Image.open(output_path) as img:
            img.verify()

        with Image.open(output_path) as img:
            out_w, out_h = img.size
            img_format = img.format
            img_mode = img.mode

        result["output"]["width"] = out_w
        result["output"]["height"] = out_h

        # 2. Megapixels Check: 4MP threshold
        mp = (out_w * out_h) / 1_000_000.0
        result["megapixels"] = round(mp, 2)
        if mp < 4.0:
            result["checks"]["resolution"] = "warn"
            result["warnings"].append(f"Resolution is {mp:.2f} MP (minimum recommended is 4.0 MP for Adobe Stock)")

        # 3. Format Container Check
        expected_format = "JPEG" if req_format.lower() in ["jpg", "jpeg"] else "PNG"
        if img_format and img_format.upper() != expected_format:
            result["checks"]["format"] = "fail"
            result["hard_failures"].append(f"Format mismatch: expected {expected_format}, got {img_format}")

        # 4. Aspect Ratio Drift Check (5% mathematically reasonable tolerance)
        orig_ratio = original_w / original_h
        out_ratio = out_w / out_h
        diff = abs(orig_ratio - out_ratio)
        if diff > 0.05:
            result["checks"]["aspect_ratio"] = "fail"
            result["hard_failures"].append(f"Aspect ratio drift: original {orig_ratio:.3f} vs output {out_ratio:.3f}")
        elif diff > 0.02:
            result["checks"]["aspect_ratio"] = "warn"
            result["warnings"].append(f"Minor aspect ratio shift: {diff:.3f}")

        # 5. PNG Transparency Check
        if req_format.lower() == "png":
            if "A" not in img_mode and img_mode != "RGBA" and img_mode != "LA":
                result["checks"]["transparency"] = "warn"
                result["warnings"].append("PNG output lacks alpha transparency channel")

        # 6. Excessive File Size Warning Check
        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        if req_format.lower() in ["jpg", "jpeg"] and file_size_mb > 50.0:
            result["checks"]["size"] = "warn"
            result["warnings"].append(f"Large output file size: {file_size_mb:.1f} MB")
        elif req_format.lower() == "png" and file_size_mb > 100.0:
            result["checks"]["size"] = "warn"
            result["warnings"].append(f"Large PNG output file size: {file_size_mb:.1f} MB")

    except Exception as e:
        logger.error(f"QC file integrity exception for {output_path}: {str(e)}")
        result["checks"]["integrity"] = "fail"
        result["hard_failures"].append(f"File integrity corrupted: {str(e)}")

    result["passed"] = len(result["hard_failures"]) == 0
    return result
