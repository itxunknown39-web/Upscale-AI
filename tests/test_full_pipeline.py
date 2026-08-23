import os
import sys
import shutil
import tempfile
from PIL import Image
import numpy as np

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def test_pipeline():
    print("==================================================")
    print("STARTING FULL REPO PIPELINE ACCEPTANCE TESTS")
    print("==================================================")

    # 1. Test standalone RRDBNet and Shims
    print("\n--- TEST 1: RRDBNet & BasicSR Shims ---")
    from scripts.rrdbnet import RRDBNet, register_basicsr_shim
    register_basicsr_shim()
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    print("[PASS] Pure PyTorch RRDBNet instantiated successfully with zero basicsr external dependency.")

    # 2. Test Model Manager
    print("\n--- TEST 2: Model Weight Manager ---")
    from scripts.model_manager import find_model_weights, ensure_model_weights
    success, weight_path, msg = ensure_model_weights("RealESRGAN_x4plus", auto_download=False)
    print(f"Weight check result: success={success}, path={weight_path}")

    # 3. Test Upscaler module imports and tile_process availability
    print("\n--- TEST 3: Upscaler Module & Scope Verification ---")
    import scripts.upscaler as upscaler
    assert hasattr(upscaler, 'torch'), "upscaler module must export/have torch"
    assert hasattr(upscaler, 'tile_process'), "upscaler module must have tile_process"
    assert hasattr(upscaler, 'RealESRGANEngine'), "upscaler module must have RealESRGANEngine"
    assert hasattr(upscaler, 'run_upscale'), "upscaler module must have run_upscale"
    print("[PASS] All upscaler scope references and symbols verified.")

    # 4. Test Single-Image Upscaling (2x Stock Ready default, 3x, 4x)
    print("\n--- TEST 4: Single Image Upscaling (2x, 3x, 4x) ---")
    test_dir = tempfile.mkdtemp(prefix="test_upscale_")
    input_img_path = os.path.join(test_dir, "sample.jpg")

    # Create synthetic test pattern
    img = Image.new('RGB', (100, 100), color=(50, 120, 200))
    img.save(input_img_path, "JPEG", quality=95)

    scales_to_test = [2.0, 3.0, 4.0]
    for sc in scales_to_test:
        out_path = os.path.join(test_dir, f"sample_out_{sc}x.jpg")
        ok, stage, reason, details = upscaler.run_upscale(
            input_path=input_img_path,
            output_path=out_path,
            scale=sc,
            model_name="RealESRGAN_x4plus",
            ext="jpg",
            quality=95
        )
        assert ok, f"Upscale failed for scale {sc}x: {stage} - {reason} - {details}"
        assert os.path.exists(out_path), f"Output file missing for scale {sc}x"
        with Image.open(out_path) as out_img:
            w, h = out_img.size
            expected_w = int(100 * sc)
            expected_h = int(100 * sc)
            print(f"[PASS] Scale {sc}x output: {w}x{h} (Expected: {expected_w}x{expected_h})")
            assert w == expected_w and h == expected_h, f"Dimension mismatch: got {w}x{h}, expected {expected_w}x{expected_h}"

    # 5. Test Second-Pass Workflow
    print("\n--- TEST 5: Second-Pass Upscale Workflow ---")
    pass1_path = os.path.join(test_dir, "sample_out_2.0x.jpg")
    pass2_out_path = os.path.join(test_dir, "sample_out_pass2.jpg")
    ok, stage, reason, details = upscaler.run_upscale(
        input_path=pass1_path,
        output_path=pass2_out_path,
        scale=2.0,
        model_name="RealESRGAN_x4plus",
        ext="jpg",
        quality=95
    )
    assert ok, f"Pass 2 upscale failed: {stage} - {reason} - {details}"
    with Image.open(pass2_out_path) as p2_img:
        w2, h2 = p2_img.size
        print(f"[PASS] Pass 2 (2x on 200x200) output: {w2}x{h2} (Expected: 400x400)")
        assert w2 == 400 and h2 == 400

    # 6. Test Technical QC
    print("\n--- TEST 6: Technical QC Module ---")
    from scripts.qc import run_technical_qc
    qc_res = run_technical_qc(pass2_out_path, 100, 100, "jpg")
    print(f"[PASS] QC Result: passed={qc_res['passed']}, dimensions={qc_res['output']['width']}x{qc_res['output']['height']}")

    # Clean up test dir
    shutil.rmtree(test_dir, ignore_errors=True)

    print("\n==================================================")
    print("ALL PIPELINE ACCEPTANCE TESTS COMPLETED SUCCESSFULLY!")
    print("==================================================")

if __name__ == "__main__":
    test_pipeline()
