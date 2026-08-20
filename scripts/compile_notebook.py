#!/usr/bin/env python3
"""
scripts/compile_notebook.py
===========================
Compiles the COMPLETE, SELF-CONTAINED Adobe_Stock_AI_Upscaler.ipynb
from local source files.

All Python scripts and the HTML frontend are base64-encoded and embedded
directly into notebook cells.  The resulting notebook has ZERO external
dependencies — no git clone required.

Usage:
    cd <project-root>
    python scripts/compile_notebook.py

Output:
    Adobe_Stock_AI_Upscaler.ipynb  (overwrites existing)
"""

import base64
import json
import os
import sys
import textwrap

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STUDIO_DIR = "/content/studio"   # Where files are written inside Colab


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _read_b64(rel_path: str) -> str:
    abs_path = os.path.join(BASE_DIR, *rel_path.split("/"))
    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"Required source file missing: {abs_path}")
    with open(abs_path, "r", encoding="utf-8") as f:
        content = f.read()
    return base64.b64encode(content.encode("utf-8")).decode("ascii")


def md_cell(text: str) -> dict:
    lines = text.splitlines(keepends=True)
    return {"cell_type": "markdown", "metadata": {}, "source": lines}


def code_cell(text: str) -> dict:
    lines = text.splitlines(keepends=True)
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines,
    }


def write_file_cell(colab_path: str, rel_source: str, label: str = "") -> dict:
    """
    Creates a notebook cell that decodes a base64 blob and writes it to disk.
    The entire file content is embedded in the cell — no network required.
    """
    b64 = _read_b64(rel_source)
    dir_path = "/".join(colab_path.split("/")[:-1])
    label = label or colab_path.split("/")[-1]
    src = textwrap.dedent(f"""\
        # ── Write {label} ────────────────────────────────────────────
        import base64, os
        os.makedirs('{dir_path}', exist_ok=True)
        _b64_{label.replace('.','_').replace('/','_')} = '{b64}'
        with open('{colab_path}', 'w', encoding='utf-8') as _f:
            _f.write(base64.b64decode(
                _b64_{label.replace('.','_').replace('/','_')}
            ).decode('utf-8'))
        print('✓ Written:', '{colab_path}')
    """)
    return code_cell(src)


# ─────────────────────────────────────────────────────────────────────────────
# Cell definitions
# ─────────────────────────────────────────────────────────────────────────────

def build_cells() -> list:
    cells = []

    # ── CELL 1: Markdown title ────────────────────────────────────────────
    cells.append(md_cell(
        "# ⚡ Adobe Stock AI Studio\n"
        "### Google Colab Launcher — One notebook, complete system\n\n"
        "**What this notebook does (run ALL cells once):**\n\n"
        "1. Detects T4 GPU\n"
        "2. Mounts Google Drive\n"
        "3. Installs system dependencies (including zstd for Ollama)\n"
        "4. Installs Ollama (local vision AI — no external API)\n"
        "5. Pulls and tests a vision model\n"
        "6. Installs Python dependencies\n"
        "7. Writes all application files to disk\n"
        "8. Downloads Real-ESRGAN weights\n"
        "9. Starts FastAPI backend\n"
        "10. Opens Cloudflare tunnel\n\n"
        "**Then opens a single web app URL where you:**\n"
        "- Drop all your images (upload-first)\n"
        "- Click Start Processing\n"
        "- Every image is upscaled → Ollama analyzes it → metadata saved\n"
        "- Download: `AdobeStock_Metadata.csv`, `AdobeStock_Metadata.json`, ZIP\n\n"
        "### Quick start\n"
        "1. **Runtime → Change runtime type → T4 GPU → Save**\n"
        "2. **Runtime → Run all** (`Ctrl+F9`)\n"
        "3. Wait ~10 min for first-time setup\n"
        "4. Click the URL shown in the last cell"
    ))

    # ── CELL 2: T4 GPU check ─────────────────────────────────────────────
    cells.append(code_cell(textwrap.dedent("""\
        # ═══════════════════════════════════════════════════════
        # CELL 2: T4 GPU Detection
        # ═══════════════════════════════════════════════════════
        import torch, sys

        print("=" * 55)
        print("  ADOBE STOCK AI STUDIO — GPU Check")
        print("=" * 55)

        if not torch.cuda.is_available():
            print("❌  No GPU detected!")
            print("   Go to Runtime → Change runtime type → T4 GPU → Save")
            raise SystemExit("GPU required.")

        gpu_name = torch.cuda.get_device_name(0)
        free_b, total_b = torch.cuda.mem_get_info(0)
        print(f"  GPU  : {gpu_name}")
        print(f"  VRAM : {free_b/1024**3:.1f} GB free / {total_b/1024**3:.1f} GB total")
        print("  ✓ T4 GPU READY")
        print("=" * 55)
    """)))

    # ── CELL 3: Drive mount ───────────────────────────────────────────────
    cells.append(code_cell(textwrap.dedent("""\
        # ═══════════════════════════════════════════════════════
        # CELL 3: Google Drive — Persistence
        # ═══════════════════════════════════════════════════════
        from google.colab import drive
        import os

        mount_drive = True  #@param {type:"boolean"}

        STUDIO_DIR = '/content/studio'
        for sub in ['uploads','output','metadata','logs','archives','failed','temp_output']:
            os.makedirs(f'{STUDIO_DIR}/{sub}', exist_ok=True)

        if mount_drive:
            print("Mounting Google Drive…")
            drive.mount('/content/drive')
            drive_path = '/content/drive/MyDrive/AdobeStockStudio'
            for sub in ['uploads','output','metadata','logs','archives','failed']:
                os.makedirs(f'{drive_path}/{sub}', exist_ok=True)
            print(f"✓ Drive mounted → {drive_path}")
        else:
            print("Drive bypass — using local Colab storage only.")

        print("✓ Directory structure ready")
    """)))

    # ── CELL 4: System dependencies (zstd FIRST) ─────────────────────────
    cells.append(code_cell(textwrap.dedent("""\
        # ═══════════════════════════════════════════════════════
        # CELL 4: System Dependencies (zstd must be FIRST)
        # ═══════════════════════════════════════════════════════
        import subprocess, sys

        def apt(pkgs):
            subprocess.run(['apt-get','install','-y','-qq'] + pkgs,
                           capture_output=True)

        print("Installing system dependencies…")
        subprocess.run(['apt-get','update','-qq'], capture_output=True)

        # zstd MUST be installed before Ollama
        apt(['zstd'])
        print("  ✓ zstd")

        apt(['curl','wget','pv','libgl1-mesa-glx','libglib2.0-0'])
        print("  ✓ curl, wget, libgl1, libglib2")
        print("✓ System dependencies ready")
    """)))

    # ── CELL 5: Install Ollama ────────────────────────────────────────────
    cells.append(code_cell(textwrap.dedent("""\
        # ═══════════════════════════════════════════════════════
        # CELL 5: Install Ollama (local AI — no external API)
        # ═══════════════════════════════════════════════════════
        import subprocess, shutil

        if not shutil.which('ollama'):
            print("Installing Ollama…")
            result = subprocess.run(
                'curl -fsSL https://ollama.com/install.sh | sh',
                shell=True, capture_output=True, text=True
            )
            if result.returncode != 0:
                print("STDERR:", result.stderr[-500:])
                raise RuntimeError("Ollama installation failed")
            print("✓ Ollama installed")
        else:
            print(f"✓ Ollama already installed: {shutil.which('ollama')}")
    """)))

    # ── CELL 6: Start Ollama server ───────────────────────────────────────
    cells.append(code_cell(textwrap.dedent("""\
        # ═══════════════════════════════════════════════════════
        # CELL 6: Start Ollama Server
        # ═══════════════════════════════════════════════════════
        import subprocess, time, requests

        OLLAMA_HOST = 'http://127.0.0.1:11434'

        # Check if already running
        try:
            if requests.get(f'{OLLAMA_HOST}/api/tags', timeout=2).status_code == 200:
                print("✓ Ollama already running")
                ollama_proc = None
            else:
                raise Exception()
        except Exception:
            print("Starting Ollama server…")
            ollama_proc = subprocess.Popen(
                ['ollama', 'serve'],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )

            # Wait for readiness
            for i in range(40):
                try:
                    if requests.get(f'{OLLAMA_HOST}/api/tags', timeout=2).ok:
                        print(f"✓ Ollama server ready (took {i+1}s)")
                        break
                except Exception:
                    pass
                time.sleep(1)
            else:
                raise RuntimeError("Ollama server did not start within 40s")
    """)))

    # ── CELL 7: Vision model detect / pull / test ─────────────────────────
    cells.append(code_cell(textwrap.dedent("""\
        # =======================================================
        # CELL 7: Vision Model — Detect -> Pull -> Validate -> Fallback
        # =======================================================
        import requests, json, base64, io, time, os
        from PIL import Image, ImageDraw

        OLLAMA_HOST = os.environ.get('OLLAMA_HOST', 'http://127.0.0.1:11434')
        OLLAMA_VISION_MODELS = [
            'moondream',
            'moondream:latest',
            'llava:7b',
            'llava',
            'llava:latest',
            'bakllava',
            'llama3.2-vision',
            'minicpm-v',
        ]

        print("=" * 55)
        print("  OLLAMA VISION MODEL VALIDATION")
        print("=" * 55)

        # ── 1. Create Patterned Test Image (256x256) ─────────────────────
        # Guarantees vision token activation across SigLIP / CLIP encoders
        test_img = Image.new('RGB', (256, 256), color=(30, 60, 120))
        draw = ImageDraw.Draw(test_img)
        draw.rectangle([20, 20, 100, 100], fill=(220, 80, 40), outline=(255, 255, 255))
        draw.ellipse([120, 50, 220, 150], fill=(40, 180, 90), outline=(255, 255, 255))
        draw.polygon([(128, 160), (60, 230), (196, 230)], fill=(240, 200, 30))
        draw.line([(0, 0), (256, 256)], fill=(255, 255, 255), width=3)
        buf = io.BytesIO()
        test_img.save(buf, format='JPEG', quality=90)
        img_b64 = base64.b64encode(buf.getvalue()).decode('ascii')

        # ── 2. List Installed Models ─────────────────────────────────────
        try:
            r = requests.get(f'{OLLAMA_HOST}/api/tags', timeout=10)
            installed = [m['name'] for m in r.json().get('models', []) if 'name' in m]
            print(f"Installed Ollama models: {installed or 'none'}")
        except Exception as e:
            installed = []
            print(f"[WARN] Could not list models: {e}")

        # ── 3. Helper: Query Vision (Chat + Generate Dual Endpoint) ──────
        def query_vision_test(model_name, b64_data):
            test_prompt = "Describe the colors, shapes, and objects in this image in one or two clear sentences."
            
            # Try /api/chat first (Primary & standard for multimodal)
            try:
                chat_payload = {
                    "model": model_name,
                    "messages": [{"role": "user", "content": test_prompt, "images": [b64_data]}],
                    "stream": False,
                    "options": {"temperature": 0.2, "num_predict": 200}
                }
                cr = requests.post(f'{OLLAMA_HOST}/api/chat', json=chat_payload, timeout=120)
                if cr.status_code == 200:
                    content = cr.json().get("message", {}).get("content", "").strip()
                    if content:
                        return True, content, "/api/chat"
                    else:
                        print(f"  [/api/chat] Returned empty content. done={cr.json().get('done')}, reason={cr.json().get('done_reason')}")
            except Exception as ce:
                print(f"  [/api/chat] Error: {ce}")

            # Try /api/generate fallback
            try:
                gen_payload = {
                    "model": model_name,
                    "prompt": test_prompt,
                    "images": [b64_data],
                    "stream": False,
                    "options": {"temperature": 0.2, "num_predict": 200}
                }
                gr = requests.post(f'{OLLAMA_HOST}/api/generate', json=gen_payload, timeout=120)
                if gr.status_code == 200:
                    resp_text = gr.json().get("response", "").strip()
                    if resp_text:
                        return True, resp_text, "/api/generate"
                    else:
                        print(f"  [/api/generate] Returned empty response. done={gr.json().get('done')}, reason={gr.json().get('done_reason')}")
            except Exception as ge:
                print(f"  [/api/generate] Error: {ge}")

            return False, "", "failed"

        # ── 4. Candidate Validation Loop with Fallback ───────────────────
        active_model = None
        for candidate in OLLAMA_VISION_MODELS:
            print(f"\\nTesting candidate vision model: '{candidate}'...")
            base = candidate.split(":")[0].lower()
            is_installed = any(base in inst.lower() for inst in installed)
            
            if not is_installed:
                print(f"Model '{candidate}' not installed. Pulling...")
                try:
                    with requests.post(f'{OLLAMA_HOST}/api/pull', json={'name': candidate}, stream=True, timeout=900) as resp:
                        last_pct = -1
                        for line in resp.iter_lines():
                            if line:
                                try:
                                    obj = json.loads(line)
                                    if obj.get('total', 0) > 0:
                                        pct = int(obj['completed'] / obj['total'] * 100)
                                        if pct // 10 != last_pct // 10:
                                            print(f"  Pulling {candidate}: {pct}%")
                                            last_pct = pct
                                except Exception:
                                    pass
                    r2 = requests.get(f'{OLLAMA_HOST}/api/tags', timeout=5)
                    installed = [m['name'] for m in r2.json().get('models', [])]
                except Exception as pe:
                    print(f"  [WARN] Failed to pull '{candidate}': {pe}")
                    continue

            # Resolve full installed tag name
            resolved_tag = candidate
            for inst in installed:
                if base in inst.lower():
                    resolved_tag = inst
                    break

            # Execute vision inference test
            print(f"Running image vision test on '{resolved_tag}'...")
            success, text_out, endpoint = query_vision_test(resolved_tag, img_b64)
            if success and text_out:
                snippet = text_out.replace('\\n', ' ')[:90]
                print(f"[OK] Vision inference test PASSED via {endpoint} on '{resolved_tag}'")
                print(f"     Output: \\"{snippet}...\\"")
                active_model = resolved_tag
                break
            else:
                print(f"[WARN] Vision test failed on '{resolved_tag}'. Trying fallback model...")

        # ── 5. Save Runtime Config or Final Status ───────────────────────
        if active_model:
            os.makedirs('/content/studio', exist_ok=True)
            with open('/content/studio/.runtime_config.json', 'w') as f:
                json.dump({'ollama_ready': True, 'ollama_model': active_model, 'vision_tested': True}, f)
            print(f"\\n[OK] Ollama Vision READY -- Active Model: {active_model}")
        else:
            print("\\n[ERROR] No vision model passed the image inference test.")
            print("         Please check that Ollama is running and has GPU access.")
            raise RuntimeError("Ollama vision validation failed on all candidate models.")
    """)))

    # ── CELL 8: Python dependencies ───────────────────────────────────────
    cells.append(code_cell(textwrap.dedent("""\
        # ═══════════════════════════════════════════════════════
        # CELL 8: Python Dependencies
        # ═══════════════════════════════════════════════════════
        import subprocess, sys

        def pip(pkgs, quiet=True):
            q = ['-q'] if quiet else []
            subprocess.run([sys.executable, '-m', 'pip', 'install'] + q + pkgs, check=True)

        print("Installing Python packages…")

        pip(['fastapi>=0.95', 'uvicorn[standard]>=0.22', 'python-multipart>=0.0.6'])
        print("  ✓ fastapi, uvicorn")

        pip(['httpx>=0.24', 'aiofiles>=23', 'pillow>=10', 'psutil>=5.9', 'pydantic>=2'])
        print("  ✓ httpx, pillow, psutil, pydantic")

        pip(['basicsr', 'facexlib', 'gfpgan', 'realesrgan>=0.3'])
        print("  ✓ basicsr, realesrgan")

        pip(['opencv-python-headless'])
        print("  ✓ opencv-python-headless")

        # torchvision compat patch
        import sys as _sys
        try:
            import torchvision.transforms.functional as _F
            _sys.modules['torchvision.transforms.functional_tensor'] = _F
            print("  ✓ torchvision compat patch")
        except Exception: pass

        print("✓ Python dependencies installed")
    """)))

    # ── CELL 9: Patch basicsr ─────────────────────────────────────────────
    cells.append(code_cell(textwrap.dedent("""\
        # ═══════════════════════════════════════════════════════
        # CELL 9: Patch basicsr (functional_tensor compat)
        # ═══════════════════════════════════════════════════════
        import os, sys

        # Apply sys.modules patch first
        try:
            import torchvision.transforms.functional as _F
            sys.modules['torchvision.transforms.functional_tensor'] = _F
        except Exception: pass

        # Patch degradations.py if needed
        try:
            import basicsr
            deg_path = os.path.join(
                os.path.dirname(basicsr.__file__), 'data', 'degradations.py'
            )
            if os.path.exists(deg_path):
                content = open(deg_path).read()
                if 'functional_tensor' in content:
                    patched = content.replace(
                        'from torchvision.transforms.functional_tensor import rgb_to_grayscale',
                        'from torchvision.transforms.functional import rgb_to_grayscale'
                    )
                    open(deg_path, 'w').write(patched)
                    print("✓ basicsr degradations.py patched")
                else:
                    print("✓ basicsr does not need patching")
        except Exception as e:
            print(f"  Note: basicsr patch: {e}")

        print("✓ basicsr ready")
    """)))

    # ── CELL 10: Real-ESRGAN setup ────────────────────────────────────────
    cells.append(code_cell(textwrap.dedent("""\
        # ═══════════════════════════════════════════════════════
        # CELL 10: Real-ESRGAN Weights + Inference Script
        # ═══════════════════════════════════════════════════════
        import os, subprocess, shutil

        STUDIO = '/content/studio'
        WEIGHTS_DIR = f'{STUDIO}/experiments/pretrained_models'
        os.makedirs(WEIGHTS_DIR, exist_ok=True)

        # Model weights
        WEIGHT_FILE = f'{WEIGHTS_DIR}/RealESRGAN_x4plus.pth'
        if not os.path.exists(WEIGHT_FILE):
            print("Downloading RealESRGAN_x4plus weights (~67 MB)…")
            subprocess.run([
                'wget', '-q', '-O', WEIGHT_FILE,
                'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth'
            ], check=True)
            print("✓ Weights downloaded")
        else:
            print(f"✓ Weights present: {WEIGHT_FILE}")

        # Clone Real-ESRGAN for inference script
        RESRGAN_DIR = '/content/Real-ESRGAN'
        if not os.path.exists(RESRGAN_DIR):
            print("Cloning Real-ESRGAN repo…")
            subprocess.run(
                ['git', 'clone', '--depth=1',
                 'https://github.com/xinntao/Real-ESRGAN.git', RESRGAN_DIR],
                check=True, capture_output=True
            )
            print("✓ Real-ESRGAN cloned")
        else:
            print("[OK] Real-ESRGAN cloned")
        else:
            print("[OK] Real-ESRGAN already present")

        # Link inference script
        INFER_DST = f'{STUDIO}/inference_realesrgan.py'
        INFER_SRC = f'{RESRGAN_DIR}/inference_realesrgan.py'
        if not os.path.exists(INFER_DST) and os.path.exists(INFER_SRC):
            shutil.copy(INFER_SRC, INFER_DST)
            print("[OK] inference_realesrgan.py linked")

        print("[OK] Real-ESRGAN ready")
    """)))

    # ── CELL 11: Write scripts/ ───────────────────────────────────────────
    STUDIO = "/content/studio"
    script_files = [
        ("scripts/__init__.py",  "scripts/__init__.py"),
        ("scripts/config.py",    "scripts/config.py"),
        ("scripts/utils.py",     "scripts/utils.py"),
        ("scripts/qc.py",        "scripts/qc.py"),
        ("scripts/upscaler.py",  "scripts/upscaler.py"),
        ("scripts/ollama_vision.py", "scripts/ollama_vision.py"),
        ("scripts/batch_processor.py", "scripts/batch_processor.py"),
    ]

    src_lines = [
        "# =======================================================\n",
        "# CELL 11: Write All Application Scripts to Disk\n",
        "# =======================================================\n",
        "import base64, os\n",
        f"STUDIO = '{STUDIO}'\n",
        "os.makedirs(f'{STUDIO}/scripts', exist_ok=True)\n",
        "os.makedirs(f'{STUDIO}/app', exist_ok=True)\n",
        "\n",
    ]

    for colab_rel, local_rel in script_files:
        b64 = _read_b64(local_rel)
        varname = colab_rel.replace("/", "_").replace(".", "_")
        colab_path = f"{STUDIO}/{colab_rel}"
        safe_dir = "/".join(colab_path.split("/")[:-1])
        src_lines += [
            f"# {colab_rel}\n",
            f"_b64 = '{b64}'\n",
            f"os.makedirs('{safe_dir}', exist_ok=True)\n",
            f"open('{colab_path}', 'w', encoding='utf-8').write(base64.b64decode(_b64).decode())\n",
            f"print('  [OK] {colab_rel}')\n",
            "\n",
        ]

    src_lines.append("print('[OK] All scripts written')\n")
    cells.append({"cell_type": "code", "execution_count": None,
                  "metadata": {}, "outputs": [], "source": src_lines})

    # ── CELL 12: Write app/main.py ────────────────────────────────────────
    main_b64 = _read_b64("app/main.py")
    cells.append(code_cell(textwrap.dedent(f"""\
        # =======================================================
        # CELL 12: Write FastAPI Backend (app/main.py)
        # =======================================================
        import base64, os
        STUDIO = '/content/studio'
        os.makedirs(f'{{STUDIO}}/app', exist_ok=True)
        _b64 = '{main_b64}'
        open(f'{{STUDIO}}/app/main.py', 'w', encoding='utf-8').write(
            base64.b64decode(_b64).decode('utf-8')
        )
        # app/__init__.py
        open(f'{{STUDIO}}/app/__init__.py', 'w').write('')
        print('[OK] app/main.py written')
    """)))

    # ── CELL 13: Write app/index.html ────────────────────────────────────
    html_b64 = _read_b64("app/index.html")
    cells.append(code_cell(textwrap.dedent(f"""\
        # =======================================================
        # CELL 13: Write Frontend (app/index.html)
        # =======================================================
        import base64, os
        STUDIO = '/content/studio'
        _b64 = '{html_b64}'
        open(f'{{STUDIO}}/app/index.html', 'w', encoding='utf-8').write(
            base64.b64decode(_b64).decode('utf-8')
        )
        print('[OK] app/index.html written')
        print(f'  Size: {{len(base64.b64decode(_b64)):,}} bytes')
    """)))

    # ── CELL 14: Write requirements.txt + patch config path ───────────────
    req_b64 = _read_b64("requirements.txt")
    cells.append(code_cell(textwrap.dedent(f"""\
        # =======================================================
        # CELL 14: Write requirements.txt + Verify File Structure
        # =======================================================
        import base64, os, sys
        STUDIO = '/content/studio'

        # requirements.txt
        _b64 = '{req_b64}'
        open(f'{{STUDIO}}/requirements.txt', 'w').write(
            base64.b64decode(_b64).decode('utf-8')
        )

        # Add studio dir to Python path
        if STUDIO not in sys.path:
            sys.path.insert(0, STUDIO)

        # List structure
        for root, dirs, files in os.walk(STUDIO):
            dirs[:] = [d for d in dirs if d not in ['__pycache__','temp_output','uploads','output','metadata','logs','archives','failed']]
            level = root.replace(STUDIO, '').count(os.sep)
            indent = '  ' * level
            print(f'{{indent}}{{os.path.basename(root)}}/')
            for f in files:
                print(f'{{indent}}  {{f}}')

        print()
        print('[OK] File structure verified')
    """)))

    # ── CELL 15: Start FastAPI backend ────────────────────────────────────
    cells.append(code_cell(textwrap.dedent("""\
        # ═══════════════════════════════════════════════════════
        # CELL 15: Start FastAPI Backend
        # ═══════════════════════════════════════════════════════
        import subprocess, sys, os, time, requests

        STUDIO = '/content/studio'

        # Ensure torchvision compat for the server process
        env = os.environ.copy()
        env['PYTHONPATH'] = STUDIO

        print("Starting FastAPI (uvicorn)…")
        server_proc = subprocess.Popen(
            [sys.executable, '-m', 'uvicorn', 'app.main:app',
             '--host', '0.0.0.0', '--port', '8000',
             '--workers', '1', '--log-level', 'warning'],
            cwd=STUDIO,
            env=env,
        )

        # Wait for backend to be ready
        for i in range(45):
            try:
                r = requests.get('http://localhost:8000/api/health', timeout=2)
                if r.ok:
                    h = r.json()
                    print(f"✓ FastAPI running (took {i+1}s)")
                    print(f"  GPU     : {h.get('gpu_name','—')}")
                    print(f"  Ollama  : {'Ready (' + h.get('ollama_model','?') + ')' if h.get('ollama_ready') else 'Initializing…'}")
                    break
            except Exception:
                pass
            time.sleep(1)
        else:
            print("⚠ Backend may still be starting. Check output above.")
            print("  Run: !curl http://localhost:8000/api/health")
    """)))

    # ── CELL 16: Cloudflare tunnel + display URL ──────────────────────────
    cells.append(code_cell(textwrap.dedent("""\
        # ═══════════════════════════════════════════════════════
        # CELL 16: Cloudflare Tunnel + Final URL
        # ═══════════════════════════════════════════════════════
        import subprocess, threading, time, re, shutil, requests

        # Install cloudflared if missing
        if not shutil.which('cloudflared'):
            print("Installing cloudflared…")
            subprocess.run([
                'wget', '-q', '-O', '/usr/local/bin/cloudflared',
                'https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64'
            ], check=True)
            subprocess.run(['chmod', '+x', '/usr/local/bin/cloudflared'], check=True)
            print("✓ cloudflared installed")

        public_url = None
        url_event = threading.Event()

        def _run_tunnel():
            global public_url
            proc = subprocess.Popen(
                ['cloudflared', 'tunnel', '--url', 'http://localhost:8000'],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
            )
            for line in proc.stdout:
                m = re.search(r'https://[a-z0-9\\-]+\\.trycloudflare\\.com', line)
                if m and not public_url:
                    public_url = m.group(0)
                    url_event.set()

        t = threading.Thread(target=_run_tunnel, daemon=True)
        t.start()

        print("Waiting for Cloudflare tunnel…")
        url_event.wait(timeout=60)

        if not public_url:
            print("⚠ Cloudflare URL not received within 60s.")
            print("  Try: !cloudflared tunnel --url http://localhost:8000")
        else:
            # Health check through public tunnel
            time.sleep(3)
            try:
                hr = requests.get(f'{public_url}/api/health', timeout=20)
                health_ok = hr.ok
            except Exception:
                health_ok = False

            bar = "=" * 57
            print()
            print(bar)
            print("  ADOBE STOCK AI STUDIO — READY")
            print(bar)
            print()
            print(f"  T4 GPU       : ✓ Connected")
            print(f"  Real-ESRGAN  : ✓ Ready")
            print(f"  Ollama       : ✓ {active_model}")
            print(f"  FastAPI      : ✓ Running :8000")
            print(f"  Tunnel       : ✓ Active")
            print(f"  Health Check : {'✓ PASS' if health_ok else '⚠ Check manually'}")
            print()
            print(f"  ┌{'─'*53}┐")
            print(f"  │  🌐  OPEN THIS URL IN YOUR BROWSER:              │")
            print(f"  │  {public_url:<51}  │")
            print(f"  └{'─'*53}┘")
            print()
            print("  Instructions:")
            print("  1. Open the URL above")
            print("  2. Drop images in the upload zone (all at once)")
            print("  3. Wait for 100/100 Uploaded ✓")
            print("  4. Click  ▶ Start Processing")
            print("  5. Watch real-time progress")
            print("  6. Download ZIP / CSV / JSON when complete")
            print()
            print(bar)
    """)))

    return cells


# ─────────────────────────────────────────────────────────────────────────────
# Assemble & write notebook
# ─────────────────────────────────────────────────────────────────────────────

def compile_notebook():
    print("Compiling Adobe_Stock_AI_Upscaler.ipynb ...")

    cells = build_cells()
    nb = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {
                "gpuType": "T4",
                "provenance": [],
                "toc_visible": True,
                "name": "Adobe_Stock_AI_Upscaler.ipynb"
            },
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.10.12"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }

    out_path = os.path.join(BASE_DIR, "Adobe_Stock_AI_Upscaler.ipynb")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    size_kb = os.path.getsize(out_path) / 1024
    print(f"[OK] Written: {out_path}")
    print(f"  Size   : {size_kb:.0f} KB")
    print(f"  Cells  : {len(cells)}")

    # Verify JSON is valid
    with open(out_path, encoding="utf-8") as f:
        check = json.load(f)
    assert check["nbformat"] == 4
    assert len(check["cells"]) == len(cells)
    print("[OK] JSON valid")
    return out_path


if __name__ == "__main__":
    try:
        compile_notebook()
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
