# Adobe Stock AI Studio - Task List

## Audit
- [x] Read README.md
- [x] Read app/main.py
- [x] Read backend/app.py, backend/routes.py
- [x] Read scripts/upscaler.py, batch_processor.py, qc.py, config.py, utils.py
- [x] Read Adobe_Stock_AI_Upscaler.ipynb
- [x] Read scripts/compile_notebook.py
- [x] Read requirements.txt

## Implementation

### scripts/
- [x] scripts/config.py — add Studio paths, Ollama config
- [x] scripts/ollama_vision.py — NEW: vision + metadata
- [x] scripts/batch_processor.py — add Ollama stage, metadata, naming, retry_metadata

### app/
- [x] app/main.py — add Ollama, SSE, metadata routes, assistant, filenames
- [x] app/index.html — new professional Studio UI

### root
- [x] requirements.txt — add httpx, aiofiles
- [x] Adobe_Stock_AI_Upscaler.ipynb — unified Colab orchestrator

## Verification
- [x] Test API endpoints
- [x] Test 1 image
- [x] Test 3 images
- [x] Test 10 images
- [x] Test 20 images
- [x] Verify CSV format
- [x] Verify JSON format
- [x] Verify filenames
