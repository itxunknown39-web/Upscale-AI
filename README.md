# Adobe Stock AI Studio

<div align="center">

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/itxunknown39-web/Upscale-AI/blob/main/Adobe_Stock_AI_Upscaler.ipynb)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.95+-009688.svg)](https://fastapi.tiangolo.com)
[![Ollama](https://img.shields.io/badge/Ollama-Local%20Vision-black.svg)](https://ollama.com)
[![Real-ESRGAN](https://img.shields.io/badge/Real--ESRGAN-x4plus-orange.svg)](https://github.com/xinntao/Real-ESRGAN)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Unified Google Colab T4 Production Suite for Adobe Stock Contributor Workflow**  
*4× AI Image Upscaling • Local Vision AI Analysis • Adobe Stock CSV/JSON Metadata • Live Studio UI*

</div>

---

## ⚡ Quick Start (Google Colab)

Click the badge below to open and launch the unified studio directly in Google Colab:

<div align="center">

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/itxunknown39-web/Upscale-AI/blob/main/Adobe_Stock_AI_Upscaler.ipynb)

</div>

### 3-Step Colab Launch:
1. **Set GPU**: Go to **Runtime → Change runtime type → T4 GPU → Save**.
2. **Run All**: Press `Ctrl + F9` (or select **Runtime → Run all**).
3. **Open Studio**: Open the public Cloudflare tunnel URL printed in the final cell (e.g. `https://xxxx.trycloudflare.com`).

---

## ✨ Features

- 🚀 **Single Master Notebook Architecture**: Everything runs from one unified notebook (`Adobe_Stock_AI_Upscaler.ipynb`). No secondary notebooks, manual analyzers, or separate apps needed.
- 🖼️ **Real-ESRGAN 4× Upscaling**: High-fidelity super-resolution with tile-based memory optimization (`tile=400`) and subprocess isolation to prevent VRAM crashes on NVIDIA T4 (16 GB).
- 👁️ **Local Ollama Vision AI**: 100% on-device vision analysis using `Moondream`, `LLaVA`, `LLaMA-3.2-Vision`, or `MiniCPM-V`. **Zero external API keys, zero paid subscriptions, zero OpenRouter dependency**.
- 📋 **Adobe Stock Metadata Engine**:
  - **Commercial Titles**: Subject-first, ≤200 characters, automated removal of banned hype words (`beautiful`, `amazing`, `stunning`, etc.).
  - **Keywords**: Ordered by relevance, deduplicated, lowercased, capped at Adobe Stock's limit of 49 keywords.
  - **Numeric Category Mapping**: Automatic classification into Adobe Stock categories (1–22).
  - **Releases Support**: Compliant metadata schema ready for direct submission.
- 📁 **Master CSV & JSON Export**:
  - `AdobeStock_Metadata.csv` (UTF-8-SIG with BOM for native Excel compatibility, exact headers: `Filename,Title,Keywords,Category,Releases`).
  - `AdobeStock_Metadata.json` (Structured dictionary of the entire batch).
- 🏷️ **Standardized Filename Mapping**: Collision-safe sequential output filenames (`stock_image_up1.jpg`, `stock_image_up2.jpg`, ... `stock_image_upN.jpg`).
- 📥 **Upload-First Sequential Pipeline**: Staged multi-image upload completes before processing begins, preventing network bottlenecks during heavy GPU execution.
- 🛡️ **Queue Failure Isolation**: If metadata generation encounters an error on a specific image, the successful upscale is preserved, and the queue automatically proceeds to subsequent images.
- 🔄 **Metadata Inspector & In-Place Editing**: Edit titles, add/remove keyword chips, and update categories in real-time with instant master CSV/JSON regeneration.
- 💬 **Built-in Deterministic Chatbot**: Query batch status, failure reasons, GPU statistics, and CSV readiness directly in the chat panel.
- 💾 **Google Drive Persistence & ZIP Archival**: Processed assets and metadata automatically persist to `/MyDrive/AdobeStockStudio/` and can be downloaded as a single ZIP bundle.

---

## 🏗️ Architecture & Workflow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                       Adobe Stock AI Studio Web UI                      │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
                             (1) Upload All
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      FastAPI Backend (:8000)                            │
│  - Staging in uploads/                                                  │
│  - SSE Event Stream (/api/events)                                       │
│  - Metadata Store & Export Handlers                                     │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
                         (2) Start Sequential Queue
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                   Sequential Batch Worker Pipeline                      │
│                                                                         │
│   Image N ──► Real-ESRGAN 4x Upscale (GPU Lock + Tile=400)              │
│                 │                                                       │
│                 ▼                                                       │
│               Technical QC (6 checks: Res, Integrity, Aspect, Size)    │
│                 │                                                       │
│                 ▼                                                       │
│               Ollama Local Vision Analysis (/api/chat + /api/generate)  │
│                 │                                                       │
│                 ▼                                                       │
│               Adobe Stock Metadata (Title, 49 Keywords, Category)       │
│                 │                                                       │
│                 ▼                                                       │
│               Save stock_image_upN.jpg + Append to CSV & JSON           │
│                 │                                                       │
│                 ▼                                                       │
│               Next Image in Queue ──► (Repeat until batch complete)     │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 📂 Project Structure

```
├── Adobe_Stock_AI_Upscaler.ipynb  # Single Master Google Colab Notebook
├── README.md                      # Documentation & Quick Start Guide
├── requirements.txt               # Backend & ML dependencies
├── .gitignore                     # Git ignore rules
│
├── app/                           # FastAPI Application & UI
│   ├── index.html                 # Dark Studio Frontend (HTML5, Vanilla CSS, JS)
│   ├── main.py                    # FastAPI Backend API & SSE Event Stream
│   └── __init__.py
│
└── scripts/                       # Modular Python Engine
    ├── batch_processor.py         # Sequential worker, filename counter, CSV/JSON writers
    ├── compile_notebook.py        # Self-contained notebook builder
    ├── config.py                  # Central settings, paths, Adobe category mappings
    ├── ollama_vision.py           # Centralized OllamaVisionClient & validation
    ├── qc.py                      # 6-point technical quality control
    ├── upscaler.py                # Real-ESRGAN subprocess wrapper with VRAM protection
    ├── utils.py                   # System monitoring & filename utilities
    └── __init__.py
```

---

## 📊 Adobe Stock CSV Specification

The generated `AdobeStock_Metadata.csv` strictly adheres to Adobe Stock Contributor guidelines:

| Column Header | Format | Description | Example |
|---|---|---|---|
| `Filename` | String | Sequential output filename | `stock_image_up1.jpg` |
| `Title` | String (≤200 chars) | Subject-first factual title, no hype words | `Modern minimalist geometric abstract background` |
| `Keywords` | String (≤49 items) | Comma-separated relevant tags, lowercase | `abstract, background, modern, design, geometric` |
| `Category` | Integer (1–22) | Adobe Stock category ID | `22` (Abstract/Backgrounds) |
| `Releases` | String | Model/Property release (optional) | *Empty if not required* |

---

## 💻 Local Development / Self-Hosted Setup

If running locally on a dedicated Linux/Windows machine with an NVIDIA GPU:

### 1. Prerequisites
- Python 3.10+
- [Ollama](https://ollama.com) installed and running:
  ```bash
  ollama serve
  ollama pull moondream
  ```
- PyTorch with CUDA support

### 2. Installation
```bash
# Clone the repository
git clone https://github.com/itxunknown39-web/Upscale-AI.git
cd Upscale-AI

# Install dependencies
pip install -r requirements.txt
pip install basicsr facexlib gfpgan realesrgan
```

### 3. Launch Backend
```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```
Open `http://localhost:8000` in your web browser.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
