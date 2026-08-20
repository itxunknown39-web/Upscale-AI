# Adobe Stock AI Upscaler

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/itxunknown39-web/Upscale-AI/blob/main/Adobe_Stock_AI_Upscaler.ipynb)

A premium, cloud-powered batch AI super-resolution upscaler designed to prepare low-resolution image inputs (e.g. from Midjourney or Stable Diffusion) to meet the minimum **4 Megapixel (MP)** technical requirement for Adobe Stock submissions.

---

## 🚀 Open in Google Colab

Click the badge below to open the main orchestrator notebook directly in Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/itxunknown39-web/Upscale-AI/blob/main/Adobe_Stock_AI_Upscaler.ipynb)

---

## 🎨 Features

*   **AI Image Upscaling**: Powered by Real-ESRGAN super-resolution model for clean, sharp enhancements (photo and illustration variants).
*   **NVIDIA T4 GPU Support**: Automatically detects PyTorch CUDA acceleration to run inference in seconds.
*   **Upscale Scale Modes**: Choice of **2x**, **4x** (default), or custom **Target Resolution** (maintains original aspect ratios via smart float scaling).
*   **Tiling & VRAM Protection**: Passes `--tile 400` arguments to the Real-ESRGAN CLI to prevent CUDA Out of Memory (OOM) allocation crashes on the T4 GPU.
*   **Subprocess Isolation**: Isolates upscaling subprocesses from the web server. If a corrupted image or memory overflow crashes the inference script, the FastAPI backend remains online to log the failure and process the next queue item.
*   **Technical Quality Control (QC)**: Performs 6 post-upscaling verification checks (Resolution, Integrity, Format, Aspect Ratio drift, Transparency preservation, and File Size limits).
*   **Google Drive Persistence**: Auto-saves completed files permanently in Google Drive under `/MyDrive/AdobeStockUpscaler/` (if mounted), falling back to local storage if Drive mount is bypassed.
*   **ZIP Batch Export**: Package all successfully upscaled batch items in a single zip archive called `AdobeStock_Upscaled_YYYY-MM-DD.zip`.
*   **Interactive Comparison Slider**: Premium obsidian-dark web UI modal featuring a side-by-side comparison slider to inspect upscaled textures.
*   **Dynamic ETA Dashboard**: Live progress tracking showing speed (sec/img), completed count, failed count, percentage, and dynamic ETA calculations.

---

## 💻 How to Use

### Deploying via Google Colab:
1. Click the **Open In Colab** badge above.
2. Select **T4 GPU** as the hardware accelerator under *Runtime -> Change runtime type*.
3. Click *Runtime -> Run all* (`Ctrl + F9`).
4. Mount your Google Drive when prompted.
5. Wait for dependencies installation, repo cloning, model weights download, and background servers startup.
6. Open the public URL printed at the bottom:
   `👉 https://xxxx.trycloudflare.com 👈`
7. Drag and drop your images into the UI, select settings, and click **Start Batch Upscale**.
8. Once completed, download individual images or click **Download ZIP Package** to export the entire batch.

### Running Locally (Testing/Mock Mode):
1. Install Python package dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Start the FastAPI server locally:
   ```bash
   python backend/app.py
   ```
3. Open your browser and go to:
   `http://127.0.0.1:8000`
   *Note: If `inference_realesrgan.py` is absent, the backend automatically runs in developer mock mode, performing high-quality PIL Lanczos resizing with simulated processing delays to test the UI.*

---

## ⚙️ Architecture

```
Colab
 └── GitHub Repo (Cloned)
      └── FastAPI Backend
           ├── Real-ESRGAN (Subprocess isolated) ──> T4 GPU
           ├── Frontend Web UI (Decoupled static assets)
           ├── Cloudflare Tunnel (Public secure forwarding)
           └── Google Drive (Persistent output mapping)
```

---

## 📂 Project Structure

```
Upscale-AI/
├── Adobe_Stock_AI_Upscaler.ipynb  # Main Colab orchestrator & launcher
├── scripts/
│   ├── config.py                  # Settings, upload constraints, & directories
│   ├── upscaler.py                # Subprocess-isolated Real-ESRGAN CLI runner
│   ├── batch_processor.py         # Worker thread queue & incremental runs logger
│   ├── qc.py                      # 6 automated Pillow validation checks
│   └── utils.py                   # RAM/VRAM resource monitors & non-overwriting helpers
├── backend/
│   ├── app.py                     # FastAPI application setup & static folder mounting
│   └── routes.py                  # HTTP Endpoints (Upload, Process, status poll, downloads)
├── frontend/
│   ├── index.html                 # Decoupled UI HTML5 layout
│   ├── style.css                  # Premium dark-theme glassmorphic stylesheet
│   └── app.js                     # Javascript event handlers, table drawer & timers
├── requirements.txt               # Library dependencies
├── .gitignore                     # Ignored weights, temp inputs & logs
└── README.md                      # Documentation (this file)
```

---

## 📦 Requirements

*   A Google Account (to run Google Colab).
*   Active Google Colab Runtime utilizing **NVIDIA T4 GPU**.
*   Google Drive storage (recommended for persistent output folders).
*   Modern web browser supporting Javascript.

---

## ⚠️ Adobe Stock Submission Note

Passing the technical QC checks (labeled as **"Technical QC Passed"**) verifies compatibility with technical guidelines (integrity, minimum resolution of 4 Megapixels, aspect ratio preservation). 

It does **not** guarantee commercial acceptance by Adobe Stock curators. You must manually review each upscaled image for AI generation artifacts, similarity errors, anatomical issues, text failures, and commercial logo/trademark issues before submitting.
