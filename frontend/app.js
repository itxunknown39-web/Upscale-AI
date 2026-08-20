// Client-Side Application State
const state = {
  files: [],             // Local file items list
  processing: false,     // Active batch processing state
  cancelRequested: false,// Cancellation flag
  healthTimer: null,     // Health poll interval reference
  uploadConcurrency: 4,  // Default parallel upload pool size (2 / 4 / 6)
  activeUploads: new Map(), // active XHR uploads for cancellation
  processingDurations: [], // inference durations in seconds
  uploadStats: {
    totalBytes: 0,
    uploadedBytes: 0,
    startTime: 0,
    currentSpeedMBs: 0.0
  }
};

// Fingerprints map to prevent duplicate uploads
const uploadedFingerprints = new Set();

// DOM Element Selectors
const dropzone = document.getElementById('dropzone');
const fileUploader = document.getElementById('file-uploader');
const queueTbody = document.getElementById('queue-tbody');
const emptyRow = document.getElementById('empty-row');
const queueBadge = document.getElementById('queue-badge');

const statTotal = document.getElementById('stat-total');
const statCompleted = document.getElementById('stat-completed');
const statFailed = document.getElementById('stat-failed');
const statSpeed = document.getElementById('stat-speed');

const btnStart = document.getElementById('btn-start');
const btnCancel = document.getElementById('btn-cancel');
const btnClearQueue = document.getElementById('btn-clear-queue');
const btnDownloadAll = document.getElementById('btn-download-all');
const btnRetryFailed = document.getElementById('btn-retry-failed');

const progressWrapper = document.getElementById('progress-wrapper');
const progressStatusText = document.getElementById('progress-status-text');
const progressPercent = document.getElementById('progress-percent');
const progressBarFill = document.getElementById('progress-bar-fill');
const progressDetails = document.getElementById('progress-details');
const progressEta = document.getElementById('progress-eta');

const statusIndicator = document.getElementById('status-indicator');
const statusText = document.getElementById('status-text');
const systemMetrics = document.getElementById('system-metrics');
const systemRam = document.getElementById('system-ram');
const systemVram = document.getElementById('system-vram');

const scaleModeRadios = document.getElementsByName('scale_mode');
const outputFormatRadios = document.getElementsByName('output_format');
const jpegQualitySlider = document.getElementById('jpeg-quality');
const jpegQualityVal = document.getElementById('jpeg-quality-val');
const jpegQualityGroup = document.getElementById('jpeg-quality-group');
const targetResInputs = document.getElementById('target-res-inputs');
const targetWidthInput = document.getElementById('target-width');
const targetHeightInput = document.getElementById('target-height');
const uploadConcurrencySelect = document.getElementById('upload-concurrency');

// Modal Elements
const previewModal = document.getElementById('preview-modal');
const modalClose = document.getElementById('modal-close');
const modalFilename = document.getElementById('modal-filename');
const modalDimensions = document.getElementById('modal-dimensions');
const modalQcStatus = document.getElementById('modal-qc-status');
const modalQcMp = document.getElementById('modal-qc-mp');
const modalQcRatio = document.getElementById('modal-qc-ratio');
const imgOriginal = document.getElementById('img-original');
const imgUpscaled = document.getElementById('img-upscaled');
const imgUpscaledWrapper = document.getElementById('img-upscaled-wrapper');
const handleLine = document.getElementById('handle-line');
const handleButton = document.getElementById('handle-button');
const sliderRange = document.getElementById('slider-range');
const modalDownloadBtn = document.getElementById('modal-download-btn');

// Error Diagnostic Modal Elements
const errorModal = document.getElementById('error-details-modal');
const errorModalClose = document.getElementById('error-modal-close');
const errorModalStage = document.getElementById('error-modal-stage');
const errorModalReason = document.getElementById('error-modal-reason');
const errorModalTrace = document.getElementById('error-modal-trace');

// Utility: Byte Formatter
function formatBytes(bytes) {
  if (bytes === 0) return '0 Bytes';
  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

// Structured Queue Lifecycle Logger
function logQueueLifecycle(fileItem, queueIndex, workerId, currentState) {
  const timestamp = new Date().toISOString();
  console.log(`[QUEUE] Worker ${workerId} | File: ${fileItem.name} (Index ${queueIndex}) | State: ${currentState} | Time: ${timestamp}`);
}

// Client-Side File Validation (Zero Base64, Native ObjectURL)
async function validateAndReadImage(file) {
  const ext = file.name.split('.').pop().toLowerCase();
  if (!['jpg', 'jpeg', 'png'].includes(ext)) {
    throw new Error(`Unsupported format '.${ext}'. Only JPG, JPEG, and PNG images are allowed.`);
  }

  const maxMb = 100;
  if (file.size > maxMb * 1024 * 1024) {
    throw new Error(`File size (${formatBytes(file.size)}) exceeds limit of ${maxMb}MB.`);
  }

  return new Promise((resolve, reject) => {
    const objectUrl = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => {
      const mp = (img.width * img.height) / 1_000_000.0;
      resolve({
        width: img.width,
        height: img.height,
        megapixels: parseFloat(mp.toFixed(2)),
        thumbnail: objectUrl
      });
    };
    img.onerror = () => {
      URL.revokeObjectURL(objectUrl);
      reject(new Error("Corrupted or unreadable image file."));
    };
    img.src = objectUrl;
  });
}

// Queue Drag-and-Drop Event Listeners
dropzone.addEventListener('click', () => {
  if (!state.processing) fileUploader.click();
});

dropzone.addEventListener('dragover', (e) => {
  e.preventDefault();
  if (!state.processing) dropzone.classList.add('dragover');
});

dropzone.addEventListener('dragleave', () => {
  dropzone.classList.remove('dragover');
});

dropzone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropzone.classList.remove('dragover');
  if (state.processing) return;
  if (e.dataTransfer.files.length > 0) {
    handleFileSelection(Array.from(e.dataTransfer.files));
  }
});

fileUploader.addEventListener('change', (e) => {
  if (e.target.files.length > 0) {
    handleFileSelection(Array.from(e.target.files));
    fileUploader.value = '';
  }
});

// File Batch Processing Handler
async function handleFileSelection(rawFiles) {
  let addedCount = 0;

  for (const rawFile of rawFiles) {
    const fingerprint = `${rawFile.name}_${rawFile.size}_${rawFile.lastModified}`;
    if (uploadedFingerprints.has(fingerprint)) {
      console.warn(`Skipping duplicate file: ${rawFile.name}`);
      continue;
    }

    try {
      const meta = await validateAndReadImage(rawFile);
      uploadedFingerprints.add(fingerprint);

      const localItem = {
        id: 'loc_' + Math.random().toString(36).substr(2, 9),
        serverId: null,
        blob: rawFile,
        name: rawFile.name,
        size: rawFile.size,
        width: meta.width,
        height: meta.height,
        megapixels: meta.megapixels,
        thumbnail: meta.thumbnail,
        status: 'ready', // ready, queued, uploading, uploaded, processing, completed, failed, cancelled
        uploadPercent: 0,
        uploadSpeed: 0.0,
        output: null,
        qc: { passed: true, hard_failures: [], warnings: [] },
        error_stage: '',
        error_reason: '',
        error_details: '',
        processing_seconds: 0.0
      };

      state.files.push(localItem);
      addedCount++;
    } catch (err) {
      alert(`Validation error for '${rawFile.name}': ${err.message}`);
    }
  }

  if (addedCount > 0) {
    updateQueueTable();
    updateOverallProgress();
    btnStart.disabled = false;
  }
}

// Update Table Display
function updateQueueTable() {
  if (state.files.length === 0) {
    emptyRow.style.display = 'table-row';
    queueBadge.textContent = '0 files';
    btnStart.disabled = true;
    btnClearQueue.disabled = true;
    btnRetryFailed.style.display = 'none';
  } else {
    emptyRow.style.display = 'none';
    queueBadge.textContent = `${state.files.length} file${state.files.length > 1 ? 's' : ''}`;
    if (!state.processing) btnClearQueue.disabled = false;
  }

  const total = state.files.length;
  const completed = state.files.filter(f => f.status === 'completed').length;
  const failed = state.files.filter(f => f.status === 'failed').length;

  statTotal.textContent = total;
  statCompleted.textContent = completed;
  statFailed.textContent = failed;

  btnRetryFailed.style.display = (failed > 0 && !state.processing) ? 'inline-flex' : 'none';
  btnDownloadAll.disabled = (completed === 0 || state.processing);

  // Clear current rows except empty row
  Array.from(queueTbody.children).forEach(child => {
    if (child.id !== 'empty-row') queueTbody.removeChild(child);
  });

  const scaleMode = document.querySelector('input[name="scale_mode"]:checked')?.value || '4';
  const targetW = parseInt(targetWidthInput.value) || 3840;
  const targetH = parseInt(targetHeightInput.value) || 2160;

  state.files.forEach(file => {
    const tr = document.createElement('tr');
    
    // Output dimensions
    let outW, outH, outMp;
    if (file.output) {
      outW = file.output.width;
      outH = file.output.height;
      outMp = file.output.megapixels;
    } else {
      if (scaleMode === 'target') {
        const scaleW = targetW / file.width;
        const scaleH = targetH / file.height;
        const scale = Math.max(scaleW, scaleH);
        outW = Math.round(file.width * scale);
        outH = Math.round(file.height * scale);
      } else {
        const factor = parseInt(scaleMode) || 4;
        outW = file.width * factor;
        outH = file.height * factor;
      }
      outMp = ((outW * outH) / 1_000_000.0).toFixed(2);
    }

    const outMpClass = parseFloat(outMp) >= 4.0 ? 'pass' : 'warn';

    // Render Status Pill
    let statusPillHtml = '';
    switch (file.status) {
      case 'ready':
        statusPillHtml = `<span class="status-pill ready">Ready</span>`;
        break;
      case 'queued':
        statusPillHtml = `<span class="status-pill queued">Queued</span>`;
        break;
      case 'uploading':
        const pctStr = file.uploadPercent ? `${file.uploadPercent}%` : '0%';
        const spdStr = file.uploadSpeed ? ` • ${file.uploadSpeed} MB/s` : '';
        statusPillHtml = `<span class="status-pill uploading">Uploading (${pctStr}${spdStr})</span>`;
        break;
      case 'uploaded':
        statusPillHtml = `<span class="status-pill uploaded">Uploaded</span>`;
        break;
      case 'processing':
        statusPillHtml = `<span class="status-pill processing">Processing...</span>`;
        break;
      case 'completed':
        statusPillHtml = `<span class="status-pill completed">Completed</span>`;
        break;
      case 'failed':
        statusPillHtml = `<span class="status-pill failed">Failed</span>`;
        break;
      case 'cancelled':
        statusPillHtml = `<span class="status-pill cancelled">Cancelled</span>`;
        break;
      default:
        statusPillHtml = `<span class="status-pill ready">${file.status}</span>`;
    }

    let qcBadgesHtml = '';
    let errorSubtext = '';
    if (file.status === 'completed') {
      const durStr = file.processing_seconds ? ` (${file.processing_seconds}s)` : '';
      qcBadgesHtml = `<span class="qc-badge pass">Technical QC Passed${durStr}</span>`;
    } else if (file.status === 'failed') {
      const stageStr = file.error_stage || 'Error';
      const reasonStr = file.error_reason || 'Inference / QC check failed';
      const badgeLabel = stageStr.toLowerCase().includes('upload') ? 'Upload Failed' : (stageStr.toLowerCase().includes('qc') ? 'QC Failed' : 'Inference Failed');
      qcBadgesHtml = `<span class="qc-badge fail" title="${reasonStr}">${badgeLabel}</span>`;
      errorSubtext = `
        <div style="margin-top: 0.25rem;">
          <span style="color: var(--state-error); font-size: 0.7rem; font-weight: 500;">${stageStr}: ${reasonStr}</span>
          ${file.error_details ? `<button class="btn-text-link" onclick="openErrorModal('${file.id}')" style="font-size:0.68rem; color: #a5b4fc; text-decoration: underline; background:none; border:none; cursor:pointer; margin-left:0.4rem;">View Error</button>` : ''}
        </div>
      `;
    } else if (file.status === 'uploading') {
      qcBadgesHtml = `<span style="color: #c084fc; font-size:0.75rem;">Uploading to server...</span>`;
    } else if (file.status === 'uploaded') {
      qcBadgesHtml = `<span style="color: #38bdf8; font-size:0.75rem;">Uploaded • Awaiting AI slot</span>`;
    } else if (file.status === 'processing') {
      qcBadgesHtml = `<span style="color: var(--accent-secondary); font-size:0.75rem;">Running Real-ESRGAN...</span>`;
    } else {
      qcBadgesHtml = `<span style="color: var(--text-dark); font-size:0.75rem;">Awaiting upscale</span>`;
    }

    let actionsHtml = '';
    if (file.status === 'completed') {
      actionsHtml = `
        <div class="row-actions-container">
          <button class="btn-icon btn-preview-trigger" title="Interactive Comparison" onclick="openPreview('${file.id}')">
            <svg viewBox="0 0 24 24"><path d="M12,9A3,3 0 0,0 9,12A3,3 0 0,0 12,15A3,3 0 0,0 15,12A3,3 0 0,0 12,9M12,4.5C7,4.5 2.73,7.61 1,12C2.73,16.39 7,19.5 12,19.5C17,19.5 21.27,16.39 23,12C21.27,7.61 17,4.5 12,4.5M12,17A5,5 0 0,1 7,12A5,5 0 0,1 12,7A5,5 0 0,1 17,12A5,5 0 0,1 12,17Z"/></svg>
          </button>
          <button class="btn-icon" title="Download Image" onclick="downloadProcessedImage('${file.id}')">
            <svg viewBox="0 0 24 24"><path d="M5,20H19V18H5V20M19,9H15V3H9V9H5L12,16L19,9Z"/></svg>
          </button>
        </div>
      `;
    } else if ((file.status === 'ready' || file.status === 'failed') && !state.processing) {
      actionsHtml = `
        <button class="btn-icon" title="Remove" onclick="removeQueueItem('${file.id}')" style="color: var(--state-error);">
          <svg viewBox="0 0 24 24"><path d="M19,4H15.5L14.5,3H9.5L8.5,4H5V6H19M6,19A2,2 0 0,0 8,21H16A2,2 0 0,0 18,19V7H6V19Z"/></svg>
        </button>
      `;
    }

    tr.innerHTML = `
      <td>
        <div class="img-preview-cell">
          <img src="${file.thumbnail}" alt="Thumbnail">
        </div>
      </td>
      <td>
        <div class="filename-container">
          <span class="filename-text" title="${file.name}">${file.name}</span>
          <span class="file-size-text">${formatBytes(file.size)}</span>
          ${errorSubtext}
        </div>
      </td>
      <td>
        <div class="resolution-badge-group">
          <span class="resolution-text">${file.width}×${file.height}</span>
          <span class="mp-badge">${file.megapixels} MP</span>
        </div>
      </td>
      <td>
        <div class="resolution-badge-group">
          <span class="resolution-text" style="color: var(--accent-secondary);">${outW}×${outH}</span>
          <span class="mp-badge ${outMpClass}">${outMp} MP</span>
        </div>
      </td>
      <td>${statusPillHtml}</td>
      <td>${qcBadgesHtml}</td>
      <td style="text-align: center;">${actionsHtml}</td>
    `;

    queueTbody.appendChild(tr);
  });
}

// Master Progress and Speed Updater
function updateOverallProgress() {
  const total = state.files.length;
  if (total === 0) {
    progressWrapper.style.display = 'none';
    statSpeed.textContent = '--';
    return;
  }

  const completed = state.files.filter(f => f.status === 'completed').length;
  const failed = state.files.filter(f => f.status === 'failed').length;
  const uploading = state.files.filter(f => f.status === 'uploading').length;
  const processing = state.files.filter(f => f.status === 'processing').length;
  const processed = completed + failed;

  const pct = total > 0 ? Math.min(100, Math.round((processed / total) * 100)) : 0;
  progressBarFill.style.width = pct + '%';
  progressPercent.textContent = pct + '%';

  // Calculate speed & ETA metrics
  let speedText = '--';
  let etaText = 'ETA: Calculating...';

  if (state.processingDurations.length > 0) {
    const avgDuration = state.processingDurations.reduce((a, b) => a + b, 0) / state.processingDurations.length;
    const remaining = total - processed;
    const etaSec = Math.max(0, Math.round(remaining * avgDuration));
    const mins = Math.floor(etaSec / 60);
    const secs = etaSec % 60;
    etaText = remaining > 0 ? `ETA: ${mins}m ${secs}s` : 'Complete';
    speedText = `${avgDuration.toFixed(1)} s/img`;

    if (state.uploadStats.currentSpeedMBs > 0 && uploading > 0) {
      statSpeed.textContent = `${avgDuration.toFixed(1)} s/img (Up: ${state.uploadStats.currentSpeedMBs.toFixed(2)} MB/s)`;
    } else {
      statSpeed.textContent = speedText;
    }
  } else if (state.uploadStats.currentSpeedMBs > 0) {
    statSpeed.textContent = `Upload: ${state.uploadStats.currentSpeedMBs.toFixed(2)} MB/s`;
  } else {
    statSpeed.textContent = '--';
  }

  progressDetails.textContent = `Processed ${processed} of ${total} files`;
  progressEta.textContent = etaText;

  if (state.processing) {
    if (processing > 0) {
      const activeFile = state.files.find(f => f.status === 'processing');
      progressStatusText.textContent = `Upscaling: ${activeFile ? activeFile.name : 'Active image...'}`;
    } else if (uploading > 0) {
      progressStatusText.textContent = `Uploading images (${uploading} active)...`;
    } else {
      progressStatusText.textContent = `Batch queue processing (${processed}/${total})...`;
    }
  }
}

// Remove Item from Queue
window.removeQueueItem = function(fileId) {
  if (state.processing) return;
  const idx = state.files.findIndex(f => f.id === fileId);
  if (idx !== -1) {
    const f = state.files[idx];
    uploadedFingerprints.delete(`${f.name}_${f.size}_${f.blob.lastModified}`);
    if (f.thumbnail && f.thumbnail.startsWith('blob:')) {
      URL.revokeObjectURL(f.thumbnail);
    }
    state.files.splice(idx, 1);
    updateQueueTable();
    updateOverallProgress();
  }
};

// Clear Queue
btnClearQueue.addEventListener('click', () => {
  if (state.processing) return;
  state.files.forEach(f => {
    if (f.thumbnail && f.thumbnail.startsWith('blob:')) {
      URL.revokeObjectURL(f.thumbnail);
    }
  });
  uploadedFingerprints.clear();
  state.files = [];
  state.processingDurations = [];
  updateQueueTable();
  updateOverallProgress();
  btnDownloadAll.disabled = true;
});

// Diagnostic Error Parser for Upload and Processing Failures
function parseHttpError(xhrOrRes, status, responseText, errorType, endpoint) {
  const origin = window.location.origin;
  let errorDetail = '';

  try {
    const json = JSON.parse(responseText);
    errorDetail = json.detail || json.message || JSON.stringify(json);
  } catch {
    const titleMatch = (responseText || '').match(/<title>(.*?)<\/title>/i);
    if (titleMatch) {
      errorDetail = titleMatch[1].trim();
    } else {
      errorDetail = (responseText || '').slice(0, 300).trim();
    }
  }

  let reason = '';
  if (errorType === 'timeout') {
    reason = 'Request Timed Out';
  } else if (errorType === 'abort') {
    reason = 'Request Aborted';
  } else if (status === 0) {
    reason = 'Network / Connection Reset (Failed to reach server)';
  } else {
    switch (status) {
      case 413:
        reason = `413 Payload Too Large (${errorDetail || 'File exceeds limit'})`;
        break;
      case 400:
        reason = `400 Bad Request (${errorDetail || 'Invalid parameters'})`;
        break;
      case 404:
        reason = `404 Not Found (Endpoint '${endpoint}' missing)`;
        break;
      case 422:
        reason = `422 Unprocessable Entity (${errorDetail})`;
        break;
      case 500:
        reason = `500 Internal Server Error (${errorDetail || 'Backend exception'})`;
        break;
      case 502:
        reason = `502 Bad Gateway (${errorDetail || 'Cloudflare tunnel / backend unreachable'})`;
        break;
      case 503:
        reason = `503 Service Unavailable (${errorDetail || 'Server overloaded'})`;
        break;
      case 504:
        reason = `504 Gateway Timeout (${errorDetail || 'Cloudflare proxy timed out'})`;
        break;
      default:
        reason = `HTTP ${status} (${errorDetail || 'Error'})`;
    }
  }

  const details = [
    `Diagnostics Report`,
    `Status: ${status || 'Network Error'} (${errorType || 'HTTP Error'})`,
    `Endpoint: ${endpoint}`,
    `Origin: ${origin}`,
    `Reason: ${reason}`,
    `Server Output: ${errorDetail || responseText || 'None'}`
  ].join('\n');

  return {
    stage: endpoint.includes('upload') ? 'Upload (Network/HTTP)' : 'AI Inference / Server',
    reason: reason,
    details: details
  };
}

// Single Native Streaming File Upload with Accurate Speed Tracking and Timeout Protection
function uploadSingleFileWithTimeout(fileItem, timeoutMs = 120000) {
  return new Promise((resolve, reject) => {
    // If already uploaded on server, return existing serverId
    if (fileItem.serverId) {
      fileItem.status = 'uploaded';
      fileItem.uploadPercent = 100;
      return resolve(fileItem.serverId);
    }

    const xhr = new XMLHttpRequest();
    state.activeUploads.set(fileItem.id, xhr);

    fileItem.status = 'uploading';
    fileItem.uploadPercent = 0;
    fileItem.uploadSpeed = 0.0;
    updateQueueTable();

    const uploadStart = performance.now();
    let lastLoaded = 0;

    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable && e.total > 0) {
        const now = performance.now();
        const elapsedSec = (now - uploadStart) / 1000;
        const pct = Math.round((e.loaded / e.total) * 100);
        fileItem.uploadPercent = pct;

        if (elapsedSec > 0.05) {
          const speedMBs = (e.loaded / elapsedSec) / (1024 * 1024);
          fileItem.uploadSpeed = parseFloat(speedMBs.toFixed(2));
          state.uploadStats.currentSpeedMBs = fileItem.uploadSpeed;
        }

        updateQueueTable();
        updateOverallProgress();
      }
    };

    xhr.onload = () => {
      state.activeUploads.delete(fileItem.id);
      if (xhr.status >= 200 && xhr.status < 300) {
        try {
          const data = JSON.parse(xhr.responseText);
          const serverId = data.files && data.files.length > 0 ? data.files[0].id : null;
          if (!serverId) {
            throw new Error("No server ID returned from upload endpoint");
          }
          fileItem.serverId = serverId;
          fileItem.status = 'uploaded';
          fileItem.uploadPercent = 100;
          updateQueueTable();
          resolve(serverId);
        } catch (jsonErr) {
          const diag = parseHttpError(xhr, xhr.status, xhr.responseText, 'json_parse', '/api/upload');
          reject(diag);
        }
      } else {
        const diag = parseHttpError(xhr, xhr.status, xhr.responseText, 'http_error', '/api/upload');
        reject(diag);
      }
    };

    xhr.onerror = () => {
      state.activeUploads.delete(fileItem.id);
      const diag = parseHttpError(xhr, 0, '', 'network_error', '/api/upload');
      reject(diag);
    };

    xhr.ontimeout = () => {
      state.activeUploads.delete(fileItem.id);
      const diag = parseHttpError(xhr, 0, '', 'timeout', '/api/upload');
      reject(diag);
    };

    xhr.onabort = () => {
      state.activeUploads.delete(fileItem.id);
      const diag = parseHttpError(xhr, 0, '', 'abort', '/api/upload');
      reject(diag);
    };

    xhr.open('POST', '/api/upload', true);
    xhr.timeout = timeoutMs;

    const formData = new FormData();
    formData.append("files", fileItem.blob, fileItem.name);
    xhr.send(formData);
  });
}

// Single AI Inference Execution with Timeout Protection
async function processSingleFileWithTimeout(serverId, fileItem, processingParams, timeoutMs = 180000) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const res = await fetch('/api/process-file', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        file_id: serverId,
        upscale_factor: processingParams.upscale_factor,
        output_format: processingParams.output_format,
        jpeg_quality: processingParams.jpeg_quality,
        model: processingParams.model,
        target_width: processingParams.target_width,
        target_height: processingParams.target_height
      }),
      signal: controller.signal
    });

    clearTimeout(timer);

    const responseText = await res.text();
    if (!res.ok) {
      return parseHttpError(null, res.status, responseText, 'http_error', '/api/process-file');
    }

    return JSON.parse(responseText);
  } catch (err) {
    clearTimeout(timer);
    if (err.name === 'AbortError') {
      return {
        status: 'failed',
        error_stage: 'AI Inference Timeout',
        error_reason: `Inference timed out after ${timeoutMs / 1000}s`,
        error_details: `The inference worker did not return a response within ${timeoutMs / 1000} seconds.`
      };
    }
    return {
      status: 'failed',
      error_stage: 'Network / Inference Error',
      error_reason: err.message || 'Fetch Exception',
      error_details: String(err.stack || err)
    };
  }
}

// Master Pipelined Worker Pool Dispatcher
async function runWorkerPool(itemsToProcess, concurrency, processingParams) {
  state.processing = true;
  state.cancelRequested = false;
  toggleUIState(true);

  progressWrapper.style.display = 'block';
  progressStatusText.textContent = `Starting pipeline (${concurrency} workers)...`;
  updateOverallProgress();

  // Create work queue
  const workQueue = itemsToProcess.map((item, idx) => ({
    fileItem: item,
    queueIndex: idx + 1
  }));

  const numWorkers = Math.min(concurrency, workQueue.length);
  console.log(`[QUEUE] Starting Worker Pool with ${numWorkers} concurrent workers for ${workQueue.length} files.`);

  async function workerLoop(workerId) {
    while (workQueue.length > 0 && !state.cancelRequested) {
      const task = workQueue.shift();
      if (!task) break;

      const { fileItem, queueIndex } = task;

      try {
        // Step 1: Uploading
        logQueueLifecycle(fileItem, queueIndex, workerId, 'uploading');
        fileItem.status = 'uploading';
        updateQueueTable();
        updateOverallProgress();

        const serverId = await uploadSingleFileWithTimeout(fileItem, 120000);

        if (state.cancelRequested) {
          fileItem.status = 'cancelled';
          updateQueueTable();
          break;
        }

        // Step 2: Uploaded
        logQueueLifecycle(fileItem, queueIndex, workerId, 'uploaded');
        fileItem.status = 'uploaded';
        updateQueueTable();
        updateOverallProgress();

        // Step 3: Processing (Real-ESRGAN AI Inference)
        logQueueLifecycle(fileItem, queueIndex, workerId, 'processing');
        fileItem.status = 'processing';
        updateQueueTable();
        updateOverallProgress();

        const result = await processSingleFileWithTimeout(serverId, fileItem, processingParams, 180000);

        if (result.status === 'completed') {
          fileItem.status = 'completed';
          fileItem.name = result.name || fileItem.name;
          fileItem.output = result.output;
          fileItem.qc = result.qc;
          fileItem.processing_seconds = result.processing_seconds || 0.0;
          if (result.processing_seconds) {
            state.processingDurations.push(result.processing_seconds);
          }
          logQueueLifecycle(fileItem, queueIndex, workerId, 'completed');
        } else {
          fileItem.status = 'failed';
          fileItem.error_stage = result.error_stage || 'Inference Error';
          fileItem.error_reason = result.error_reason || 'Inference execution failed';
          fileItem.error_details = result.error_details || '';
          logQueueLifecycle(fileItem, queueIndex, workerId, 'failed');
        }
      } catch (errDiag) {
        fileItem.status = 'failed';
        fileItem.error_stage = errDiag.stage || 'Pipeline Failure';
        fileItem.error_reason = errDiag.reason || (errDiag.message || 'Worker Exception');
        fileItem.error_details = errDiag.details || String(errDiag.stack || errDiag);
        logQueueLifecycle(fileItem, queueIndex, workerId, 'failed');
        console.error(`[QUEUE] Worker ${workerId} exception on ${fileItem.name}:`, errDiag);
      } finally {
        // ALWAYS update table and progress in finally block to ensure worker release
        updateQueueTable();
        updateOverallProgress();
      }
    }

    console.log(`[QUEUE] Worker ${workerId} has completed its queue assignments.`);
  }

  // Kick off concurrent worker tasks
  const workers = [];
  for (let w = 1; w <= numWorkers; w++) {
    workers.push(workerLoop(w));
  }

  await Promise.all(workers);

  state.processing = false;
  toggleUIState(false);
  updateQueueTable();
  updateOverallProgress();

  if (state.cancelRequested) {
    progressStatusText.textContent = "Batch Run Cancelled.";
    progressEta.textContent = "--";
  } else {
    const completed = state.files.filter(f => f.status === 'completed').length;
    const failed = state.files.filter(f => f.status === 'failed').length;
    progressStatusText.textContent = `Batch Complete! (${completed} Completed, ${failed} Failed)`;
    progressEta.textContent = "Finished";
  }
}

// Start Batch Processing Button Click
btnStart.addEventListener('click', async () => {
  if (state.files.length === 0 || state.processing) return;

  const processable = state.files.filter(f => f.status === 'ready' || f.status === 'failed' || f.status === 'queued');
  if (processable.length === 0) {
    alert("No valid files available for processing.");
    return;
  }

  // Reset status to queued for visual clarity
  processable.forEach(f => {
    f.status = 'queued';
    f.error_stage = '';
    f.error_reason = '';
    f.error_details = '';
  });
  updateQueueTable();

  state.uploadConcurrency = parseInt(uploadConcurrencySelect.value) || 4;
  const scaleMode = document.querySelector('input[name="scale_mode"]:checked')?.value || '4';
  const upscaleModel = document.querySelector('input[name="upscale_model"]:checked')?.value || 'RealESRGAN_x4plus';
  const outputFormat = document.querySelector('input[name="output_format"]:checked')?.value || 'jpg';
  const jpegQuality = parseInt(jpegQualitySlider.value) || 95;
  const targetWidth = parseInt(targetWidthInput.value) || 3840;
  const targetHeight = parseInt(targetHeightInput.value) || 2160;

  const processingParams = {
    upscale_factor: scaleMode === 'target' ? 4 : (parseInt(scaleMode) || 4),
    output_format: outputFormat,
    jpeg_quality: jpegQuality,
    model: upscaleModel,
    target_width: scaleMode === 'target' ? targetWidth : null,
    target_height: scaleMode === 'target' ? targetHeight : null
  };

  await runWorkerPool(processable, state.uploadConcurrency, processingParams);
});

// Retry Failed Items Click Handler
btnRetryFailed.addEventListener('click', async () => {
  if (state.processing) return;
  btnRetryFailed.disabled = true;

  const failedItems = state.files.filter(f => f.status === 'failed');
  if (failedItems.length === 0) return;

  failedItems.forEach(f => {
    f.status = 'queued';
    f.error_stage = '';
    f.error_reason = '';
    f.error_details = '';
  });
  updateQueueTable();

  state.uploadConcurrency = parseInt(uploadConcurrencySelect.value) || 4;
  const scaleMode = document.querySelector('input[name="scale_mode"]:checked')?.value || '4';
  const upscaleModel = document.querySelector('input[name="upscale_model"]:checked')?.value || 'RealESRGAN_x4plus';
  const outputFormat = document.querySelector('input[name="output_format"]:checked')?.value || 'jpg';
  const jpegQuality = parseInt(jpegQualitySlider.value) || 95;
  const targetWidth = parseInt(targetWidthInput.value) || 3840;
  const targetHeight = parseInt(targetHeightInput.value) || 2160;

  const processingParams = {
    upscale_factor: scaleMode === 'target' ? 4 : (parseInt(scaleMode) || 4),
    output_format: outputFormat,
    jpeg_quality: jpegQuality,
    model: upscaleModel,
    target_width: scaleMode === 'target' ? targetWidth : null,
    target_height: scaleMode === 'target' ? targetHeight : null
  };

  await runWorkerPool(failedItems, state.uploadConcurrency, processingParams);
});

// Cancel Batch Execution
btnCancel.addEventListener('click', async () => {
  if (!state.processing) return;
  state.cancelRequested = true;
  btnCancel.disabled = true;
  progressStatusText.textContent = "Aborting batch workers...";

  // Abort active upload XHRs
  for (const [id, xhr] of state.activeUploads.entries()) {
    try { xhr.abort(); } catch {}
  }
  state.activeUploads.clear();

  // Cancel remaining queued items
  state.files.forEach(f => {
    if (f.status === 'queued' || f.status === 'ready') {
      f.status = 'cancelled';
    }
  });

  try {
    await fetch('/api/cancel', { method: 'POST' });
  } catch (err) {
    console.warn("Cancel request exception:", err);
  }

  updateQueueTable();
  updateOverallProgress();
});

// Download Archive & Single Images
btnDownloadAll.addEventListener('click', () => {
  window.location.href = '/api/download-all';
});

window.downloadProcessedImage = function(fileId) {
  const file = state.files.find(f => f.id === fileId);
  if (file) {
    window.location.href = `/api/download/${encodeURIComponent(file.name)}`;
  }
};

// UI Element Toggle Logic
function toggleUIState(isDisabled) {
  btnStart.disabled = isDisabled;
  btnCancel.disabled = !isDisabled;
  btnClearQueue.disabled = isDisabled;
  fileUploader.disabled = isDisabled;
  btnRetryFailed.disabled = isDisabled;

  Array.from(scaleModeRadios).forEach(r => r.disabled = isDisabled);
  Array.from(outputFormatRadios).forEach(r => r.disabled = isDisabled);
  Array.from(document.getElementsByName('upscale_model')).forEach(r => r.disabled = isDisabled);
  jpegQualitySlider.disabled = isDisabled;
  targetWidthInput.disabled = isDisabled;
  targetHeightInput.disabled = isDisabled;
  uploadConcurrencySelect.disabled = isDisabled;

  dropzone.style.pointerEvents = isDisabled ? 'none' : 'auto';
  dropzone.style.opacity = isDisabled ? 0.5 : 1;
}

// Health Poller
async function checkHealth() {
  try {
    const res = await fetch('/api/health');
    if (!res.ok) throw new Error("Offline");
    const data = await res.json();

    statusIndicator.className = "status-dot active";
    if (data.gpu) {
      statusText.textContent = `GPU: ${data.gpu_name}`;
    } else {
      statusText.textContent = "CPU Mode (No GPU)";
      statusIndicator.className = "status-dot warning";
    }

    if (data.ram_usage && data.vram_usage) {
      systemMetrics.style.display = 'flex';
      systemRam.textContent = `${data.ram_usage.used.toFixed(1)}/${data.ram_usage.total.toFixed(0)} GB`;
      systemVram.textContent = data.gpu ? `${data.vram_usage.used.toFixed(1)}/${data.vram_usage.total.toFixed(0)} GB` : 'N/A';
    }
  } catch (err) {
    statusIndicator.className = "status-dot error";
    statusText.textContent = "Server Offline";
    systemMetrics.style.display = 'none';
  }
}
state.healthTimer = setInterval(checkHealth, 3000);
checkHealth();

// Radio Button Event Attachment
Array.from(scaleModeRadios).forEach(radio => {
  radio.addEventListener('change', (e) => {
    targetResInputs.style.display = e.target.value === 'target' ? 'flex' : 'none';
    updateQueueTable();
  });
});

Array.from(outputFormatRadios).forEach(radio => {
  radio.addEventListener('change', (e) => {
    jpegQualityGroup.style.display = e.target.value === 'png' ? 'none' : 'block';
  });
});

jpegQualitySlider.addEventListener('input', (e) => {
  jpegQualityVal.textContent = e.target.value + '%';
});

uploadConcurrencySelect.addEventListener('change', (e) => {
  state.uploadConcurrency = parseInt(e.target.value) || 4;
});

// Comparison Slider Modal Handlers
window.openPreview = function(fileId) {
  const file = state.files.find(f => f.id === fileId);
  if (!file || file.status !== 'completed') return;

  modalFilename.textContent = file.name;
  const outW = file.output ? file.output.width : '--';
  const outH = file.output ? file.output.height : '--';
  const outMp = file.output ? file.output.megapixels : '--';
  modalDimensions.textContent = `Original: ${file.width}×${file.height} (${file.megapixels} MP) | Upscaled: ${outW}×${outH} (${outMp} MP)`;

  imgOriginal.src = file.thumbnail;
  imgUpscaled.src = `/api/download/${encodeURIComponent(file.name)}?t=${Date.now()}`;

  modalQcStatus.className = `qc-badge ${file.qc && file.qc.passed ? 'pass' : 'warn'}`;
  modalQcStatus.textContent = file.qc && file.qc.passed ? 'PASS' : 'WARN';
  modalQcMp.textContent = `${outMp} MP`;

  const fileRatio = (file.width / file.height).toFixed(3);
  const outRatio = file.output ? (file.output.width / file.output.height).toFixed(3) : '--';
  modalQcRatio.textContent = `${fileRatio} (${fileRatio === outRatio ? 'Match' : 'Drift'})`;

  sliderRange.value = 50;
  updateSliderPosition(50);

  modalDownloadBtn.onclick = () => downloadProcessedImage(file.id);
  previewModal.classList.add('active');
};

modalClose.addEventListener('click', () => previewModal.classList.remove('active'));
previewModal.addEventListener('click', (e) => {
  if (e.target === previewModal) previewModal.classList.remove('active');
});

function updateSliderPosition(val) {
  imgUpscaledWrapper.style.clipPath = `polygon(0 0, ${val}% 0, ${val}% 100%, 0 100%)`;
  handleLine.style.left = `${val}%`;
  handleButton.style.left = `${val}%`;
}

sliderRange.addEventListener('input', (e) => updateSliderPosition(e.target.value));

// Diagnostic Technical Log Modal Handler
window.openErrorModal = function(fileId) {
  const file = state.files.find(f => f.id === fileId);
  if (!file) return;

  errorModalStage.textContent = file.error_stage || 'Unknown';
  errorModalReason.textContent = file.error_reason || 'Inference error';
  errorModalTrace.textContent = file.error_details || 'No extended trace logs available.';

  errorModal.classList.add('active');
};

errorModalClose.addEventListener('click', () => errorModal.classList.remove('active'));
errorModal.addEventListener('click', (e) => {
  if (e.target === errorModal) errorModal.classList.remove('active');
});
