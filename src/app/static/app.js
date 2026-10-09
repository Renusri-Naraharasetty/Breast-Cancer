/**
 * BCMID Multimodal & Single-Modality Breast Cancer Classification Research Demo
 * Client Application Logic
 * 
 * Supports:
 * - Single-Modality: Mammogram (DenseNet-121, τ = 0.2932)
 * - Single-Modality: Ultrasound (ResNet-50, τ = 0.3844)
 * - Multimodal: Mammogram + Ultrasound (Graph Transformer, τ = 0.2565)
 * 
 * DISCLAIMER: Research prototype — not for clinical diagnosis.
 */

document.addEventListener("DOMContentLoaded", () => {
  // State
  let mammoFile = null;
  let usFile = null;
  let stepInterval = null;

  // DOM Elements - Uploads
  const dropZoneMammo = document.getElementById("drop-zone-mammo");
  const fileInputMammo = document.getElementById("file-mammo");
  const dropContentMammo = document.getElementById("drop-content-mammo");
  const previewMammoContainer = document.getElementById("preview-mammo-container");
  const previewMammoImg = document.getElementById("preview-mammo-img");
  const previewMammoName = document.getElementById("preview-mammo-name");
  const btnRemoveMammo = document.getElementById("btn-remove-mammo");

  const dropZoneUs = document.getElementById("drop-zone-us");
  const fileInputUs = document.getElementById("file-us");
  const dropContentUs = document.getElementById("drop-content-us");
  const previewUsContainer = document.getElementById("preview-us-container");
  const previewUsImg = document.getElementById("preview-us-img");
  const previewUsName = document.getElementById("preview-us-name");
  const btnRemoveUs = document.getElementById("btn-remove-us");

  // Mode Indicator Elements
  const activeModeBadge = document.getElementById("active-mode-badge");
  const activeModelBadge = document.getElementById("active-model-badge");
  const activeThresholdChip = document.getElementById("active-threshold-chip");

  // Controls & Status
  const btnAnalyze = document.getElementById("btn-analyze");
  const btnReset = document.getElementById("btn-reset");
  const btnSpinner = document.getElementById("btn-spinner");
  const btnText = document.getElementById("btn-text");
  const statusSummary = document.getElementById("status-summary");
  const errorBanner = document.getElementById("error-banner");
  const errorText = document.getElementById("error-text");
  const errorClose = document.getElementById("error-close");
  const loadingPanel = document.getElementById("loading-panel");
  const loadingTitle = document.getElementById("loading-title");
  const loadingStep = document.getElementById("loading-step");

  // Samples
  const btnSample1 = document.getElementById("btn-load-sample-1");
  const btnSample2 = document.getElementById("btn-load-sample-2");

  // Results Elements
  const resultsSection = document.getElementById("results-section");
  const resModePill = document.getElementById("res-mode-pill");
  const resModelPill = document.getElementById("res-model-pill");
  const resClassTitle = document.getElementById("res-class-title");
  const resBadge = document.getElementById("res-badge");
  const resBadgeText = document.getElementById("res-badge-text");
  const resStatusIcon = document.getElementById("res-status-icon");
  const resProbVal = document.getElementById("res-prob-val");
  const resGaugeFill = document.getElementById("res-gauge-fill");
  const resGaugeThresholdLine = document.getElementById("res-gauge-threshold-line");
  const resThresholdMarkerLabel = document.getElementById("res-threshold-marker-label");
  const resGaugeNote = document.getElementById("res-gauge-note");
  const resProbMalignant = document.getElementById("res-prob-malignant");
  const resProbBenign = document.getElementById("res-prob-benign");
  const resThreshold = document.getElementById("res-threshold");
  const resConfidence = document.getElementById("res-confidence");
  const resConfidenceSub = document.getElementById("res-confidence-sub");
  const resMetricMode = document.getElementById("res-metric-mode");
  const resMetricModel = document.getElementById("res-metric-model");
  const resGuidanceText = document.getElementById("res-guidance-text");

  // Explainability Elements
  const expTag = document.getElementById("exp-tag");
  const expTitle = document.getElementById("exp-title");
  const expInfoPill = document.getElementById("exp-info-pill");
  const expDesc = document.getElementById("exp-desc");
  const explainabilityImg = document.getElementById("explainability-img");
  const attentionBox = document.getElementById("attention-box");
  const attentionPills = document.getElementById("attention-pills");

  // Dynamic Pipeline Simulation Steps
  const PIPELINE_STEPS_MULTIMODAL = [
    "Applying clinical evaluation transforms (Resize 224x224 & ImageNet Normalization)...",
    "Extracting deep CNN representations via dual-branch backbones...",
    "Computing Graph Attention fusion & Transformer cross-modality interaction...",
    "Evaluating calibrated operating decision threshold (τ = 0.2565)...",
    "Generating multimodal attention heatmaps and report..."
  ];

  const PIPELINE_STEPS_MAMMO = [
    "Applying clinical mammogram transforms (Resize 224x224 & Normalization)...",
    "Extracting feature embeddings via DenseNet-121 backbone...",
    "Evaluating validation-selected operating threshold (τ = 0.2932)...",
    "Computing DenseNet-121 Grad-CAM saliency activation map...",
    "Generating single-modality diagnostic report..."
  ];

  const PIPELINE_STEPS_US = [
    "Applying ultrasound preprocessing transforms (Resize 224x224 & Normalization)...",
    "Extracting feature embeddings via ResNet-50 backbone...",
    "Evaluating validation-selected operating threshold (τ = 0.3844)...",
    "Computing ResNet-50 Grad-CAM saliency activation map...",
    "Generating single-modality diagnostic report..."
  ];

  // Helper: Update UI Controls and Mode Indicator State
  function updateControlsState() {
    const hasMammo = Boolean(mammoFile);
    const hasUs = Boolean(usFile);
    const hasAny = hasMammo || hasUs;

    btnAnalyze.disabled = !hasAny;

    if (hasMammo && hasUs) {
      // Multimodal Mode
      activeModeBadge.textContent = "Multimodal Analysis";
      activeModeBadge.className = "mode-badge badge-multimodal";
      activeModelBadge.textContent = "Model: Graph Transformer";
      activeThresholdChip.textContent = "Operating Cutoff: τ = 0.2565";
      activeThresholdChip.className = "threshold-chip chip-multimodal";

      btnText.textContent = "⚡ Run Multimodal Analysis (Graph Transformer)";
      statusSummary.textContent = "Both modalities loaded. Ready for multimodal Graph Transformer inference.";
      statusSummary.style.color = "var(--accent-cyan)";

    } else if (hasMammo && !hasUs) {
      // Mammogram Only
      activeModeBadge.textContent = "Single-Modality Analysis — Mammogram";
      activeModeBadge.className = "mode-badge badge-mammo";
      activeModelBadge.textContent = "Model: DenseNet-121";
      activeThresholdChip.textContent = "Operating Cutoff: τ = 0.2932";
      activeThresholdChip.className = "threshold-chip chip-mammo";

      btnText.textContent = "⚡ Run Mammogram Analysis (DenseNet-121)";
      statusSummary.textContent = "Mammogram loaded. Ready for single-modality DenseNet-121 inference.";
      statusSummary.style.color = "var(--accent-cyan)";

    } else if (!hasMammo && hasUs) {
      // Ultrasound Only
      activeModeBadge.textContent = "Single-Modality Analysis — Ultrasound";
      activeModeBadge.className = "mode-badge badge-us";
      activeModelBadge.textContent = "Model: ResNet-50";
      activeThresholdChip.textContent = "Operating Cutoff: τ = 0.3844";
      activeThresholdChip.className = "threshold-chip chip-us";

      btnText.textContent = "⚡ Run Ultrasound Analysis (ResNet-50)";
      statusSummary.textContent = "Ultrasound loaded. Ready for single-modality ResNet-50 inference.";
      statusSummary.style.color = "var(--accent-rose)";

    } else {
      // Empty State
      activeModeBadge.textContent = "Waiting for Scan Upload...";
      activeModeBadge.className = "mode-badge badge-empty";
      activeModelBadge.textContent = "Upload ≥ 1 image";
      activeThresholdChip.textContent = "Operating Cutoff: --";
      activeThresholdChip.className = "threshold-chip chip-empty";

      btnText.textContent = "⚡ Run Analysis";
      statusSummary.textContent = "Please upload at least one image (Mammogram or Ultrasound) to begin.";
      statusSummary.style.color = "var(--text-muted)";
    }
  }

  // Helper: Show Error
  function showError(message) {
    errorText.textContent = message;
    errorBanner.classList.remove("hidden");
  }

  // Helper: Hide Error
  function hideError() {
    errorBanner.classList.add("hidden");
  }

  errorClose.addEventListener("click", hideError);

  // Setup File Input & Drag and Drop for a Card
  function setupUploadZone(zone, input, content, previewContainer, previewImg, nameLabel, isMammo) {
    zone.addEventListener("click", (e) => {
      if (e.target.closest(".btn-remove")) return;
      input.click();
    });

    ["dragenter", "dragover"].forEach(eventName => {
      zone.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
        zone.classList.add("drag-over");
      }, false);
    });

    ["dragleave", "drop"].forEach(eventName => {
      zone.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
        zone.classList.remove("drag-over");
      }, false);
    });

    zone.addEventListener("drop", (e) => {
      const dt = e.dataTransfer;
      if (dt.files && dt.files.length > 0) {
        handleFileSelect(dt.files[0], isMammo);
      }
    });

    input.addEventListener("change", (e) => {
      if (e.target.files && e.target.files.length > 0) {
        handleFileSelect(e.target.files[0], isMammo);
      }
    });
  }

  function handleFileSelect(file, isMammo) {
    hideError();

    const validTypes = ["image/png", "image/jpeg", "image/tiff", "image/bmp"];
    const validExts = [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"];
    const ext = "." + file.name.split(".").pop().toLowerCase();

    if (!validTypes.includes(file.type) && !validExts.includes(ext)) {
      showError(`Unsupported file format '${ext}'. Please upload PNG, JPG, or TIFF image files.`);
      return;
    }

    if (isMammo) {
      mammoFile = file;
      previewMammoImg.src = URL.createObjectURL(file);
      previewMammoName.textContent = file.name;
      dropContentMammo.classList.add("hidden");
      previewMammoContainer.classList.remove("hidden");
    } else {
      usFile = file;
      previewUsImg.src = URL.createObjectURL(file);
      previewUsName.textContent = file.name;
      dropContentUs.classList.add("hidden");
      previewUsContainer.classList.remove("hidden");
    }

    updateControlsState();
  }

  // Remove Buttons
  btnRemoveMammo.addEventListener("click", (e) => {
    e.stopPropagation();
    mammoFile = null;
    fileInputMammo.value = "";
    previewMammoContainer.classList.add("hidden");
    dropContentMammo.classList.remove("hidden");
    updateControlsState();
  });

  btnRemoveUs.addEventListener("click", (e) => {
    e.stopPropagation();
    usFile = null;
    fileInputUs.value = "";
    previewUsContainer.classList.add("hidden");
    dropContentUs.classList.remove("hidden");
    updateControlsState();
  });

  // Initialize upload zones
  setupUploadZone(
    dropZoneMammo,
    fileInputMammo,
    dropContentMammo,
    previewMammoContainer,
    previewMammoImg,
    previewMammoName,
    true
  );

  setupUploadZone(
    dropZoneUs,
    fileInputUs,
    dropContentUs,
    previewUsContainer,
    previewUsImg,
    previewUsName,
    false
  );

  // Load Multimodal Test Samples (Labels Omitted)
  async function loadSample(caseId, btnElement) {
    hideError();
    const originalText = btnElement.innerHTML;
    btnElement.disabled = true;
    btnElement.innerHTML = `<span class="btn-spinner"></span> Loading Scans...`;

    try {
      const cacheBust = Date.now();
      const [mammoResp, usResp] = await Promise.all([
        fetch(`/api/samples/${caseId}/mammogram?t=${cacheBust}`),
        fetch(`/api/samples/${caseId}/ultrasound?t=${cacheBust}`)
      ]);

      if (!mammoResp.ok || !usResp.ok) {
        throw new Error("Failed to load research sample scans from backend.");
      }

      const mammoBlob = await mammoResp.blob();
      const usBlob = await usResp.blob();

      const sampleLabel = caseId === "case_1" ? "Low-Risk Sample" : "High-Risk Sample";
      const samplePrefix = caseId === "case_1" ? "low_risk" : "high_risk";

      mammoFile = new File([mammoBlob], `${samplePrefix}_mammogram.png`, { type: "image/png" });
      usFile = new File([usBlob], `${samplePrefix}_ultrasound.png`, { type: "image/png" });

      previewMammoImg.src = URL.createObjectURL(mammoBlob);
      previewMammoName.textContent = `${sampleLabel} — Mammogram`;
      dropContentMammo.classList.add("hidden");
      previewMammoContainer.classList.remove("hidden");

      previewUsImg.src = URL.createObjectURL(usBlob);
      previewUsName.textContent = `${sampleLabel} — Ultrasound`;
      dropContentUs.classList.add("hidden");
      previewUsContainer.classList.remove("hidden");

      updateControlsState();
    } catch (err) {
      showError(err.message || "Failed to load research sample data.");
    } finally {
      btnElement.disabled = false;
      btnElement.innerHTML = originalText;
    }
  }

  btnSample1.addEventListener("click", () => loadSample("case_1", btnSample1));
  btnSample2.addEventListener("click", () => loadSample("case_2", btnSample2));

  // Run Inference (Single-Modality or Multimodal)
  btnAnalyze.addEventListener("click", async () => {
    if (!mammoFile && !usFile) {
      showError("Please upload at least one image (Mammogram or Ultrasound) before analyzing.");
      return;
    }

    hideError();
    resultsSection.classList.add("hidden");
    loadingPanel.classList.remove("hidden");

    btnAnalyze.disabled = true;
    btnSpinner.classList.remove("hidden");

    // Select pipeline animation steps
    let currentSteps = PIPELINE_STEPS_MULTIMODAL;
    if (mammoFile && usFile) {
      loadingTitle.textContent = "Processing Multimodal Input (Graph Transformer)";
      currentSteps = PIPELINE_STEPS_MULTIMODAL;
    } else if (mammoFile) {
      loadingTitle.textContent = "Processing Mammogram (DenseNet-121)";
      currentSteps = PIPELINE_STEPS_MAMMO;
    } else {
      loadingTitle.textContent = "Processing Ultrasound (ResNet-50)";
      currentSteps = PIPELINE_STEPS_US;
    }

    btnText.textContent = "Analyzing Scan(s)...";

    let stepIdx = 0;
    loadingStep.textContent = currentSteps[0];
    stepInterval = setInterval(() => {
      stepIdx = (stepIdx + 1) % currentSteps.length;
      loadingStep.textContent = currentSteps[stepIdx];
    }, 650);

    const formData = new FormData();
    if (mammoFile) formData.append("mammogram", mammoFile);
    if (usFile) formData.append("ultrasound", usFile);

    try {
      const response = await fetch("/api/analyze", {
        method: "POST",
        body: formData
      });

      if (!response.ok) {
        let errData;
        try {
          errData = await response.json();
        } catch {
          errData = { detail: `Server error HTTP ${response.status}` };
        }
        throw new Error(errData.detail || "Inference request failed.");
      }

      const data = await response.json();
      renderResults(data);

    } catch (err) {
      showError(err.message || "Failed to complete model analysis.");
    } finally {
      clearInterval(stepInterval);
      loadingPanel.classList.add("hidden");
      btnSpinner.classList.add("hidden");
      updateControlsState();
    }
  });

  // Render Inference Results
  function renderResults(data) {
    const pred = data.prediction;
    const isSuspicious = pred.is_positive_suspicious;
    const probMalignantPct = (pred.prob_malignant * 100);
    const thresholdPct = (pred.calibrated_threshold * 100);

    // Mode & Model Pill
    resModePill.textContent = data.analysis_mode;
    resModelPill.textContent = "Model: " + data.model_name;

    // Header & Badge
    resClassTitle.textContent = pred.class_name;
    
    if (isSuspicious) {
      resBadge.className = "results-status-badge badge-suspicious";
      resBadgeText.textContent = "Suspicious Finding (High Risk)";
      resStatusIcon.textContent = "▲";
      resGaugeFill.className = "gauge-fill gauge-suspicious";
    } else {
      resBadge.className = "results-status-badge badge-benign";
      resBadgeText.textContent = "Negative / Benign (Low Risk)";
      resStatusIcon.textContent = "●";
      resGaugeFill.className = "gauge-fill gauge-benign";
    }

    // Gauge bar
    resProbVal.textContent = `${probMalignantPct.toFixed(2)}%`;
    resGaugeFill.style.width = `${Math.min(100, Math.max(0, probMalignantPct))}%`;
    resGaugeThresholdLine.style.left = `${Math.min(100, Math.max(0, thresholdPct))}%`;
    resThresholdMarkerLabel.textContent = `Threshold: τ = ${pred.calibrated_threshold.toFixed(4)}`;
    resGaugeNote.textContent = `▲ Cutoff (${pred.calibrated_threshold.toFixed(4)})`;

    // Metric cards
    resProbMalignant.textContent = pred.prob_malignant.toFixed(4);
    resProbBenign.textContent = pred.prob_benign.toFixed(4);
    resThreshold.textContent = pred.calibrated_threshold.toFixed(4);
    resConfidence.textContent = pred.confidence_level;
    
    const margin = (pred.prob_malignant - pred.calibrated_threshold);
    const marginText = margin >= 0 ? `+${margin.toFixed(4)} above cutoff` : `${margin.toFixed(4)} below cutoff`;
    resConfidenceSub.textContent = marginText;

    resMetricMode.textContent = data.analysis_mode;
    resMetricModel.textContent = data.model_name;

    // Guidance text
    resGuidanceText.textContent = pred.clinical_guidance;

    // Explainability handling
    if (data.explainability) {
      const exp = data.explainability;

      if (exp.type === "gradcam") {
        // Unimodal Grad-CAM
        attentionBox.classList.add("hidden");
        expTag.textContent = "GRAD-CAM SALIENCY HEATMAP";
        expTitle.textContent = `${data.model_name} Grad-CAM Saliency Analysis`;
        expInfoPill.textContent = `${data.model_name} Activation Mapping`;
        expDesc.textContent = `Visualizes discriminative anatomical regions identified by the convolutional backbone (${data.model_name}) using Gradient-weighted Class Activation Mapping.`;
      } else {
        // Multimodal Attention
        attentionBox.classList.remove("hidden");
        expTag.textContent = "EXPLAINABILITY & ATTENTION ANALYSIS";
        expTitle.textContent = "Cross-Modality Graph Attention Interaction";
        expInfoPill.textContent = "GAT + Transformer Attention";
        expDesc.textContent = "Visualizes multi-head attention weights between the Mammogram and Ultrasound representation tokens in the Graph Attention Layer and the Self-Attention Transformer fusion module.";

        // Attention Pills
        attentionPills.innerHTML = "";
        if (exp.attention_weights) {
          const weights = exp.attention_weights;
          const pillData = [
            { label: "Transformer: Mammo → US", val: weights.transformer_mammo_to_us },
            { label: "Transformer: US → Mammo", val: weights.transformer_us_to_mammo },
            { label: "Transformer: Mammo Self", val: weights.transformer_mammo_self },
            { label: "Transformer: US Self", val: weights.transformer_us_self }
          ];

          pillData.forEach(item => {
            if (item.val !== undefined && item.val !== null) {
              const pill = document.createElement("div");
              pill.className = "attention-pill";
              pill.innerHTML = `<span>${item.label}:</span> <strong>${Number(item.val).toFixed(3)}</strong>`;
              attentionPills.appendChild(pill);
            }
          });
        }
      }

      if (exp.plot_base64) {
        explainabilityImg.src = exp.plot_base64;
      }
    }

    // Reveal results and scroll smoothly
    resultsSection.classList.remove("hidden");
    resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  // Reset Workspace
  btnReset.addEventListener("click", () => {
    hideError();
    clearInterval(stepInterval);
    loadingPanel.classList.add("hidden");
    resultsSection.classList.add("hidden");

    mammoFile = null;
    usFile = null;
    fileInputMammo.value = "";
    fileInputUs.value = "";

    previewMammoContainer.classList.add("hidden");
    dropContentMammo.classList.remove("hidden");
    previewMammoImg.src = "";

    previewUsContainer.classList.add("hidden");
    dropContentUs.classList.remove("hidden");
    previewUsImg.src = "";

    updateControlsState();
    window.scrollTo({ top: 0, behavior: "smooth" });
  });

  // Initial setup check
  updateControlsState();
});
