# Phase D2 - Experiment D1: Frozen C3 Benchmark Reproduction Report

**Experiment ID:** `exp_d1_c3_repro`  
**Execution Timestamp:** 2026-10-15T12:20:09Z  
**Hardware Environment:** NVIDIA GeForce MX550 (2048 MB VRAM), CUDA 12.x, PyTorch 2.5.1  
**Project Workspace:** `C:\Users\admin\Music\Breast Cancer1`  
**Phase D Output Path:** `outputs/phase_d/experiments/exp_d1_c3_repro/`  

---

## 1. Executive Summary

Phase D2, Experiment D1 successfully executed a bit-exact, deterministic reproduction of the frozen Candidate C3 benchmark (`exp_c3_git1_focal_gamma1_5`) under strict methodological controls. 

### Key Findings
1. **100% Metric Concordance:** Experiment D1 selected **Epoch 4**, reproducing all validation metrics to four decimal places:
   - **Composite Score:** **0.8351** (Exact Match: $0.8351$)
   - **Validation ROC-AUC:** **0.9281** (Exact Match: $0.9281$)
   - **Validation PR-AUC:** **0.7661** (Exact Match: $0.7661$)
   - **Validation F1-Score:** **0.7059** (Exact Match: $0.7059$)
   - **Validation Sensitivity:** **75.00%** ($6/8$) (Exact Match: $75.00\%$)
   - **Validation Specificity:** **92.50%** ($37/40$) (Exact Match: $92.50\%$)
   - **Validation Balanced Accuracy:** **83.75%** (Exact Match: $83.75\%$)
   - **Validation Loss:** **0.1679** (Exact Match: $0.1679$)
   - **Confusion Matrix:** `[37, 3, 2, 6]` (Bit-for-bit identical)
2. **Deterministic Prediction Concordance:** Every single one of the 48 validation patient-level continuous predicted probabilities matches the saved Phase C C3 predictions to full IEEE 754 floating-point precision ($100.000\%$ agreement).
3. **Hardware Efficiency:** Peak VRAM was measured at **105.52 MB** with a total wall-clock runtime of **48.40 seconds** (0.81 minutes), proving that Phase D iterations are exceptionally lightweight and hardware-safe on the 2.0 GB GeForce MX550.
4. **Zero Test Contamination:** The held-out test cohort ($N=47$) was strictly quarantined and never loaded or evaluated.

---

## 2. Image-Encoder Zero-Contamination Audit

Before initiating training, an audit was conducted on the upstream image-encoder training manifests, scripts, and logs to verify that the EfficientNet-B0 mammogram encoder (`exp1`) and ViT-B/16 ultrasound encoder (`exp2`) were trained and selected without using validation or test patients for gradient updates.

### 2.1 Code and Dataloader Partitioning
- **Source File:** `src/training/trainer.py`
  - Lines 136–151: Dataloaders are instantiated exclusively for `split="train"` and `split="val"`. No dataloader for `split="test"` is instantiated or referenced in the training code.
  - Lines 196–219: The gradient update loop (`scaler.scale(loss).backward()` and `scaler.step(optimizer)`) executes **strictly and exclusively** during iteration over `train_loader`.
  - Lines 224 & 33–50: Validation data is processed exclusively through `evaluate_validation()`, which is enclosed in `with torch.no_grad():`. Gradients are disabled, and validation tensors never interact with `optimizer.step()`.
- **Source File:** `src/data/dataset.py`
  - Lines 63–75: Patient filtering enforces `allowed_folders = set(splits_data["folder_patient_splits"][split])`. When `split="train"`, only the 220 train patient folders are indexed. Validation ($N=48$) and test ($N=47$) folders are completely excluded from the training dataset.

### 2.2 Manifest and Historical Log Evidence
- **Manifest:** `outputs/experiments/phase4_training_summary.json`
  - `exp1_efficientnet_mammogram`: Best checkpoint selected at Epoch 1 on `val_loss=0.6349` (`val_roc_auc=0.5131`). Early stopped at Epoch 5. Checkpoint saved to `outputs/experiments/exp1_efficientnet_mammogram/best_checkpoint.pth`.
  - `exp2_vit_ultrasound`: Best checkpoint selected at Epoch 1 on `val_loss=0.5189` (`val_roc_auc=0.7284`). Early stopped at Epoch 5. Checkpoint saved to `outputs/experiments/exp2_vit_ultrasound/best_checkpoint.pth`.
- **Feature Extraction Provenance (`scripts/extract_multimodal_features.py`):**
  - Mammogram feature vectors (1280-dim) were extracted using `exp1` best checkpoint in `model.eval()` mode under `torch.no_grad()`.
  - Ultrasound feature vectors (768-dim) were extracted using `exp2` best checkpoint in `model.eval()` mode under `torch.no_grad()`.
  - Neither feature extraction process executed backpropagation, ensuring completely frozen representations.

---

## 3. Experiment D1 Configuration & Implementation Parameters

Experiment D1 reproduced the verified C3 architecture and training configuration without modification:

| Parameter | C3 Reference Value | D1 Reproduction Value | Status |
| :--- | :--- | :--- | :---: |
| **Model Architecture** | `GraphInfusedTransformer` | `GraphInfusedTransformer` | Identical |
| **Mammogram Feature Dim** | 1280 (EfficientNet-B0) | 1280 (EfficientNet-B0) | Identical |
| **Ultrasound Feature Dim** | 768 (ViT-B/16) | 768 (ViT-B/16) | Identical |
| **Hidden Embedding Dim** | 256 | 256 | Identical |
| **GAT Layers & Heads** | 2 layers, 4 attention heads | 2 layers, 4 attention heads | Identical |
| **Transformer Layers & Heads**| 2 layers, 8 attention heads | 2 layers, 8 attention heads | Identical |
| **Feedforward Dim ($d_{ff}$)** | 1024 | 1024 | Identical |
| **Dropout Rate** | 0.30 | 0.30 | Identical |
| **Loss Function** | ClassBalancedFocalLoss | ClassBalancedFocalLoss | Identical |
| **Focal Gamma ($\gamma$)** | 1.5 | 1.5 | Identical |
| **Class Weighting ($\alpha$)** | `[1.0, 5.1111]` (normalized $w_0=1.0$) | `[1.0, 5.1111]` (normalized $w_0=1.0$) | Identical |
| **Loss Reduction** | Mean divided by $\sum \alpha_t$ across batch | Mean divided by $\sum \alpha_t$ across batch | Identical |
| **Optimizer** | AdamW ($\text{lr}=10^{-4}, \text{wd}=10^{-2}$) | AdamW ($\text{lr}=10^{-4}, \text{wd}=10^{-2}$) | Identical |
| **Learning Rate Scheduler** | CosineAnnealingLR ($T_{\max}=20, \eta_{\min}=10^{-6}$) | CosineAnnealingLR ($T_{\max}=20, \eta_{\min}=10^{-6}$) | Identical |
| **Batch Size & Grad Accum** | $8 \times 2 = 16$ effective batch size | $8 \times 2 = 16$ effective batch size | Identical |
| **Precision** | Mixed Precision (`torch.amp.autocast`) | Mixed Precision (`torch.amp.autocast`) | Identical |
| **Random Seed** | 42 | 42 | Identical |
| **Selection Protocol** | $S = 0.5 \text{AUC} + 0.3 \text{PR} + 0.2 \text{F1}$ | $S = 0.5 \text{AUC} + 0.3 \text{PR} + 0.2 \text{F1}$ | Identical |
| **Early Stopping Rule** | Patience = 8 epochs | Patience = 8 epochs | Identical |

---

## 4. Epoch-by-Epoch Training Trajectory

Below is the epoch-by-epoch trajectory logged during Experiment D1 execution:

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc | Val Sens | Val Spec | Val BalAcc | Val F1 | Val ROC-AUC | Val PR-AUC | Composite Score ($S$) | Selection Flag |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 0.2140 | 67.3% | 0.2464 | 83.3% | 0.00% | 100.0% | 50.00% | 0.0000 | 0.8969 | 0.6439 | 0.6416 | Best (Epoch 1) |
| 2 | 0.1879 | 80.9% | 0.1791 | 62.5% | 100.0% | 55.0% | 77.50% | 0.4706 | 0.8812 | 0.5551 | 0.7013 | Best (Epoch 2) |
| 3 | 0.1333 | 85.9% | 0.1445 | 83.3% | 50.0% | 90.0% | 70.00% | 0.5000 | 0.9172 | 0.7029 | 0.7695 | Best (Epoch 3) |
| **4** | **0.1590** | **90.9%** | **0.1679** | **89.6%** | **75.0%** | **92.5%** | **83.75%** | **0.7059** | **0.9281** | **0.7661** | **0.8351** | **SELECTED BEST** |
| 5 | 0.0737 | 87.7% | 0.4927 | 87.5% | 25.0% | 100.0% | 62.50% | 0.4000 | 0.9375 | 0.7558 | 0.7755 | — |
| 6 | 0.1422 | 92.7% | 0.1391 | 79.2% | 75.0% | 80.0% | 77.50% | 0.5455 | 0.9094 | 0.7063 | 0.7757 | — |
| 7 | 0.0537 | 95.9% | 0.2771 | 85.4% | 50.0% | 92.5% | 71.25% | 0.5333 | 0.9313 | 0.7795 | 0.8062 | — |
| 8 | 0.0260 | 98.2% | 0.4937 | 87.5% | 37.5% | 97.5% | 67.50% | 0.5000 | 0.8969 | 0.6510 | 0.7437 | — |
| 9 | 0.0239 | 98.6% | 0.5919 | 89.6% | 37.5% | 100.0% | 68.75% | 0.5455 | 0.9219 | 0.7500 | 0.7950 | — |
| 10 | 0.0102 | 100.0% | 0.3775 | 89.6% | 62.5% | 95.0% | 78.75% | 0.6667 | 0.9281 | 0.7594 | 0.8252 | — |
| 11 | 0.0056 | 99.1% | 0.7613 | 87.5% | 25.0% | 100.0% | 62.50% | 0.4000 | 0.9156 | 0.7283 | 0.7563 | — |
| 12 | 0.0235 | 99.1% | 0.7972 | 87.5% | 25.0% | 100.0% | 62.50% | 0.4000 | 0.9031 | 0.6942 | 0.7398 | Stopped (Patience=8) |

---

## 5. Side-by-Side Comparison: D1 vs. Frozen C3 Benchmark

| Metric | Frozen C3 Benchmark | Experiment D1 Repro | Absolute Difference | Verification Status |
| :--- | :---: | :---: | :---: | :---: |
| **Selected Epoch** | **4** | **4** | **0** | **Exact Match** |
| **Composite Score ($S$)** | **0.8351** | **0.8351** | **0.0000** | **Exact Match** |
| **Validation ROC-AUC** | **0.9281** | **0.9281** | **0.0000** | **Exact Match** |
| **Validation PR-AUC** | **0.7661** | **0.7661** | **0.0000** | **Exact Match** |
| **Validation F1-Score** | **0.7059** | **0.7059** | **0.0000** | **Exact Match** |
| **Validation Sensitivity** | **75.00%** ($6/8$) | **75.00%** ($6/8$) | **0.0000** | **Exact Match** |
| **Validation Specificity** | **92.50%** ($37/40$) | **92.50%** ($37/40$) | **0.0000** | **Exact Match** |
| **Validation Balanced Accuracy** | **83.75%** | **83.75%** | **0.0000** | **Exact Match** |
| **Validation Loss** | **0.1679** | **0.1679** | **0.0000** | **Exact Match** |
| **Training Loss at Best Epoch** | **0.1590** | **0.1590** | **0.0000** | **Exact Match** |
| **Loss Overfitting Gap** | **+0.0089** | **+0.0089** | **0.0000** | **Exact Match** |
| **Confusion Matrix `[TN, FP, FN, TP]`** | `[37, 3, 2, 6]` | `[37, 3, 2, 6]` | `[0, 0, 0, 0]` | **Exact Match** |

### Discrepancy Analysis
- **Discrepancies Encountered:** **None.**
- The reproduction achieved zero variance across all continuous loss values, decision boundaries, classification matrices, and composite score calculations.
- This proves that the Phase D infrastructure, dataloading pipeline, feature alignment, and random seeds are fully synchronized with Phase C.

---

## 6. Resource Profiling & Hardware Feasibility

| Resource Dimension | Measurement | Budget / Threshold | Safety Margin |
| :--- | :---: | :---: | :---: |
| **Peak GPU VRAM** | **105.52 MB** | 2048 MB (Hardware Total) | **94.8% headroom** |
| **Total Wall-Clock Time** | **48.40 seconds** | < 10 minutes | **12.4x faster than limit** |
| **Average Epoch Time** | **4.03 seconds** | — | High iterative throughput |
| **GPU Utilization Profile** | Highly stable | No thermal throttling | Safe for extended exploration |

---

## 7. Artifact Integrity Verification

The following files have been saved exclusively under Phase D directories:
1. `outputs/phase_d/experiments/exp_d1_c3_repro/best_checkpoint.pth` (18.3 MB)
2. `outputs/phase_d/experiments/exp_d1_c3_repro/config.json` (1.2 KB)
3. `outputs/phase_d/experiments/exp_d1_c3_repro/training_history.csv` (1.3 KB)
4. `outputs/phase_d/experiments/exp_d1_c3_repro/training_history.json` (4.9 KB)
5. `outputs/phase_d/experiments/exp_d1_c3_repro/val_predictions.csv` (1.9 KB)
6. `outputs/phase_d/experiments/exp_d1_c3_repro/validation_metrics.json` (2.2 KB)
7. `outputs/phase_d/tables/c3_reproduction_comparison.csv` (1.2 KB)
8. `outputs/phase_d/tables/c3_reproduction_comparison.md` (1.8 KB)
9. `outputs/phase_d/reports/phase_d2_repro_comparison.md` (This document)

All original Phase C artifacts under `outputs/experiments/phase_c/` remain completely untouched and preserved as read-only references.

---

## 8. Gatekeeping Recommendation

Phase D2 Experiment D1 is complete and verified. The baseline reproduction is established with zero discrepancies. As instructed, execution is paused here pending user review. Experiments D2–D5 will not be initiated without explicit approval.
