# Phase D8: Pre-Evaluation Integrity & Gatekeeping Audit Report

**Date & Time:** October 16, 2026 | 10:30 UTC  
**Auditor:** Antigravity AI Research Team  
**Evaluation Target:** Candidate C3 (Frozen Baseline Benchmark) vs. Candidate D5 (Phase D Leading Candidate)  
**Execution Context:** Locked Final Held-Out Evaluation on Quarantined Test Cohort ($N=47$)  
**Status:** **PASSED — ALL 7 PRE-EVALUATION INTEGRITY CHECKS SATISFIED WITH ZERO BLOCKERS**

---

## 1. Executive Summary & Verification Protocol

In strict adherence to the Phase D8 governance rules, this audit was executed **prior to inspecting any test-set predictions, labels, or metrics for Candidate D5**. The primary objective was to verify the scientific isolation, provenance, architectural parity, and split integrity across all upstream assets before unblinding the held-out test cohort.

All seven mandatory pre-evaluation verification gates were verified against authoritative manifests, filesystem hashes, database tables, and model state dictionaries.

---

## 2. Mandatory Verification Gates

### Gate 1: Review of Prior Phase D Reports & Manifest
- **Phase D Manifest (`outputs/phase_d/phase_d_manifest.json`):** Verified. Formally defines Candidate C3 as the benchmark comparator (`outputs/experiments/phase_c/exp_c3_git1_focal_gamma1_5/best_checkpoint.pth`, Epoch 4, Validation Composite Score $S=0.8351$).
- **Phase D0 Audit Report (`outputs/phase_d/phase_d0_audit_report.md`):** Verified. Confirmed cohort counts ($332$ raw folders, $17$ non-diagnostic exclusions, $311$ unique biological root patients, $315$ patient folders across train/val/test).
- **Phase D1 Reproduction Report (`outputs/phase_d/reports/phase_d2_repro_comparison.md`):** Verified. Confirmed bit-exact deterministic reproduction of C3 on validation data ($S=0.8351$, ROC-AUC $0.9281$, PR-AUC $0.7661$, F1 $0.7059$).
- **Phase D2–D5 Comparison Report (`outputs/phase_d/reports/phase_d2_d5_comparison.md`):** Verified. Confirmed selection of Candidate D5 ($\text{lr}=5\times 10^{-5}$) as the top validation candidate ($S=0.8468$, Sensitivity $87.50\%$, Specificity $92.50\%$, Val loss $0.1390$) using strictly validation-only data.
- **Phase D6 Multi-Seed Stability Report (`outputs/phase_d/reports/phase_d6_multi_seed_stability_report.md`):** Verified. Confirmed stability across seeds `42`, `123`, `456` on validation data ($S=0.8532 \pm 0.0200$, Sens $83.33\% \pm 7.22\%$, Spec $91.67\% \pm 1.44\%$).
- **Phase D7 Threshold & Calibration Report (`outputs/phase_d/reports/phase_d7_threshold_calibration_report.md`):** Verified. Demonstrated that canonical threshold $\tau = 0.50$ is the unique optimal operating point under all predefined selection rules (constrained specificity $\ge 90\%$, Youden index, unadjusted prior alignment). Held-out test set remained $100\%$ quarantined throughout.
- **Verdict:** **PASSED.**

---

### Gate 2: Split Manifest & Provenance Verification
- **Split File Path:** `data/splits/patient_stratified_split.json`
- **SHA-256 Hash:** `b1e28b17c34b618dc0937964a677ea8b06884ae5c8ea8bd6d7e48c17e006a8ef`
- **Cohort Breakdown:**
  - **Training:** 220 patient folders representing 217 unique biological root patients.
  - **Validation:** 48 patient folders representing 47 unique biological root patients.
  - **Held-Out Test:** 47 patient folders representing 47 unique biological root patients.
  - **Total:** 315 patient folders, 311 unique biological root patients.
- **Repeat-Visit Disjoint Verification:** Exactly 4 root patients underwent longitudinal or bilateral visits:
  - `01566y21` (2 folders) $\rightarrow$ Training partition exclusively.
  - `04412y22` (2 folders) $\rightarrow$ Training partition exclusively.
  - `03128y21` (2 folders) $\rightarrow$ Validation partition exclusively.
  - `05128y22` (2 folders) $\rightarrow$ Training partition exclusively.
- **Patient Leakage Audit:**
  - $\text{Train} \cap \text{Val} = \emptyset$ ($0$ overlapping folders, $0$ overlapping root patients).
  - $\text{Train} \cap \text{Test} = \emptyset$ ($0$ overlapping folders, $0$ overlapping root patients).
  - $\text{Val} \cap \text{Test} = \emptyset$ ($0$ overlapping folders, $0$ overlapping root patients).
- **Verdict:** **PASSED.** 100% strictly disjoint separation confirmed at both folder and biological patient level.

---

### Gate 3: Validation-Only Selection of Checkpoints
- **Candidate C3 Checkpoint:**
  - File: `outputs/experiments/phase_c/exp_c3_git1_focal_gamma1_5/best_checkpoint.pth`
  - SHA-256: `9a60e5110973660b2d6ab4fd777c47e73124174678e84b8457494da757dafbc8`
  - Selected Epoch: **Epoch 4** (Validation Composite Score $S=0.8351$).
- **Candidate D5 Checkpoint:**
  - File: `outputs/phase_d/experiments/exp_d5_lr5e-5/best_checkpoint.pth`
  - SHA-256: `e79338f82492e654ceab798250092c86e0e229cace83ea5d28a67a98db8c9e7b`
  - Selected Epoch: **Epoch 4** (Validation Composite Score $S=0.8468$).
- **Selection Isolation:** Both checkpoints were selected strictly using validation metrics and the predeclared composite score formula:
  $$S = 0.5 \times \text{ROC-AUC} + 0.3 \times \text{PR-AUC} + 0.2 \times \text{F1}$$
  Zero test predictions or test labels were accessed during training, tuning, or model selection.
- **Verdict:** **PASSED.**

---

### Gate 4: Architectural, Feature, and Inference Parity
- **Network Class:** `GraphInfusedTransformer` (`src/models/git.py`)
- **Mammogram Feature Encoder:** EfficientNet-B0 (1280-dim feature vector pre-extracted from frozen checkpoint `outputs/experiments/exp1_efficientnet_mammogram/best_checkpoint.pth`).
- **Ultrasound Feature Encoder:** ViT-B/16 (768-dim feature vector pre-extracted from frozen checkpoint `outputs/experiments/exp2_vit_ultrasound/best_checkpoint.pth`).
- **Feature Cache:** `data/features/patient_multimodal_features.pt` (SHA-256: `4a7ae4f20085a2f9be61dc9d2f73d2b84f782cd080f18860703d09b2b3f11bb6`).
- **State Dict Parity:**
  - C3 State Dict Keys: 82 parameters / buffers.
  - D5 State Dict Keys: 82 parameters / buffers.
  - Shape Agreement: **100.0% identical shapes across all 82 tensors**.
  - Total Parameters: 4,576,132 in state dict (2,897,922 trainable parameters).
- **Label Semantics:**
  - Class 0 (`LOW_RISK`): BI-RADS Categories 1, 2, 3.
  - Class 1 (`SUSPICIOUS_ACTIONABLE`): BI-RADS Categories 4A, 4B, 4C, 5, 6.
- **Inference Pipeline:** Single forward pass in `model.eval()` mode under `torch.no_grad()` with PyTorch AMP autocast. Identical side-aware graph adjacency and multi-view cross-attention aggregation.
- **Verdict:** **PASSED.** Complete mathematical and code parity verified.

---

### Gate 5: Held-Out Test Cohort Composition
- **Authoritative Split Record:** `data/splits/patient_stratified_split.json` ("test" array).
- **Patient Cohort Count:** Exactly **47 patient folders**.
- **Biological Root Patient Count:** Exactly **47 unique root patients** (zero repeat visits in test cohort).
- **Class Breakdown in Authoritative TSV (`BCMID_annotations.tsv`):**
  - **Class 0 (`LOW_RISK`):** Exactly **40 patients** ($85.11\%$).
    - BI-RADS 1: 9 patients
    - BI-RADS 2: 24 patients
    - BI-RADS 3: 7 patients
  - **Class 1 (`SUSPICIOUS_ACTIONABLE`):** Exactly **7 patients** ($14.89\%$).
    - BI-RADS 4B: 1 patient (`00506y21`)
    - BI-RADS 5: 2 patients (`1019y18`, `7708y181`)
    - BI-RADS 6: 4 patients (`00263y23`, `00278y20`, `03911y20`, `03964y22`)
- **Exclusion Verification:** The 17 non-diagnostic folders (11 `NONE`, 5 `INDETERMINATE`, 1 `Category 0`) remain strictly excluded.
- **Cache Alignment:** All 47 test patient folders are present in `data/features/patient_multimodal_features.pt` with 0 missing samples and 100% label concordance.
- **Verdict:** **PASSED.**

---

### Gate 6: Data Integrity & Cross-Split Anomaly Checks
- **Duplicate Patients:** 0 duplicates in test cohort.
- **Repeated Examinations:** 0 repeated examinations in test cohort.
- **Missing Predictions / Failures:** 0 failed tensors, 0 empty views.
- **Label Mismatches:** 0 mismatches between TSV and cached features.
- **Patient Overlap:** 0 patient overlap across train, validation, and test.
- **Verdict:** **PASSED.**

---

### Gate 7: Software Environment, Hashes & Exact Evaluation Command
- **Operating System:** Windows 11 Enterprise (Version 10.0.26200)
- **Python Version:** 3.13.5 (Anaconda 64-bit)
- **PyTorch Version:** 2.7.1+cu118
- **CUDA Device:** NVIDIA GeForce MX550 (Compute Capability 7.5, 2048 MB VRAM)
- **Key Libraries:** `scikit-learn==1.6.1`, `pandas==2.2.3`, `numpy==2.1.3`, `scipy==1.15.3`, `statsmodels==0.14.4`
- **Random Seed:** 42
- **Frozen Primary Threshold:** $\tau = 0.50$ (applied to both C3 and D5)
- **Exact Evaluation Command:**
  ```powershell
  python scripts/evaluate_phase_d8_locked.py
  ```
- **Verdict:** **PASSED.**

---

## 3. Formal Gatekeeping Declaration

All seven mandatory pre-evaluation integrity checks have been formally verified. Zero blockers, zero data leaks, zero label anomalies, and zero configuration discrepancies exist. 

**Approval is granted to proceed with the locked final held-out test evaluation comparing Frozen Candidate C3 and Candidate D5.**
