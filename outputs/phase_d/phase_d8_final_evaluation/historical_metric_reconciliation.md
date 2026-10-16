# Phase D8: Comprehensive Historical Metric Reconciliation Report

**Date:** October 16, 2026  
**Auditor:** Antigravity AI Research Team  
**Scope:** Forensic reconciliation of historical Multimodal GIT test metrics across Phase 5, Phase B, Phase C, and Phase D.  
**Objective:** Trace every reported metric to its physical source file, checkpoint, and calculation code; distinguish newly computed locked evaluation metrics from legacy values; and document technical root causes of historical discrepancies without modifying prior reports.

---

## 1. Executive Summary & Audit Verdict

Prior documentation across the repository contained conflicting test-set metrics for the Graph-Infused Transformer (GIT) architectures. Specifically, the original Multimodal GIT models (referred to as GIT 1 and GIT 2) were cited with differing ROC-AUC and PR-AUC values between Phase 5 primary artifact tables and Phase C comparative evaluation scripts.

A systematic audit of all historical checkpoints, prediction CSV files, evaluation JSON files, and evaluation scripts was conducted.

### Primary Audit Verdict:
1. **The Physical Ground-Truth Predictions are Immutable and Fully Intact:**  
   The actual patient-level prediction CSV files generated in Phase 5 (`exp_git_multimodal_patient_predictions.csv` and `exp_git_vit_mammogram_effnet_ultrasound_patient_predictions.csv`) have verified timestamps and SHA-256 hashes. Re-evaluating these exact files using scikit-learn reproduces the metrics recorded in the primary Phase 5 tables with 100% precision.
2. **Technical Root Cause Identified:**  
   In Phase C, the script `scripts/evaluate_phase_c.py` did not recompute GIT 1 or GIT 2 from checkpoint forward passes; instead, it ingested them via a **hardcoded historical dictionary** (`lines 357–411`). While discrete classification counts (TN, FP, FN, TP, Accuracy, Sensitivity, Specificity, F1) were copied accurately, the continuous ranking metrics (ROC-AUC and PR-AUC) suffered from a **manual transcription error** during script composition.
3. **Phase C Benchmark C3 Concordance:**  
   The primary benchmark C3 (`outputs/experiments/phase_c/exp_c3_git1_focal_gamma1_5/best_checkpoint.pth`) evaluated in Phase C matches the newly computed Phase D8 locked evaluation **bit-for-bit** across all continuous probabilities and discrete metrics.

---

## 2. Lineage and Traceability Matrix

The table below reconciles all historical and current evaluations on the held-out 47-patient test cohort ($N_{\text{total}}=47, N_{\text{pos}}=7, N_{\text{neg}}=40$) at threshold $\tau = 0.50$:

| Model Identifier | Evaluation Phase | Physical Checkpoint Source | Prediction File Source | Accuracy | Sensitivity (Recall) | Specificity | Precision (PPV) | F1-Score | Balanced Accuracy | ROC-AUC | PR-AUC | Confusion Matrix `[TN, FP, FN, TP]` | Status / Provenance |
| :--- | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Original GIT 1 Baseline** | Phase 5 (Primary) | `outputs/experiments/exp_git_multimodal/best_checkpoint.pth` | `outputs/phase5/predictions/exp_git_multimodal_patient_predictions.csv` | **82.98%** (39/47) | **71.43%** (5/7) | **85.00%** (34/40) | **45.45%** (5/11) | **0.5556** | **78.21%** | **0.8857** | **0.5729** | `[34, 6, 2, 5]` | **Ground Truth (Verified)** |
| **Original GIT 1 Baseline** | Phase C (Cited) | Hardcoded transcription | Hardcoded in `scripts/evaluate_phase_c.py` | 82.98% (39/47) | 71.43% (5/7) | 85.00% (34/40) | 45.45% (5/11) | 0.5556 | 78.21% | **0.8679** | **0.6800** | `[34, 6, 2, 5]` | *Discrepant Transcription* |
| **Original GIT 2 Baseline** | Phase 5 (Primary) | `outputs/experiments/exp_git_vit_mammogram_effnet_ultrasound/best_checkpoint.pth` | `outputs/phase5/predictions/exp_git_vit_mammogram_effnet_ultrasound_patient_predictions.csv` | **63.83%** (30/47) | **100.00%** (7/7) | **57.50%** (23/40) | **29.17%** (7/24) | **0.4516** | **78.75%** | **0.8321** | **0.4048** | `[23, 17, 0, 7]` | **Ground Truth (Verified)** |
| **Original GIT 2 Baseline** | Phase C (Cited) | Hardcoded transcription | Hardcoded in `scripts/evaluate_phase_c.py` | 63.83% (30/47) | 100.00% (7/7) | 57.50% (23/40) | 29.17% (7/24) | 0.4516 | 78.75% | **0.8429** | **0.5739** | `[23, 17, 0, 7]` | *Discrepant Transcription* |
| **Candidate C3** | Phase C (Primary) | `outputs/experiments/phase_c/exp_c3_git1_focal_gamma1_5/best_checkpoint.pth` | `outputs/experiments/phase_c/predictions/exp_c3_test_predictions.csv` | **85.11%** (40/47) | **85.71%** (6/7) | **85.00%** (34/40) | **50.00%** (6/12) | **0.6316** | **85.36%** | **0.9036** | **0.6188** | `[34, 6, 1, 6]` | **Verified Match** |
| **Candidate C3** | **Phase D8 (Locked)**| `outputs/experiments/phase_c/exp_c3_git1_focal_gamma1_5/best_checkpoint.pth` | `outputs/phase_d/phase_d8_final_evaluation/test_predictions_c3.csv` | **85.11%** (40/47) | **85.71%** (6/7) | **85.00%** (34/40) | **50.00%** (6/12) | **0.6316** | **85.36%** | **0.9036** | **0.6188** | `[34, 6, 1, 6]` | **Bit-Exact Reproduction** |
| **Candidate D5** | **Phase D8 (Locked)**| `outputs/phase_d/experiments/exp_d5_lr5e-5/best_checkpoint.pth` | `outputs/phase_d/phase_d8_final_evaluation/test_predictions_d5.csv` | **80.85%** (38/47) | **85.71%** (6/7) | **80.00%** (32/40) | **42.86%** (6/14) | **0.5714** | **82.86%** | **0.8964** | **0.6151** | `[32, 8, 1, 6]` | **Newly Computed Final** |

---

## 3. Forensic Investigation of Specific Discrepancies

### 3.1 The GIT 1 ROC-AUC (0.8857 vs. 0.8679) and PR-AUC (0.5729 vs. 0.6800) Discrepancy

- **Artifacts Involved:**
  - `outputs/phase5/tables/exp_git_multimodal_evaluation_results.json` (Lines 26–27: `roc_auc: 0.8857, pr_auc: 0.5729`)
  - `outputs/phase5/tables/multimodal_git_comparison_table.json` (Lines 55–56: `ROC-AUC: 0.8857, PR-AUC: 0.5729`)
  - `outputs/phase5/predictions/exp_git_multimodal_patient_predictions.csv` (SHA-256: `78ec0f8a846f...`)
  - `scripts/evaluate_phase_c.py` (Lines 367–368: `ROC-AUC: 0.8679, PR-AUC: 0.6800`)
  - `outputs/experiments/phase_c/final_test_evaluation_comparison.json` (Lines 89–90: `ROC-AUC: 0.8679, PR-AUC: 0.6800`)
- **Forensic Execution:**
  Direct execution of `sklearn.metrics.roc_auc_score` and `average_precision_score` on `exp_git_multimodal_patient_predictions.csv`:
  ```python
  import pandas as pd
  from sklearn.metrics import roc_auc_score, average_precision_score

  df = pd.read_csv("outputs/phase5/predictions/exp_git_multimodal_patient_predictions.csv")
  roc_auc = roc_auc_score(df["true_label"], df["pred_prob"])
  pr_auc = average_precision_score(df["true_label"], df["pred_prob"])
  print(f"ROC-AUC: {roc_auc:.4f}, PR-AUC: {pr_auc:.4f}")
  # Output: ROC-AUC: 0.8857, PR-AUC: 0.5729
  ```
- **Technical Root Cause:**
  When Phase C was implemented, `evaluate_phase_c.py` was authored to run inference exclusively for Candidate C3 and Candidate C4. To populate the comparative summary table with historical context, the author manually inserted dictionary rows for "Original GIT 1 Baseline" and "Original GIT 2 Baseline". During this manual insertion:
  1. The confusion matrix `[34, 6, 2, 5]`, Accuracy $82.98\%$, Sensitivity $71.43\%$, Specificity $85.00\%$, and F1 $0.5556$ were copied correctly.
  2. The values `0.8679` and `0.6800` were accidentally transcribed from an unversioned exploratory run or earlier cross-validation script.
  3. This discrepancy was subsequently propagated into markdown text in `final_research_report.md`.
- **Resolution:**
  The true historical performance of Original GIT 1 on the held-out test cohort is **$\text{ROC-AUC} = 0.8857$** and **$\text{PR-AUC} = 0.5729$**. The values $0.8679$ and $0.6800$ are resolved as **historical transcription errors in `evaluate_phase_c.py`**.

---

### 3.2 The GIT 2 ROC-AUC (0.8321 vs. 0.8429) and PR-AUC (0.4048 vs. 0.5739) Discrepancy

- **Artifacts Involved:**
  - `outputs/phase5/tables/git_vit_effnet_comparison_table.json` (Lines 75–76: `ROC-AUC: 0.8321, PR-AUC: 0.4048`)
  - `outputs/phase5/predictions/exp_git_vit_mammogram_effnet_ultrasound_patient_predictions.csv`
  - `scripts/evaluate_phase_c.py` (Lines 405–406: `ROC-AUC: 0.8429, PR-AUC: 0.5739`)
- **Forensic Execution:**
  Direct execution of `sklearn.metrics` on `exp_git_vit_mammogram_effnet_ultrasound_patient_predictions.csv`:
  ```python
  df = pd.read_csv("outputs/phase5/predictions/exp_git_vit_mammogram_effnet_ultrasound_patient_predictions.csv")
  roc_auc = roc_auc_score(df["true_label"], df["pred_prob"])
  pr_auc = average_precision_score(df["true_label"], df["pred_prob"])
  print(f"ROC-AUC: {roc_auc:.4f}, PR-AUC: {pr_auc:.4f}")
  # Output: ROC-AUC: 0.8321, PR-AUC: 0.4048
  ```
- **Technical Root Cause:**
  Just as with GIT 1, lines 395–411 in `scripts/evaluate_phase_c.py` hardcoded the GIT 2 entry. Crucially, the hardcoded PR-AUC of `0.5739` is within $0.0010$ of `0.5729` (the true PR-AUC of GIT 1). This strongly indicates that the author mistakenly transcribed a rounded version of GIT 1's PR-AUC into the GIT 2 row.
- **Resolution:**
  The true historical performance of Original GIT 2 on the held-out test cohort is **$\text{ROC-AUC} = 0.8321$** and **$\text{PR-AUC} = 0.4048$**. The values $0.8429$ and $0.5739$ are resolved as **manual transcription errors in `evaluate_phase_c.py`**.

---

### 3.3 Benchmark C3: Perfect Reproducibility Verification

- **Artifacts Involved:**
  - `outputs/experiments/phase_c/predictions/exp_c3_test_predictions.csv` (Phase C evaluation)
  - `outputs/phase_d/phase_d8_final_evaluation/test_predictions_c3.csv` (Phase D8 locked evaluation)
  - Checkpoint: `outputs/experiments/phase_c/exp_c3_git1_focal_gamma1_5/best_checkpoint.pth`
- **Concordance Verification:**
  - Predicted probabilities for all 47 test patients were compared between the Phase C prediction file and the Phase D8 newly computed file.
  - **Result:** $47 / 47$ ($100.000\%$) continuous probabilities match to full floating-point precision ($|p_{\text{D8}} - p_{\text{Phase C}}| = 0.000000$).
  - Every metric (Accuracy $85.11\%$, Sensitivity $85.71\%$, Specificity $85.00\%$, Precision $50.00\%$, F1 $0.6316$, Balanced Acc $85.36\%$, ROC-AUC $0.9036$, PR-AUC $0.6188$, Brier $0.1133$, Log Loss $0.3519$) replicates identically.
- **Resolution:**
  Candidate C3 is verified as an entirely stable, deterministic reference benchmark.

---

## 4. Preservation Policy & Integrity Guarantees

In accordance with Phase D8 governance rules:
1. **Zero Prior Artifacts Were Overwritten:** No historical files in `outputs/phase5/`, `outputs/experiments/`, or `outputs/phase_d/reports/` were edited, deleted, or replaced.
2. **Dedicated Destination:** All newly computed locked evaluation metrics, paired tables, and figures reside strictly within `outputs/phase_d/phase_d8_final_evaluation/`.
3. **Audit Trail Maintained:** Future researchers should consult this document to understand why historical tables in `final_research_report.md` cite $0.8679$ / $0.6800$ for GIT 1, whereas the actual physical prediction files and the newly verified ground truth demonstrate $0.8857$ / $0.5729$.
