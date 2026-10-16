# Phase D9: Comprehensive Reproducibility & Scientific Integrity Audit Checklist

**Date:** October 16, 2026  
**Auditor:** Antigravity AI Research Team  
**Evaluation Scope:** Complete lifecycle audit across Phase 1–5, Phase C, and Phase D (D0–D9).  
**Repository:** `c:\Users\admin\Music\Breast Cancer1`

---

## 1. Patient-Level Split Isolation & Data Hygiene

- [x] **Root-Patient Partitioning Enforced:** Partitioning was executed strictly on `root_patient_id` rather than folder names or image files.
- [x] **Zero Split Overlap:**
  - $\text{Train} \cap \text{Val} = \emptyset$ (0 overlapping root patients, 0 overlapping folders).
  - $\text{Train} \cap \text{Test} = \emptyset$ (0 overlapping root patients, 0 overlapping folders).
  - $\text{Val} \cap \text{Test} = \emptyset$ (0 overlapping root patients, 0 overlapping folders).
- [x] **Repeat Examination Containment:** Exactly 4 root patients had repeat visits (`01566y21`, `04412y22`, `03128y21`, `05128y22`). All associated folders remain strictly confined within their assigned partition (3 patients / 6 folders in Train; 1 patient / 2 folders in Validation; 0 in Held-Out Test).
- [x] **Exclusion Criteria Preserved:** Exactly 17 non-diagnostic folders (11 `NONE`, 5 `INDETERMINATE`, 1 `Category 0`) were completely isolated and never accessed for training, validation, or test evaluation.
- [x] **Cohort Count Verification:**
  - Raw folders: 332
  - Excluded folders: 17
  - Eligible folders: 315
  - Unique biological root patients: 311
  - Training folders: 220 (217 root patients)
  - Validation folders: 48 (47 root patients)
  - Held-out test folders: 47 (47 root patients)
  - The historical "314" citation was an arithmetic typo in an earlier table row ($220 + 48 + 47 = 315$).

---

## 2. Checkpoint & Configuration Provenance

- [x] **Candidate C3 Checkpoint:**
  - Path: `outputs/experiments/phase_c/exp_c3_git1_focal_gamma1_5/best_checkpoint.pth`
  - File Size: 18,337,197 bytes
  - SHA-256: `9a60e5110973660b2d6ab4fd777c47e73124174678e84b8457494da757dafbc8`
  - Selected Epoch: **Epoch 4** (Validation Composite Score $S=0.8351$)
  - Architecture: `GraphInfusedTransformer` (1280-dim MG, 768-dim US, hidden=256, 2 GAT layers, 2 Transformer layers, dropout=0.30)
  - Loss: ClassBalancedFocalLoss ($\gamma=1.5, \alpha=[1.0, 5.1111]$)
  - Learning Rate: $10^{-4}$ (AdamW, weight decay $0.01$)
- [x] **Candidate D5 Checkpoint:**
  - Path: `outputs/phase_d/experiments/exp_d5_lr5e-5/best_checkpoint.pth`
  - File Size: 18,337,581 bytes
  - SHA-256: `e79338f82492e654ceab798250092c86e0e229cace83ea5d28a67a98db8c9e7b`
  - Selected Epoch: **Epoch 4** (Validation Composite Score $S=0.8468$)
  - Architecture: `GraphInfusedTransformer` (Exact structural match with C3)
  - Loss: ClassBalancedFocalLoss ($\gamma=1.5, \alpha=[1.0, 5.1111]$)
  - Learning Rate: $5\times 10^{-5}$ (AdamW, weight decay $0.01$)
- [x] **Feature Cache Provenance:**
  - Path: `data/features/patient_multimodal_features.pt`
  - SHA-256: `4a7ae4f20085a2f9be61dc9d2f73d2b84f782cd080f18860703d09b2b3f11bb6`
  - Encoders: Frozen EfficientNet-B0 (1280-dim from Exp 1 best checkpoint) and Frozen ViT-B/16 (768-dim from Exp 2 best checkpoint).
  - Feature extraction executed under `torch.no_grad()` with zero backpropagation into vision encoders.

---

## 3. Test-Set Usage History & Quarantining

- [x] **Phase 1–4:** Zero test-set usage. Unimodal models trained and selected strictly on training and validation splits.
- [x] **Phase 5:** Held-out test set accessed once to establish unimodal and initial multimodal GIT baselines.
- [x] **Phase C:** Development of focal loss candidates C1–C5 conducted strictly on training ($N=220$) and validation ($N=48$). Model selection rule ($S$) applied strictly to validation data. Selected C3 benchmark evaluated on test cohort.
- [x] **Phase D0–D7:** Complete quarantine enforced.
  - Phase D0: Pre-audit with zero test loading.
  - Phase D1: C3 reproduction verified on validation data only.
  - Phase D2–D5: Controlled hyperparameter search evaluated strictly on validation data.
  - Phase D6: Multi-seed stability (seeds 42, 123, 456) evaluated strictly on validation data.
  - Phase D7: Threshold trade-off ($\tau \in [0.30, 0.70]$) and calibration curves evaluated strictly on validation data.
- [x] **Phase D8:** Quarantined test cohort unblinded under locked script [`scripts/evaluate_phase_d8_locked.py`](file:///c:/Users/admin/Music/Breast%20Cancer1/scripts/evaluate_phase_d8_locked.py) for exactly one predefined evaluation at $\tau = 0.50$.
- [x] **Phase D9:** Documentation and consolidation only. Zero execution of test inference or retraining.

---

## 4. Software Environment & Execution Determinism

- [x] **Operating System:** Windows 11 Enterprise (Version 10.0.26200)
- [x] **Python:** 3.13.5 (Anaconda 64-bit)
- [x] **PyTorch:** 2.7.1+cu118
- [x] **CUDA Hardware:** NVIDIA GeForce MX550 (2048 MB VRAM, Driver Version 537.13)
- [x] **Core Analytics Libraries:**
  - `scikit-learn==1.6.1`
  - `pandas==2.2.3`
  - `numpy==2.1.3`
  - `scipy==1.15.3`
  - `statsmodels==0.14.4`
  - `matplotlib==3.10.0`
  - `seaborn==0.13.2`
- [x] **Random Seed Controls:**
  - Training seeds: Seed 42 for primary models; seeds 123 and 456 for Phase D6 multi-seed testing.
  - Bootstrap seed: Fixed at `seed=42` across all confidence interval resamples ($B=2000$).
  - CUDA AMP deterministic autocasting verified.

---

## 5. Metric Calculations & Statistical Tests

- [x] **Point Metrics Calculated Directly from Patient-Level Predictions:**
  - Accuracy: $(TP + TN) / N$
  - Sensitivity: $TP / (TP + FN)$
  - Specificity: $TN / (TN + FP)$
  - Precision: $TP / (TP + FP)$
  - Negative Predictive Value: $TN / (TN + FN)$
  - F1-Score: $2 \times (\text{Precision} \times \text{Recall}) / (\text{Precision} + \text{Recall})$
  - Balanced Accuracy: $0.5 \times (\text{Sensitivity} + \text{Specificity})$
  - ROC-AUC: Trapezoidal integration via `sklearn.metrics.roc_auc_score`
  - PR-AUC: Weighted mean of precisions via `sklearn.metrics.average_precision_score`
  - Brier Score: Mean squared error via `sklearn.metrics.brier_score_loss`
  - Log Loss: Binary cross-entropy via `sklearn.metrics.log_loss`
- [x] **Analytical and Empirical Confidence Intervals:**
  - Wilson Score intervals computed via `statsmodels.stats.proportion.proportion_confint(method='wilson')`.
  - Clopper-Pearson exact binomial intervals computed via `scipy.stats.beta`.
  - Stratified bootstrap intervals computed over $B=2000$ resamples.
- [x] **Paired Comparison Tests:**
  - McNemar exact two-sided binomial test computed via `scipy.stats.binomtest(k=min(b,c), n=b+c, p=0.5)`.
  - Wilcoxon paired signed-rank test computed via `scipy.stats.wilcoxon(p_c3, p_d5)`.

---

## 6. Historical Reconciliation Audit

- [x] **GIT 1 Baseline:**
  - Physical CSV [`outputs/phase5/predictions/exp_git_multimodal_patient_predictions.csv`](file:///c:/Users/admin/Music/Breast%20Cancer1/outputs/phase5/predictions/exp_git_multimodal_patient_predictions.csv) yields ROC-AUC $0.8857$ and PR-AUC $0.5729$ with CM `[34, 6, 2, 5]`.
  - Legacy citations of $0.8679$ and $0.6800$ in `scripts/evaluate_phase_c.py` traced to manual transcription errors in a hardcoded dictionary.
- [x] **GIT 2 Baseline:**
  - Physical CSV [`outputs/phase5/predictions/exp_git_vit_mammogram_effnet_ultrasound_patient_predictions.csv`](file:///c:/Users/admin/Music/Breast%20Cancer1/outputs/phase5/predictions/exp_git_vit_mammogram_effnet_ultrasound_patient_predictions.csv) yields ROC-AUC $0.8321$ and PR-AUC $0.4048$ with CM `[23, 17, 0, 7]`.
  - Legacy citations of $0.8429$ and $0.5739$ traced to manual transcription errors.
- [x] **Candidate C3 Benchmark:**
  - Physical CSV [`outputs/experiments/phase_c/predictions/exp_c3_test_predictions.csv`](file:///c:/Users/admin/Music/Breast%20Cancer1/outputs/experiments/phase_c/predictions/exp_c3_test_predictions.csv) matches newly computed [`outputs/phase_d/phase_d8_final_evaluation/test_predictions_c3.csv`](file:///c:/Users/admin/Music/Breast%20Cancer1/outputs/phase_d/phase_d8_final_evaluation/test_predictions_c3.csv) bit-for-bit ($100.00\%$ concordance across all 47 predictions).
- [x] **Earlier Reports Preserved:** All historical reports and tables in `outputs/phase5/` and `outputs/experiments/` remain unmodified.

---

## 7. Known Scientific Limitations

- [x] **Small Positive Test Sample Size ($N_{\text{pos}}=7$):**
  - Moving one positive patient shifts sensitivity by $\pm 14.29\%$.
  - Sensitivity 95% Clopper-Pearson exact CI spans $[42.13\%, 99.64\%]$.
  - Statistical power to detect minor calibration differences is inherently limited.
- [x] **Single-Center Retrospective Bias:** BCMID originates from a single hospital in Alexandria, Egypt. External multi-center prospective validation is required before drawing clinical conclusions.
- [x] **Research Scope Boundary:** The system is an automated radiologic triage research tool (`LOW_RISK` vs `SUSPICIOUS_ACTIONABLE`), not a tissue pathology diagnostic device.
- [x] **No Post-Hoc Threshold Optimization:** The canonical threshold $\tau = 0.50$ was maintained; no attempt was made to search for flattering post-hoc operating points on the test cohort.
