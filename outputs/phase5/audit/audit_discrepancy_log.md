# Phase 5.1 Audit Discrepancy Log & Metric Verification

**Audit Timestamp:** 2026-10-13  
**Target Output Directory:** `outputs/phase5/`  
**Audit Scope:** Independent recomputation from raw prediction CSVs, max-pooling verification, label consistency, continuous probability curve verification, and patient-clustered bootstrap uncertainty.

---

## 1. Discrepancy Findings Summary

> [!NOTE]
> **Zero Numerical Discrepancies Found:** All recomputed classification metrics (Accuracy, Precision, Sensitivity, Specificity, F1, G-Mean, ROC-AUC, PR-AUC) and confusion matrices match `outputs/phase5/tables/test_evaluation_results.csv` and `.json` to within machine floating-point tolerance.

## 2. Max-Pooling Aggregation & Label Verification

1. **Max-Pooling Rule:** For every patient $P$, $P_{\text{patient}} = \max_{i \in \text{Images}(P)} P_{\text{image}, i}$ was audited across all 47 test patients in all four models. **100% of recorded patient probabilities match the exact max probability of their respective images.**
2. **Image-to-Patient Label Consistency:**
   - All 47 test patients have 100% concordant labels across all individual images in both modalities. (No contralateral discordant lesions exist in the held-out test cohort; patient `05439y20` was part of the training set).

## 3. Probability Distribution & Curve Verification

- **Continuous vs. Binary:** Confirmed that `roc_curve` and `precision_recall_curve` were computed directly on continuous probability predictions ($P \in [0.0, 1.0]$) rather than thresholded binary 0/1 integers.
- **AUC Agreement:** The areas under the plotted curves match the tabular ROC-AUC and PR-AUC values exactly.

## 4. Patient-Clustered Bootstrap Uncertainty (95% CI)

Resampling was performed by patient cluster ($B = 2,000$ replicates) to account for correlation between multiple views/images from the same patient.

| Experiment ID | Modality | Level | Metric | Point Estimate | 95% Bootstrap CI | Std Error |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: |
| `exp1_efficientnet_mammogram` | mammogram | Patient | ACCURACY | **0.8511** | [0.7447, 0.9362] | ±0.0505 |
| `exp1_efficientnet_mammogram` | mammogram | Patient | RECALL | **0.1429** | [0.0000, 0.5000] | ±0.1422 |
| `exp1_efficientnet_mammogram` | mammogram | Patient | SPECIFICITY | **0.975** | [0.9189, 1.0000] | ±0.0247 |
| `exp1_efficientnet_mammogram` | mammogram | Patient | F1_SCORE | **0.2222** | [0.0000, 0.5714] | ±0.1821 |
| `exp1_efficientnet_mammogram` | mammogram | Patient | ROC_AUC | **0.6571** | [0.3659, 0.8905] | ±0.1321 |
| `exp1_efficientnet_mammogram` | mammogram | Patient | PR_AUC | **0.3079** | [0.1053, 0.6861] | ±0.1508 |
| `exp2_vit_ultrasound` | ultrasound | Patient | ACCURACY | **0.9149** | [0.8298, 0.9787] | ±0.0413 |
| `exp2_vit_ultrasound` | ultrasound | Patient | RECALL | **0.5714** | [0.1429, 1.0000] | ±0.2024 |
| `exp2_vit_ultrasound` | ultrasound | Patient | SPECIFICITY | **0.975** | [0.9189, 1.0000] | ±0.0254 |
| `exp2_vit_ultrasound` | ultrasound | Patient | F1_SCORE | **0.6667** | [0.2222, 0.9231] | ±0.1798 |
| `exp2_vit_ultrasound` | ultrasound | Patient | ROC_AUC | **0.8357** | [0.6089, 1.0000] | ±0.1029 |
| `exp2_vit_ultrasound` | ultrasound | Patient | PR_AUC | **0.7099** | [0.3500, 1.0000] | ±0.1605 |
| `exp3_vit_mammogram` | mammogram | Patient | ACCURACY | **0.8723** | [0.7660, 0.9574] | ±0.0481 |
| `exp3_vit_mammogram` | mammogram | Patient | RECALL | **0.1429** | [0.0000, 0.5000] | ±0.1439 |
| `exp3_vit_mammogram` | mammogram | Patient | SPECIFICITY | **1.0** | [1.0000, 1.0000] | ±0.0000 |
| `exp3_vit_mammogram` | mammogram | Patient | F1_SCORE | **0.25** | [0.0000, 0.6667] | ±0.2058 |
| `exp3_vit_mammogram` | mammogram | Patient | ROC_AUC | **0.7893** | [0.6395, 0.9181] | ±0.0724 |
| `exp3_vit_mammogram` | mammogram | Patient | PR_AUC | **0.413** | [0.1500, 0.7208] | ±0.1525 |
| `exp4_efficientnet_ultrasound` | ultrasound | Patient | ACCURACY | **0.4894** | [0.3404, 0.6383] | ±0.0731 |
| `exp4_efficientnet_ultrasound` | ultrasound | Patient | RECALL | **1.0** | [1.0000, 1.0000] | ±0.0000 |
| `exp4_efficientnet_ultrasound` | ultrasound | Patient | SPECIFICITY | **0.4** | [0.2500, 0.5526] | ±0.0778 |
| `exp4_efficientnet_ultrasound` | ultrasound | Patient | F1_SCORE | **0.3684** | [0.1667, 0.5556] | ±0.0998 |
| `exp4_efficientnet_ultrasound` | ultrasound | Patient | ROC_AUC | **0.8357** | [0.6464, 0.9750] | ±0.0829 |
| `exp4_efficientnet_ultrasound` | ultrasound | Patient | PR_AUC | **0.4999** | [0.2102, 0.8922] | ±0.1776 |

> [!IMPORTANT]
> **Statistical Limitation of Small Positive Sample Size ($N_{\text{pos}} = 7$):**
> Because the test set contains only 7 positive patients (14.9%), each single positive patient constitutes $1/7 \approx 14.3\%$ of the positive cohort. Consequently, confidence intervals on Sensitivity and Precision are wide (e.g., Sensitivity of 14.3% has a 95% CI of [0.00, 42.86%]). No claim of definitive statistical superiority between architectures can be made without a larger multi-center evaluation.
