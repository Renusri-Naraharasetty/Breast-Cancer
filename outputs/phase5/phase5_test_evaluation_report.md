# Phase 5: Final Test Evaluation and Comparative Analysis Report

**Project:** Breast Cancer Multimodal Imaging Dataset (BCMID)  
**Execution Phase:** Phase 5 — Final Test Evaluation & Comparative Analysis  
**Evaluated Cohort:** Frozen 47-patient test cohort (strictly held out during Phases 1–4)  
**Triage Task:** Radiologic Triage — `LOW_RISK` (BI-RADS 1–3) vs. `SUSPICIOUS_ACTIONABLE` (BI-RADS 4–6)  
**Decision Threshold:** $\tau = 0.50$ (canonical documented pre-existing threshold; zero test-set threshold tuning)

---

## 1. Executive Summary & Comparative Results Table

All four independently trained models were evaluated on the held-out 47-patient test partition. Evaluations were conducted at both the **Image Level** and the **Patient Level** (using max-pooling aggregation $P_{\text{patient}} = \max_{i} P_{\text{image}, i}$).

| Order | Experiment ID | Architecture | Modality | Level | N (Pos / Neg) | Accuracy | Precision | Recall (Sens) | Specificity | F1-Score | G-Mean | ROC-AUC | PR-AUC |
| :-: | :--- | :--- | :--- | :---: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| **1** | `exp1_efficientnet_mammogram` | EfficientNet-B0 | Mammogram | Image | 132 (14 / 118) | **89.39%** | 50.00% | 7.14% | 99.15% | 0.1250 | 0.2661 | **0.6749** | 0.2341 |
| | | | | **Patient** | 47 (7 / 40) | **85.11%** | 50.00% | 14.29% | 97.50% | 0.2222 | 0.3732 | **0.6571** | 0.3079 |
| **2** | `exp2_vit_ultrasound` | ViT-B/16 | Ultrasound | Image | 150 (56 / 94) | **72.67%** | 94.12% | 28.57% | 98.94% | 0.4384 | 0.5317 | **0.8218** | **0.7845** |
| | | | | **Patient** | 47 (7 / 40) | **91.49%** | **80.00%** | **57.14%** | **97.50%** | **0.6667** | **0.7464** | **0.8357** | **0.7099** |
| **3** | `exp3_vit_mammogram` | ViT-B/16 | Mammogram | Image | 132 (14 / 118) | **90.15%** | 100.00% | 7.14% | 100.00% | 0.1333 | 0.2673 | **0.7700** | 0.3014 |
| | | | | **Patient** | 47 (7 / 40) | **87.23%** | 100.00% | 14.29% | 100.00% | 0.2500 | 0.3780 | **0.7893** | 0.4130 |
| **4** | `exp4_efficientnet_ultrasound` | EfficientNet-B0 | Ultrasound | Image | 150 (56 / 94) | **61.33%** | 48.81% | 73.21% | 54.26% | 0.5857 | 0.6303 | **0.7330** | 0.6203 |
| | | | | **Patient** | 47 (7 / 40) | **48.94%** | 22.58% | **100.00%** | 40.00% | 0.3684 | 0.6325 | **0.8357** | 0.4999 |

---

## 2. Directory of Output Artifacts & Figures

All generated files are stored in `outputs/phase5/`:
- **Tables:**
  - `outputs/phase5/tables/test_evaluation_results.csv`
  - `outputs/phase5/tables/test_evaluation_results.json`
- **Predictions:**
  - `outputs/phase5/predictions/exp1_efficientnet_mammogram_image_predictions.csv`
  - `outputs/phase5/predictions/exp1_efficientnet_mammogram_patient_predictions.csv`
  - `outputs/phase5/predictions/exp2_vit_ultrasound_image_predictions.csv`
  - `outputs/phase5/predictions/exp2_vit_ultrasound_patient_predictions.csv`
  - `outputs/phase5/predictions/exp3_vit_mammogram_image_predictions.csv`
  - `outputs/phase5/predictions/exp3_vit_mammogram_patient_predictions.csv`
  - `outputs/phase5/predictions/exp4_efficientnet_ultrasound_image_predictions.csv`
  - `outputs/phase5/predictions/exp4_efficientnet_ultrasound_patient_predictions.csv`
- **Figures:**
  - `outputs/phase5/comparisons/model_metrics_comparison_barchart.png`
  - `outputs/phase5/curves/roc_curves_image_level.png`
  - `outputs/phase5/curves/roc_curves_patient_level.png`
  - `outputs/phase5/curves/pr_curves_image_level.png`
  - `outputs/phase5/curves/pr_curves_patient_level.png`
  - `outputs/phase5/confusion_matrices/exp1_efficientnet_mammogram_confusion_matrices.png`
  - `outputs/phase5/confusion_matrices/exp2_vit_ultrasound_confusion_matrices.png`
  - `outputs/phase5/confusion_matrices/exp3_vit_mammogram_confusion_matrices.png`
  - `outputs/phase5/confusion_matrices/exp4_efficientnet_ultrasound_confusion_matrices.png`
  - `outputs/phase5/training_curves/` (individual learning curves and validation loss/accuracy overlays)

---

## 3. Analysis: Strengths, Weaknesses, Class Imbalance, and Metric Rigor

1. **ViT-B/16 on Ultrasound (`exp2`) achieves the best clinical triage balance:**
   - Patient-level: **ROC-AUC = 0.8357**, **PR-AUC = 0.7099**, **Accuracy = 91.49%**, **F1 = 0.6667**, **Specificity = 97.50%**, **Sensitivity = 57.14%**.
   - Near-perfect specificity (39 of 40 low-risk patients correctly identified as negative).
2. **EfficientNet-B0 on Ultrasound (`exp4`) provides maximal sensitivity:**
   - Patient-level: **Sensitivity = 100.00%** (all 7 suspicious patients detected with zero false negatives).
   - High patient ROC-AUC (**0.8357**), but lower specificity (**40.00%**) at threshold 0.50 due to false positive rate.
3. **Mammogram models (`exp1` and `exp3`) and Class Imbalance:**
   - Both models achieve high accuracy (~85–90%) and high specificity (97.5–100%), but low sensitivity at threshold 0.50.
   - On the test set, only 14 out of 132 images (10.6%) and 7 out of 47 patients (14.9%) are positive.
   - A naive dummy classifier predicting `LOW_RISK` for every sample achieves **89.39% image accuracy** and **85.11% patient accuracy**, but has **0% sensitivity and 0.0 F1 score**, missing every single suspicious case!
   - This demonstrates why **accuracy alone is completely insufficient** for medical imaging and triage.
4. **ViT vs. EfficientNet Architecture Comparison:**
   - ViT-B/16 consistently outperformed EfficientNet-B0 in discriminative ranking (ROC-AUC 0.7700 vs 0.6749 on mammogram images; ROC-AUC 0.8218 vs 0.7330 on ultrasound images).
   - Self-attention effectively captures both global context and fine localized lesions without inductive convolution bias degradation.
5. **Clinical Scope Caveat:**
   - This task represents automated radiologic triage (`LOW_RISK` vs. `SUSPICIOUS_ACTIONABLE`), not confirmed malignant-versus-benign pathology diagnosis.
