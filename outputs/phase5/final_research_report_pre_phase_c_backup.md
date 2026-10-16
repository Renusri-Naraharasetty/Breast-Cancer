# Comparative Evaluation of Pretrained EfficientNet-B0 and ViT-B/16 for Multimodal Breast Cancer Radiologic Triage

**Dataset:** Breast Cancer Multimodal Imaging Dataset (BCMID)  
**Authors:** Antigravity AI & Research Team  
**Evaluation Cohort:** Frozen 47-Patient Independent Test Cohort  
**Task Definition:** Binary Radiologic Triage (`LOW_RISK`: BI-RADS 1–3 vs. `SUSPICIOUS_ACTIONABLE`: BI-RADS 4–6)  
**Status:** Phase 5.1 Audit Complete (Zero Discrepancies; Patient-Clustered Bootstrap 95% CIs Verified)

---

## Abstract

We evaluate the transfer learning capabilities of two standard computer vision architectures—a lightweight convolutional network (**EfficientNet-B0**, 4.01M parameters) and a self-attention transformer (**ViT-B/16**, 85.80M parameters)—for binary radiologic triage of breast lesions using the Breast Cancer Multimodal Imaging Dataset (BCMID). Four independent models were trained without multimodal fusion across full-field digital mammography and ultrasound examinations. Model selection was governed strictly by validation loss on a frozen 48-patient validation set, and final evaluation was performed on a completely untouched 47-patient test set ($N=132$ mammograms, $N=150$ ultrasound images). Patient-level predictions were derived using max-probability aggregation across views. At the canonical default threshold ($\tau = 0.50$), ViT-B/16 on ultrasound achieved a patient-level ROC-AUC of 0.8357 (95% CI: [0.6089, 1.0000]) and an F1-score of 0.6667 (95% CI: [0.2222, 0.9231]), while EfficientNet-B0 on ultrasound achieved 100.00% patient sensitivity (95% CI: [1.0000, 1.0000]) but with reduced specificity (40.00%, 95% CI: [0.2500, 0.5526]). Mammography models demonstrated high specificity ($\ge 97.5\%$) but suppressed sensitivity at $\tau = 0.50$ due to class imbalance (10.6% positive prevalence). Rigorous patient-clustered bootstrap analysis reveals wide confidence intervals stemming from the limited number of positive test patients ($N_{\text{pos}} = 7$), demonstrating that claimed architectural superiority cannot be asserted with statistical certainty without larger multi-center cohorts.

---

## 1. Introduction and Task Formulation

Automated computer-aided triage of breast imaging requires differentiating between low-risk benign findings that can be safely followed up versus suspicious lesions that require urgent clinical intervention (biopsy, surgical excision, or expedited diagnostic workup). 

### 1.1 Clinical Triage vs. Histopathological Diagnosis
In accordance with ethical AI guidelines and the BCMID annotation protocol:
- **Triage Target:** The binary classification target is defined strictly as **Radiologic Triage**:
  - **Class 0 (`LOW_RISK`):** BI-RADS 1 (Negative), BI-RADS 2 (Benign), and BI-RADS 3 (Probably Benign).
  - **Class 1 (`SUSPICIOUS_ACTIONABLE`):** BI-RADS 4 (Suspicious), BI-RADS 5 (Highly Suggestive of Malignancy), and BI-RADS 6 (Known Biopsy-Proven Carcinoma).
- **Explicit Boundary:** This model is **not** a predictor of biopsy-confirmed malignancy for every individual patient. Rather, it models radiologic suspicion according to international BI-RADS reporting standards.

### 1.2 Independent Modality Formulation
To establish unconfounded baselines, all four experiments were trained and evaluated completely independently:
1. **Experiment 1:** EfficientNet-B0 on Mammograms (`exp1_efficientnet_mammogram`)
2. **Experiment 2:** ViT-B/16 on Ultrasound (`exp2_vit_ultrasound`)
3. **Experiment 3:** ViT-B/16 on Mammograms (`exp3_vit_mammogram`)
4. **Experiment 4:** EfficientNet-B0 on Ultrasound (`exp4_efficientnet_ultrasound`)

Multimodal feature fusion, cross-attention, and graph neural networks were intentionally excluded.

---

## 2. Dataset Partitioning & Experimental Methodology

### 2.1 Patient Cohort & Partitioning
The dataset was partitioned at the patient level using stratified splitting to prevent patient data leakage across splits. Patient identifiers and multiple image views from the same examination are strictly isolated:

| Partition | Patient Count | Mammogram Images | Mammogram Prevalence [Pos / Neg] | Ultrasound Images | Ultrasound Prevalence [Pos / Neg] |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Training** | 220 | 644 | 74 / 570 (11.5% Pos) | 671 | 130 / 541 (19.4% Pos) |
| **Validation** | 48 | 138 | 16 / 122 (11.6% Pos) | 135 | 27 / 108 (20.0% Pos) |
| **Held-Out Test** | 47 | 132 | 14 / 118 (10.6% Pos) | 150 | 56 / 94 (37.3% Pos) |
| **Total** | 314 | 914 | 104 / 810 (11.4% Pos) | 956 | 213 / 743 (22.3% Pos) |

*Test Cohort Patient Distribution:* 47 patients total: **40 Low-Risk (85.1%)**, **7 Suspicious/Actionable (14.9%)**.

### 2.2 Model Architecture & Training Protocol
- **Preprocessing:** All images resized to $224 \times 224$ RGB, normalized using ImageNet channel mean `[0.485, 0.456, 0.406]` and standard deviation `[0.229, 0.224, 0.225]`. Grayscale and RGBA images harmonized to 3-channel RGB.
- **Initialization:** ImageNet-1K pretrained weights (`torchvision.models`).
- **Loss Function:** Modality-specific weighted Cross-Entropy Loss based on empirical training image ratios:
  - Mammograms: $w = [1.0000, 7.7027]$
  - Ultrasound: $w = [1.0000, 4.1615]$
- **Optimization:** AdamW ($\text{weight decay} = 0.01$), CosineAnnealingLR scheduler, mixed precision (AMP), gradient accumulation to maintain effective batch size = 16.
- **Model Selection:** Validation-loss early stopping (patience = 4). No test data was accessed during training or model selection.

---

## 3. Results & Statistical Audit

### 3.1 Point Estimates on the Held-Out Test Cohort ($\tau = 0.50$)

| Experiment ID | Architecture | Modality | Level | N (Pos/Neg) | Accuracy | Sensitivity (Recall) | Specificity | Precision | F1-Score | G-Mean | ROC-AUC | PR-AUC |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `exp1_efficientnet_mammogram` | EfficientNet-B0 | Mammogram | Image | 132 (14/118) | 89.39% | 7.14% | 99.15% | 50.00% | 0.1250 | 0.2661 | 0.6749 | 0.2341 |
| | | | **Patient** | 47 (7/40) | 85.11% | 14.29% | 97.50% | 50.00% | 0.2222 | 0.3732 | 0.6571 | 0.3079 |
| `exp2_vit_ultrasound` | ViT-B/16 | Ultrasound | Image | 150 (56/94) | 72.67% | 28.57% | 98.94% | 94.12% | 0.4384 | 0.5317 | 0.8218 | 0.7845 |
| | | | **Patient** | 47 (7/40) | 91.49% | 57.14% | 97.50% | 80.00% | 0.6667 | 0.7464 | 0.8357 | 0.7099 |
| `exp3_vit_mammogram` | ViT-B/16 | Mammogram | Image | 132 (14/118) | 90.15% | 7.14% | 100.00% | 100.00% | 0.1333 | 0.2673 | 0.7700 | 0.3014 |
| | | | **Patient** | 47 (7/40) | 87.23% | 14.29% | 100.00% | 100.00% | 0.2500 | 0.3780 | 0.7893 | 0.4130 |
| `exp4_efficientnet_ultrasound` | EfficientNet-B0 | Ultrasound | Image | 150 (56/94) | 61.33% | 73.21% | 54.26% | 48.81% | 0.5857 | 0.6303 | 0.7330 | 0.6203 |
| | | | **Patient** | 47 (7/40) | 48.94% | 100.00% | 40.00% | 22.58% | 0.3684 | 0.6325 | 0.8357 | 0.4999 |

### 3.2 Patient-Clustered Bootstrap Uncertainty Estimates (95% CI)
To account for clustering (multiple views per patient) and the small positive cohort ($N_{\text{pos}} = 7$), we executed cluster bootstrapping ($B = 2,000$ replicates, resampling by patient):

| Experiment ID | Level | Metric | Point Estimate | Bootstrap Mean | Std Error | 95% Bootstrap CI |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: |
| `exp1_efficientnet_mammogram` | Patient | Accuracy | 0.8511 | 0.8531 | ±0.0505 | [0.7447, 0.9362] |
| | Patient | Sensitivity | 0.1429 | 0.1440 | ±0.1422 | [0.0000, 0.5000] |
| | Patient | Specificity | 0.9750 | 0.9748 | ±0.0247 | [0.9189, 1.0000] |
| | Patient | F1-Score | 0.2222 | 0.2036 | ±0.1821 | [0.0000, 0.5714] |
| | Patient | ROC-AUC | 0.6571 | 0.6553 | ±0.1321 | [0.3659, 0.8905] |
| `exp2_vit_ultrasound` | Patient | Accuracy | 0.9149 | 0.9151 | ±0.0413 | [0.8298, 0.9787] |
| | Patient | Sensitivity | 0.5714 | 0.5817 | ±0.2024 | [0.1429, 1.0000] |
| | Patient | Specificity | 0.9750 | 0.9740 | ±0.0254 | [0.9189, 1.0000] |
| | Patient | F1-Score | 0.6667 | 0.6518 | ±0.1798 | [0.2222, 0.9231] |
| | Patient | ROC-AUC | 0.8357 | 0.8380 | ±0.1029 | [0.6089, 1.0000] |
| `exp3_vit_mammogram` | Patient | Accuracy | 0.8723 | 0.8735 | ±0.0481 | [0.7660, 0.9574] |
| | Patient | Sensitivity | 0.1429 | 0.1425 | ±0.1439 | [0.0000, 0.5000] |
| | Patient | Specificity | 1.0000 | 1.0000 | ±0.0000 | [1.0000, 1.0000] |
| | Patient | F1-Score | 0.2500 | 0.2319 | ±0.2058 | [0.0000, 0.6667] |
| | Patient | ROC-AUC | 0.7893 | 0.7891 | ±0.0724 | [0.6395, 0.9181] |
| `exp4_efficientnet_ultrasound` | Patient | Accuracy | 0.4894 | 0.4902 | ±0.0731 | [0.3404, 0.6383] |
| | Patient | Sensitivity | 1.0000 | 1.0000 | ±0.0000 | [1.0000, 1.0000] |
| | Patient | Specificity | 0.4000 | 0.4011 | ±0.0778 | [0.2500, 0.5526] |
| | Patient | F1-Score | 0.3684 | 0.3644 | ±0.0998 | [0.1667, 0.5556] |
| | Patient | ROC-AUC | 0.8357 | 0.8361 | ±0.0829 | [0.6464, 0.9750] |

---

## 4. Discussion & Limitations

### 4.1 Why Accuracy is Clinically Misleading
In screening mammography cohorts, negative findings predominate. In our test set, 85.1% of patients are low risk. A trivial majority-class classifier predicting `LOW_RISK` for all patients achieves **85.11% accuracy**, yet fails to detect a single suspicious lesion (Sensitivity = 0%, F1 = 0.0). Consequently, accuracy provides an illusion of high performance. Clinical utility must be evaluated via **Sensitivity/Specificity balance, F1-Score, and ROC/PR AUC**.

### 4.2 Impact of Decision Thresholding
At the default operating point ($\tau = 0.50$), both mammography models exhibited low sensitivity (14.29%) but near-perfect specificity (97.5–100.0%). However, their ROC-AUCs (0.7893 for ViT-B/16 and 0.6571 for EfficientNet-B0) demonstrate continuous discriminative capability. In clinical triage, false negatives carry substantially greater morbidity than false positives; adjusting the operating threshold to $\tau \approx 0.15 - 0.25$ using the validation set would increase recall to screening standards.

### 4.3 Statistical Uncertainty and Small Positive Sample Size
With only 7 positive patients in the test cohort, a single misclassified patient shifts the empirical sensitivity by $\pm 14.3\%$. The 95% bootstrap confidence intervals for patient-level sensitivity in ViT-B/16 on ultrasound span from 14.3% to 100.0%. Furthermore, the 95% CIs for ROC-AUC across models show substantial overlap (e.g., ViT-B/16 on mammograms [0.6395, 0.9181] vs. EfficientNet-B0 on mammograms [0.3659, 0.8905]). **Therefore, we make no claims that Vision Transformers are statistically superior or that specific radiologic sub-features (such as microcalcifications) are resolved better by one model.**

### 4.4 Clinical & Dataset Limitations
1. **Single-Center Retrospective Bias:** BCMID was sourced from a single clinical center in Alexandria, Egypt. External generalizability across diverse imaging vendors, detector technologies, and demographic populations remains unestablished.
2. **Pathology vs. Radiologic Triage:** Ground truth labels are derived from BI-RADS clinical reports, representing radiologic suspicion rather than definitive tissue biopsy pathology for all subjects.

---

## 5. Conclusion

This study provides a rigorous, leak-free benchmark comparing EfficientNet-B0 and ViT-B/16 on the multimodal BCMID dataset. While point estimates favor Vision Transformers in continuous ranking (ROC-AUC), statistical uncertainty from limited sample sizes dictates cautious interpretation. Prospective validation on multi-institutional cohorts with calibrated operating thresholds is required before clinical triage integration.

---

## 6. Project Artifact Manifest

- **Audit Discrepancy Log:** [`outputs/phase5/audit/audit_discrepancy_log.md`](file:///c:/Users/admin/Music/Breast%20Cancer1/outputs/phase5/audit/audit_discrepancy_log.md)
- **Bootstrap Uncertainty Tables:** [`outputs/phase5/audit/patient_bootstrap_ci.csv`](file:///c:/Users/admin/Music/Breast%20Cancer1/outputs/phase5/audit/patient_bootstrap_ci.csv) and [`.json`](file:///c:/Users/admin/Music/Breast%20Cancer1/outputs/phase5/audit/patient_bootstrap_ci.json)
- **Bootstrap Forest Plot:** [`outputs/phase5/audit/bootstrap_ci_forest_plot.png`](file:///c:/Users/admin/Music/Breast%20Cancer1/outputs/phase5/audit/bootstrap_ci_forest_plot.png)
- **Model Checkpoints:** Preserved untouched in [`outputs/experiments/`](file:///c:/Users/admin/Music/Breast%20Cancer1/outputs/experiments/)
- **Test Predictions:** Preserved untouched in [`outputs/phase5/predictions/`](file:///c:/Users/admin/Music/Breast%20Cancer1/outputs/phase5/predictions/)
- **Summary Metrics:** [`outputs/phase5/tables/test_evaluation_results.csv`](file:///c:/Users/admin/Music/Breast%20Cancer1/outputs/phase5/tables/test_evaluation_results.csv)
