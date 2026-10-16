# Comparative Evaluation of Single-Modality Transfer Learning and Multimodal Graph-Infused Transformers for Breast Cancer Radiologic Triage

**Dataset:** Breast Cancer Multimodal Imaging Dataset (BCMID)  
**Authors:** Antigravity AI & Research Team  
**Evaluation Cohort:** Frozen 47-Patient Independent Held-Out Test Cohort ($N_{\text{pos}} = 7, N_{\text{neg}} = 40$)  
**Task Definition:** Binary Radiologic Triage (`LOW_RISK`: BI-RADS 1–3 vs. `SUSPICIOUS_ACTIONABLE`: BI-RADS 4–6)  
**Status:** Phase C Audit Complete (Frozen Test Evaluation, Patient-Clustered Bootstrap 95% CIs Verified)  

---

## Abstract

We present a comprehensive evaluation of transfer learning and multimodal fusion architectures for binary radiologic triage of breast lesions using the Breast Cancer Multimodal Imaging Dataset (BCMID). The study comprises two sequential phases: (1) an independent single-modality baseline benchmark comparing lightweight convolutional networks (**EfficientNet-B0**, 4.01M parameters) and vision transformers (**ViT-B/16**, 85.80M parameters) across full-field digital mammography and breast ultrasound; and (2) a controlled multimodal fusion study (**Phase C**) investigating **Graph-Infused Transformers (GIT)** combining mammography and ultrasound feature representations with side-aware graph attention and cross-attention mechanisms. Controlled exploration of loss functions and regularization across five experiments (C1–C5) identified **Candidate C3** (focal loss with $\gamma = 1.5$ and class weighting $\alpha = [1.0000, 5.1111]$) as the primary candidate based on a predeclared validation composite score ($0.8351$), with **Candidate C4** (weighted cross-entropy with lower learning rate and higher regularization) retained as a secondary candidate.

On the frozen, untouched 47-patient test cohort ($N_{\text{pos}} = 7, N_{\text{neg}} = 40$), Candidate C3 achieved a **ROC-AUC of 0.9036** (95% CI: `[0.7884, 0.9849]`), **Sensitivity of 85.71%** ($6/7$ suspicious lesions detected; 95% CI: `[0.5000, 1.0000]`), **Specificity of 85.00%** ($34/40$ true negatives; 95% CI: `[0.7250, 0.9487]`), and **F1-score of 0.6316** (95% CI: `[0.3077, 0.8462]`) at the canonical classification threshold of $\tau = 0.50$. Compared with the original GIT 1 baseline at the same threshold, C3 detected one additional suspicious patient ($6/7$ vs. $5/7$, $+14.28\%$ sensitivity) while maintaining the exact same specificity ($85.00\%$, $6$ false positives), though its PR-AUC ($0.6188$) was lower than the original baseline ($0.6800$). Rigorous patient-clustered bootstrap analysis ($B = 2,000$ replicates) demonstrates substantial statistical uncertainty due to the limited number of positive test cases ($N_{\text{pos}} = 7$), with overlapping confidence intervals across architectures. These findings support the research potential of multimodal graph-based fusion for radiologic triage but do not establish clinical diagnostic efficacy. Prospective validation on large, multi-institutional cohorts is required.

---

## 1. Introduction and Task Formulation

Automated computer-aided triage of breast imaging requires differentiating between low-risk benign findings that can be safely followed up versus suspicious lesions that require urgent clinical intervention (biopsy, surgical excision, or expedited diagnostic workup). 

### 1.1 Clinical Triage vs. Histopathological Diagnosis
In accordance with ethical AI guidelines and the BCMID annotation protocol:
- **Triage Target:** The binary classification target is defined strictly as **Radiologic Triage**:
  - **Class 0 (`LOW_RISK`):** BI-RADS 1 (Negative), BI-RADS 2 (Benign), and BI-RADS 3 (Probably Benign).
  - **Class 1 (`SUSPICIOUS_ACTIONABLE`):** BI-RADS 4 (Suspicious), BI-RADS 5 (Highly Suggestive of Malignancy), and BI-RADS 6 (Known Biopsy-Proven Carcinoma).
- **Explicit Boundary:** This model is **not** a predictor of biopsy-confirmed malignancy for every individual patient. Rather, it models radiologic suspicion according to international BI-RADS reporting standards.

### 1.2 Independent Single-Modality Benchmark (Experiments 1–4)
To establish unconfounded baselines, four independent experiments were trained and evaluated without multimodal fusion:
1. **Experiment 1:** EfficientNet-B0 on Mammograms (`exp1_efficientnet_mammogram`)
2. **Experiment 2:** ViT-B/16 on Ultrasound (`exp2_vit_ultrasound`)
3. **Experiment 3:** ViT-B/16 on Mammograms (`exp3_vit_mammogram`)
4. **Experiment 4:** EfficientNet-B0 on Ultrasound (`exp4_efficientnet_ultrasound`)

### 1.3 Controlled Multimodal Graph Fusion (Phase C)
Clinical breast assessment relies on complementary imaging: mammography captures tissue architecture, microcalcifications, and asymmetric densities, while ultrasound resolves lesion echogenicity, margins, and posterior acoustic features. The Multimodal Graph-Infused Transformer (GIT) constructs a patient-level graph incorporating side-aware spatial relationships (left, right, bilateral) across multiple views, followed by cross-attention between modalities. Phase C evaluates whether focal loss formulation and controlled regularization improve calibration and triage accuracy over standard weighted cross-entropy.

---

## 2. Dataset Partitioning & Preprocessing Methodology

### 2.1 Patient Cohort & Stratified Partitioning
The dataset was partitioned at the patient level using stratified splitting to strictly prevent patient data leakage across splits. Patient identifiers and multiple image views from the same examination are completely isolated:

| Partition | Patient Count | Mammogram Images | Mammogram Prevalence [Pos / Neg] | Ultrasound Images | Ultrasound Prevalence [Pos / Neg] |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Training** | 220 | 644 | 74 / 570 (11.5% Pos) | 671 | 130 / 541 (19.4% Pos) |
| **Validation** | 48 | 138 | 16 / 122 (11.6% Pos) | 135 | 27 / 108 (20.0% Pos) |
| **Held-Out Test** | 47 | 132 | 14 / 118 (10.6% Pos) | 150 | 56 / 94 (37.3% Pos) |
| **Total** | 314 | 914 | 104 / 810 (11.4% Pos) | 956 | 213 / 743 (22.3% Pos) |

*Patient Distribution by Split:*
- **Training Cohort:** $220$ patients total: **$184$ Low-Risk (83.6%)**, **$36$ Suspicious/Actionable (16.4%)** ($\text{ratio} = 5.1111$).
- **Validation Cohort:** $48$ patients total: **$40$ Low-Risk (83.3%)**, **$8$ Suspicious/Actionable (16.7%)** ($\text{ratio} = 5.0000$).
- **Held-Out Test Cohort:** $47$ patients total: **$40$ Low-Risk (85.1%)**, **$7$ Suspicious/Actionable (14.9%)** ($\text{ratio} = 5.7143$).

### 2.2 Model Architecture & Preprocessing
- **Preprocessing:** Images resized to $224 \times 224$ RGB, normalized using ImageNet channel mean `[0.485, 0.456, 0.406]` and standard deviation `[0.229, 0.224, 0.225]`. Grayscale and RGBA images harmonized to 3-channel RGB.
- **Hardware Optimization:** Mixed precision (`torch.amp`), gradient accumulation to maintain effective batch size $16$, memory footprint $< 100$ MB VRAM via pre-extracted frozen backbone features.
- **Leak-Free Protocol:** Held-out test data was untouched throughout feature extraction, training, validation sweeps, and threshold selection.

---

## 3. Phase 1–4 Benchmark: Single-Modality Unimodal Results

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

### 3.2 Single-Modality Patient-Clustered Bootstrap Uncertainty Estimates (95% CI)
Clustered bootstrap resampling ($B = 2,000$ replicates by patient):

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

## 4. Phase C: Controlled Multimodal GIT Architecture & Training Methodology

### 4.1 Architecture Specification
The Multimodal Graph-Infused Transformer (GIT) integrates pre-extracted image features from both modalities:
- **Mammogram Stream:** EfficientNet-B0 backbone ($1,280$-dimensional feature vectors) for GIT 1 / Phase C models.
- **Ultrasound Stream:** ViT-B/16 backbone ($768$-dimensional feature vectors) for GIT 1 / Phase C models.
- **Graph Attention Network (GAT):** 2 GAT layers with 4 attention heads, hidden dimension $256$, with anatomical side encoding (Left, Right, Bilateral).
- **Cross-Attention Transformer:** 2 transformer encoder layers with 8 heads, feedforward dimension $1,024$, operating on fused view tokens.
- **Classification Head:** Linear projection producing binary logits (`LOW_RISK` vs. `SUSPICIOUS_ACTIONABLE`).

### 4.2 Loss Formulations & Class Weighting
1. **Weighted Cross-Entropy (Weighted BCE):**
   $$\text{WCE}(p_t) = -\alpha_t \log(p_t), \quad \alpha = [1.0000, 5.1111]$$
   Accounting for the $184$ negative vs. $36$ positive training distribution ($184/36 = 5.1111$).
2. **Class-Balanced Focal Loss:**
   $$\text{FL}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t), \quad \mathcal{L} = \frac{\sum_{i=1}^B \alpha_{y_i} (1 - p_{i, y_i})^\gamma (-\log p_{i, y_i})}{\sum_{i=1}^B \alpha_{y_i}}$$
   - When $\gamma = 0$, $\text{FL}(p_t) \equiv \text{WCE}(p_t)$ (numerically verified with $\Delta = 0.0000000$).
   - When $\gamma > 0$, easy negative samples ($p_t \approx 1$) are modulated, focusing optimization on borderline, ambiguous lesions.

### 4.3 Predeclared Checkpoint Selection Criterion
To eliminate post-hoc checkpoint cherry-picking, all Phase C candidate checkpoints were selected strictly on the 48-patient validation set using a predeclared composite score:
$$\text{Composite Score} = 0.5 \times \text{ROC-AUC} + 0.3 \times \text{PR-AUC} + 0.2 \times \text{F1}$$
Tie-breaking rule: lower validation loss.

### 4.4 Controlled Experimental Series (C1–C5) on Validation Data
All five models were trained using identical patient splits, seed ($42$), dataloaders, and evaluation code:

| Experiment | Configuration Details | Selected Epoch | Val Loss | Val ROC-AUC | Val PR-AUC | Val F1 | Val Sens (Recall) | Val Spec | Val Bal Acc | Composite Score | Confusion Matrix `[TN, FP, FN, TP]` | Overfitting Loss Gap @ Best |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Orig GIT 1** | Weighted BCE, $\text{lr}=10^{-4}, \text{wd}=0.01, \text{drop}=0.3$ | 3 (Min Loss) | 0.4069 | 0.9250 | 0.7436 | 0.5333 | 50.0% (4/8) | 92.5% (37/40) | 71.25% | 0.7922 | `[37, 3, 4, 4]` | $+0.0144$ |
| **C1 (Repro)** | Weighted BCE, $\text{lr}=10^{-4}, \text{wd}=0.01, \text{drop}=0.3$ | 4 (Comp) | 0.4272 | 0.9187 | 0.7432 | 0.6316 | 75.0% (6/8) | 87.5% (35/40) | 81.25% | 0.8086 | `[35, 5, 2, 6]` | $+0.0731$ |
| **C2** | Focal ($\gamma=2.0$), $\text{lr}=10^{-4}, \text{wd}=0.01, \text{drop}=0.3$ | 10 (Comp) | 0.2537 | 0.9219 | 0.7516 | 0.7059 | 75.0% (6/8) | 92.5% (37/40) | 83.75% | 0.8276 | `[37, 3, 2, 6]` | $+0.2462$ |
| **C3 (Primary)** | **Focal ($\gamma=1.5$), $\text{lr}=10^{-4}, \text{wd}=0.01, \text{drop}=0.3$** | **4 (Comp)** | **0.1679** | **0.9281** | **0.7661** | **0.7059** | **75.0% (6/8)** | **92.5% (37/40)** | **83.75%** | **0.8351** | **`[37, 3, 2, 6]`** | **$+0.0089$** |
| **C4 (Secondary)** | Weighted BCE, $\text{lr}=5\times 10^{-5}, \text{wd}=0.02, \text{drop}=0.4$ | 7 (Comp) | 0.4664 | 0.9219 | 0.7667 | **0.7143** | 62.5% (5/8) | **97.5% (39/40)** | 80.00% | 0.8338 | `[39, 1, 3, 5]` | $+0.1697$ |
| **C5** | Focal ($\gamma=2.0$), $\text{lr}=5\times 10^{-5}, \text{wd}=0.02, \text{drop}=0.4$ | 8 (Comp) | 0.1227 | 0.9219 | **0.7708** | 0.6000 | 75.0% (6/8) | 85.0% (34/40) | 80.00% | 0.8122 | `[34, 6, 2, 6]` | $+0.0640$ |

### 4.5 Candidate Selection and Operating Threshold Lock
- **Candidate C3 was frozen as the Primary Model** based on achieving the highest validation composite score ($0.8351$), highest ROC-AUC ($0.9281$), and near-perfect loss calibration at selection ($\text{Train Loss}=0.1590, \text{Val Loss}=0.1679, \Delta=0.0089$).
- **Candidate C4 was retained as the Secondary Model** ($0.8338$) for comparative assessment of conservative regularization.
- **Operating Thresholds Locked Strictly on Validation Data:**
  - **Candidate C3:** Default $\tau = 0.50$; Validation-Selected Operating Threshold $\tau^* = 0.44$ (unanimous optimum across Max F1, Max Balanced Accuracy, and Sensitivity $\ge 85\%$).
  - **Candidate C4:** Default $\tau = 0.50$; Validation-Selected Operating Threshold $\tau^* = 0.21$.

---

## 5. Phase C: Final Held-Out Test Cohort Evaluation

### 5.1 Test Performance Comparison Table ($N=47$ Patients, $7$ Positive / $40$ Negative)

| Model Designation | Evaluation Protocol | Threshold ($\tau$) | Accuracy | Sensitivity (Recall) | Specificity | Precision | F1-Score | Balanced Accuracy | G-Mean | ROC-AUC | PR-AUC | Confusion Matrix `[TN, FP, FN, TP]` |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **C3 (Focal $\gamma=1.5$, Primary)** | **Canonical Baseline** | **0.50** | **85.11%** | **85.71% (6/7)** | **85.00% (34/40)** | **50.00% (6/12)** | **0.6316** | **85.36%** | **0.8536** | **0.9036** | **0.6188** | **`[34, 6, 1, 6]`** |
| **C3 (Focal $\gamma=1.5$, Primary)** | Validation-Selected | 0.44 | 76.60% | 85.71% (6/7) | 75.00% (30/40) | 37.50% (6/16) | 0.5217 | 80.36% | 0.8018 | 0.9036 | 0.6188 | `[30, 10, 1, 6]` |
| **C4 (W-BCE Reg, Secondary)** | Canonical Baseline | 0.50 | 85.11% | 71.43% (5/7) | 87.50% (35/40) | 50.00% (5/10) | 0.5882 | 79.46% | 0.7906 | 0.8821 | 0.5050 | `[35, 5, 2, 5]` |
| **C4 (W-BCE Reg, Secondary)** | Validation-Selected | 0.21 | 74.47% | 85.71% (6/7) | 72.50% (29/40) | 35.29% (6/17) | 0.5000 | 79.11% | 0.7883 | 0.8821 | 0.5050 | `[29, 11, 1, 6]` |
| **Original GIT 1 Baseline** | Canonical Baseline | 0.50 | 82.98% | 71.43% (5/7) | 85.00% (34/40) | 45.45% (5/11) | 0.5556 | 78.21% | 0.7792 | 0.8679 | 0.6800 | `[34, 6, 2, 5]` |
| **Original GIT 1 Baseline** | Validation-Selected (Phase B) | 0.42 | 78.72% | 85.71% (6/7) | 77.50% (31/40) | 40.00% (6/15) | 0.5455 | 81.61% | 0.8150 | 0.8679 | 0.6800 | `[31, 9, 1, 6]` |
| **Original GIT 2 Baseline** | Canonical Baseline | 0.50 | 63.83% | 100.00% (7/7) | 57.50% (23/40) | 29.17% (7/24) | 0.4516 | 78.75% | 0.7583 | 0.8429 | 0.5739 | `[23, 17, 0, 7]` |
| **Original GIT 2 Baseline** | Validation-Selected (Phase B) | 0.76 | 70.21% | 85.71% (6/7) | 67.50% (27/40) | 31.58% (6/19) | 0.4615 | 76.61% | 0.7606 | 0.8429 | 0.5739 | `[27, 13, 1, 6]` |

### 5.2 Patient-Clustered Bootstrap Uncertainty Estimates (95% CI)
Derived from $B = 2,000$ cluster-bootstrap resamples on the test cohort:

| Model & Evaluation Protocol | Metric | Point Estimate | Bootstrap Mean ± Std | 95% Bootstrap CI |
| :--- | :--- | :---: | :---: | :---: |
| **C3 (Primary, $\tau = 0.50$)** | Sensitivity | 0.8571 | 0.8618 ± 0.1403 | `[0.5000, 1.0000]` |
| | Specificity | 0.8500 | 0.8485 ± 0.0560 | `[0.7250, 0.9487]` |
| | Precision | 0.5000 | 0.4937 ± 0.1476 | `[0.2100, 0.7857]` |
| | F1-Score | 0.6316 | 0.6145 ± 0.1365 | `[0.3077, 0.8462]` |
| | Balanced Accuracy | 0.8536 | 0.8551 ± 0.0756 | `[0.6859, 0.9634]` |
| | ROC-AUC | 0.9036 | 0.9038 ± 0.0510 | `[0.7884, 0.9849]` |
| | PR-AUC | 0.6188 | 0.6413 ± 0.1688 | `[0.2583, 0.9334]` |
| **C3 (Primary, $\tau = 0.44$)** | Sensitivity | 0.8571 | 0.8618 ± 0.1403 | `[0.5000, 1.0000]` |
| | Specificity | 0.7500 | 0.7472 ± 0.0696 | `[0.6000, 0.8780]` |
| | Precision | 0.3750 | 0.3697 ± 0.1237 | `[0.1429, 0.6429]` |
| | F1-Score | 0.5217 | 0.5058 ± 0.1305 | `[0.2353, 0.7500]` |
| | Balanced Accuracy | 0.8036 | 0.8045 ± 0.0776 | `[0.6278, 0.9231]` |
| **C4 (Secondary, $\tau = 0.50$)** | Sensitivity | 0.7143 | 0.7173 ± 0.1846 | `[0.3333, 1.0000]` |
| | Specificity | 0.8750 | 0.8739 ± 0.0517 | `[0.7674, 0.9714]` |
| | Precision | 0.5000 | 0.4935 ± 0.1640 | `[0.1816, 0.8000]` |
| | F1-Score | 0.5882 | 0.5694 ± 0.1510 | `[0.2500, 0.8333]` |
| | ROC-AUC | 0.8821 | 0.8826 ± 0.0546 | `[0.7619, 0.9737]` |
| | PR-AUC | 0.5050 | 0.5602 ± 0.1720 | `[0.2286, 0.8969]` |

### 5.3 Comparative Analysis: C3 vs. Original GIT 1 Baseline
1. **Detection Gain at $\tau = 0.50$:**
   Candidate C3 successfully detected **$6$ of $7$ suspicious patients** on the held-out test cohort ($85.71\%$ sensitivity), compared with $5$ of $7$ ($71.43\%$) for the Original GIT 1 baseline at the same threshold. Crucially, this $+14.28\%$ sensitivity improvement was achieved **without any increase in false alarms** ($6$ false positives, $85.00\%$ specificity for both models).
2. **Elimination of Threshold Shifting Penalty:**
   In Original GIT 1, achieving $85.71\%$ sensitivity required shifting the operating threshold down to $\tau = 0.42$, which degraded specificity from $85.00\%$ to $77.50\%$ ($9$ false positives). Candidate C3 achieved $85.71\%$ sensitivity naturally at the default threshold $\tau = 0.50$ with $85.00\%$ specificity.
3. **Nuanced Examination of PR-AUC:**
   While Candidate C3 demonstrated higher ROC-AUC ($0.9036$ vs. $0.8679$), higher F1 ($0.6316$ vs. $0.5556$), and higher balanced accuracy ($85.36\%$ vs. $78.21\%$), its test PR-AUC was **$0.6188$**, which is lower than the reported Original GIT 1 baseline PR-AUC of **$0.6800$**. This occurred because the single missed cancer case in C3 ($00171y21$) was assigned a lower continuous probability ($0.218$), creating an earlier drop in precision along the precision-recall curve, whereas Original GIT 1 placed its two missed cases closer to the decision boundary. This trade-off underscores why reporting multiple complementary metrics is essential.

---

## 6. Discussion & Critical Research Scope

### 6.1 Clinical Utility vs. Accuracy
In screening mammography cohorts, negative findings predominate. In our test set, 85.1% of patients are low risk. A trivial majority-class classifier predicting `LOW_RISK` for all patients achieves **85.11% accuracy**, yet fails to detect a single suspicious lesion (Sensitivity = 0%, F1 = 0.0). Consequently, accuracy provides an illusion of high performance. Clinical utility must be evaluated via **Sensitivity/Specificity balance, F1-Score, and ROC/PR AUC**.

### 6.2 Impact of Decision Thresholding
At the default operating point ($\tau = 0.50$), both unimodal mammography models exhibited low sensitivity (14.29%) but high specificity (97.5–100.0%). In contrast, multimodal fusion in C3 achieved balanced operating performance directly at $\tau = 0.50$ ($85.71\%$ sensitivity, $85.00\%$ specificity). Furthermore, evaluating the validation-selected threshold ($\tau^* = 0.44$) revealed that lowering the threshold did not capture the remaining missed lesion ($p=0.218$) but increased false positives from $6$ to $10$, demonstrating the stability of the default $\tau = 0.50$ operating point for C3.

### 6.3 Statistical Uncertainty and Small Positive Sample Size ($N_{\text{pos}} = 7$)
The held-out test cohort contains exactly **seven positive patients** ($N_{\text{pos}} = 7, N_{\text{neg}} = 40$). A single misclassified patient shifts the empirical sensitivity by $\pm 14.29\%$. The 95% bootstrap confidence interval for C3 sensitivity spans from $50.00\%$ to $100.00\%$, and its ROC-AUC confidence interval spans from $0.7884$ to $0.9849$, overlapping with the Original GIT 1 baseline (`[0.6696, 0.9856]`). **Consequently, we make no claims of definitive architectural superiority or clinical diagnostic superiority based on this limited test sample.**

### 6.4 Clinical & Dataset Limitations
1. **Single-Center Retrospective Bias:** BCMID was sourced from a single clinical center in Alexandria, Egypt. External generalizability across diverse imaging vendors, detector technologies, and demographic populations remains unestablished.
2. **Pathology vs. Radiologic Triage:** Ground truth labels are derived from BI-RADS clinical reports, representing radiologic suspicion rather than definitive tissue biopsy pathology for all subjects.
3. **Research Scope:** This work represents **exploratory proof-of-concept research on multimodal graph-based transfer learning for radiologic triage**, not a clinically validated diagnostic tool.

---

## 7. Conclusion

This study provides a controlled, leak-free investigation of transfer learning and multimodal graph fusion for radiologic breast triage. Candidate C3, trained with class-balanced focal loss ($\gamma = 1.5$), demonstrated the strongest overall validation profile and achieved improved test sensitivity ($85.71\%$ vs. $71.43\%$) and test ROC-AUC ($0.9036$ vs. $0.8679$) at default threshold $\tau = 0.50$ relative to the weighted BCE baseline, though with a reduced PR-AUC ($0.6188$ vs. $0.6800$). Due to the limited size of the positive test cohort ($N_{\text{pos}} = 7$) and wide confidence intervals, these results must be interpreted as encouraging preliminary evidence rather than definitive clinical validation. Multi-center prospective studies with hundreds of biopsy-confirmed malignancies are required.

---

## 8. Project Artifact Manifest

- **Phase C Evaluation Script:** [`scripts/evaluate_phase_c.py`](file:///c:/Users/admin/Music/Breast%20Cancer1/scripts/evaluate_phase_c.py)
- **Phase C Final Comparison Table (CSV):** [`outputs/experiments/phase_c/final_test_evaluation_comparison.csv`](file:///c:/Users/admin/Music/Breast%20Cancer1/outputs/experiments/phase_c/final_test_evaluation_comparison.csv)
- **Phase C Final Evaluation Results (JSON):** [`outputs/experiments/phase_c/phase_c_final_evaluation_results.json`](file:///c:/Users/admin/Music/Breast%20Cancer1/outputs/experiments/phase_c/phase_c_final_evaluation_results.json)
- **Phase C Validation Comparison (CSV):** [`outputs/experiments/phase_c/phase_c_validation_comparison.csv`](file:///c:/Users/admin/Music/Breast%20Cancer1/outputs/experiments/phase_c/phase_c_validation_comparison.csv)
- **Candidate C3 Checkpoint:** [`outputs/experiments/phase_c/exp_c3_git1_focal_gamma1_5/best_checkpoint.pth`](file:///c:/Users/admin/Music/Breast%20Cancer1/outputs/experiments/phase_c/exp_c3_git1_focal_gamma1_5/best_checkpoint.pth)
- **Candidate C3 Test Predictions:** [`outputs/experiments/phase_c/predictions/exp_c3_test_predictions.csv`](file:///c:/Users/admin/Music/Breast%20Cancer1/outputs/experiments/phase_c/predictions/exp_c3_test_predictions.csv)
- **Candidate C4 Checkpoint:** [`outputs/experiments/phase_c/exp_c4_git1_low_lr_reg/best_checkpoint.pth`](file:///c:/Users/admin/Music/Breast%20Cancer1/outputs/experiments/phase_c/exp_c4_git1_low_lr_reg/best_checkpoint.pth)
- **Candidate C4 Test Predictions:** [`outputs/experiments/phase_c/predictions/exp_c4_test_predictions.csv`](file:///c:/Users/admin/Music/Breast%20Cancer1/outputs/experiments/phase_c/predictions/exp_c4_test_predictions.csv)
- **Original Report Backup:** [`outputs/phase5/final_research_report_pre_phase_c_backup.md`](file:///c:/Users/admin/Music/Breast%20Cancer1/outputs/phase5/final_research_report_pre_phase_c_backup.md)
- **Phase 5 Single-Modality Results:** [`outputs/phase5/tables/test_evaluation_results.csv`](file:///c:/Users/admin/Music/Breast%20Cancer1/outputs/phase5/tables/test_evaluation_results.csv)
- **Phase 5 Audit Discrepancy Log:** [`outputs/phase5/audit/audit_discrepancy_log.md`](file:///c:/Users/admin/Music/Breast%20Cancer1/outputs/phase5/audit/audit_discrepancy_log.md)
