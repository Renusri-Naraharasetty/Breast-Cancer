# Scientific Limitations & Strategic Future Research Directions

**Document:** Phase D9 — Limitations and Future Work  
**Project:** Multimodal Radiologic Breast Cancer Triage using Graph-Infused Transformers  
**Repository:** `c:\Users\admin\Music\Breast Cancer1`

---

## 1. Primary Methodological & Sample Size Constraints

### 1.1 The Small Held-Out Test Cohort & Severe Binomial Discreteness ($N_{\text{pos}} = 7$)
The foremost limitation of this research is the sample size of the held-out test cohort ($N_{\text{test}} = 47$), and specifically the small number of high-risk suspicious cases:
- Exactly **$N_{\text{pos}} = 7$ patients** constitute the positive class (`SUSPICIOUS_ACTIONABLE`, BI-RADS 4B, 5, 6).
- **Mathematical Penalty of Coarse Discreteness:** Each positive patient represents **$1/7 \approx 14.2857\%$ of the sensitivity metric**. A model detecting 6 of 7 cases receives a sensitivity score of $85.71\%$; missing one additional case drops sensitivity to $71.43\%$; catching all 7 yields $100.00\%$.
- **Wide Exact Confidence Bounds:** The 95% Clopper-Pearson exact binomial confidence interval for $6/7$ sensitivity spans **$42.13\%$ to $99.64\%$** (a width of $57.51\%$).
- **Implication for Claims of Superiority:** Because confidence intervals are wide and the margin of difference between C3 and D5 is driven by **zero additional detected cancers** and only two borderline false positives, **no statistical claim of superiority or inferiority can be made**. Results must be interpreted strictly as descriptive benchmark observations on this specific test cohort.

### 1.2 Validation-to-Test Generalization Divergence
In Phase D2, Candidate D5 ($\text{lr}=5\times 10^{-5}$) achieved a higher composite validation score ($S = 0.8468$ vs. C3's $0.8351$) by recovering one additional true positive on the validation partition ($7/8$ vs. $6/8$ cases detected, sensitivity $87.50\%$ vs. $75.00\%$).
- However, when evaluated on the held-out test partition in Phase D8, D5 caught the **exact same 6 of 7 cancers** as C3, while misclassifying **two additional benign patients as suspicious** (specificity dropping from $85.00\%$ to $80.00\%$).
- **Scientific Takeaway:** This divergence illustrates the classic hazard of small-sample model selection. In an imbalanced validation set with $N_{\text{pos}}=8$, capturing one borderline case ($p=0.5034$) is easily mistaken for a structural breakthrough, but does not necessarily translate to a recall advantage on independent held-out data.

---

## 2. Clinical & Dataset Limitations

### 2.1 Single-Center Retrospective Origin
- **Source Institution:** The BCMID dataset was retrospectively collected from a single clinical institution: **Ayady Almostakbal Hospital in Alexandria, Egypt**, between 2019 and 2022.
- **Demographic & Equipment Homogeneity:** Imaging protocols, radiological acquisition systems, digital mammography detector technologies, ultrasound transducer frequencies, and patient demographics reflect the local clinical setting.
- **External Generalizability:** The models have not been evaluated on external data from other medical centers, countries, or healthcare tiers. Performance on alternate imaging hardware (e.g., Hologic, GE, Siemens mammography systems; Philips, Canon ultrasound systems) remains unquantified.

### 2.2 Radiologic Triage vs. Definitive Pathological Diagnosis
- **Task Scope:** The classification objective is **automated radiologic triage** (`LOW_RISK` [BI-RADS 1–3] vs. `SUSPICIOUS_ACTIONABLE` [BI-RADS 4–6]).
- **Label Origin:** Labels are extracted from radiologist narrative findings and BI-RADS assessment categories. While BI-RADS 6 cases represent biopsy-confirmed malignancies and BI-RADS 5 represents $>95\%$ malignancy probability, lower categories (BI-RADS 1–3 and 4A/4B) represent radiologic suspicion rather than universal histological truth.
- **Deployment Boundary:** This software **does not diagnose breast cancer** and cannot substitute for histological tissue biopsy or clinical radiological review.

### 2.3 Variable Multi-View Imaging & View Asymmetry
- In clinical practice, patients undergo variable imaging protocols (e.g., 2 to 4 mammographic views: L/R CC and MLO; 1 to 16 targeted ultrasound views).
- While the GIT architecture handles variable views through dynamic padding masks and graph attention, extreme view imbalances (e.g., patients with only 1 ultrasound image vs. patients with 16 targeted sweeps) may introduce feature variance.

---

## 3. Strategic Directions for Future Research

### 3.1 Multi-Center Prospective Validation Cohorts
The essential prerequisite for any clinical translation of multimodal radiologic triage is evaluating on multi-center prospective cohorts:
1. **Large-Scale Cohort Expansion:** Assembling test cohorts with hundreds of biopsy-proven malignancies ($N_{\text{pos}} \ge 250$) across diverse breast densities (ACR Density Categories A, B, C, and D) to shrink sensitivity confidence intervals to within $\pm 3\%$.
2. **Multi-Vendor Heterogeneity:** Evaluating model robustness across varied digital mammography (FFDM), digital breast tomosynthesis (DBT), and automated breast ultrasound (ABUS) hardware.

### 3.2 End-to-End Fine-Tuning with Memory-Efficient Architectures
In this study, the image feature encoders (EfficientNet-B0 and ViT-B/16) were frozen after unimodal pre-training to respect hardware constraints on the 2.0 GB NVIDIA GeForce MX550.
- With expanded GPU hardware (e.g., NVIDIA A100 or H100 with 40–80 GB VRAM), future studies should explore **joint end-to-end backpropagation** through both the image encoders and the Graph-Infused Transformer.
- Parameter-efficient fine-tuning (PEFT) techniques, such as Low-Rank Adaptation (LoRA), could enable gradient updates in the vision transformer layers without exceeding memory budgets.

### 3.3 Side-Aware Contrastive Pre-Training
- Exploring self-supervised multimodal contrastive pre-training (e.g., aligning CC and MLO mammogram views with corresponding ipsilateral ultrasound sweeps before fine-tuning on triage labels).
- Enforcing anatomical symmetry constraints (comparing ipsilateral suspicious findings against the contralateral breast to replicate the radiologist's comparative screening workflow).

### 3.4 Conformal Prediction for Risk-Controlled Triage
- In medical triage, predicting a point probability may be insufficient for high-stakes clinical decisions.
- Incorporating **conformal prediction sets** that guarantee a user-specified sensitivity (e.g., $95\%$ or $99\%$ guarantee) with finite-sample coverage guarantees, allowing uncertain or borderline cases to be automatically flagged for expert human radiological review.
