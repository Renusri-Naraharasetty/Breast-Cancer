# Multimodal Radiologic Breast Cancer Triage using Graph-Infused Transformers (GIT)

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Research Audit: PASS](https://img.shields.io/badge/Research%20Audit-READY%20FOR%20SUBMISSION-success.svg)](outputs/phase_d/phase_d10_final_integrity_audit/final_integrity_audit_report.md)

This repository contains the authoritative research implementation, experimental pipelines, locked evaluation suite, and publication deliverables for multimodal radiologic breast cancer triage on the **Breast Cancer Multimodal Imaging Dataset (BCMID)**.

The framework pairs Full-Field Digital Mammography (FFDM) and targeted Ultrasound (US) sweeps using frozen unimodal vision backbones, anatomical side-aware Graph Attention Networks (GAT), and cross-attention Transformers to triage patient examinations into **Low-Risk Screening (`LOW_RISK`, BI-RADS 1–3)** versus **Suspicious / Actionable Screening (`SUSPICIOUS_ACTIONABLE`, BI-RADS 4A–6)**.

---

## 1. Architectural Pipeline

```
  Mammograms (CC / MLO)                 Ultrasound Sweeps
   (Frozen EfficientNet-B0)              (Frozen ViT-B/16)
          │                                     │
          ▼                                     ▼
   Mammogram Embeddings                  Ultrasound Embeddings
     [N, 1280]                             [M, 768]
          │                                     │
          └──────────────────┬──────────────────┘
                             ▼
            Side-Aware Graph Builder (GAT Layers)
            • View-to-view message passing
            • Anatomical side relational biases (Ipsilateral / Contralateral)
                             │
                             ▼
            Multimodal Cross-Attention Transformer
            • Patient-level cross-modal interaction
            • Residual token fusion
                             │
                             ▼
            Linear Classification Head (Softmax)
            • Class-Balanced Focal Loss (gamma = 1.5, alpha = [1.0, 5.11])
            • Canonical Operating Threshold tau = 0.50
                             │
                             ▼
            Radiologic Triage Prediction:
            P(SUSPICIOUS_ACTIONABLE | Mammography + Ultrasound)
```

---

## 2. BCMID Dataset & Patient-Level Split Governance

The dataset was curated from **Ayady Almostakbal Hospital in Alexandria, Egypt** (2019–2022) and hosted on [Zenodo (Record 15546909)](https://zenodo.org/records/15546909).

### Cohort Accounting & Anti-Leakage Protocol
To prevent data leakage across multiple visits or bilateral folders, all partitions were generated strictly at the **biological root-patient level** (`root_patient_id`):

| Cohort Partition | Unique Root Patients | Patient Folders | Class 0 (`LOW_RISK`) | Class 1 (`SUSPICIOUS`) | Class Ratio (Neg/Pos) | Repeat Exam Groups | Images (MG / US) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Training (70%)** | 217 | **220** | 184 (83.64%) | 36 (16.36%) | 5.1111 : 1 | Groups R1, R2, R3 (6 folders) | 644 / 671 |
| **Validation (15%)**| 47 | **48** | 40 (83.33%) | 8 (16.67%) | 5.0000 : 1 | Group R4 (2 folders) | 138 / 135 |
| **Held-Out Test (15%)**| 47 | **47** | 40 (85.11%) | 7 (14.89%) | 5.7143 : 1 | Zero repeat visits (100% unique) | 132 / 150 |
| **Total Eligible Cohort** | **311** | **315** | **264** (83.81%) | **51** (16.19%) | 5.1765 : 1 | 4 Root Patients (8 folders) | **914 / 956** |

- **Exclusions ($N=17$):** 11 non-diagnostic / missing reports (`NONE`), 5 indeterminate reports (`INDETERMINATE`), and 1 incomplete assessment (`Category 0`).
- **Partition Intersections:** Strictly null ($\text{Train} \cap \text{Val} = \emptyset$, $\text{Train} \cap \text{Test} = \emptyset$, $\text{Val} \cap \text{Test} = \emptyset$).

---

## 3. Locked Final Held-Out Evaluation (Threshold $\tau = 0.50$)

Following upstream development and quarantine during Phase D1–D7, the 47-patient held-out test cohort was unblinded once in Phase D8 for locked evaluation comparing the frozen reference benchmark **C3** (`exp_c3_git1_focal_gamma1_5`) against the leading tuned candidate **D5** (`exp_d5_lr5e-5`):

| Performance Metric | Frozen Benchmark C3 | Leading Candidate D5 | Delta ($\text{D5} - \text{C3}$) | Exact Statistical Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **Accuracy** | **85.11%** (40/47) | **80.85%** (38/47) | $-4.26\%$ | C3 achieves higher overall classification accuracy |
| **Sensitivity (Recall)** | **85.71%** (6/7) | **85.71%** (6/7) | $0.00\%$ | **Identical detection yield (both identify 6 of 7 suspicious cases)** |
| **Specificity** | **85.00%** (34/40) | **80.00%** (32/40) | $-5.00\%$ | C3 clears 2 additional benign patients without false alarms |
| **Positive Predictive Value (PPV)** | **50.00%** (6/12) | **42.86%** (6/14) | $-7.14\%$ | C3 yields higher biopsy precision on flagged examinations |
| **Negative Predictive Value (NPV)** | **97.14%** (34/35) | **96.97%** (32/33) | $-0.17\%$ | High reassurance on low-risk screening triage clearance |
| **F1-Score** | **0.6316** | **0.5714** | $-0.0602$ | Harmonic mean favors C3 |
| **Balanced Accuracy** | **85.36%** | **82.86%** | $-2.50\%$ | Unweighted average of sensitivity and specificity |
| **ROC-AUC** | **0.9036** | **0.8964** | $-0.0072$ | Overlapping 95% bootstrap confidence intervals |
| **PR-AUC (Average Precision)** | **0.6188** | **0.6151** | $-0.0037$ | Baseline prevalence: 14.89% |
| **Brier Score** | **0.1133** | **0.1281** | $+0.0148$ (worse) | C3 exhibits lower squared continuous probability error |
| **Log Loss (Cross-Entropy)** | **0.3519** | **0.4075** | $+0.0556$ (worse) | C3 penalizes ground-truth cross-entropy less |
| **Confusion Matrix `[TN, FP, FN, TP]`** | `[34, 6, 1, 6]` | `[32, 8, 1, 6]` | 2 extra FPs in D5 | Driven entirely by 2 borderline benign patients |
| **Exact McNemar Test (Paired)** | Reference | $p = 0.5000$ | — | **No statistically significant difference ($95.74\%$ agreement)** |
| **Wilcoxon Signed-Rank Test** | Reference | $p = 0.0074$ | — | Slight upward probability shift in D5 ($\Delta p = +0.0409$) |

### Key Scientific Findings
1. **Benchmark C3 Reaffirmation:** Candidate D5 demonstrated an apparent sensitivity advantage on validation data ($87.5\%$ vs $75.0\%$), but **failed to improve sensitivity on held-out test data**. Both models detected the exact same 6 of 7 suspicious lesions and missed the identical single positive case.
2. **False Alarm Penalty:** Candidate D5 incurred 2 additional false-positive classifications on benign patients, lowering specificity from 85.00% to 80.00% and precision from 50.00% to 42.86%.
3. **Statistical Non-Significance:** The exact McNemar test ($p = 0.5000$) demonstrates that the paired classification difference is not statistically significant. Benchmark C3 is reaffirmed as the primary, most defensible model.

---

## 4. Phase D10 Final Research Integrity Audit Deliverables

All publication-ready reports, audited tables, de-identified figures, and multi-format manuscript exports are located in [`outputs/phase_d/phase_d10_final_integrity_audit/`](outputs/phase_d/phase_d10_final_integrity_audit/):

- **Master Research Manuscript:**
  - Markdown Source: [`final_research_manuscript.md`](outputs/phase_d/phase_d10_final_integrity_audit/final_research_manuscript.md)
  - Compiled PDF: [`final_research_manuscript.pdf`](outputs/phase_d/phase_d10_final_integrity_audit/final_research_manuscript.pdf)
  - Editable Word Document: [`final_research_manuscript.docx`](outputs/phase_d/phase_d10_final_integrity_audit/final_research_manuscript.docx)
  - Standalone HTML: [`final_research_manuscript.html`](outputs/phase_d/phase_d10_final_integrity_audit/final_research_manuscript.html)
- **Executive Synthesis:** [`executive_summary.md`](outputs/phase_d/phase_d10_final_integrity_audit/executive_summary.md)
- **Integrity Audit Report:** [`final_integrity_audit_report.md`](outputs/phase_d/phase_d10_final_integrity_audit/final_integrity_audit_report.md)
- **Submission Readiness Checklist:** [`submission_readiness_checklist.md`](outputs/phase_d/phase_d10_final_integrity_audit/submission_readiness_checklist.md) (16/16 PASS)
- **Reconciliation Audit CSVs:**
  - Dataset Counts: [`dataset_count_discrepancies.csv`](outputs/phase_d/phase_d10_final_integrity_audit/dataset_count_discrepancies.csv)
  - Metric Consistency: [`metric_consistency_audit.csv`](outputs/phase_d/phase_d10_final_integrity_audit/metric_consistency_audit.csv)
- **Privacy & Literature Verification:** [`privacy_and_citation_audit.md`](outputs/phase_d/phase_d10_final_integrity_audit/privacy_and_citation_audit.md)
- **Machine-Readable Manifest:** [`phase_d10_manifest.json`](outputs/phase_d/phase_d10_final_integrity_audit/phase_d10_manifest.json)

---

## 5. Quickstart & Reproducibility

### Installation
```bash
git clone https://github.com/Renusri-Naraharasetty/Breast-Cancer.git
cd Breast-Cancer
pip install torch torchvision scikit-learn pandas numpy scipy matplotlib statsmodels
```

### Reproduce Locked Phase D8 Evaluation
```bash
python scripts/evaluate_phase_d8_locked.py
```

### Run Phase D10 Automated Integrity Verification Checks
```bash
python scratch/run_final_automated_checks.py
```

### Run Multimodal Sample Inference
```bash
python predict_sample.py
```

---

## 6. Ethical Considerations & Clinical Boundaries

> [!CAUTION]
> **Research Prototype Disclaimer**: This system is an automated radiologic triage research tool (`LOW_RISK` vs. `SUSPICIOUS_ACTIONABLE`), not an autonomous tissue pathology diagnostic device. It does not replace clinical breast examination, expert radiological interpretation, or tissue biopsy.
>
> **Cohort Constraints**: The held-out test cohort contains $N_{\text{pos}} = 7$ suspicious cases, yielding wide exact 95% Clopper-Pearson confidence intervals ($[42.13\%, 99.64\%]$ for sensitivity). Results reflect internal descriptive observations on retrospective data from a single hospital and require external multi-center clinical validation before translational deployment.

---

## 7. Citation

If you use this codebase, methodology, or results in your research, please cite:

```bibtex
@article{bcmid_git_triage_2026,
  title   = {Multimodal Deep Learning for Radiologic Breast Cancer Triage: Graph-Infused Transformers on Paired Mammogram and Ultrasound Imaging},
  author  = {Antigravity AI Research Team and Collaborators},
  journal = {Research Consolidation Report, BCMID Project},
  year    = {2026},
  url     = {https://github.com/Renusri-Naraharasetty/Breast-Cancer}
}
```

---

## 8. License

This repository is licensed under the [MIT License](LICENSE). The BCMID imaging dataset is governed by its original Zenodo open-access terms (Record 15546909).
