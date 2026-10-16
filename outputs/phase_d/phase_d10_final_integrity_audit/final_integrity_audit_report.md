# Phase D10: Final Research Integrity Audit & Submission Readiness Report

**Project:** Multimodal Radiologic Breast Cancer Triage using Graph-Infused Transformers (GIT)  
**Corpus:** Breast Cancer Multimodal Imaging Dataset (BCMID)  
**Execution Phase:** Phase D10 — Final Research Integrity Audit and Submission Readiness  
**Date:** October 16, 2026  
**Auditor:** Antigravity AI Research Team  
**Final Verdict:** **READY FOR SUBMISSION REVIEW**

---

## 1. Executive Summary & Audit Overview

Phase D10 was executed as a comprehensive, independent verification and integrity audit of all Phase D9 deliverables. In accordance with strict scientific guidelines:
- **Zero model modifications, training, fine-tuning, or inference executions occurred.**
- **No threshold sweeps or post-hoc operating point adjustments were performed.**
- **All original Phase C, Phase 5, Phase D0–D8, and Phase D9 artifacts remain preserved and untouched.**
- **All corrections and audit deliverables are organized in `outputs/phase_d/phase_d10_final_integrity_audit/`.**

The audit achieved the following core outcomes:
1. **Authoritative Dataset Reconciliation:** Re-verified all cohort counts from `BCMID_annotations.tsv` and `data/splits/patient_stratified_split.json`. Identified that the discrepancy between the assistant's ephemeral chat turn summary (165/55 train, 38/10 val) and the authoritative audit (184/36 train, 40/8 val) was an ephemeral reporting error in chat text; the underlying Phase D9 manuscript and table artifacts already contained the verified counts (184/36 train, 40/8 val).
2. **Definitive Locked Metric Verification:** Verified every test metric for Benchmark C3 and Candidate D5 at threshold $\tau = 0.50$ against the Phase D8 JSON and CSV outputs. 100% numerical concordance confirmed.
3. **Patient Privacy & De-Identification:** Eradicated 8 patient-level identifiers from publication-facing Markdown, Word, PDF, and HTML manuscripts, and re-generated Figure 9 (`paired_prediction_scatter.png`) with de-identified labels (`Discordant Case A`, `Discordant Case B`).
4. **Clinical Framing & Medical Wording:** Replaced all inaccurate pathology claims ("cancers detected", "malignancies caught") with rigorous radiologic triage terminology ("suspicious/actionable lesions identified", "high-risk cases triaged"). Reaffirmed that the objective is AI-assisted radiologic triage (`LOW_RISK` [BI-RADS 1–3] vs. `SUSPICIOUS_ACTIONABLE` [BI-RADS 4A–6]), not autonomous pathology diagnosis.
5. **Transparent Test-Set History Disclosure:** Explicitly disclosed that while Phase D8 was a locked internal evaluation with a quarantined test set during Phase D1–D7, preliminary test evaluations occurred historically during Phase C exploratory benchmarking.
6. **Scholarly Citation & Bibliography Verification:** Validated all 22 citations against authoritative databases (CrossRef, PubMed, IEEE Xplore, Zenodo Record 15546909) and aligned all in-text bracketed callouts (`[1]` to `[22]`).
7. **Multi-Format Publication Deliverables:** Successfully compiled corrected Markdown, DOCX, self-contained HTML, and publication-grade PDF documents with embedded 300 DPI figures and complete tables.

---

## 2. Comprehensive Evidence & Artifact Inventory

The following primary files and source artifacts were inspected:

| Artifact Path | Artifact Description | Audit Purpose | Verification Outcome |
| :--- | :--- | :--- | :---: |
| `BCMID_annotations.tsv` | Primary radiologic annotations file (332 rows) | Cohort ground truth, BI-RADS mapping, exclusions | **Verified (332 total, 17 excluded, 315 eligible)** |
| `data/splits/patient_stratified_split.json` | Authoritative patient-level split manifest | Partition lists, root patient groupings | **Verified (220 train, 48 val, 47 test)** |
| `data/features/patient_multimodal_features.pt` | Pre-extracted feature cache | Tensor shapes, patient coverage | **Verified (315 patients, MG 1280, US 768)** |
| `outputs/phase_d/phase_d0_audit_report.md` | Pre-experiment audit report | Historical count reconciliation & protocol | **Verified (311 roots, 315 folders)** |
| `outputs/phase_d/phase_d8_final_evaluation/test_metrics_c3.json` | Primary C3 test metrics | Locked evaluation metrics at $\tau = 0.50$ | **Verified (100% concordance)** |
| `outputs/phase_d/phase_d8_final_evaluation/test_metrics_d5.json` | Primary D5 test metrics | Locked evaluation metrics at $\tau = 0.50$ | **Verified (100% concordance)** |
| `outputs/phase_d/phase_d8_final_evaluation/paired_comparison.json` | Paired statistical tests | McNemar & Wilcoxon tests, discordant details | **Verified (McNemar $p=0.5000$)** |
| `outputs/phase_d/phase_d8_final_evaluation/historical_metric_reconciliation.md` | Historical audit report | GIT1 / GIT2 metric reconciliation | **Verified (Hardcode error resolved)** |
| `outputs/phase_d/phase_d9_research_consolidation/final_research_manuscript.md` | Phase D9 manuscript source | Content, structure, citations, privacy | **Audited (Discrepancies identified & corrected)** |
| `outputs/phase_d/phase_d9_research_consolidation/figures/` | Phase D9 publication figures | Image resolution, readability, privacy | **Audited (Figure 9 re-generated de-identified)** |

---

## 3. Dataset Count Reconciliation

### 3.1 Authoritative Cohort Hierarchy
$$\begin{aligned}
\text{Total Raw Patient Folders} &= 332 \\
\text{Excluded Non-Diagnostic Folders} &= 17 \quad (11 \text{ NONE}, 5 \text{ INDETERMINATE}, 1 \text{ Category 0}) \\
\text{Audited Eligible Folders} &= 315 \\
\text{Unique Biological Root Patients} &= 311 \\
\text{Repeat-Examination Root Patients} &= 4 \quad (\text{8 folders total; 4 extra examination folders})
\end{aligned}$$

### 3.2 Partition Breakdown & Root Patient Grouping
To ensure zero data leakage across longitudinal or bilateral examinations, all repeat visits for the same root patient were partitioned into the same split:

| Cohort Partition | Unique Biological Patients | Total Patient Folders | Class 0 (`LOW_RISK`) | Class 1 (`SUSPICIOUS`) | Class Ratio (Neg / Pos) | Multi-Visit Root Patient Groups | Total Images (MG / US) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Training (70%)** | 217 | **220** | 184 (83.64%) | 36 (16.36%) | 5.1111 : 1 | Groups R1, R2, R3 (6 folders) | 644 / 671 |
| **Validation (15%)**| 47 | **48** | 40 (83.33%) | 8 (16.67%) | 5.0000 : 1 | Group R4 (2 folders) | 138 / 135 |
| **Held-Out Test (15%)**| 47 | **47** | 40 (85.11%) | 7 (14.89%) | 5.7143 : 1 | Zero repeat visits (100% unique) | 132 / 150 |
| **Total Eligible Cohort** | **311** | **315** | **264** (83.81%) | **51** (16.19%) | 5.1765 : 1 | 4 Root Patients (8 folders) | **914 / 956** |

### 3.3 Leakage & Isolation Audit
- $\text{Train} \cap \text{Val} = \emptyset$ (0 overlapping root patients or folders).
- $\text{Train} \cap \text{Test} = \emptyset$ (0 overlapping root patients or folders).
- $\text{Val} \cap \text{Test} = \emptyset$ (0 overlapping root patients or folders).
- Zero excluded non-diagnostic cases ($N=17$) appear in any evaluation split.

### 3.4 Resolution of the Phase D9 Chat Summary Discrepancy
In the assistant's previous chat response, a summary row mistakenly stated:
`Train: 220 folders (217 unique patients; 165 LOW_RISK, 55 SUSPICIOUS_ACTIONABLE)`
`Validation: 48 folders (47 unique patients; 38 LOW_RISK, 10 SUSPICIOUS_ACTIONABLE)`
An exhaustive audit confirmed that:
1. This was purely an ephemeral transcription error in the chat response text.
2. The actual saved Phase D9 manuscript (`final_research_manuscript.md` lines 89–94) and table files already contained the correct, verified figures: **Train: 184 Class 0 / 36 Class 1; Val: 40 Class 0 / 8 Class 1**.
3. All underlying feature caches and training scripts loaded the verified 184/36 and 40/8 distributions.
4. The corrected manuscript in `outputs/phase_d/phase_d10_final_integrity_audit/corrected_manuscript/` strictly reinforces the verified counts.

---

## 4. Locked Test Metric Verification (Held-Out Test Set, $\tau = 0.50$)

All reported test metrics were audited against `test_metrics_c3.json`, `test_metrics_d5.json`, and `paired_comparison.json`:

| Performance Metric | Frozen C3 Benchmark | Candidate D5 | Delta ($\text{D5} - \text{C3}$) | Verification Status | Exact Statistical Interpretation |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Test Cohort Size ($N$)** | 47 | 47 | 0 | **Verified** | Identical patient cohort |
| **Positive Class Cases ($N_{\text{pos}}$)** | 7 | 7 | 0 | **Verified** | 7 radiologically suspicious cases |
| **Negative Class Cases ($N_{\text{neg}}$)** | 40 | 40 | 0 | **Verified** | 40 routine/benign cases |
| **Accuracy** | **85.11%** (40/47) | **80.85%** (38/47) | $-4.26\%$ | **Verified** | Higher proportion correct in C3 |
| **Sensitivity (Recall)** | **85.71%** (6/7) | **85.71%** (6/7) | $0.00\%$ | **Verified** | **Identical detection (6 of 7 caught)** |
| **Specificity** | **85.00%** (34/40) | **80.00%** (32/40) | $-5.00\%$ | **Verified** | C3 clears 2 more healthy patients |
| **Positive Predictive Value (PPV)** | **50.00%** (6/12) | **42.86%** (6/14) | $-7.14\%$ | **Verified** | Higher biopsy yield in C3 |
| **Negative Predictive Value (NPV)** | **97.14%** (34/35) | **96.97%** (32/33) | $-0.17\%$ | **Verified** | High reassurance on negative triage |
| **F1-Score** | **0.6316** | **0.5714** | $-0.0602$ | **Verified** | Harmonic balance favors C3 |
| **Balanced Accuracy** | **85.36%** | **82.86%** | $-2.50\%$ | **Verified** | Unweighted mean of sens & spec |
| **ROC-AUC** | **0.9036** | **0.8964** | $-0.0072$ | **Verified** | Overlapping bootstrap intervals |
| **PR-AUC (Prevalence: 14.89%)** | **0.6188** | **0.6151** | $-0.0037$ | **Verified** | Overlapping bootstrap intervals |
| **Brier Score** | **0.1133** | **0.1281** | $+0.0148$ | **Verified** | Lower squared probability error in C3 |
| **Log Loss (Cross-Entropy)** | **0.3519** | **0.4075** | $+0.0556$ | **Verified** | Lower cross-entropy penalty in C3 |
| **Confusion Matrix `[TN, FP, FN, TP]`** | `[34, 6, 1, 6]` | `[32, 8, 1, 6]` | 2 extra FPs in D5 | **Verified** | C3 FP=6, D5 FP=8; both FN=1 |
| **Paired Agreement** | 45 / 47 (95.74%) | 45 / 47 (95.74%) | — | **Verified** | High inter-model concordance |
| **Exact McNemar Test** | Reference | $p = 0.5000$ | — | **Verified** | **Difference not statistically significant** |
| **Wilcoxon Signed-Rank Test** | Reference | $p = 0.0074$ | — | **Verified** | Slight upward probability shift in D5 |

### Scientific Conclusions:
1. **No Sensitivity Advantage:** Halving the learning rate ($\text{lr} = 5\times 10^{-5}$) in D5 yielded an apparent sensitivity increase on validation data ($87.5\%$ vs. $75.0\%$, $+1$ patient), but this **failed to generalize to the held-out test cohort**. Both models identified the exact same 6 of 7 suspicious lesions and missed the identical single BI-RADS 5 lesion.
2. **Elevated False Alarm Burden:** D5 caused 2 additional false-positive classifications on benign patients (`Discordant Case A` and `Discordant Case B`), reducing specificity from 85.00% to 80.00% and precision from 50.00% to 42.86%.
3. **Statistical Non-Significance:** The exact McNemar test ($p = 0.5000$) demonstrates that the paired classification performance difference is not statistically significant.
4. **Benchmark Reaffirmation:** **Benchmark C3 is reaffirmed as the stronger, more parsimonious, and better-calibrated reference model.**

---

## 5. Patient Privacy & De-Identification Audit

### 5.1 Vulnerabilities Found in Phase D9 Deliverables
A comprehensive scan identified raw patient codes in public-facing files:
- `final_research_manuscript.md` contained 8 raw patient identifiers (e.g., longitudinal repeat roots and specific test-case IDs).
- `executive_summary.md` exposed the IDs of the two discordant false-positive cases.
- `figures/paired_prediction_scatter.png` annotated the discordant scatter points with explicit patient IDs (`Pt 2310y19`, `Pt 4105y18`).
- Compiled Word, HTML, and PDF files inherited these identifiers.

### 5.2 Corrective De-Identification Implemented
In `outputs/phase_d/phase_d10_final_integrity_audit/corrected_manuscript/`:
- All patient IDs have been replaced with non-identifying categorical designations (`Discordant Case A`, `Discordant Case B`, `Mutual False-Negative Case C`, `Validation True-Positive Case V1`, and `Multi-Visit Groups R1–R4`).
- Figure 9 (`paired_prediction_scatter.png`) was re-rendered with clean, de-identified callout labels.
- Zero raw patient IDs remain in public deliverables (verified by automated regex scan).
- The raw mapping tables are strictly restricted to internal audit files.

---

## 6. Scientific & Clinical Wording Audit

The manuscript was thoroughly reviewed to enforce appropriate medical terminology:
1. **Radiologic Triage vs. Pathology Diagnosis:** The target is AI-assisted radiologic triage (`LOW_RISK` [BI-RADS 1–3] vs. `SUSPICIOUS_ACTIONABLE` [BI-RADS 4A–6]), designed to prioritize suspicious screening studies for urgent radiologist review and diagnostic tissue biopsy. It is **not** an autonomous breast cancer diagnostic device.
2. **Eradication of "Cancers Detected" Language:** Replaced all claims stating that models detected "cancers" or "malignancies" with precise statements that models triaged "radiologically suspicious / actionable lesions" or "high-risk lesions". In screening imaging without routine surgical pathology across all benign cases, BI-RADS 4A–6 lesions represent suspicious radiologic findings requiring biopsy, not proven invasive carcinomas.
3. **Disclosure of Uncertainty ($N_{\text{pos}} = 7$):** Highlighted that because the held-out test cohort contains only 7 positive cases, exact 95% Clopper-Pearson confidence intervals for sensitivity span from **$42.13\%$ to $99.64\%$**. A single patient shift alters sensitivity by $\pm 14.29\%$. No claims of proven clinical superiority can be justified from a sample of this size.
4. **Single-Center Retrospective Origin:** Emphasized that data were collected retrospectively from a single institution (Ayady Almostakbal Hospital, Alexandria, Egypt) and require multi-center external validation before clinical claims can be evaluated.

---

## 7. Scholarly Literature & Citation Verification

- All 22 citations in `references.md` and Section 13 of the manuscript were verified against scholarly repositories.
- Authors, years, titles, journal names, and DOIs are authentic and accurate.
- In-text bracketed citation callouts were updated to ensure that every reference listed in Section 13 is explicitly cited in the body of the manuscript (including statistical methods: McNemar `[19]`, Wilcoxon `[20]`, Clopper-Pearson `[21]`, and Wilson `[22]`).
- Zero fabricated, hallucinated, or unverified citations exist.

---

## 8. Artifact Generation & Deliverable Verification

All corrected deliverables have been generated and saved under `outputs/phase_d/phase_d10_final_integrity_audit/`:

1. `phase_d10_final_integrity_audit.md` (this report)
2. `dataset_count_discrepancies.csv` (complete reconciliation audit table)
3. `metric_consistency_audit.csv` (complete metric concordance audit table)
4. `privacy_and_citation_audit.md` (privacy scan and literature verification log)
5. `final_submission_checklist.md` (formal 16-point submission readiness checklist)
6. `phase_d10_manifest.json` (machine-readable audit and file manifest)
7. `corrected_manuscript/`:
   - `final_research_manuscript.md` (authoritative de-identified Markdown source)
   - `final_research_manuscript.pdf` (compiled publication PDF with embedded de-identified figures)
   - `final_research_manuscript.docx` (compiled editable Word document)
   - `final_research_manuscript.html` (self-contained HTML manuscript)
   - `executive_summary.md` (corrected executive research synthesis)
   - `limitations_and_future_work.md` (comprehensive boundary and future directions document)
   - `references.md` (verified 22-reference scholarly bibliography)
   - `final_results_tables.csv` (master consolidated numerical results table)
   - `figures/` (complete directory of 10 publication figures, including de-identified Figure 9)
   - `tables/` (complete directory of 14 CSV and Markdown publication tables)

---

## 9. Final Decision & Status

### **FINAL AUDIT STATUS: READY FOR SUBMISSION REVIEW**

**Summary Statement:**
All numerical, scientific, clinical, privacy, citation, and artifact consistency requirements have been rigorously satisfied. The data splits are leakage-free, all metrics agree bit-for-bit with authoritative test-set outputs, patient identifiers have been eliminated from all public deliverables, and the medical claims strictly respect the boundary of radiologic triage. The publication deliverables are complete, verified, and ready for formal scientific dissemination.
