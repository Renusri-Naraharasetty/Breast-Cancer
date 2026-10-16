# Phase D9: Executive Research Summary & Scientific Synthesis

**Project:** Multimodal Radiologic Breast Cancer Triage using Graph-Infused Transformers (GIT)  
**Corpus:** Breast Cancer Multimodal Imaging Dataset (BCMID)  
**Execution Phase:** Phase D9 — Research Consolidation, Reproducibility Audit & Publication Readiness  
**Date:** October 16, 2026  
**Auditor:** Antigravity AI Research Team  
**Status:** **LOCKED & VERIFIED FINAL DELIVERABLES**

---

## 1. Project Background & Research Trajectory

Automated radiologic breast triage aims to prioritize high-risk, suspicious breast lesions (`SUSPICIOUS_ACTIONABLE`, BI-RADS 4A–6) requiring urgent tissue biopsy, while clearing routine benign or negative examinations (`LOW_RISK`, BI-RADS 1–3) without unnecessary invasive procedures.

This project conducted a multi-phase, scientifically controlled investigation evaluating unimodal vision backbones and graph-infused multimodal fusion on the BCMID dataset:

```
[Phase 1-4: Unimodal Models]
EfficientNet-B0 & ViT-B/16 on Mammography & Ultrasound (Frozen Encoders)
            │
            ▼
[Phase 5: Multimodal GIT Baselines]
GIT 1 (EffNet MG + ViT US) & GIT 2 (ViT MG + EffNet US)
            │
            ▼
[Phase C: Focal Loss & Benchmark Selection]
Systematic exploration of class-balanced focal loss -> Selected C3 (gamma=1.5, Epoch 4)
            │
            ▼
[Phase D0: Pre-Experiment Audit]
Cohort count reconciliation (311 roots, 315 folders) & strict test quarantine
            │
            ▼
[Phase D1-D5: Controlled Ablations]
Deterministic C3 reproduction (D1) + single-factor ablations (D2-D5) -> Selected D5 (lr=5e-5)
            │
            ▼
[Phase D6: Multi-Seed Stability]
Variance testing across seeds 42, 123, 456 on validation data (N=48)
            │
            ▼
[Phase D7: Threshold & Calibration Analysis]
Operating point mapping (tau in [0.30, 0.70]) & ECE analysis -> Reaffirmed tau=0.50
            │
            ▼
[Phase D8: Locked Final Held-Out Evaluation]
One-time automated test unblinding (N=47, N_pos=7) -> C3 vs D5
            │
            ▼
[Phase D9: Research Consolidation & Reproducibility Audit]
Full forensic audit, publication-quality deliverables, and manuscript finalization
```

---

## 2. Definitive Final Model Comparison (Locked Held-Out Test Cohort)

Both models were evaluated on the exact same 47 test patients at canonical threshold $\tau = 0.50$:

| Performance Metric | Frozen C3 Benchmark ($\text{lr}=10^{-4}$) | Candidate D5 ($\text{lr}=5\times 10^{-5}$) | Delta (D5 - C3) | Statistical Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **Accuracy** | **85.11%** (40/47) | **80.85%** (38/47) | $-4.26\%$ | Higher overall proportion correct in C3 |
| **Sensitivity (Recall)** | **85.71%** (6/7) | **85.71%** (6/7) | $0.00\%$ | **Identical suspicious lesion detection yield (6 of 7 triaged)** |
| **Specificity** | **85.00%** (34/40) | **80.00%** (32/40) | $-5.00\%$ | C3 clears 2 more healthy patients without false alarms |
| **Precision (PPV)** | **50.00%** (6/12) | **42.86%** (6/14) | $-7.14\%$ | Positive biopsy yield is higher in C3 |
| **Negative Predictive Value** | **97.14%** (34/35) | **96.97%** (32/33) | $-0.17\%$ | High diagnostic reassurance on negative results |
| **F1-Score** | **0.6316** | **0.5714** | $-0.0602$ | Harmonic mean favors C3 |
| **Balanced Accuracy** | **85.36%** | **82.86%** | $-2.50\%$ | Unweighted average of sensitivity and specificity |
| **ROC-AUC** | **0.9036** | **0.8964** | $-0.0072$ | Overlapping 95% bootstrap intervals |
| **PR-AUC (Prevalence: 14.89%)** | **0.6188** | **0.6151** | $-0.0037$ | Overlapping 95% bootstrap intervals |
| **Brier Score** | **0.1133** | **0.1281** | $+0.0148$ | C3 exhibits lower squared probability error |
| **Log Loss** | **0.3519** | **0.4075** | $+0.0556$ | C3 penalizes ground-truth cross-entropy less |
| **Confusion Matrix `[TN, FP, FN, TP]`** | `[34, 6, 1, 6]` | `[32, 8, 1, 6]` | TN: $-2$, FP: $+2$ | Driven entirely by 2 borderline benign patients |
| **McNemar Exact Test (Paired)** | Reference Baseline | $p = 0.5000$ | — | **Not statistically significant** |
| **Wilcoxon Signed-Rank Test** | Reference Baseline | $p = 0.0074$ | — | Minor upward probability shift in D5 ($\Delta p = +0.041$) |

---

## 3. Key Scientific Conclusions & Evidence

1. **Candidate D5 Does Not Surpass Benchmark C3 on Held-Out Test Data:**
   - In validation testing ($N_{\text{val}}=48, N_{\text{pos}}=8$), D5 demonstrated an apparent recall advantage ($87.5\%$ vs. $75.0\%$ sensitivity, $+1$ patient detected).
   - On the independent held-out test cohort ($N_{\text{test}}=47, N_{\text{pos}}=7$), D5 detected the **exact same 6 malignancies as C3**, while generating **two additional false positives** (`Discordant Case A` and `Discordant Case B`).
   - Consequently, **Benchmark C3 descriptively outperforms Candidate D5** across all specificity, precision, accuracy, F1, and loss dimensions.
2. **Statistical Indistinguishability:**
   - C3 and D5 make identical classifications on **45 out of 47 patients ($95.74\%$ agreement)**.
   - McNemar's exact two-sided binomial test on discordant pairs ($b=2, c=0$) yields **$p = 0.5000$**. The descriptive difference between models is not statistically significant.
3. **Reaffirmation of Benchmark C3 as Primary Reference:**
   - Because halving the learning rate ($\text{lr}=5\times 10^{-5}$) provided zero sensitivity benefit on held-out test data while increasing false alarms by $33\%$, **Candidate C3 remains the primary, most defensible benchmark architecture**.
4. **Resolution of Historical Metrics:**
   - Forensic tracing confirmed that the physical test prediction CSVs generated in Phase 5 yield ROC-AUC $0.8857$ and PR-AUC $0.5729$ for GIT 1, and ROC-AUC $0.8321$ and PR-AUC $0.4048$ for GIT 2.
   - Earlier citations of $0.8679$ / $0.6800$ were manual transcription errors in a hardcoded dictionary within `scripts/evaluate_phase_c.py`.
   - The Phase C C3 evaluation matches the newly computed Phase D8 locked evaluation **bit-for-bit** across all 47 predictions ($100.00\%$ concordance).

---

## 4. Fundamental Limitations & Safety Boundaries

1. **Small Positive Sample Size ($N_{\text{pos}} = 7$):**  
   The held-out test cohort contains only seven positive cases. A single patient classification shifts sensitivity by $\pm 14.29\%$, and exact 95% Clopper-Pearson bounds span from **$42.13\%$ to $99.64\%$**. No claims of definitive clinical superiority can be made from a cohort of this size.
2. **Internal Validation Only:**  
   This study represents an internal held-out test evaluation on retrospective data from a single medical center (Ayady Almostakbal Hospital, Alexandria, Egypt).
3. **Clinical Scope Boundary:**  
   The system is an automated radiologic triage research tool (`LOW_RISK` vs. `SUSPICIOUS_ACTIONABLE`), not a pathology diagnostic device. It does not replace tissue biopsy or radiological interpretation.
4. **Final Safety Gate Enforced:**  
   All evaluation decisions remain permanently frozen. No further tuning, training, or threshold sweeps have been performed.
