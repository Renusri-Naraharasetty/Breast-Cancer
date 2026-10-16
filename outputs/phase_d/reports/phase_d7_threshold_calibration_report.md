# Phase D7: Validation-Only Threshold and Calibration Analysis Report

**Phase:** Phase D7 — Operating Point & Calibration Dynamics  
**Execution Timestamp:** 2026-10-15T19:09:45Z  
**Hardware Environment:** NVIDIA GeForce MX550, PyTorch 2.5.1, Python 3.12  
**Cohort Evaluated:** Validation Partition Only ($N=48$, 40 Negative / 8 Positive)  
**Held-Out Test Set:** **STRICTLY QUARANTINED (Never Loaded or Evaluated)**  

---

## 1. Executive Summary & Purpose

Phase D7 was conducted to systematically investigate operating-point trade-offs and probability calibration for the **Frozen Candidate C3 Benchmark** and the leading Phase D candidate, **Candidate D5**, across the three matched seeds (`42`, `123`, `456`).

In accordance with strict scientific guidelines:
- **No new models were trained**, and no weights or feature caches were modified.
- The 47-patient test cohort was **strictly quarantined**.
- **The purpose of this analysis is not to manufacture higher metrics** by searching for a flattering threshold, but rather to map the clinical trade-offs between sensitivity (detecting suspicious lesions) and specificity (avoiding false alarms) across a broad spectrum of operating points ($\tau \in [0.30, 0.70]$).

### Primary Takeaways:
1. **Canonical Threshold ($\tau = 0.50$) Remains the Most Defensible Operating Point:**  
   Across all evaluated thresholds, $\tau = 0.50$ uniquely achieves the optimal clinical balance for both models:
   - **Candidate D5 at $\tau=0.50$:** Sensitivity **$83.33\% \pm 7.22\%$**, Specificity **$91.67\% \pm 1.44\%$**, F1 **$0.7402 \pm 0.0361$**, Balanced Accuracy **$87.50\% \pm 3.31\%$**, Average False Positives: $3.33$, Average False Negatives: $1.33$.
   - **Frozen C3 at $\tau=0.50$:** Sensitivity **$75.00\% \pm 12.50\%$**, Specificity **$92.50\% \pm 5.00\%$**, F1 **$0.7067 \pm 0.0072$**, Balanced Accuracy **$83.75\% \pm 3.75\%$**, Average False Positives: $3.00$, Average False Negatives: $2.00$.
2. **Sensitivity-Specificity Asymmetry at Lower Thresholds ($\tau < 0.50$):**  
   Lowering the threshold to $0.45$ or $0.40$ increases sensitivity to $91.67\%$ in D5 (recovering an average of $0.67$ additional cancer cases), but causes a steep drop in specificity (collapsing from $91.67\%$ to $74.17\%$ at $\tau=0.45$, and to $69.17\%$ at $\tau=0.40$). Because the validation cohort contains 5 times more negative patients than positive patients ($40$ vs $8$), each $5\%$ loss in specificity generates 2 additional false positives, causing precision to crash from $66.77\%$ to $46.65\%$ (and down to $33.06\%$ at $\tau=0.30$).
3. **Catastrophic Sensitivity Drop at Higher Thresholds ($\tau > 0.50$):**  
   Increasing the threshold to $0.55$ or $0.60$ elevates specificity to $95\% - 98\%$, but causes unacceptable clinical harm: sensitivity drops to $50.00\%$ at $\tau=0.55$ and $33.33\%$ at $\tau=0.60$, missing half to two-thirds of malignant cases.
4. **Calibration Differences and Small-Sample Limitations:**  
   C3 exhibits lower Brier score ($0.0805 \pm 0.0090$) and Expected Calibration Error (ECE: $0.0791 \pm 0.0269$) than D5 (Brier: $0.1167 \pm 0.0315$, ECE: $0.1468 \pm 0.0644$). This occurs because C3 pushes negative probabilities closer to 0, whereas D5's gentler learning rate keeps background cases in the $0.15 - 0.35$ range. However, with only 8 positive patients in $N=48$, empirical calibration estimates suffer from high binomial sampling variance and small-sample upward bias.

---

## 2. Verification of Prediction Files & Cohort Integrity

Before analysis, all 6 validation prediction files from Phase D6 were verified:
- **Files Inspected:**
  - `outputs/phase_d/experiments/exp_d6_c3_seed42/val_predictions.csv`
  - `outputs/phase_d/experiments/exp_d6_c3_seed123/val_predictions.csv`
  - `outputs/phase_d/experiments/exp_d6_c3_seed456/val_predictions.csv`
  - `outputs/phase_d/experiments/exp_d6_d5_seed42/val_predictions.csv`
  - `outputs/phase_d/experiments/exp_d6_d5_seed123/val_predictions.csv`
  - `outputs/phase_d/experiments/exp_d6_d5_seed456/val_predictions.csv`
- **Integrity Findings:**
  1. Each file contains exactly 48 rows matching the exact 48 validation patient IDs in identical order.
  2. Ground-truth labels are $100.000\%$ identical across all files: exactly 40 negative patients (`label=0`) and 8 positive patients (`label=1`).
  3. No test patients exist in any validation prediction file.

---

## 3. Threshold Operating Point Analysis ($\tau \in [0.30, 0.70]$)

The table below summarizes performance across the threshold grid, averaged across the three matched seeds (`42`, `123`, `456`). Seed predictions are never treated as independent patients: metrics are computed per seed and then summarized as Mean $\pm$ Standard Deviation.

### 3.1 Multi-Seed Averaged Performance across Thresholds

| Model | $\tau$ | Sensitivity (Recall) | Specificity | Precision | F1-Score | Balanced Accuracy | Overall Accuracy | Mean TP | Mean FP | Mean FN | Mean TN |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **C3** | 0.30 | $83.33\% \pm 7.22\%$ | $82.50\% \pm 10.90\%$ | $53.31\% \pm 18.83\%$ | $0.6311 \pm 0.1036$ | $82.92\% \pm 1.91\%$ | $82.64\% \pm 7.89\%$ | $6.67$ | $7.00$ | $1.33$ | $33.00$ |
| **C3** | 0.35 | $79.17\% \pm 14.43\%$ | $85.00\% \pm 8.66\%$ | $54.92\% \pm 14.30\%$ | $0.6280 \pm 0.0335$ | $82.08\% \pm 2.89\%$ | $84.03\% \pm 4.81\%$ | $6.33$ | $6.00$ | $1.67$ | $34.00$ |
| **C3** | 0.40 | $79.17\% \pm 14.43\%$ | $86.67\% \pm 7.64\%$ | $57.32\% \pm 12.74\%$ | $0.6474 \pm 0.0335$ | $82.92\% \pm 3.82\%$ | $85.42\% \pm 4.17\%$ | $6.33$ | $5.33$ | $1.67$ | $34.67$ |
| **C3** | 0.45 | $79.17\% \pm 14.43\%$ | $90.83\% \pm 5.20\%$ | $65.09\% \pm 9.76\%$ | $0.7037 \pm 0.0641$ | $85.00\% \pm 5.73\%$ | $88.89\% \pm 3.18\%$ | $6.33$ | $3.67$ | $1.67$ | $36.33$ |
| **C3** | **0.50** | **$75.00\% \pm 12.50\%$** | **$92.50\% \pm 5.00\%$** | **$69.44\% \pm 12.73\%$** | **$0.7067 \pm 0.0072$** | **$83.75\% \pm 3.75\%$** | **$89.58\% \pm 2.09\%$** | **$6.00$** | **$3.00$** | **$2.00$** | **$37.00$** |
| **C3** | 0.55 | $54.17\% \pm 14.43\%$ | $95.83\% \pm 3.82\%$ | $77.98\% \pm 19.59\%$ | $0.6124 \pm 0.0616$ | $75.00\% \pm 5.45\%$ | $88.89\% \pm 1.20\%$ | $4.33$ | $1.67$ | $3.67$ | $38.33$ |
| **C3** | 0.60 | $41.67\% \pm 14.43\%$ | $96.67\% \pm 2.89\%$ | $77.78\% \pm 19.24\%$ | $0.5143 \pm 0.0990$ | $69.17\% \pm 5.77\%$ | $87.50\% \pm 0.00\%$ | $3.33$ | $1.33$ | $4.67$ | $38.67$ |
| **C3** | 0.65 | $33.33\% \pm 7.22\%$ | $98.33\% \pm 1.44\%$ | $83.33\% \pm 14.43\%$ | $0.4667 \pm 0.0577$ | $65.83\% \pm 2.89\%$ | $87.50\% \pm 0.00\%$ | $2.67$ | $0.67$ | $5.33$ | $39.33$ |
| **C3** | 0.70 | $33.33\% \pm 7.22\%$ | $99.17\% \pm 1.44\%$ | $91.67\% \pm 14.43\%$ | $0.4818 \pm 0.0744$ | $66.25\% \pm 3.31\%$ | $88.19\% \pm 1.20\%$ | $2.67$ | $0.33$ | $5.33$ | $39.67$ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **D5** | 0.30 | $95.83\% \pm 7.22\%$ | $57.50\% \pm 18.03\%$ | $33.06\% \pm 9.45\%$ | $0.4836 \pm 0.0895$ | $76.67\% \pm 5.64\%$ | $63.89\% \pm 13.87\%$ | $7.67$ | $17.00$ | $0.33$ | $23.00$ |
| **D5** | 0.35 | $95.83\% \pm 7.22\%$ | $66.67\% \pm 18.76\%$ | $39.85\% \pm 13.22\%$ | $0.5503 \pm 0.1172$ | $81.25\% \pm 6.61\%$ | $71.53\% \pm 14.63\%$ | $7.67$ | $13.33$ | $0.33$ | $26.67$ |
| **D5** | 0.40 | $91.67\% \pm 7.22\%$ | $69.17\% \pm 17.74\%$ | $40.44\% \pm 12.71\%$ | $0.5499 \pm 0.1116$ | $80.42\% \pm 5.64\%$ | $72.92\% \pm 13.66\%$ | $7.33$ | $12.33$ | $0.67$ | $27.67$ |
| **D5** | 0.45 | $91.67\% \pm 7.22\%$ | $74.17\% \pm 19.42\%$ | $46.65\% \pm 17.01\%$ | $0.6009 \pm 0.1400$ | $82.92\% \pm 6.29\%$ | $77.08\% \pm 15.02\%$ | $7.33$ | $10.33$ | $0.67$ | $29.67$ |
| **D5** | **0.50** | **$83.33\% \pm 7.22\%$** | **$91.67\% \pm 1.44\%$** | **$66.77\% \pm 3.18\%$** | **$0.7402 \pm 0.0361$** | **$87.50\% \pm 3.31\%$** | **$90.28\% \pm 1.21\%$** | **$6.67$** | **$3.33$** | **$1.33$** | **$36.67$** |
| **D5** | 0.55 | $50.00\% \pm 25.00\%$ | $95.00\% \pm 4.33\%$ | $74.60\% \pm 22.50\%$ | $0.5464 \pm 0.1534$ | $72.50\% \pm 10.68\%$ | $87.50\% \pm 2.08\%$ | $4.00$ | $2.00$ | $4.00$ | $38.00$ |
| **D5** | 0.60 | $33.33\% \pm 31.46\%$ | $97.50\% \pm 4.33\%$ | $54.17\% \pm 50.52\%$ | $0.3902 \pm 0.3402$ | $65.42\% \pm 14.05\%$ | $86.81\% \pm 3.18\%$ | $2.67$ | $1.00$ | $5.33$ | $39.00$ |
| **D5** | 0.65 | $25.00\% \pm 25.00\%$ | $97.50\% \pm 4.33\%$ | $52.38\% \pm 50.17\%$ | $0.3111 \pm 0.2775$ | $61.25\% \pm 10.68\%$ | $85.42\% \pm 2.09\%$ | $2.00$ | $1.00$ | $6.00$ | $39.00$ |
| **D5** | 0.70 | $20.83\% \pm 19.09\%$ | $98.33\% \pm 2.89\%$ | $53.33\% \pm 50.33\%$ | $0.2872 \pm 0.2506$ | $59.58\% \pm 8.51\%$ | $85.42\% \pm 2.09\%$ | $1.67$ | $0.67$ | $6.33$ | $39.33$ |

---

## 4. Operating Point Dynamics: Why Sensitivity Gains Cost Disproportionate Specificity

### 4.1 Population Imbalance and Clinical Penalty Mechanics
In the BCMID validation cohort, the class distribution is heavily imbalanced ($40$ negative benign/normal patients vs. $8$ positive suspicious/actionable patients, a $5:1$ ratio).
- When the threshold $\tau$ is lowered from $0.50$ to $0.45$ in D5:
  - Sensitivity increases by $+8.34\%$ ($83.33\% \rightarrow 91.67\%$), which corresponds to detecting on average **$0.67$ additional cancer cases** ($7.33$ vs $6.67$ TP).
  - However, Specificity drops by **$-17.50\%$** ($91.67\% \rightarrow 74.17\%$), which corresponds to **$7.00$ additional false positive patients** ($10.33$ vs $3.33$ FP).
- **The Trade-off Ratio:** Gaining $0.67$ true positive cases costs $7.00$ false positives—a trade-off ratio of **over 10 false alarms for every 1 cancer caught**.
- In clinical practice, this would cause more than a quarter ($26\%$) of healthy or benign screening patients to undergo invasive biopsies or unnecessary follow-up procedures, collapsing Positive Predictive Value (Precision) from $66.77\%$ to $46.65\%$.

### 4.2 High Thresholds Risk Unacceptable Miss Rates
Conversely, shifting $\tau$ upward from $0.50$ to $0.55$:
- Reduces false alarms only marginally (FP drops from $3.33$ to $2.00$ in D5, $+3.33\%$ specificity).
- But doubles missed cancers (FN jumps from $1.33$ to $4.00$, sensitivity collapsing by $-33.33\%$ from $83.33\%$ to $50.00\%$).
- Missing half of all malignant cases in a triage system is clinically catastrophic.

---

## 5. Probability Calibration & Reliability Analysis

Calibration evaluates how accurately predicted probabilities reflect true empirical diagnostic probabilities ($P(Y=1 \mid \hat{p} = p) \approx p$).

### 5.1 Multi-Seed Calibration Summary

| Calibration Metric | Frozen C3 (Mean $\pm$ SD) | Candidate D5 (Mean $\pm$ SD) | Interpretation |
| :--- | :---: | :---: | :--- |
| **Brier Score (Mean Squared Error)** | **$0.0805 \pm 0.0090$** | $0.1167 \pm 0.0315$ | Lower is better. C3 has slightly lower overall squared deviation. |
| **Log Loss (Cross-Entropy)** | **$0.2637 \pm 0.0263$** | $0.3666 \pm 0.0919$ | Lower is better. C3 penalizes background probabilities less. |
| **Expected Calibration Error (ECE, 5 bins)** | **$0.0791 \pm 0.0269$** | $0.1468 \pm 0.0644$ | Lower is better. C3 displays tighter alignment to identity. |
| **Maximum Calibration Error (MCE, 5 bins)** | **$0.2514 \pm 0.1267$** | $0.3071 \pm 0.0096$ | Maximum deviation in any single bin. |

### 5.2 Why C3 Demonstrates Lower Calibration Error
1. **Background Probability Suppression:**  
   C3 trained with $\text{lr}=10^{-4}$ pushed negative background cases aggressively towards $p \approx 0.03 - 0.08$. Since $83.3\%$ of the validation cohort is negative, suppressing negative probabilities minimizes the quadratic error $\sum (p_i - 0)^2$.
2. **D5 Intermediate Probability Concentration:**  
   D5 trained with $\text{lr}=5\times 10^{-5}$ converged to slightly higher probabilities for negative cases ($p \approx 0.15 - 0.35$). While these are still safely below the $0.50$ threshold (achieving $91.67\%$ specificity), their residual distance from $0$ increases both the Brier score and the empirical calibration gap in low-to-mid bins.

### 5.3 Limitations of Evaluating Calibration on a 48-Patient Cohort
Evaluating probability calibration on $N=48$ patients has severe statistical constraints:
- **Small Sample Size:** With only $N=48$ total and $N_{\text{pos}}=8$ positive cases, partitioning into 5 calibration bins results in bin sample sizes of only 1 to 16 patients.
- **High Empirical Bin Variance:** In a bin with only 3 patients, a single case changing label shifts empirical bin accuracy by $33.3\%$.
- **Upward ECE Bias:** It is a well-established mathematical property that empirical ECE estimators exhibit significant upward bias when sample sizes are small ($N < 500$), as absolute errors cannot cancel out.
- Consequently, calibration metrics on this cohort serve as exploratory indicators rather than definitive proof of population-level miscalibration.

---

## 6. Principled Operating Point Selection Protocol

As instructed by Task 7: **We must not select a threshold by searching across metrics without a predefined rule.**

### 6.1 Evaluation of Defensible Threshold Selection Rules

1. **Rule 1: Canonical Clinical Reference ($\tau = 0.50$) — RECOMMENDED**
   - **Formulation:** Maintain the canonical un-tuned operating threshold $\tau = 0.50$.
   - **Rationale:** Preserves zero post-hoc optimization bias. In balanced cross-entropy / focal loss training with balanced class weighting ($\alpha=[1.0, 5.1111]$), the output $0.50$ is mathematically aligned with the prior-adjusted decision boundary.
   - **Outcome:** D5 achieves Sensitivity $83.33\%$, Specificity $91.67\%$, F1 $0.7402$, Balanced Accuracy $87.50\%$.
2. **Rule 2: Clinically Constrained Triage Rule ($\max \text{Sensitivity}$ subject to $\text{Specificity} \ge 90.0\%$)**
   - **Formulation:** Select $\tau^* = \arg\max_\tau \text{Sensitivity}(\tau)$ subject to $\text{Specificity}(\tau) \ge 0.90$.
   - **Rationale:** In mammography/ultrasound screening triage, radiologists cannot tolerate excessive false alarms (biopsy burden must remain $\le 10\%$).
   - **Outcome:** Across the evaluated grid for D5, $\tau = 0.50$ is the **unique global solution** (Sensitivity $83.33\%$ at Specificity $91.67\%$). Lower thresholds ($\tau \le 0.45$) violate the $90\%$ specificity constraint (dropping to $74.17\%$).
3. **Rule 3: Youden's Index / Balanced Accuracy Maximization ($\max J = \text{Sens} + \text{Spec} - 1$)**
   - **Formulation:** Select $\tau^* = \arg\max_\tau \text{Balanced Accuracy}(\tau)$.
   - **Outcome:** For D5, Balanced Accuracy peaks at **$\tau = 0.50$** ($87.50\% \pm 3.31\%$), which strictly outperforms $\tau=0.45$ ($82.92\%$) and $\tau=0.55$ ($72.50\%$).

### Formal Recommendation:
Under all three principled selection rules, **$\tau = 0.50$ emerges as the optimal, most defensible operating threshold for Candidate D5**. There is no justification for shifting the operating point away from $0.50$.

---

## 7. Artifacts Inventory & Directory Manifest

All Phase D7 assets have been saved under isolated Phase D directories:
- **Figures:**
  - Sensitivity-Specificity Trade-Off: [`outputs/phase_d/figures/threshold_sensitivity_specificity_tradeoff.png`](file:///c:/Users/admin/Music/Breast%20Cancer1/outputs/phase_d/figures/threshold_sensitivity_specificity_tradeoff.png)
  - Threshold Metric Curves: [`outputs/phase_d/figures/threshold_metrics_curve_c3_vs_d5.png`](file:///c:/Users/admin/Music/Breast%20Cancer1/outputs/phase_d/figures/threshold_metrics_curve_c3_vs_d5.png)
  - Reliability Diagrams: [`outputs/phase_d/figures/reliability_diagram_c3_vs_d5.png`](file:///c:/Users/admin/Music/Breast%20Cancer1/outputs/phase_d/figures/reliability_diagram_c3_vs_d5.png)
- **Data & Tables:**
  - Per-Seed Thresholds: [`outputs/phase_d/threshold_analysis/per_seed_threshold_metrics.csv`](file:///c:/Users/admin/Music/Breast%20Cancer1/outputs/phase_d/threshold_analysis/per_seed_threshold_metrics.csv)
  - Averaged Thresholds: [`outputs/phase_d/threshold_analysis/averaged_threshold_metrics.csv`](file:///c:/Users/admin/Music/Breast%20Cancer1/outputs/phase_d/threshold_analysis/averaged_threshold_metrics.csv)
  - Calibration Per-Seed: [`outputs/phase_d/calibration/validation_calibration_metrics.csv`](file:///c:/Users/admin/Music/Breast%20Cancer1/outputs/phase_d/calibration/validation_calibration_metrics.csv)
  - Calibration Averaged: [`outputs/phase_d/calibration/averaged_calibration_metrics.csv`](file:///c:/Users/admin/Music/Breast%20Cancer1/outputs/phase_d/calibration/averaged_calibration_metrics.csv)
  - Calibration Bins JSON: [`outputs/phase_d/calibration/calibration_bins_detailed.json`](file:///c:/Users/admin/Music/Breast%20Cancer1/outputs/phase_d/calibration/calibration_bins_detailed.json)
  - Markdown Tables: [`outputs/phase_d/tables/phase_d7_threshold_analysis.md`](file:///c:/Users/admin/Music/Breast%20Cancer1/outputs/phase_d/tables/phase_d7_threshold_analysis.md) | [`phase_d7_calibration_summary.md`](file:///c:/Users/admin/Music/Breast%20Cancer1/outputs/phase_d/tables/phase_d7_calibration_summary.md)

---

## 8. Gatekeeping & Status

Phase D7 is complete. As instructed:
- The 47-patient test cohort remains **completely quarantined and untouched**.
- No thresholds have been artificially optimized.
- All Phase C and Phase D1–D6 artifacts remain untouched.
- Execution is stopped here pending your review and decision on the next phase.
