# Phase D2: Controlled Multimodal GIT Experiments Report (D2–D5)

**Phase:** Phase D2 — Controlled Hyperparameter & Loss Exploration  
**Execution Timestamp:** 2026-10-15T12:57:06Z  
**Hardware Platform:** NVIDIA GeForce MX550 (2048 MB VRAM), CUDA 12.x, PyTorch 2.5.1  
**Project Root:** `C:\Users\admin\Music\Breast Cancer1`  
**Held-Out Test Set Quarantine:** **100% Maintained (Zero Access to 47-Patient Test Cohort)**  

---

## 1. Executive Summary & Overview

Following the successful reproduction of the frozen Candidate C3 benchmark in Experiment D1 ($S=0.8351$), Phase D2 conducted four controlled, single-parameter experiments (D2–D5). All other architectural and training settings were held fixed:
- Patient cohorts: Training ($N=220$), Validation ($N=48$), Test ($N=47$, quarantined)
- Random seed: 42
- Feature cache: `data/features/patient_multimodal_features.pt`
- Architecture: `GraphInfusedTransformer` (1280-dim mammogram, 768-dim ultrasound, hidden 256, 2 GAT layers, 2 Transformer layers, $d_{ff}=1024$)
- Loss implementation: `ClassBalancedFocalLoss` with $\alpha=[1.0, 5.1111]$, reduction mean
- Predeclared checkpoint selection rule: $S = 0.5 \times \text{ROC-AUC} + 0.3 \times \text{PR-AUC} + 0.2 \times \text{F1}$ (tie-breaker: lower validation loss)
- Batch handling: Batch size 8, gradient accumulation 2 (effective batch size 16)
- Early stopping: Patience 8 epochs

### Key Findings:
1. **Candidate D5 ($\text{lr} = 5\times 10^{-5}$)** achieved the highest composite score ($S = 0.8468$, $+0.0117$ over C3), driven by improved sensitivity ($87.50\%$ vs $75.00\%$, $7/8$ vs $6/8$ positive cases detected) and F1 ($0.7778$ vs $0.7059$), while perfectly maintaining specificity at $92.50\%$ ($37/40$) and achieving the lowest validation loss ($0.1390$).
2. **Candidate D2 ($\gamma = 1.0$)** produced identical discrete classification decisions and ranking metrics ($S = 0.8351$, ROC-AUC $0.9281$, PR-AUC $0.7661$) at Epoch 4, but with higher validation loss ($0.2250$ vs $0.1679$), demonstrating that milder focal modulation produces equivalent ranking with larger absolute penalty values.
3. **Candidate D3 ($\gamma = 2.0$)** exhibited delayed checkpoint selection (Epoch 10) and significant training-set memorization (Train loss $0.0075$, Train acc $99.55\%$), resulting in degraded validation composite score ($S = 0.8276$, $-0.0075$), lower ROC-AUC ($0.9219$), and severe overfitting gap ($+0.2462$).
4. **Candidate D4 ($\text{Dropout} = 0.40$)** caused severe classification imbalance: while specificity reached $100.00\%$ ($40/40$), sensitivity collapsed to $50.00\%$ ($4/8$), missing two additional malignancies (4 false negatives vs 2 in C3). Balanced accuracy dropped sharply to $75.00\%$ ($-8.75\%$), and validation loss escalated to $0.5854$.
5. **Statistical Uncertainty:** The validation cohort contains only $N=8$ positive patients. The sensitivity difference between D5 ($87.5\%$) and C3 ($75.0\%$) reflects **a single patient** (`00247y21`, BI-RADS 6, whose predicted probability moved from $0.4531$ to $0.5034$). Therefore, D5 should not be declared an undisputed clinical breakthrough based solely on validation numbers.

---

## 2. Comprehensive Comparative Results Table

| Metric / Dimension | Frozen C3 / D1 Repro | Exp D2 ($\gamma=1.0$) | Exp D3 ($\gamma=2.0$) | Exp D4 ($\text{Dropout}=0.4$) | Exp D5 ($\text{LR}=5\times 10^{-5}$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Parameter Varied** | **C3 Baseline** | $\gamma = 1.0$ | $\gamma = 2.0$ | $\text{Dropout} = 0.40$ | $\text{LR} = 5 \times 10^{-5}$ |
| **Selected Epoch** | **Epoch 4** | **Epoch 4** | Epoch 10 | Epoch 18 | **Epoch 4** |
| **Predeclared Composite Score ($S$)** | **0.8351** | **0.8351** | 0.8276 | 0.8334 | **0.8468** |
| **$\Delta$ Composite Score vs. C3** | — | $0.0000$ | $-0.0075$ | $-0.0017$ | **$+0.0117$** |
| **Validation ROC-AUC** | **0.9281** | **0.9281** | 0.9219 | **0.9313** | **0.9281** |
| **$\Delta$ ROC-AUC vs. C3** | — | $0.0000$ | $-0.0062$ | $+0.0032$ | $0.0000$ |
| **Validation PR-AUC** | **0.7661** | **0.7661** | 0.7516 | **0.7816** | 0.7573 |
| **$\Delta$ PR-AUC vs. C3** | — | $0.0000$ | $-0.0145$ | $+0.0155$ | $-0.0088$ |
| **Validation Sensitivity (Recall)** | **75.00%** (6/8) | **75.00%** (6/8) | 75.00% (6/8) | 50.00% (4/8) | **87.50%** (7/8) |
| **$\Delta$ Sensitivity vs. C3** | — | $0.00\%$ | $0.00\%$ | **$-25.00\%$** | **$+12.50\%$** (+1 TP) |
| **Validation Specificity** | **92.50%** (37/40) | **92.50%** (37/40) | 92.50% (37/40) | **100.00%** (40/40) | **92.50%** (37/40) |
| **$\Delta$ Specificity vs. C3** | — | $0.00\%$ | $0.00\%$ | $+7.50\%$ | $0.00\%$ |
| **Validation Precision** | 66.67% (6/9) | 66.67% (6/9) | 66.67% (6/9) | **100.00%** (4/4) | **70.00%** (7/10) |
| **$\Delta$ Precision vs. C3** | — | $0.00\%$ | $0.00\%$ | $+33.33\%$ | **$+3.33\%$** |
| **Validation F1-Score** | **0.7059** | **0.7059** | 0.7059 | 0.6667 | **0.7778** |
| **$\Delta$ F1-Score vs. C3** | — | $0.0000$ | $0.0000$ | $-0.0392$ | **$+0.0719$** |
| **Validation Balanced Accuracy** | **83.75%** | **83.75%** | 83.75% | 75.00% | **90.00%** |
| **$\Delta$ Balanced Accuracy vs. C3** | — | $0.00\%$ | $0.00\%$ | **$-8.75\%$** | **$+6.25\%$** |
| **Validation Accuracy** | 89.58% (43/48) | 89.58% (43/48) | 89.58% (43/48) | 91.67% (44/48) | 91.67% (44/48) |
| **Validation Loss** | 0.1679 | 0.2250 | 0.2537 | 0.5854 | **0.1390** |
| **$\Delta$ Val Loss vs. C3** | — | $+0.0571$ | $+0.0858$ | $+0.4175$ | **$-0.0289$** |
| **Confusion Matrix `[TN, FP, FN, TP]`** | `[37, 3, 2, 6]` | `[37, 3, 2, 6]` | `[37, 3, 2, 6]` | `[40, 0, 4, 4]` | `[37, 3, 1, 7]` |
| **Train Loss at Best Epoch** | 0.1590 | 0.2026 | 0.0075 | 0.0068 | 0.1585 |
| **Training Accuracy at Best Epoch** | 90.91% | 91.82% | 99.55% | 99.55% | 88.18% |
| **Overfitting Gap (Val - Train)** | +0.0089 | +0.0224 | **+0.2462 (Severe)** | **+0.5786 (Severe)** | **-0.0195 (Optimal)** |
| **Stopping Epoch (Patience=8)** | Epoch 12 | Epoch 12 | Epoch 18 | Epoch 20 (Max) | Epoch 12 |
| **Total Wall-Clock Runtime** | 48.40s | 44.51s | 64.11s | 70.84s | 71.95s |
| **Peak GPU VRAM (MB)** | 105.52 MB | 105.52 MB | 105.52 MB | 105.52 MB | 105.52 MB |

---

## 3. In-Depth Experiment-by-Experiment Analysis

### 3.1 Experiment D2: Focal Loss $\gamma = 1.0$
- **Hypothesis:** A milder focusing exponent ($\gamma = 1.0$ vs $1.5$) provides less aggressive down-weighting of easy background negatives, potentially stabilizing calibration.
- **Observed Behavior:**
  - Selected at Epoch 4, matching C3 exactly.
  - All discrete predictions and discrimination metrics are identical to C3: ROC-AUC $0.9281$, PR-AUC $0.7661$, F1 $0.7059$, Sensitivity $75.00\%$, Specificity $92.50\%$, and Composite Score $0.8351$.
  - Validation loss is higher ($0.2250$ vs $0.1679$). Because $(1 - p_t)^{1.0} \ge (1 - p_t)^{1.5}$, the absolute numerical scale of the loss is larger across all samples.
- **Conclusion:** Lowering $\gamma$ from $1.5$ to $1.0$ has negligible effect on rank ordering or classification boundaries, but yields higher unnormalized cross-entropy penalties.

### 3.2 Experiment D3: Focal Loss $\gamma = 2.0$
- **Hypothesis:** Stronger focusing ($\gamma = 2.0$) heavily suppresses easy samples, forcing the network to concentrate strictly on ambiguous boundary cases.
- **Observed Behavior:**
  - Checkpoint selection was delayed to Epoch 10.
  - By Epoch 10, the model had extensively overfitted the training partition: Training loss dropped to $0.0075$ with $99.55\%$ training accuracy ($219/220$ correct).
  - Validation metrics degraded: Composite score dropped to $0.8276$ ($-0.0075$), ROC-AUC decreased to $0.9219$, and PR-AUC declined to $0.7516$.
  - The overfitting loss gap blew up to $+0.2462$ ($\text{Val}=0.2537$ vs $\text{Train}=0.0075$).
- **Conclusion:** For this sample size ($N=220$ train), $\gamma = 2.0$ over-suppresses easy gradients too early, encouraging over-parameterized memorization of difficult training examples at the expense of validation generalizability.

### 3.3 Experiment D4: Dropout Rate $= 0.40$
- **Hypothesis:** Increasing dropout from $0.30$ to $0.40$ across GAT and Transformer layers will combat memorization and regularize representations.
- **Observed Behavior:**
  - While ROC-AUC ($0.9313$) and PR-AUC ($0.7816$) slightly increased on rank-ordering alone, the calibrated decision probabilities collapsed toward zero.
  - At the default $0.50$ threshold, the model became hyper-conservative: Specificity was a perfect $100.00\%$ ($40/40$), but Sensitivity halved to **$50.00\%$** ($4/8$), missing 4 out of 8 cancer cases.
  - Balanced Accuracy fell to $75.00\%$ ($-8.75\%$), F1 fell to $0.6667$, and validation loss inflated to $0.5854$ ($+0.4175$).
  - Training continued until the maximum budget (Epoch 20) with severe loss divergence ($+0.5786$).
- **Conclusion:** Over-regularization with $40\%$ dropout suppresses confidence on positive samples. In an oncology screening context where false negatives are catastrophic, D4's trade-off is fundamentally unacceptable.

### 3.4 Experiment D5: Learning Rate $= 5 \times 10^{-5}$
- **Hypothesis:** Halving the AdamW learning rate from $10^{-4}$ to $5\times 10^{-5}$ allows more refined feature adaptation and smoother convergence along the multimodal manifold.
- **Observed Behavior:**
  - Selected cleanly at Epoch 4, matching the early-convergence profile of C3.
  - **Highest Composite Score:** $0.8468$ ($+0.0117$ over C3).
  - **Sensitivity Improvement:** Increased from $75.00\%$ ($6/8$) to **$87.50\%$** ($7/8$), reducing false negatives from 2 to 1.
  - **Maintained High Specificity:** Kept identical at **$92.50\%$** ($37/40$), with the exact same three false positives (`0800y07`, `1724y16`, `1780y07`).
  - **F1 and Balanced Accuracy:** F1 reached $0.7778$ ($+0.0719$), and Balanced Accuracy reached $90.00\%$ ($+6.25\%$).
  - **Optimal Loss Alignment:** Lowest validation loss ($0.1390$) and a healthy, non-overfitting gap ($-0.0195$, $\text{Val}=0.1390 \le \text{Train}=0.1585$).
- **Patient-Level Transition:** The sensitivity improvement stems from patient `00247y21` (confirmed BI-RADS 6). In C3/D1, its predicted probability was $0.4531$ (false negative at threshold 0.50). In D5, the refined gradient steps calibrated this patient's probability to $0.5034$, correctly classifying it as positive without destabilizing any negative cases.
- **Statistical Humility:** Because $N_{\text{pos}}=8$, this $+12.50\%$ sensitivity leap corresponds to **exactly one patient**. It represents a promising candidate for balanced improvement, but cannot be treated as statistically conclusive on validation data alone.

---

## 4. Hardware and Computational Profiling

| Metric | Measurement across D2–D5 | Capacity / Threshold | Headroom |
| :--- | :---: | :---: | :---: |
| **Peak GPU VRAM** | **105.52 MB** | 2048 MB (GeForce MX550) | **94.8% free memory** |
| **Average Run Time** | **62.85 seconds** (1.05 min) | < 10 minutes | **9.5x faster than limit** |
| **Cumulative D2–D5 Time** | **251.41 seconds** (4.19 min) | — | Highly feasible |
| **Thermal / Memory Stability** | 100% stable, no OOM, AMP active | Safe operating limits | Zero hardware risk |

---

## 5. Artifact Directory Inventory

All Phase D2 outputs have been saved to their dedicated directories:
- **D1:** [`outputs/phase_d/experiments/exp_d1_c3_repro/`](file:///c:/Users/admin/Music/Breast%20Cancer1/outputs/phase_d/experiments/exp_d1_c3_repro/)
- **D2:** [`outputs/phase_d/experiments/exp_d2_focal_gamma1_0/`](file:///c:/Users/admin/Music/Breast%20Cancer1/outputs/phase_d/experiments/exp_d2_focal_gamma1_0/)
- **D3:** [`outputs/phase_d/experiments/exp_d3_focal_gamma2_0/`](file:///c:/Users/admin/Music/Breast%20Cancer1/outputs/phase_d/experiments/exp_d3_focal_gamma2_0/)
- **D4:** [`outputs/phase_d/experiments/exp_d4_dropout0_4/`](file:///c:/Users/admin/Music/Breast%20Cancer1/outputs/phase_d/experiments/exp_d4_dropout0_4/)
- **D5:** [`outputs/phase_d/experiments/exp_d5_lr5e-5/`](file:///c:/Users/admin/Music/Breast%20Cancer1/outputs/phase_d/experiments/exp_d5_lr5e-5/)
- **Comparative Table (CSV):** [`outputs/phase_d/tables/phase_d2_d5_comparison.csv`](file:///c:/Users/admin/Music/Breast%20Cancer1/outputs/phase_d/tables/phase_d2_d5_comparison.csv)
- **Comparative Table (MD):** [`outputs/phase_d/tables/phase_d2_d5_comparison.md`](file:///c:/Users/admin/Music/Breast%20Cancer1/outputs/phase_d/tables/phase_d2_d5_comparison.md)

---

## 6. Scientific Governance & Gatekeeping

Experiments D2–D5 are concluded. As instructed:
1. No multi-seed repeats have been run.
2. No thresholds were adjusted from the canonical $0.50$ baseline.
3. No hyperparameters were combined.
4. The held-out 47-patient test cohort remains strictly quarantined and unaccessed.
5. All further experiments are paused pending user review.
