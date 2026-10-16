# Phase D0 — Comprehensive Pre-Experiment Research Audit Report

**Date:** October 15, 2026  
**Auditor:** Antigravity AI Research Team  
**Scope:** Repository integrity, C3 reference benchmark verification, split reconciliation (311 vs. 314 vs. 315), feature cache provenance, loss formulations, and Phase D experimental protocol.  
**Non-Negotiable Enforcements:** Zero modifications to existing Phase C checkpoints or reports; complete quarantine of the 47-patient held-out test cohort.

---

## 1. Executive Summary & Audit Verdict

1. **Audit Verdict:** **PASSED WITH ZERO UNRESOLVED BLOCKERS**. The data splits, feature caches, labels, architecture, and loss implementations are fully verified, deterministic, and free of leakage.
2. **Reconciliation of Cohort Counts (311 vs. 314 vs. 315):**
   - **332 Folders Total:** Exist in raw directory `BCMID/` and `BCMID_annotations.tsv`.
   - **17 Folders Excluded:** Excluded due to missing/indeterminate diagnostic categories: `NONE` ($N=11$), `INDETERMINATE` ($N=5$), or `Category 0` ($N=1$).
   - **311 Unique Root Patients:** Exactly $311$ unique biological patients form the eligible dataset.
   - **315 Patient Folders:** Exactly $4$ root patients underwent multiple examinations (bilateral or longitudinal visits), resulting in $311 + 4 = 315$ total patient folders.
   - **The "314" Discrepancy Resolved:** The $314$ figure in earlier reports was an **arithmetic typographical error** in a table summary row ($220 + 48 + 47 = 315$, mistakenly summed as $314$). The underlying code and dataloaders always loaded all $315$ patient folders.
3. **Reference Benchmark C3 Verification:**
   - Checkpoint verified at: `outputs/experiments/phase_c/exp_c3_git1_focal_gamma1_5/best_checkpoint.pth` (Epoch 4).
   - Validation composite score: **$0.8351$** ($\text{ROC-AUC}=0.9281, \text{PR-AUC}=0.7661, \text{F1}=0.7059, \text{Sens}=75.0\%, \text{Spec}=92.5\%$).
   - Generalization loss gap at selection: **$+0.0089$** ($\text{Train Loss}=0.1590, \text{Val Loss}=0.1679$).
4. **Test Set Quarantine:**
   - The 47-patient test cohort is completely isolated. No test labels or predictions are used in Phase D development or model selection.

---

## 2. Cohort Counts & Split Integrity Audit

### 2.1 Complete Cohort Hierarchy
$$\begin{aligned}
\text{Raw Patient Folders} &= 332 \\
\text{Excluded Folders (Non-diagnostic BI-RADS)} &= 17 \\
\text{Eligible Patient Folders} &= 315 \\
\text{Eligible Unique Root Patients} &= 311 \\
\text{Repeat-Visit Folders} &= 4 \quad (\text{Root Patients: } \texttt{01566y21}, \texttt{04412y22}, \texttt{03128y21}, \texttt{05128y22})
\end{aligned}$$

### 2.2 Split Breakdown & Patient-Level Separation
Splitting was executed at the **root patient level** to ensure that multiple folders for the same patient remain strictly within the same partition:

| Partition | Root Patients | Patient Folders | Class 0 (`LOW_RISK`) | Class 1 (`SUSPICIOUS`) | Class Ratio (Neg/Pos) | Image Count (MG / US) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Training** | 217 | **220** | 184 (83.6%) | 36 (16.4%) | 5.1111 | 644 / 671 |
| **Validation** | 47 | **48** | 40 (83.3%) | 8 (16.7%) | 5.0000 | 138 / 135 |
| **Held-Out Test (Frozen)** | 47 | **47** | 40 (85.1%) | 7 (14.9%) | 5.7143 | 132 / 150 |
| **Total** | **311** | **315** | **264** | **51** | — | **914 / 956** |

- **Leakage Audit:**
  - $\text{Train} \cap \text{Val} = \emptyset$ (0 overlapping root patients or folders).
  - $\text{Train} \cap \text{Test} = \emptyset$ (0 overlapping root patients or folders).
  - $\text{Val} \cap \text{Test} = \emptyset$ (0 overlapping root patients or folders).
  - 100% strictly disjoint separation confirmed.

---

## 3. Multimodal Feature Cache Audit

- **File Path:** `data/features/patient_multimodal_features.pt`
- **Metadata:** `{'train_patients': 220, 'val_patients': 48, 'test_patients': 47, 'mg_feat_dim': 1280, 'us_feat_dim': 768}`
- **Mammogram Encoder:** EfficientNet-B0 (1280-dimensional feature vectors pre-extracted from `outputs/experiments/exp1_efficientnet_mammogram/best_checkpoint.pth`).
- **Ultrasound Encoder:** ViT-B/16 (768-dimensional feature vectors pre-extracted from `outputs/experiments/exp2_vit_ultrasound/best_checkpoint.pth`).
- **Label Concordance:** Evaluated all $315$ patients in the feature cache against `BCMID_annotations.tsv`. **Zero label or BI-RADS mismatches found (100% concordance).**

---

## 4. Exact Implementation Details of Benchmark C3

- **Model Class:** `GraphInfusedTransformer` in `src/models/git.py`.
  - Architecture: GAT (2 layers, 4 heads, hidden dim 256, side-aware embedding) + Cross-Attention Transformer (2 layers, 8 heads, feedforward dim 1024, dropout 0.3) + Linear classification head.
  - Total Trainable Parameters: $2,897,922$.
- **Loss Function:** `ClassBalancedFocalLoss` in `src/models/focal_loss.py`.
  $$\text{FL}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t), \quad \gamma = 1.5, \quad \alpha = [1.0000, 5.1111]$$
  Batch reduction: normalized by $\sum_{i=1}^B \alpha_{y_i}$.
- **Optimization:**
  - Optimizer: `torch.optim.AdamW(lr=1e-4, weight_decay=1e-2)`
  - Scheduler: `CosineAnnealingLR(T_max=20, eta_min=1e-6)`
  - Mixed Precision: `torch.amp.autocast('cuda')` with `GradScaler`
  - Gradient Clipping: `max_norm = 1.0`
  - Batch Size: 8 with 2 gradient accumulation steps (effective batch size = 16).
- **Model Selection Protocol:**
  $$S = 0.5 \times \text{ROC-AUC} + 0.3 \times \text{PR-AUC} + 0.2 \times \text{F1}$$
  - Selected Epoch: **Epoch 4** ($S = 0.8351$).
  - Early stopping patience: 8 epochs without composite score improvement (stopped after Epoch 12).

---

## 5. Phase D Experimental Plan

Following the completion of this audit, Phase D proceeds strictly within `outputs/phase_d/`:

### Step D1: Phase D Isolated Environment Setup
- Manifest saved at `outputs/phase_d/phase_d_manifest.json`.
- Directories created: `outputs/phase_d/{configs, experiments, predictions, tables, reports}`.

### Step D2: Deterministic C3 Reproduction
- Run C3 reproduction under `outputs/phase_d/experiments/exp_d1_c3_repro/`.
- Verify that validation metrics replicate C3 ($S \approx 0.8351$, Epoch 4, Loss gap $\approx +0.0089$).
- Record runtime, loss curves, and peak GPU VRAM.

### Step D3: Controlled Hyperparameter & Regularization Series
Run strictly one isolated modification at a time using development data ($N_{\text{train}}=220, N_{\text{val}}=48$):

| Experiment ID | Parameter Varied | Focal Gamma ($\gamma$) | Learning Rate | Weight Decay | Dropout Rate | Rationale |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **D1 (C3 Repro)** | Reference Baseline | 1.5 | $10^{-4}$ | $0.01$ | $0.3$ | Exact C3 reproduction verification |
| **D2 (Gamma 1.0)** | Loss Modulation | **1.0** | $10^{-4}$ | $0.01$ | $0.3$ | Evaluate intermediate focal focusing between BCE ($\gamma=0$) and C3 ($\gamma=1.5$) |
| **D3 (Gamma 2.0)** | Loss Modulation | **2.0** | $10^{-4}$ | $0.01$ | $0.3$ | Re-verify steeper focal penalty with C3's exact training schedule |
| **D4 (GIT Regularization)**| Transformer Dropout | 1.5 | $10^{-4}$ | $0.01$ | **0.4** | Test higher dropout ($0.4$ vs. $0.3$) in cross-attention head to curb late-epoch divergence |
| **D5 (Conservative LR)** | Learning Rate | 1.5 | **$5\times 10^{-5}$** | $0.01$ | $0.3$ | Evaluate half learning rate to stabilize gradient updates on focal loss |

### Step D4: Scientific Model Selection & Validation Comparison
- Evaluate all candidates D1–D5 strictly on validation data using the composite score $S$.
- Generate comprehensive validation comparison tables, confusion matrices, and generalization diagnostics.
- Present the completed validation comparison before selecting the top model.
