# Phase D2: Controlled Experiments Comparative Summary (D1–D5 vs Frozen C3)

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
| **Training Loss at Best Epoch** | 0.1590 | 0.2026 | 0.0075 | 0.0068 | 0.1585 |
| **Training Accuracy at Best Epoch** | 90.91% | 91.82% | 99.55% | 99.55% | 88.18% |
| **Overfitting Gap (Val - Train)** | +0.0089 | +0.0224 | **+0.2462 (Severe)** | **+0.5786 (Severe)** | **-0.0195 (Optimal)** |
| **Stopping Epoch (Patience=8)** | Epoch 12 | Epoch 12 | Epoch 18 | Epoch 20 (Max) | Epoch 12 |
| **Total Wall-Clock Runtime** | 48.40s | 44.51s | 64.11s | 70.84s | 71.95s |
| **Peak GPU VRAM (MB)** | 105.52 MB | 105.52 MB | 105.52 MB | 105.52 MB | 105.52 MB |
