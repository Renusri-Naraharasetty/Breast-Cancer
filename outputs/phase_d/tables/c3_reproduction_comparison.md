# Phase D2 - Experiment D1: C3 Reproduction Comparison Table

| Metric / Dimension | Frozen C3 Benchmark (Phase C) | Experiment D1 Repro (Phase D) | Absolute Delta | Verification Status |
| :--- | :---: | :---: | :---: | :---: |
| **Selected Epoch** | **Epoch 4** | **Epoch 4** | 0 | **Exact Match** |
| **Composite Score ($S$)** | **0.8351** | **0.8351** | 0.0000 | **Exact Match** |
| **Validation ROC-AUC** | **0.9281** | **0.9281** | 0.0000 | **Exact Match** |
| **Validation PR-AUC** | **0.7661** | **0.7661** | 0.0000 | **Exact Match** |
| **Validation F1-Score** | **0.7059** | **0.7059** | 0.0000 | **Exact Match** |
| **Validation Sensitivity (Recall)** | **75.00%** (6/8) | **75.00%** (6/8) | 0.0000 | **Exact Match** |
| **Validation Specificity** | **92.50%** (37/40) | **92.50%** (37/40) | 0.0000 | **Exact Match** |
| **Validation Balanced Accuracy** | **83.75%** | **83.75%** | 0.0000 | **Exact Match** |
| **Validation Precision** | 66.67% (6/9) | 66.67% (6/9) | 0.0000 | **Exact Match** |
| **Validation Accuracy** | 89.58% (43/48) | 89.58% (43/48) | 0.0000 | **Exact Match** |
| **Validation Loss** | 0.1679 | 0.1679 | 0.0000 | **Exact Match** |
| **Training Loss at Best Epoch** | 0.1590 | 0.1590 | 0.0000 | **Exact Match** |
| **Training Accuracy at Best Epoch** | 90.91% | 90.91% | 0.0000 | **Exact Match** |
| **Loss Gap (Val - Train)** | +0.0089 | +0.0089 | 0.0000 | **Exact Match** |
| **Confusion Matrix `[TN, FP, FN, TP]`** | `[37, 3, 2, 6]` | `[37, 3, 2, 6]` | `[0, 0, 0, 0]` | **Exact Match** |
| **Early Stopping Triggered** | Epoch 12 (patience=8) | Epoch 12 (patience=8) | 0 | **Exact Match** |
| **Total Wall-Clock Runtime** | ~20–30s | **48.40s** (0.81 min) | — | **Logged** |
| **Peak GPU VRAM** | <150 MB | **105.52 MB** | — | **Logged** |
| **Hardware Platform** | NVIDIA GeForce MX550 | NVIDIA GeForce MX550 | — | **Identical** |
| **Test Set Quarantine Status** | Quarantined | Quarantined | — | **100% Compliant** |
