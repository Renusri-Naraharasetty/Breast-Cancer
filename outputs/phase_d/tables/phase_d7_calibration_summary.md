# Phase D7: Validation Calibration Summary Table

**Cohorts:** 48 Validation Patients (40 Negative, 8 Positive)  
**Evaluation:** Multi-Seed Comparison (Seeds 42, 123, 456)  
**Held-Out Test Set:** **Quarantined (Never Loaded)**  

### 1. Calibration Metrics by Run

| Model | Seed | Brier Score (MSE) | Log Loss (Cross-Entropy) | ECE (5 Bins) | MCE (5 Bins) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **C3 Baseline** | 42 | 0.0877 | 0.2903 | 0.0935 | 0.3160 |
| **C3 Baseline** | 123 | 0.0835 | 0.2631 | 0.0957 | 0.3328 |
| **C3 Baseline** | 456 | 0.0704 | 0.2378 | 0.0481 | 0.1055 |
| **D5 Candidate** | 42 | 0.1158 | 0.3740 | 0.1816 | 0.3029 |
| **D5 Candidate** | 123 | 0.0857 | 0.2713 | 0.0725 | 0.3004 |
| **D5 Candidate** | 456 | 0.1487 | 0.4546 | 0.1862 | 0.3181 |

---

### 2. Aggregated Multi-Seed Calibration Summary (Mean $\pm$ SD)

| Metric | C3 Baseline Mean $\pm$ SD | D5 Candidate Mean $\pm$ SD | Delta (D5 - C3) |
| :--- | :---: | :---: | :---: |
| **Brier Score** | **$0.0805 \pm 0.0090$** | $0.1167 \pm 0.0315$ | $+0.0362$ |
| **Log Loss** | **$0.2637 \pm 0.0263$** | $0.3666 \pm 0.0919$ | $+0.1029$ |
| **Expected Calibration Error (ECE)** | **$0.0791 \pm 0.0269$** | $0.1468 \pm 0.0644$ | $+0.0677$ |
| **Maximum Calibration Error (MCE)** | **$0.2514 \pm 0.1267$** | $0.3071 \pm 0.0096$ | $+0.0557$ |

---

### 3. Detailed Reliability Bin Distribution (Seed 42 Reference)

#### C3 (Seed 42) Bins:
- Bin 1 `[0.0, 0.2]`: Count = 23 | Confidence = 0.0768 | Empirical Accuracy = 0.0435 | Gap = 0.0333
- Bin 2 `[0.2, 0.4]`: Count = 14 | Confidence = 0.3370 | Empirical Accuracy = 0.0714 | Gap = 0.2656
- Bin 3 `[0.4, 0.6]`: Count = 4 | Confidence = 0.5222 | Empirical Accuracy = 0.5000 | Gap = 0.0222
- Bin 4 `[0.6, 0.8]`: Count = 4 | Confidence = 0.6697 | Empirical Accuracy = 0.7500 | Gap = 0.0803
- Bin 5 `[0.8, 1.0]`: Count = 3 | Confidence = 0.8842 | Empirical Accuracy = 1.0000 | Gap = 0.1158

#### D5 (Seed 42) Bins:
- Bin 1 `[0.0, 0.2]`: Count = 16 | Confidence = 0.0884 | Empirical Accuracy = 0.0000 | Gap = 0.0884
- Bin 2 `[0.2, 0.4]`: Count = 14 | Confidence = 0.3204 | Empirical Accuracy = 0.0714 | Gap = 0.2490
- Bin 3 `[0.4, 0.6]`: Count = 14 | Confidence = 0.5186 | Empirical Accuracy = 0.3571 | Gap = 0.1615
- Bin 4 `[0.6, 0.8]`: Count = 3 | Confidence = 0.6971 | Empirical Accuracy = 1.0000 | Gap = 0.3029
- Bin 5 `[0.8, 1.0]`: Count = 1 | Confidence = 0.8250 | Empirical Accuracy = 1.0000 | Gap = 0.1750
