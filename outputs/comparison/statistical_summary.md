# Statistical Validation: 95% Confidence Interval Analysis

## Cohort & Statistical Design
- **Dataset**: BCMID Multimodal Breast Cancer Imaging Dataset
- **Test Cohort Size**: $N = 48$ patients
- **Class Distribution**: $n = 8$ Malignant (Class 1), $n = 40$ Benign (Class 0)
- **Malignant Prevalence**: 16.7%
- **Sensitivity Confidence Intervals**: Exact Clopper-Pearson Binomial 95% CI ($n = 8$)
- **Specificity Confidence Intervals**: Exact Clopper-Pearson Binomial 95% CI ($n = 40$)
- **AUC, Accuracy, F1 Confidence Intervals**: Patient-level Stratified Bootstrap (Complete Patients, $B = 2000$ replicates, Percentile Method)
- **Decision Thresholds**: Calibrated strictly on the independent validation set (Youden's J)

## Final Statistical Comparison Table

| Model | ROC-AUC (95% CI) | PR-AUC (95% CI) | Accuracy (95% CI) | Sensitivity (95% CI) | Specificity (95% CI) | F1 (95% CI) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **DenseNet-121** | 0.5813 (0.3500–0.8031) | 0.2234 (0.1230–0.5406) | 0.6875 (0.5625–0.8125) | 0.5000 (0.1570–0.8430) | 0.7250 (0.5611–0.8540) | 0.3478 (0.1053–0.5600) |
| **ResNet-50** | 0.8594 (0.7094–0.9719) | 0.6947 (0.4074–0.9256) | 0.8542 (0.7500–0.9375) | 0.6250 (0.2449–0.9148) | 0.9000 (0.7634–0.9721) | 0.5882 (0.3075–0.8421) |
| **Simple Fusion** | 0.7063 (0.4937–0.8875) | 0.2728 (0.1662–0.4663) | 0.7708 (0.6458–0.8750) | 0.6250 (0.2449–0.9148) | 0.8000 (0.6435–0.9095) | 0.4762 (0.2306–0.7000) |
| **Graph Transformer** | 0.8594 (0.7374–0.9563) | 0.5450 (0.2879–0.8510) | 0.7917 (0.6667–0.8958) | 0.7500 (0.3491–0.9681) | 0.8000 (0.6435–0.9095) | 0.5455 (0.3333–0.7368) |

### Confusion Matrices and Operational Parameters

| Model | Selected Threshold | TN | FP | FN | TP | Balanced Accuracy (95% CI) | Precision (95% CI) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **DenseNet-121** | 0.2932 | 29 | 11 | 4 | 4 | 0.6125 (0.4250–0.7878) | 0.2667 (0.0909–0.4545) |
| **ResNet-50** | 0.3844 | 36 | 4 | 3 | 5 | 0.7625 (0.5872–0.9250) | 0.5556 (0.3000–0.8750) |
| **Simple Fusion** | 0.2928 | 32 | 8 | 3 | 5 | 0.7125 (0.5250–0.8750) | 0.3846 (0.2000–0.6154) |
| **Graph Transformer** | 0.2565 | 32 | 8 | 2 | 6 | 0.7750 (0.6000–0.9125) | 0.4286 (0.2667–0.6364) |

## Rigorous Paired Hypothesis Testing: Graph Transformer vs. ResNet-50

- **Test ROC-AUC**: Graph Transformer = `0.8594`, ResNet-50 = `0.8594`
- **Test PR-AUC**: Graph Transformer = `0.5450`, ResNet-50 = `0.6947`
- **Sensitivity**: Graph Transformer = `75.0%` (6/8), ResNet-50 = `62.5%` (5/8)
- **Specificity**: Graph Transformer = `80.0%` (32/40), ResNet-50 = `90.0%` (36/40)
- **McNemar's Paired Test**: Exact two-sided $p = 0.5811$
  - Both models correct: 33
  - Graph Transformer correct, ResNet incorrect: 5
  - ResNet correct, Graph Transformer incorrect: 8
  - Both models incorrect: 2
- **Paired Bootstrap ROC-AUC Difference (Graph Transformer - ResNet-50)**: Mean difference = `0.0003`, 95% CI: `[-0.1125, 0.1219]`, empirical $p = 1.0000$

> [!IMPORTANT]
> **Scientific Conclusion**: There is **no statistically significant difference** in overall discriminative ability between the Multimodal Graph Transformer and the unimodal ResNet-50 on the test cohort ($p = 0.5811$ McNemar; $\Delta\text{ROC-AUC} = 0.0000$, 95% CI: $[-0.1231, +0.1270]$). The Multimodal Graph Transformer demonstrates higher point sensitivity ($75.0\%$ vs $62.5\%$), identifying 6 out of 8 malignant lesions with only 2 false negatives, whereas ResNet-50 demonstrates higher point specificity ($90.0\%$ vs $80.0\%$) and PR-AUC ($0.6947$ vs $0.5450$). Due to the limited size of the malignant test sample ($N_{\text{pos}} = 8$), the 95% confidence intervals for sensitivity are broad for all models (e.g. $[34.91\%, 96.81\%]$ for Graph Transformer and $[24.49\%, 91.48\%]$ for ResNet-50).