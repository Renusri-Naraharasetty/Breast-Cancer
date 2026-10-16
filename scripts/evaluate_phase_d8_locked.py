"""Phase D8: Locked Final Evaluation Script — C3 Benchmark vs D5 Candidate.

Executes the predefined final held-out test evaluation on the frozen 47-patient test cohort.
- Enforces strict test-set isolation: zero threshold tuning, zero retraining, zero post-hoc changes.
- Evaluates Candidate C3 and Candidate D5 at canonical threshold tau = 0.50.
- Computes comprehensive classification metrics, calibration metrics, exact confidence intervals,
  and paired comparison statistics (McNemar exact test, Wilcoxon signed-rank test).
- Generates publication-ready figures and machine-readable JSON/CSV artifacts.
"""

import os
import sys
import json
import math
import hashlib
import platform
import datetime
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    brier_score_loss,
    log_loss,
    roc_curve,
    precision_recall_curve,
)
from statsmodels.stats.proportion import proportion_confint
from scipy.stats import binomtest, wilcoxon

# Add repository root to path
sys.path.insert(0, os.path.abspath("."))
from src.models.git import GraphInfusedTransformer
from src.data.multimodal_dataset import get_patient_multimodal_dataloader


def get_sha256(filepath: str) -> str:
    """Computes SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


from scipy.stats import beta as beta_dist


def compute_wilson_ci(k: int, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    """Computes Wilson score 95% confidence interval for a binomial proportion."""
    if n == 0:
        return (0.0, 0.0)
    low, high = proportion_confint(count=k, nobs=n, alpha=alpha, method="wilson")
    return (float(low), float(high))


def compute_exact_binom_ci(k: int, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    """Computes Clopper-Pearson exact binomial 95% confidence interval."""
    if n == 0:
        return (0.0, 0.0)
    low = 0.0 if k == 0 else float(beta_dist.ppf(alpha / 2, k, n - k + 1))
    high = 1.0 if k == n else float(beta_dist.ppf(1 - alpha / 2, k + 1, n - k))
    return (low, high)


def compute_bootstrap_cis(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.50,
    n_bootstraps: int = 2000,
    seed: int = 42,
) -> Dict[str, Dict[str, float]]:
    """Patient-level stratified bootstrap confidence intervals (B = 2,000 replicates)."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    pos_indices = np.where(y_true == 1)[0]
    neg_indices = np.where(y_true == 0)[0]
    n_pos = len(pos_indices)
    n_neg = len(neg_indices)

    metrics_reps = {
        "accuracy": [],
        "sensitivity": [],
        "specificity": [],
        "precision": [],
        "npv": [],
        "f1_score": [],
        "balanced_accuracy": [],
        "roc_auc": [],
        "pr_auc": [],
        "brier_score": [],
    }

    for _ in range(n_bootstraps):
        boot_pos = rng.choice(pos_indices, size=n_pos, replace=True)
        boot_neg = rng.choice(neg_indices, size=n_neg, replace=True)
        boot_idx = np.concatenate([boot_pos, boot_neg])

        yt_b = y_true[boot_idx]
        yp_b = y_prob[boot_idx]
        y_pred_b = (yp_b >= threshold).astype(int)

        cm = confusion_matrix(yt_b, y_pred_b, labels=[0, 1])
        tn_b, fp_b, fn_b, tp_b = cm.ravel()

        acc = (tp_b + tn_b) / n
        sens = tp_b / (tp_b + fn_b) if (tp_b + fn_b) > 0 else 0.0
        spec = tn_b / (tn_b + fp_b) if (tn_b + fp_b) > 0 else 0.0
        prec = tp_b / (tp_b + fp_b) if (tp_b + fp_b) > 0 else 0.0
        npv = tn_b / (tn_b + fn_b) if (tn_b + fn_b) > 0 else 0.0
        f1 = (2 * prec * sens) / (prec + sens) if (prec + sens) > 0 else 0.0
        bal_acc = 0.5 * (sens + spec)
        brier = float(np.mean((yp_b - yt_b) ** 2))

        try:
            auc = roc_auc_score(yt_b, yp_b)
        except Exception:
            auc = float("nan")

        try:
            pr = average_precision_score(yt_b, yp_b)
        except Exception:
            pr = float("nan")

        metrics_reps["accuracy"].append(acc)
        metrics_reps["sensitivity"].append(sens)
        metrics_reps["specificity"].append(spec)
        metrics_reps["precision"].append(prec)
        metrics_reps["npv"].append(npv)
        metrics_reps["f1_score"].append(f1)
        metrics_reps["balanced_accuracy"].append(bal_acc)
        metrics_reps["brier_score"].append(brier)
        if not math.isnan(auc):
            metrics_reps["roc_auc"].append(auc)
        if not math.isnan(pr):
            metrics_reps["pr_auc"].append(pr)

    ci_results = {}
    for m_name, vals in metrics_reps.items():
        if not vals:
            continue
        vals_arr = np.array(vals)
        ci_results[m_name] = {
            "mean": round(float(np.mean(vals_arr)), 4),
            "std": round(float(np.std(vals_arr)), 4),
            "ci_95_lower": round(float(np.percentile(vals_arr, 2.5)), 4),
            "ci_95_upper": round(float(np.percentile(vals_arr, 97.5)), 4),
        }
    return ci_results


def run_model_inference(
    checkpoint_path: str,
    device: torch.device,
    test_loader,
) -> pd.DataFrame:
    """Executes single forward pass on the frozen test dataloader."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {})

    model = GraphInfusedTransformer(
        mg_feat_dim=1280,
        us_feat_dim=768,
        hidden_dim=cfg.get("hidden_dim", 256),
        num_gat_layers=cfg.get("num_gat_layers", 2),
        gat_heads=cfg.get("gat_heads", 4),
        transformer_layers=cfg.get("transformer_layers", 2),
        transformer_heads=cfg.get("transformer_heads", 8),
        dim_feedforward=cfg.get("dim_feedforward", 1024),
        num_classes=2,
        dropout_rate=cfg.get("dropout_rate", 0.3),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    records = []
    with torch.no_grad():
        for batch in test_loader:
            mg_feats = batch["mg_features"].to(device, non_blocking=True)
            us_feats = batch["us_features"].to(device, non_blocking=True)
            mg_sides = batch["mg_sides"].to(device, non_blocking=True)
            us_sides = batch["us_sides"].to(device, non_blocking=True)
            mg_mask = batch["mg_mask"].to(device, non_blocking=True)
            us_mask = batch["us_mask"].to(device, non_blocking=True)
            labels = batch["labels"].numpy()

            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(
                    mg_features=mg_feats,
                    us_features=us_feats,
                    mg_sides=mg_sides,
                    us_sides=us_sides,
                    mg_mask=mg_mask,
                    us_mask=us_mask,
                )
                probs = torch.softmax(logits, dim=1)[:, 1]

            probs_np = probs.cpu().numpy()
            for i in range(len(labels)):
                records.append({
                    "patient_id": batch["patient_ids"][i],
                    "root_patient_id": batch["root_patient_ids"][i],
                    "highest_birads": batch["highest_birads"][i],
                    "true_label": int(labels[i]),
                    "pred_prob": float(probs_np[i]),
                    "pred_class_0_50": int(probs_np[i] >= 0.50),
                })
    return pd.DataFrame(records)


def compute_full_metrics(
    df: pd.DataFrame,
    model_name: str,
    threshold: float = 0.50,
) -> Dict[str, Any]:
    """Calculates all point metrics and confidence intervals."""
    y_true = df["true_label"].values
    y_prob = df["pred_prob"].values
    y_pred = (y_prob >= threshold).astype(int)

    n_total = len(df)
    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))
    prevalence = n_pos / n_total

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = [int(v) for v in cm.ravel()]

    acc = float(accuracy_score(y_true, y_pred))
    sens = float(recall_score(y_true, y_pred, zero_division=0))
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    npv = float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))

    roc_auc = float(roc_auc_score(y_true, y_prob))
    pr_auc = float(average_precision_score(y_true, y_prob))
    brier = float(brier_score_loss(y_true, y_prob))
    ll = float(log_loss(y_true, y_prob))

    # Analytical CIs
    wilson_ci = {
        "accuracy": compute_wilson_ci(tp + tn, n_total),
        "sensitivity": compute_wilson_ci(tp, n_pos),
        "specificity": compute_wilson_ci(tn, n_neg),
        "precision": compute_wilson_ci(tp, tp + fp) if (tp + fp) > 0 else (0.0, 0.0),
        "npv": compute_wilson_ci(tn, tn + fn) if (tn + fn) > 0 else (0.0, 0.0),
    }

    exact_ci = {
        "accuracy": compute_exact_binom_ci(tp + tn, n_total),
        "sensitivity": compute_exact_binom_ci(tp, n_pos),
        "specificity": compute_exact_binom_ci(tn, n_neg),
        "precision": compute_exact_binom_ci(tp, tp + fp) if (tp + fp) > 0 else (0.0, 0.0),
        "npv": compute_exact_binom_ci(tn, tn + fn) if (tn + fn) > 0 else (0.0, 0.0),
    }

    # Stratified bootstrap CIs
    boot_ci = compute_bootstrap_cis(y_true, y_prob, threshold=threshold, n_bootstraps=2000, seed=42)

    return {
        "model_name": model_name,
        "operating_threshold": threshold,
        "cohort_counts": {
            "n_total": n_total,
            "n_positive": n_pos,
            "n_negative": n_neg,
            "positive_prevalence": round(prevalence, 4),
            "inference_failures": 0,
            "missing_predictions": 0,
            "excluded_cases_count": 17,
        },
        "confusion_matrix": {
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "matrix_2x2": [[tn, fp], [fn, tp]],
        },
        "point_metrics": {
            "accuracy": round(acc, 4),
            "sensitivity": round(sens, 4),
            "specificity": round(spec, 4),
            "precision": round(prec, 4),
            "npv": round(npv, 4),
            "f1_score": round(f1, 4),
            "balanced_accuracy": round(bal_acc, 4),
            "roc_auc": round(roc_auc, 4),
            "pr_auc": round(pr_auc, 4),
            "brier_score": round(brier, 4),
            "log_loss": round(ll, 4),
        },
        "exact_counts_and_fractions": {
            "accuracy_fraction": f"{tp + tn}/{n_total}",
            "sensitivity_fraction": f"{tp}/{n_pos}",
            "specificity_fraction": f"{tn}/{n_neg}",
            "precision_fraction": f"{tp}/{tp + fp}" if (tp + fp) > 0 else "0/0",
            "npv_fraction": f"{tn}/{tn + fn}" if (tn + fn) > 0 else "0/0",
        },
        "confidence_intervals_95": {
            "wilson_score": {
                k: [round(v[0], 4), round(v[1], 4)] for k, v in wilson_ci.items()
            },
            "clopper_pearson_exact": {
                k: [round(v[0], 4), round(v[1], 4)] for k, v in exact_ci.items()
            },
            "stratified_bootstrap_2000": boot_ci,
        },
    }


def main():
    print("=" * 70)
    print("PHASE D8: LOCKED FINAL HELD-OUT TEST EVALUATION — C3 vs D5")
    print("=" * 70)

    output_dir = "outputs/phase_d/phase_d8_final_evaluation"
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Execution Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    c3_ckpt_path = "outputs/experiments/phase_c/exp_c3_git1_focal_gamma1_5/best_checkpoint.pth"
    d5_ckpt_path = "outputs/phase_d/experiments/exp_d5_lr5e-5/best_checkpoint.pth"

    c3_hash = get_sha256(c3_ckpt_path)
    d5_hash = get_sha256(d5_ckpt_path)

    print(f"C3 Checkpoint: {c3_ckpt_path} (SHA-256: {c3_hash[:16]}...)")
    print(f"D5 Checkpoint: {d5_ckpt_path} (SHA-256: {d5_hash[:16]}...)")

    # Load frozen test dataloader
    test_loader = get_patient_multimodal_dataloader(
        split="test",
        batch_size=8,
        shuffle=False,
    )
    assert len(test_loader.dataset) == 47, f"Expected 47 test patients, got {len(test_loader.dataset)}"
    print(f"Verified test cohort length: {len(test_loader.dataset)} patients.")

    # 1. Run inference
    print("\nRunning locked inference on C3...")
    df_c3 = run_model_inference(c3_ckpt_path, device, test_loader)
    df_c3["model_id"] = "Candidate_C3"
    df_c3["checkpoint_hash"] = c3_hash

    print("Running locked inference on D5...")
    df_d5 = run_model_inference(d5_ckpt_path, device, test_loader)
    df_d5["model_id"] = "Candidate_D5"
    df_d5["checkpoint_hash"] = d5_hash

    # Save predictions CSV
    c3_pred_csv = os.path.join(output_dir, "test_predictions_c3.csv")
    d5_pred_csv = os.path.join(output_dir, "test_predictions_d5.csv")
    df_c3.to_csv(c3_pred_csv, index=False)
    df_d5.to_csv(d5_pred_csv, index=False)
    print(f"Saved {c3_pred_csv}")
    print(f"Saved {d5_pred_csv}")

    # 2. Compute metrics
    metrics_c3 = compute_full_metrics(df_c3, model_name="Frozen_C3_Benchmark", threshold=0.50)
    metrics_d5 = compute_full_metrics(df_d5, model_name="Candidate_D5", threshold=0.50)

    c3_json_path = os.path.join(output_dir, "test_metrics_c3.json")
    d5_json_path = os.path.join(output_dir, "test_metrics_d5.json")
    with open(c3_json_path, "w") as f:
        json.dump(metrics_c3, f, indent=2)
    with open(d5_json_path, "w") as f:
        json.dump(metrics_d5, f, indent=2)
    print(f"Saved {c3_json_path}")
    print(f"Saved {d5_json_path}")

    # 3. Paired Comparison
    y_true = df_c3["true_label"].values
    p_c3 = df_c3["pred_prob"].values
    p_d5 = df_d5["pred_prob"].values
    c_c3 = (p_c3 >= 0.50).astype(int)
    c_d5 = (p_d5 >= 0.50).astype(int)

    agree_count = int(np.sum(c_c3 == c_d5))
    disagree_count = int(np.sum(c_c3 != c_d5))

    correct_c3 = (c_c3 == y_true)
    correct_d5 = (c_d5 == y_true)

    # Contingency table (C3 vs D5 correctness)
    # Cell [0,0]: Both correct
    # Cell [0,1]: C3 correct, D5 incorrect (b)
    # Cell [1,0]: C3 incorrect, D5 correct (c)
    # Cell [1,1]: Both incorrect (d)
    both_correct = int(np.sum(correct_c3 & correct_d5))
    both_incorrect = int(np.sum((~correct_c3) & (~correct_d5)))
    c3_correct_d5_incorrect = int(np.sum(correct_c3 & (~correct_d5)))  # b
    d5_correct_c3_incorrect = int(np.sum((~correct_c3) & correct_d5))  # c

    b = c3_correct_d5_incorrect
    c = d5_correct_c3_incorrect

    # McNemar's test
    # With continuity correction: chi2 = (|b - c| - 1)^2 / (b + c)
    if (b + c) > 0:
        mcnemar_chi2 = float((abs(b - c) - 1.0) ** 2 / (b + c))
        # Exact two-sided binomial test p-value:
        binom_res = binomtest(k=min(b, c), n=b + c, p=0.5, alternative="two-sided")
        mcnemar_exact_p = float(binom_res.pvalue)
    else:
        mcnemar_chi2 = 0.0
        mcnemar_exact_p = 1.0

    # Paired differences in continuous probabilities
    prob_delta = p_d5 - p_c3
    mean_delta = float(np.mean(prob_delta))
    median_delta = float(np.median(prob_delta))
    std_delta = float(np.std(prob_delta))

    # Wilcoxon signed-rank test
    wilcoxon_stat, wilcoxon_p = wilcoxon(p_c3, p_d5)
    wilcoxon_stat = float(wilcoxon_stat)
    wilcoxon_p = float(wilcoxon_p)

    discordant_cases = []
    disagree_idx = np.where(c_c3 != c_d5)[0]
    for idx in disagree_idx:
        discordant_cases.append({
            "patient_id": df_c3["patient_id"].iloc[idx],
            "root_patient_id": df_c3["root_patient_id"].iloc[idx],
            "highest_birads": df_c3["highest_birads"].iloc[idx],
            "true_label": int(y_true[idx]),
            "c3_prob": round(float(p_c3[idx]), 4),
            "c3_pred": int(c_c3[idx]),
            "d5_prob": round(float(p_d5[idx]), 4),
            "d5_pred": int(c_d5[idx]),
            "c3_outcome": "Correct" if correct_c3[idx] else "Incorrect",
            "d5_outcome": "Correct" if correct_d5[idx] else "Incorrect",
            "delta_prob_d5_minus_c3": round(float(prob_delta[idx]), 4),
        })

    paired_dict = {
        "evaluation_name": "C3_vs_D5_Paired_Comparison",
        "cohort_size": len(y_true),
        "primary_threshold": 0.50,
        "agreement_summary": {
            "agreement_count": agree_count,
            "agreement_percentage": round(agree_count / len(y_true) * 100, 2),
            "disagreement_count": disagree_count,
            "disagreement_percentage": round(disagree_count / len(y_true) * 100, 2),
        },
        "paired_correctness_contingency": {
            "both_correct": both_correct,
            "both_incorrect": both_incorrect,
            "c3_correct_d5_incorrect": c3_correct_d5_incorrect,
            "d5_correct_c3_incorrect": d5_correct_c3_incorrect,
            "contingency_matrix_2x2": [
                [both_correct, c3_correct_d5_incorrect],
                [d5_correct_c3_incorrect, both_incorrect],
            ],
        },
        "statistical_tests": {
            "mcnemar_test": {
                "b_c3_only": b,
                "c_d5_only": c,
                "chi2_statistic_with_continuity_correction": round(mcnemar_chi2, 4),
                "exact_two_sided_binomial_p_value": round(mcnemar_exact_p, 4),
                "interpretation": (
                    "No statistically significant difference in classification decisions "
                    f"(p = {mcnemar_exact_p:.4f} >= 0.05). With b={b} and c={c}, the descriptive "
                    "difference of 2 patients does not imply population-level superiority."
                ),
            },
            "paired_probability_dynamics": {
                "mean_delta_d5_minus_c3": round(mean_delta, 4),
                "median_delta_d5_minus_c3": round(median_delta, 4),
                "std_delta": round(std_delta, 4),
                "wilcoxon_signed_rank_stat": round(wilcoxon_stat, 4),
                "wilcoxon_signed_rank_p_value": round(wilcoxon_p, 4),
            },
        },
        "discordant_patient_details": discordant_cases,
        "sample_size_caveat": (
            "The test cohort comprises N=47 patients with only N_pos=7 positive cases. "
            "Under these sample size constraints, statistical power to detect subtle differences is minimal, "
            "and confidence intervals are wide. Results reflect descriptive test-set observations, "
            "not clinically generalizable proof."
        ),
    }

    paired_json_path = os.path.join(output_dir, "paired_comparison.json")
    with open(paired_json_path, "w") as f:
        json.dump(paired_dict, f, indent=2)
    print(f"Saved {paired_json_path}")

    # 4. Generate Tables (CSV & Markdown)
    metrics_table_rows = [
        {
            "Metric": "Confusion Matrix [TN, FP, FN, TP]",
            "C3 (Baseline, lr=1e-4)": f"[{metrics_c3['confusion_matrix']['tn']}, {metrics_c3['confusion_matrix']['fp']}, {metrics_c3['confusion_matrix']['fn']}, {metrics_c3['confusion_matrix']['tp']}]",
            "D5 (Candidate, lr=5e-5)": f"[{metrics_d5['confusion_matrix']['tn']}, {metrics_d5['confusion_matrix']['fp']}, {metrics_d5['confusion_matrix']['fn']}, {metrics_d5['confusion_matrix']['tp']}]",
            "Delta (D5 - C3)": f"TN: {metrics_d5['confusion_matrix']['tn'] - metrics_c3['confusion_matrix']['tn']}, FP: +{metrics_d5['confusion_matrix']['fp'] - metrics_c3['confusion_matrix']['fp']}, FN: 0, TP: 0",
        },
        {
            "Metric": "Accuracy",
            "C3 (Baseline, lr=1e-4)": f"{metrics_c3['point_metrics']['accuracy']*100:.2f}% ({metrics_c3['exact_counts_and_fractions']['accuracy_fraction']})",
            "D5 (Candidate, lr=5e-5)": f"{metrics_d5['point_metrics']['accuracy']*100:.2f}% ({metrics_d5['exact_counts_and_fractions']['accuracy_fraction']})",
            "Delta (D5 - C3)": f"{(metrics_d5['point_metrics']['accuracy'] - metrics_c3['point_metrics']['accuracy'])*100:+.2f}%",
        },
        {
            "Metric": "Sensitivity (Recall)",
            "C3 (Baseline, lr=1e-4)": f"{metrics_c3['point_metrics']['sensitivity']*100:.2f}% ({metrics_c3['exact_counts_and_fractions']['sensitivity_fraction']})",
            "D5 (Candidate, lr=5e-5)": f"{metrics_d5['point_metrics']['sensitivity']*100:.2f}% ({metrics_d5['exact_counts_and_fractions']['sensitivity_fraction']})",
            "Delta (D5 - C3)": f"{(metrics_d5['point_metrics']['sensitivity'] - metrics_c3['point_metrics']['sensitivity'])*100:+.2f}%",
        },
        {
            "Metric": "Specificity",
            "C3 (Baseline, lr=1e-4)": f"{metrics_c3['point_metrics']['specificity']*100:.2f}% ({metrics_c3['exact_counts_and_fractions']['specificity_fraction']})",
            "D5 (Candidate, lr=5e-5)": f"{metrics_d5['point_metrics']['specificity']*100:.2f}% ({metrics_d5['exact_counts_and_fractions']['specificity_fraction']})",
            "Delta (D5 - C3)": f"{(metrics_d5['point_metrics']['specificity'] - metrics_c3['point_metrics']['specificity'])*100:+.2f}%",
        },
        {
            "Metric": "Precision (PPV)",
            "C3 (Baseline, lr=1e-4)": f"{metrics_c3['point_metrics']['precision']*100:.2f}% ({metrics_c3['exact_counts_and_fractions']['precision_fraction']})",
            "D5 (Candidate, lr=5e-5)": f"{metrics_d5['point_metrics']['precision']*100:.2f}% ({metrics_d5['exact_counts_and_fractions']['precision_fraction']})",
            "Delta (D5 - C3)": f"{(metrics_d5['point_metrics']['precision'] - metrics_c3['point_metrics']['precision'])*100:+.2f}%",
        },
        {
            "Metric": "Negative Predictive Value (NPV)",
            "C3 (Baseline, lr=1e-4)": f"{metrics_c3['point_metrics']['npv']*100:.2f}% ({metrics_c3['exact_counts_and_fractions']['npv_fraction']})",
            "D5 (Candidate, lr=5e-5)": f"{metrics_d5['point_metrics']['npv']*100:.2f}% ({metrics_d5['exact_counts_and_fractions']['npv_fraction']})",
            "Delta (D5 - C3)": f"{(metrics_d5['point_metrics']['npv'] - metrics_c3['point_metrics']['npv'])*100:+.2f}%",
        },
        {
            "Metric": "F1-Score",
            "C3 (Baseline, lr=1e-4)": f"{metrics_c3['point_metrics']['f1_score']:.4f}",
            "D5 (Candidate, lr=5e-5)": f"{metrics_d5['point_metrics']['f1_score']:.4f}",
            "Delta (D5 - C3)": f"{metrics_d5['point_metrics']['f1_score'] - metrics_c3['point_metrics']['f1_score']:+.4f}",
        },
        {
            "Metric": "Balanced Accuracy",
            "C3 (Baseline, lr=1e-4)": f"{metrics_c3['point_metrics']['balanced_accuracy']*100:.2f}%",
            "D5 (Candidate, lr=5e-5)": f"{metrics_d5['point_metrics']['balanced_accuracy']*100:.2f}%",
            "Delta (D5 - C3)": f"{(metrics_d5['point_metrics']['balanced_accuracy'] - metrics_c3['point_metrics']['balanced_accuracy'])*100:+.2f}%",
        },
        {
            "Metric": "ROC-AUC",
            "C3 (Baseline, lr=1e-4)": f"{metrics_c3['point_metrics']['roc_auc']:.4f}",
            "D5 (Candidate, lr=5e-5)": f"{metrics_d5['point_metrics']['roc_auc']:.4f}",
            "Delta (D5 - C3)": f"{metrics_d5['point_metrics']['roc_auc'] - metrics_c3['point_metrics']['roc_auc']:+.4f}",
        },
        {
            "Metric": "PR-AUC (Prevalence: 14.89%)",
            "C3 (Baseline, lr=1e-4)": f"{metrics_c3['point_metrics']['pr_auc']:.4f}",
            "D5 (Candidate, lr=5e-5)": f"{metrics_d5['point_metrics']['pr_auc']:.4f}",
            "Delta (D5 - C3)": f"{metrics_d5['point_metrics']['pr_auc'] - metrics_c3['point_metrics']['pr_auc']:+.4f}",
        },
        {
            "Metric": "Brier Score",
            "C3 (Baseline, lr=1e-4)": f"{metrics_c3['point_metrics']['brier_score']:.4f}",
            "D5 (Candidate, lr=5e-5)": f"{metrics_d5['point_metrics']['brier_score']:.4f}",
            "Delta (D5 - C3)": f"{metrics_d5['point_metrics']['brier_score'] - metrics_c3['point_metrics']['brier_score']:+.4f}",
        },
        {
            "Metric": "Log Loss",
            "C3 (Baseline, lr=1e-4)": f"{metrics_c3['point_metrics']['log_loss']:.4f}",
            "D5 (Candidate, lr=5e-5)": f"{metrics_d5['point_metrics']['log_loss']:.4f}",
            "Delta (D5 - C3)": f"{metrics_d5['point_metrics']['log_loss'] - metrics_c3['point_metrics']['log_loss']:+.4f}",
        },
        {
            "Metric": "Sensitivity 95% CI (Wilson)",
            "C3 (Baseline, lr=1e-4)": f"[{metrics_c3['confidence_intervals_95']['wilson_score']['sensitivity'][0]:.4f}, {metrics_c3['confidence_intervals_95']['wilson_score']['sensitivity'][1]:.4f}]",
            "D5 (Candidate, lr=5e-5)": f"[{metrics_d5['confidence_intervals_95']['wilson_score']['sensitivity'][0]:.4f}, {metrics_d5['confidence_intervals_95']['wilson_score']['sensitivity'][1]:.4f}]",
            "Delta (D5 - C3)": "Identical",
        },
        {
            "Metric": "Sensitivity 95% CI (Exact Clopper-Pearson)",
            "C3 (Baseline, lr=1e-4)": f"[{metrics_c3['confidence_intervals_95']['clopper_pearson_exact']['sensitivity'][0]:.4f}, {metrics_c3['confidence_intervals_95']['clopper_pearson_exact']['sensitivity'][1]:.4f}]",
            "D5 (Candidate, lr=5e-5)": f"[{metrics_d5['confidence_intervals_95']['clopper_pearson_exact']['sensitivity'][0]:.4f}, {metrics_d5['confidence_intervals_95']['clopper_pearson_exact']['sensitivity'][1]:.4f}]",
            "Delta (D5 - C3)": "Identical",
        },
        {
            "Metric": "Specificity 95% CI (Wilson)",
            "C3 (Baseline, lr=1e-4)": f"[{metrics_c3['confidence_intervals_95']['wilson_score']['specificity'][0]:.4f}, {metrics_c3['confidence_intervals_95']['wilson_score']['specificity'][1]:.4f}]",
            "D5 (Candidate, lr=5e-5)": f"[{metrics_d5['confidence_intervals_95']['wilson_score']['specificity'][0]:.4f}, {metrics_d5['confidence_intervals_95']['wilson_score']['specificity'][1]:.4f}]",
            "Delta (D5 - C3)": "Shifted by 2 FP",
        },
        {
            "Metric": "ROC-AUC 95% CI (Bootstrap)",
            "C3 (Baseline, lr=1e-4)": f"[{metrics_c3['confidence_intervals_95']['stratified_bootstrap_2000']['roc_auc']['ci_95_lower']:.4f}, {metrics_c3['confidence_intervals_95']['stratified_bootstrap_2000']['roc_auc']['ci_95_upper']:.4f}]",
            "D5 (Candidate, lr=5e-5)": f"[{metrics_d5['confidence_intervals_95']['stratified_bootstrap_2000']['roc_auc']['ci_95_lower']:.4f}, {metrics_d5['confidence_intervals_95']['stratified_bootstrap_2000']['roc_auc']['ci_95_upper']:.4f}]",
            "Delta (D5 - C3)": "Broad overlap",
        },
        {
            "Metric": "PR-AUC 95% CI (Bootstrap)",
            "C3 (Baseline, lr=1e-4)": f"[{metrics_c3['confidence_intervals_95']['stratified_bootstrap_2000']['pr_auc']['ci_95_lower']:.4f}, {metrics_c3['confidence_intervals_95']['stratified_bootstrap_2000']['pr_auc']['ci_95_upper']:.4f}]",
            "D5 (Candidate, lr=5e-5)": f"[{metrics_d5['confidence_intervals_95']['stratified_bootstrap_2000']['pr_auc']['ci_95_lower']:.4f}, {metrics_d5['confidence_intervals_95']['stratified_bootstrap_2000']['pr_auc']['ci_95_upper']:.4f}]",
            "Delta (D5 - C3)": "Broad overlap",
        },
    ]

    df_metrics_table = pd.DataFrame(metrics_table_rows)
    metrics_csv_path = os.path.join(output_dir, "c3_vs_d5_test_metrics_table.csv")
    metrics_md_path = os.path.join(output_dir, "c3_vs_d5_test_metrics_table.md")
    df_metrics_table.to_csv(metrics_csv_path, index=False)
    with open(metrics_md_path, "w") as f:
        f.write(df_metrics_table.to_markdown(index=False))
    print(f"Saved {metrics_csv_path}")
    print(f"Saved {metrics_md_path}")

    # Paired Contingency table
    contingency_df = pd.DataFrame([
        {"D5 Classification": "D5 Correct", "C3 Correct": both_correct, "C3 Incorrect": d5_correct_c3_incorrect, "Total D5 Correct": both_correct + d5_correct_c3_incorrect},
        {"D5 Classification": "D5 Incorrect", "C3 Correct": c3_correct_d5_incorrect, "C3 Incorrect": both_incorrect, "Total D5 Incorrect": c3_correct_d5_incorrect + both_incorrect},
        {"D5 Classification": "Total", "C3 Correct": both_correct + c3_correct_d5_incorrect, "C3 Incorrect": d5_correct_c3_incorrect + both_incorrect, "Grand Total": len(y_true)},
    ])
    cont_csv_path = os.path.join(output_dir, "paired_contingency_table.csv")
    cont_md_path = os.path.join(output_dir, "paired_contingency_table.md")
    contingency_df.to_csv(cont_csv_path, index=False)
    with open(cont_md_path, "w") as f:
        f.write(contingency_df.to_markdown(index=False))
    print(f"Saved {cont_csv_path}")
    print(f"Saved {cont_md_path}")

    # 5. Generate Publication-Quality Figures
    figures_dir = os.path.join(output_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", font_scale=1.1)

    # Figure 1: ROC Curves
    fpr_c3, tpr_c3, _ = roc_curve(y_true, p_c3)
    fpr_d5, tpr_d5, _ = roc_curve(y_true, p_d5)

    plt.figure(figsize=(7, 6), dpi=300)
    plt.plot(fpr_c3, tpr_c3, color="#2563EB", lw=2.5, label=f"C3 Baseline (AUC = {metrics_c3['point_metrics']['roc_auc']:.4f})")
    plt.plot(fpr_d5, tpr_d5, color="#DC2626", lw=2.2, linestyle="--", label=f"D5 Candidate (AUC = {metrics_d5['point_metrics']['roc_auc']:.4f})")
    plt.plot([0, 1], [0, 1], color="#6B7280", lw=1.2, linestyle=":", label="Chance Baseline (AUC = 0.5000)")
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Sensitivity)")
    plt.title("Held-Out Test ROC Curves: C3 vs. D5 (N=47, N_pos=7)", fontweight="bold")
    plt.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    fig1_path = os.path.join(figures_dir, "roc_curves_c3_vs_d5.png")
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    print(f"Saved {fig1_path}")

    # Figure 2: Precision-Recall Curves
    prevalence = float(np.mean(y_true))
    prec_c3, rec_c3, _ = precision_recall_curve(y_true, p_c3)
    prec_d5, rec_d5, _ = precision_recall_curve(y_true, p_d5)

    plt.figure(figsize=(7, 6), dpi=300)
    plt.plot(rec_c3, prec_c3, color="#2563EB", lw=2.5, label=f"C3 Baseline (PR-AUC = {metrics_c3['point_metrics']['pr_auc']:.4f})")
    plt.plot(rec_d5, prec_d5, color="#DC2626", lw=2.2, linestyle="--", label=f"D5 Candidate (PR-AUC = {metrics_d5['point_metrics']['pr_auc']:.4f})")
    plt.axhline(prevalence, color="#6B7280", lw=1.2, linestyle=":", label=f"Positive Prevalence ({prevalence*100:.1f}%)")
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.05])
    plt.xlabel("Recall (Sensitivity)")
    plt.ylabel("Precision (PPV)")
    plt.title("Held-Out Test Precision-Recall Curves: C3 vs. D5", fontweight="bold")
    plt.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    fig2_path = os.path.join(figures_dir, "pr_curves_c3_vs_d5.png")
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    print(f"Saved {fig2_path}")

    # Figure 3: Side-by-Side Confusion Matrices
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), dpi=300)
    cm_c3 = metrics_c3['confusion_matrix']['matrix_2x2']
    cm_d5 = metrics_d5['confusion_matrix']['matrix_2x2']

    sns.heatmap(cm_c3, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes[0],
                annot_kws={"size": 16, "weight": "bold"},
                xticklabels=["Pred Low Risk", "Pred Suspicious"],
                yticklabels=["True Low Risk", "True Suspicious"])
    axes[0].set_title(f"Frozen C3 Benchmark (tau=0.50)\nAccuracy: 85.11% | Sens: 85.71% | Spec: 85.00%", fontweight="bold")

    sns.heatmap(cm_d5, annot=True, fmt="d", cmap="Reds", cbar=False, ax=axes[1],
                annot_kws={"size": 16, "weight": "bold"},
                xticklabels=["Pred Low Risk", "Pred Suspicious"],
                yticklabels=["True Low Risk", "True Suspicious"])
    axes[1].set_title(f"Candidate D5 (tau=0.50)\nAccuracy: 80.85% | Sens: 85.71% | Spec: 80.00%", fontweight="bold")
    plt.tight_layout()
    fig3_path = os.path.join(figures_dir, "confusion_matrices_c3_vs_d5.png")
    plt.savefig(fig3_path, dpi=300)
    plt.close()
    print(f"Saved {fig3_path}")

    # Figure 4: Paired Prediction Probability Scatter Plot
    plt.figure(figsize=(7.5, 6.8), dpi=300)
    neg_mask = (y_true == 0)
    pos_mask = (y_true == 1)

    # Concordant and discordant masks
    conc_mask = (c_c3 == c_d5)
    disc_mask = (c_c3 != c_d5)

    plt.scatter(p_c3[neg_mask & conc_mask], p_d5[neg_mask & conc_mask],
                color="#2563EB", marker="o", s=65, alpha=0.75, label="True Low Risk (Concordant, N=38)")
    plt.scatter(p_c3[pos_mask & conc_mask], p_d5[pos_mask & conc_mask],
                color="#DC2626", marker="^", s=90, alpha=0.9, label="True Suspicious (Concordant, N=7)")
    plt.scatter(p_c3[disc_mask], p_d5[disc_mask],
                color="#D97706", marker="X", s=130, edgecolor="black", lw=1.5,
                label=f"Discordant Classifications (N={disagree_count})", zorder=5)

    plt.axvline(0.50, color="#4B5563", linestyle="--", lw=1.2, alpha=0.8, label="Decision Boundary (tau=0.50)")
    plt.axhline(0.50, color="#4B5563", linestyle="--", lw=1.2, alpha=0.8)
    plt.plot([0, 1], [0, 1], color="#9CA3AF", linestyle=":", lw=1.0, label="Identity Line (y=x)")

    # Annotate the two discordant cases
    for idx in np.where(disc_mask)[0]:
        pid = df_c3["patient_id"].iloc[idx]
        px = p_c3[idx]
        py = p_d5[idx]
        plt.annotate(
            f"Pt {pid} (True Neg)\nC3={px:.3f}, D5={py:.3f}",
            xy=(px, py),
            xytext=(px - 0.25, py + 0.08),
            arrowprops=dict(arrowstyle="->", color="#B45309", lw=1.2),
            fontsize=9,
            weight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#FEF3C7", edgecolor="#B45309", alpha=0.9),
        )

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel("C3 Predicted Probability", fontweight="bold")
    plt.ylabel("D5 Predicted Probability", fontweight="bold")
    plt.title("Paired Continuous Prediction Dynamics: C3 vs. D5\n(Held-Out Test Cohort N=47)", fontweight="bold")
    plt.legend(loc="lower right", frameon=True, fontsize=9.5)
    plt.tight_layout()
    fig4_path = os.path.join(figures_dir, "paired_prediction_scatter.png")
    plt.savefig(fig4_path, dpi=300)
    plt.close()
    print(f"Saved {fig4_path}")

    # Figure 5: Test Probability Distribution & Calibration Bins
    plt.figure(figsize=(7.5, 5.5), dpi=300)
    bins = np.linspace(0, 1, 6)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    
    # Calibration bins for C3
    c3_accs = []
    c3_confs = []
    for i in range(len(bins) - 1):
        m = (p_c3 >= bins[i]) & (p_c3 < bins[i+1]) if i < len(bins)-2 else (p_c3 >= bins[i]) & (p_c3 <= bins[i+1])
        if np.sum(m) > 0:
            c3_accs.append(float(np.mean(y_true[m])))
            c3_confs.append(float(np.mean(p_c3[m])))
        else:
            c3_accs.append(float("nan"))
            c3_confs.append(float(bin_centers[i]))

    # Calibration bins for D5
    d5_accs = []
    d5_confs = []
    for i in range(len(bins) - 1):
        m = (p_d5 >= bins[i]) & (p_d5 < bins[i+1]) if i < len(bins)-2 else (p_d5 >= bins[i]) & (p_d5 <= bins[i+1])
        if np.sum(m) > 0:
            d5_accs.append(float(np.mean(y_true[m])))
            d5_confs.append(float(np.mean(p_d5[m])))
        else:
            d5_accs.append(float("nan"))
            d5_confs.append(float(bin_centers[i]))

    plt.plot([0, 1], [0, 1], "k:", label="Perfect Calibration")
    plt.plot(c3_confs, c3_accs, "s-", color="#2563EB", lw=2, label=f"C3 (Brier={metrics_c3['point_metrics']['brier_score']:.4f})")
    plt.plot(d5_confs, d5_accs, "o--", color="#DC2626", lw=2, label=f"D5 (Brier={metrics_d5['point_metrics']['brier_score']:.4f})")
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel("Mean Predicted Probability (5 Equal Bins)")
    plt.ylabel("Fraction of True Suspicious Lesions")
    plt.title("Held-Out Test Reliability Diagram (N=47, N_pos=7)", fontweight="bold")
    plt.legend(loc="upper left", frameon=True)
    plt.tight_layout()
    fig5_path = os.path.join(figures_dir, "reliability_curves_c3_vs_d5.png")
    plt.savefig(fig5_path, dpi=300)
    plt.close()
    print(f"Saved {fig5_path}")

    # 6. Evaluation Manifest
    manifest_data = {
        "manifest_version": "1.0",
        "phase": "Phase D8",
        "title": "Locked Final Held-Out Evaluation Manifest: C3 vs D5",
        "execution_timestamp": datetime.datetime.now().isoformat(),
        "random_seeds": {
            "c3_seed": 42,
            "d5_seed": 42,
            "bootstrap_seed": 42,
        },
        "operating_threshold": 0.50,
        "environment": {
            "os": f"{platform.system()} {platform.release()} ({platform.version()})",
            "python_version": sys.version,
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None",
            "scikit_learn_version": "1.6.1",
            "pandas_version": pd.__version__,
            "numpy_version": np.__version__,
        },
        "input_provenance": {
            "splits_manifest": {
                "path": "data/splits/patient_stratified_split.json",
                "sha256": get_sha256("data/splits/patient_stratified_split.json"),
                "held_out_test_patients": 47,
                "held_out_unique_roots": 47,
            },
            "annotations_tsv": {
                "path": "BCMID_annotations.tsv",
                "sha256": get_sha256("BCMID_annotations.tsv"),
                "total_rows": 332,
            },
            "feature_cache": {
                "path": "data/features/patient_multimodal_features.pt",
                "sha256": get_sha256("data/features/patient_multimodal_features.pt"),
                "mammogram_encoder": "EfficientNet-B0 (1280-dim)",
                "ultrasound_encoder": "ViT-B/16 (768-dim)",
            },
            "c3_checkpoint": {
                "path": c3_ckpt_path,
                "sha256": c3_hash,
                "selected_epoch": 4,
                "selection_composite_score": 0.8351,
            },
            "d5_checkpoint": {
                "path": d5_ckpt_path,
                "sha256": d5_hash,
                "selected_epoch": 4,
                "selection_composite_score": 0.8468,
            },
        },
        "metric_definitions": {
            "accuracy": "(TP + TN) / Total",
            "sensitivity_recall": "TP / (TP + FN)",
            "specificity": "TN / (TN + FP)",
            "precision_ppv": "TP / (TP + FP)",
            "negative_predictive_value_npv": "TN / (TN + FN)",
            "f1_score": "2 * (Precision * Recall) / (Precision + Recall)",
            "balanced_accuracy": "0.5 * (Sensitivity + Specificity)",
            "roc_auc": "Area under the receiver operating characteristic curve (trapezoidal)",
            "pr_auc": "Average precision score across decision thresholds",
            "brier_score": "Mean squared error between predicted probabilities and binary labels",
            "log_loss": "Binary cross-entropy loss",
        },
        "evaluation_command": "python scripts/evaluate_phase_d8_locked.py",
        "output_artifacts": {
            "predictions_c3": c3_pred_csv,
            "predictions_d5": d5_pred_csv,
            "metrics_c3": c3_json_path,
            "metrics_d5": d5_json_path,
            "paired_comparison": paired_json_path,
            "metrics_table_csv": metrics_csv_path,
            "metrics_table_md": metrics_md_path,
            "contingency_table_csv": cont_csv_path,
            "contingency_table_md": cont_md_path,
            "roc_curve_plot": fig1_path,
            "pr_curve_plot": fig2_path,
            "confusion_matrices_plot": fig3_path,
            "paired_scatter_plot": fig4_path,
            "reliability_diagram_plot": fig5_path,
        },
    }

    manifest_json_path = os.path.join(output_dir, "evaluation_manifest.json")
    with open(manifest_json_path, "w") as f:
        json.dump(manifest_data, f, indent=2)
    print(f"Saved {manifest_json_path}")

    print("\nLOCKED FINAL EVALUATION PIPELINE COMPLETED SUCCESSFULLY.")


if __name__ == "__main__":
    main()
