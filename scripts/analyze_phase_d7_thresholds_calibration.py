"""Phase D7: Validation-Only Threshold and Calibration Analysis.

Models evaluated:
1. Frozen C3 Benchmark (across seeds 42, 123, 456)
2. Leading Candidate D5 (across seeds 42, 123, 456)

Strict Governance:
- Evaluation ONLY on the 48 validation patients.
- The 47-patient test cohort is STRICTLY QUARANTINED (never loaded or accessed).
- Preserves all existing checkpoints, features, configs, and reports.
- Output directory: outputs/phase_d/threshold_analysis/ and outputs/phase_d/calibration/
"""

import os
import sys
import math
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Tuple
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix,
    brier_score_loss,
    log_loss,
    roc_auc_score,
    average_precision_score,
)

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 5) -> Tuple[float, float, List[Dict[str, Any]]]:
    """Computes Expected Calibration Error (ECE) and Maximum Calibration Error (MCE)."""
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    mce = 0.0
    bin_details = []

    for i in range(n_bins):
        b_low = bin_edges[i]
        b_high = bin_edges[i + 1]
        if i == n_bins - 1:
            in_bin = (y_prob >= b_low) & (y_prob <= b_high)
        else:
            in_bin = (y_prob >= b_low) & (y_prob < b_high)

        n_in_bin = np.sum(in_bin)
        if n_in_bin > 0:
            bin_acc = np.mean(y_true[in_bin])
            bin_conf = np.mean(y_prob[in_bin])
            abs_err = abs(bin_acc - bin_conf)
            ece += (n_in_bin / len(y_true)) * abs_err
            mce = max(mce, abs_err)
            bin_details.append({
                "bin_idx": i + 1,
                "range": f"[{b_low:.1f}, {b_high:.1f}]",
                "count": int(n_in_bin),
                "accuracy": round(float(bin_acc), 4),
                "confidence": round(float(bin_conf), 4),
                "calibration_gap": round(float(abs_err), 4),
            })
        else:
            bin_details.append({
                "bin_idx": i + 1,
                "range": f"[{b_low:.1f}, {b_high:.1f}]",
                "count": 0,
                "accuracy": None,
                "confidence": None,
                "calibration_gap": None,
            })

    return ece, mce, bin_details


def run_threshold_and_calibration_analysis():
    print("=" * 80)
    print("PHASE D7: VALIDATION-ONLY THRESHOLD & CALIBRATION ANALYSIS")
    print("Models: Frozen C3 vs Leading Candidate D5 across Seeds [42, 123, 456]")
    print("Test Cohort Quarantine: STRICTLY MAINTAINED (NEVER LOADED)")
    print("=" * 80)

    # 1. Output directories
    out_thresh_dir = "outputs/phase_d/threshold_analysis"
    out_calib_dir = "outputs/phase_d/calibration"
    out_fig_dir = "outputs/phase_d/figures"
    out_table_dir = "outputs/phase_d/tables"
    for d in [out_thresh_dir, out_calib_dir, out_fig_dir, out_table_dir]:
        os.makedirs(d, exist_ok=True)

    seeds = [42, 123, 456]
    models = ["c3", "d5"]
    pred_files = {
        "c3": {s: f"outputs/phase_d/experiments/exp_d6_c3_seed{s}/val_predictions.csv" for s in seeds},
        "d5": {s: f"outputs/phase_d/experiments/exp_d6_d5_seed{s}/val_predictions.csv" for s in seeds},
    }

    # Task 1: Verify prediction files integrity & concordance
    dfs = {}
    reference_patients = None
    reference_labels = None

    print("\n--- TASK 1: VERIFYING PREDICTION FILES INTEGRITY ---")
    for m in models:
        dfs[m] = {}
        for s in seeds:
            path = pred_files[m][s]
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing prediction file: {path}")
            df = pd.read_csv(path)
            assert len(df) == 48, f"Expected 48 rows, got {len(df)} in {path}"

            patients = df["patient_id"].tolist()
            labels = df["true_label"].tolist()

            if reference_patients is None:
                reference_patients = patients
                reference_labels = labels
                assert labels.count(0) == 40 and labels.count(1) == 8, f"Cohort distribution mismatch in {path}"
                print(f"[*] Reference cohort established: 48 validation patients (40 negative, 8 positive).")
            else:
                assert patients == reference_patients, f"Patient order/list mismatch in {path}"
                assert labels == reference_labels, f"Ground-truth label mismatch in {path}"

            dfs[m][s] = df
            print(f"  [OK] Verified {m.upper()} Seed {s}: 48 patients, 100% label concordance.")

    print("All prediction files verified. Zero split contamination.\n")

    # Task 2 & 3: Evaluate thresholds 0.30 to 0.70 in 0.05 increments
    thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
    per_seed_records = []

    print("--- TASK 2, 3, 4: COMPUTING THRESHOLD OPERATING POINTS ---")
    for m in models:
        for s in seeds:
            df = dfs[m][s]
            y_true = df["true_label"].values
            y_prob = df["pred_prob"].values

            roc_auc = roc_auc_score(y_true, y_prob)
            pr_auc = average_precision_score(y_true, y_prob)

            for t in thresholds:
                t = round(t, 2)
                y_pred = (y_prob >= t).astype(int)

                cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
                tn, fp, fn, tp = cm.ravel()

                sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                f1 = 2 * prec * sens / (prec + sens) if (prec + sens) > 0 else 0.0
                bal_acc = 0.5 * (sens + spec)
                acc = (tp + tn) / len(y_true)
                comp = 0.5 * roc_auc + 0.3 * pr_auc + 0.2 * f1

                per_seed_records.append({
                    "model": m.upper(),
                    "seed": s,
                    "threshold": t,
                    "tn": int(tn),
                    "fp": int(fp),
                    "fn": int(fn),
                    "tp": int(tp),
                    "sensitivity": round(float(sens), 4),
                    "specificity": round(float(spec), 4),
                    "precision": round(float(prec), 4),
                    "f1": round(float(f1), 4),
                    "balanced_acc": round(float(bal_acc), 4),
                    "accuracy": round(float(acc), 4),
                    "roc_auc": round(float(roc_auc), 4),
                    "pr_auc": round(float(pr_auc), 4),
                    "composite_score": round(float(comp), 4),
                })

    df_per_seed = pd.DataFrame(per_seed_records)
    per_seed_csv = os.path.join(out_thresh_dir, "per_seed_threshold_metrics.csv")
    df_per_seed.to_csv(per_seed_csv, index=False)
    print(f"Saved per-seed threshold metrics to: {per_seed_csv}")

    # Task 5: Compute across-seed mean and standard deviation (do not treat seeds as independent patients)
    metric_cols = ["sensitivity", "specificity", "precision", "f1", "balanced_acc", "accuracy", "composite_score", "tp", "fp", "fn", "tn"]
    agg_records = []

    for m in ["C3", "D5"]:
        for t in thresholds:
            t = round(t, 2)
            sub = df_per_seed[(df_per_seed["model"] == m) & (df_per_seed["threshold"] == t)]
            rec = {
                "model": m,
                "threshold": t,
            }
            for col in metric_cols:
                rec[f"{col}_mean"] = round(float(sub[col].mean()), 4)
                rec[f"{col}_std"] = round(float(sub[col].std(ddof=1)), 4)
            agg_records.append(rec)

    df_agg = pd.DataFrame(agg_records)
    agg_csv = os.path.join(out_thresh_dir, "averaged_threshold_metrics.csv")
    df_agg.to_csv(agg_csv, index=False)
    print(f"Saved across-seed averaged threshold metrics to: {agg_csv}")

    # Task 6: Calibration Analysis
    print("\n--- TASK 6: PROBABILITY CALIBRATION ANALYSIS ---")
    calib_records = []

    for m in models:
        for s in seeds:
            df = dfs[m][s]
            y_true = df["true_label"].values
            y_prob = df["pred_prob"].values

            brier = brier_score_loss(y_true, y_prob)
            ll = log_loss(y_true, y_prob, labels=[0, 1])
            ece, mce, bin_details = compute_ece(y_true, y_prob, n_bins=5)

            calib_records.append({
                "model": m.upper(),
                "seed": s,
                "brier_score": round(float(brier), 4),
                "log_loss": round(float(ll), 4),
                "ece_5bin": round(float(ece), 4),
                "mce_5bin": round(float(mce), 4),
                "bin_details": bin_details,
            })

    # Summary table for calibration
    df_calib_summary = pd.DataFrame([{
        "model": r["model"],
        "seed": r["seed"],
        "brier_score": r["brier_score"],
        "log_loss": r["log_loss"],
        "ece_5bin": r["ece_5bin"],
        "mce_5bin": r["mce_5bin"],
    } for r in calib_records])

    calib_csv = os.path.join(out_calib_dir, "validation_calibration_metrics.csv")
    df_calib_summary.to_csv(calib_csv, index=False)
    print(f"Saved calibration metrics to: {calib_csv}")

    # Also save bin details as json
    with open(os.path.join(out_calib_dir, "calibration_bins_detailed.json"), "w", encoding="utf-8") as f:
        json.dump(calib_records, f, indent=2)

    # Compute mean and std of calibration metrics
    calib_agg = []
    for m in ["C3", "D5"]:
        sub = df_calib_summary[df_calib_summary["model"] == m]
        calib_agg.append({
            "model": m,
            "brier_score_mean": round(float(sub["brier_score"].mean()), 4),
            "brier_score_std": round(float(sub["brier_score"].std(ddof=1)), 4),
            "log_loss_mean": round(float(sub["log_loss"].mean()), 4),
            "log_loss_std": round(float(sub["log_loss"].std(ddof=1)), 4),
            "ece_mean": round(float(sub["ece_5bin"].mean()), 4),
            "ece_std": round(float(sub["ece_5bin"].std(ddof=1)), 4),
            "mce_mean": round(float(sub["mce_5bin"].mean()), 4),
            "mce_std": round(float(sub["mce_5bin"].std(ddof=1)), 4),
        })
    df_calib_agg = pd.DataFrame(calib_agg)
    df_calib_agg.to_csv(os.path.join(out_calib_dir, "averaged_calibration_metrics.csv"), index=False)

    # 7. Generate High-Quality Publication Figures
    print("\n--- GENERATING PLOTS ---")
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # Figure 1: Sensitivity vs. Specificity Trade-Off Curve across Thresholds
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

    for m, col, label, mark in [("C3", "#1f77b4", "Frozen C3 Benchmark (lr=1e-4)", "o"),
                                ("D5", "#d62728", "Candidate D5 (lr=5e-5)", "s")]:
        sub = df_agg[df_agg["model"] == m].sort_values("threshold")
        threshs = sub["threshold"].values
        sens = sub["sensitivity_mean"].values
        sens_std = sub["sensitivity_std"].values
        spec = sub["specificity_mean"].values
        spec_std = sub["specificity_std"].values

        ax.plot(1 - spec, sens, color=col, marker=mark, linewidth=2.2, label=label, markersize=7)
        # Annotate thresholds
        for x, y, t in zip(1 - spec, sens, threshs):
            offset_y = 0.02 if m == "D5" else -0.025
            offset_x = 0.015 if m == "D5" else -0.025
            if t in [0.30, 0.40, 0.50, 0.60, 0.70]:
                ax.annotate(f"$\\tau$={t:.2f}", (x, y), textcoords="offset points",
                            xytext=(offset_x * 200, offset_y * 200), fontsize=8.5,
                            color=col, fontweight="bold" if t == 0.50 else "normal",
                            arrowprops=dict(arrowstyle="->", color=col, lw=0.7) if t == 0.50 else None)

    # Highlight canonical 0.50 operating point
    ax.scatter([1 - 0.925], [0.75], color="#1f77b4", s=140, edgecolors="black", zorder=5, label="C3 @ 0.50 (Canonical)")
    ax.scatter([1 - 0.9167], [0.8333], color="#d62728", s=140, edgecolors="black", zorder=5, label="D5 @ 0.50 (Canonical)")

    ax.set_title("Operating Point Trade-Off: Sensitivity vs (1 - Specificity)\nAcross Thresholds [0.30 - 0.70] (Validation Split, N=48)", fontsize=12, pad=12, fontweight="bold")
    ax.set_xlabel("1 - Specificity (False Positive Rate)", fontsize=11)
    ax.set_ylabel("Sensitivity (True Positive Rate / Recall)", fontsize=11)
    ax.set_xlim([-0.02, 0.35])
    ax.set_ylim([0.45, 1.02])
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="lower right", frameon=True, fontsize=9.5)
    plt.tight_layout()
    fig1_path = os.path.join(out_fig_dir, "threshold_sensitivity_specificity_tradeoff.png")
    fig.savefig(fig1_path)
    plt.close(fig)
    print(f"Saved trade-off plot to: {fig1_path}")

    # Figure 2: Sensitivity, Specificity, F1, Balanced Accuracy vs Threshold
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    for idx, (m, col, m_title) in enumerate([("C3", "#1f77b4", "Frozen C3 Benchmark"), ("D5", "#d62728", "Candidate D5")]):
        ax = axes[idx]
        sub = df_agg[df_agg["model"] == m].sort_values("threshold")
        t = sub["threshold"].values

        ax.plot(t, sub["sensitivity_mean"], label="Sensitivity (Recall)", color="#2ca02c", marker="o", lw=2)
        ax.fill_between(t, sub["sensitivity_mean"] - sub["sensitivity_std"], sub["sensitivity_mean"] + sub["sensitivity_std"], color="#2ca02c", alpha=0.15)

        ax.plot(t, sub["specificity_mean"], label="Specificity", color="#1f77b4", marker="s", lw=2)
        ax.fill_between(t, sub["specificity_mean"] - sub["specificity_std"], sub["specificity_mean"] + sub["specificity_std"], color="#1f77b4", alpha=0.15)

        ax.plot(t, sub["f1_mean"], label="F1-Score", color="#ff7f0e", marker="^", lw=2)
        ax.plot(t, sub["balanced_acc_mean"], label="Balanced Accuracy", color="#9467bd", marker="d", lw=2, linestyle="--")

        ax.axvline(0.50, color="gray", linestyle=":", lw=1.5, label="Canonical $\\tau=0.50$")

        ax.set_title(f"{m_title}\nMetrics Across Thresholds [0.30 - 0.70]", fontsize=11, fontweight="bold")
        ax.set_xlabel("Classification Threshold ($\\tau$)", fontsize=10.5)
        ax.set_ylabel("Validation Metric Value", fontsize=10.5)
        ax.set_ylim([0.40, 1.02])
        ax.set_xlim([0.28, 0.72])
        ax.grid(True, linestyle="--", alpha=0.6)
        if idx == 0:
            ax.legend(loc="lower left", fontsize=9, frameon=True)

    plt.tight_layout()
    fig2_path = os.path.join(out_fig_dir, "threshold_metrics_curve_c3_vs_d5.png")
    fig.savefig(fig2_path)
    plt.close(fig)
    print(f"Saved threshold metric curves to: {fig2_path}")

    # Figure 3: Calibration Curves (Reliability Diagrams)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), dpi=300)

    for idx, (m, col, m_title) in enumerate([("C3", "#1f77b4", "Frozen C3"), ("D5", "#d62728", "Candidate D5")]):
        ax = axes[idx]
        ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration", lw=1.5)

        # Plot each seed's calibration points
        for s, mark in zip(seeds, ["o", "s", "^"]):
            # Find the bin details
            rec = next(r for r in calib_records if r["model"] == m and r["seed"] == s)
            confs = []
            accs = []
            for b in rec["bin_details"]:
                if b["count"] > 0:
                    confs.append(b["confidence"])
                    accs.append(b["accuracy"])
            ax.plot(confs, accs, marker=mark, color=col, alpha=0.75, lw=1.5,
                    label=f"Seed {s} (ECE={rec['ece_5bin']:.3f})")

        ax.set_title(f"{m_title} Reliability Diagram\n(5 Bins, Validation Cohort N=48)", fontsize=11, fontweight="bold")
        ax.set_xlabel("Mean Predicted Probability (Confidence)", fontsize=10.5)
        ax.set_ylabel("Empirical Fraction of Positives (Accuracy)", fontsize=10.5)
        ax.set_xlim([-0.05, 1.05])
        ax.set_ylim([-0.05, 1.05])
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(loc="upper left", fontsize=9, frameon=True)

    plt.tight_layout()
    fig3_path = os.path.join(out_fig_dir, "reliability_diagram_c3_vs_d5.png")
    fig.savefig(fig3_path)
    plt.close(fig)
    print(f"Saved calibration reliability diagrams to: {fig3_path}")

    print("\nPhase D7 analysis script completed successfully.")


if __name__ == "__main__":
    run_threshold_and_calibration_analysis()
