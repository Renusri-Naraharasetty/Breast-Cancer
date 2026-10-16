"""Plots threshold sweep curves for GIT 1 and GIT 2 validation sets."""

import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_sweeps():
    csv1 = "outputs/phase5/tables/threshold_optimization_val_sweep_git1.csv"
    csv2 = "outputs/phase5/tables/threshold_optimization_val_sweep_git2.csv"
    out_png = "outputs/phase5/comparisons/threshold_optimization_curves.png"

    df1 = pd.read_csv(csv1)
    df2 = pd.read_csv(csv2)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    # Plot GIT 1
    ax = axes[0]
    ax.plot(df1["threshold"], df1["f1_score"], label="F1-Score", color="#2563EB", linewidth=2.2)
    ax.plot(df1["threshold"], df1["balanced_accuracy"], label="Balanced Accuracy", color="#059669", linewidth=2.2)
    ax.plot(df1["threshold"], df1["sensitivity"], label="Sensitivity", color="#DC2626", linestyle="--", linewidth=1.8)
    ax.plot(df1["threshold"], df1["specificity"], label="Specificity", color="#7C3AED", linestyle=":", linewidth=1.8)

    ax.axvline(0.50, color="#6B7280", linestyle="-.", label="Baseline (0.50)", alpha=0.8)
    ax.axvline(0.42, color="#2563EB", linestyle="--", label="Max F1/BalAcc (0.42)", alpha=0.9)
    ax.axvline(0.19, color="#DC2626", linestyle=":", label="High Sens >=90% (0.19)", alpha=0.9)

    ax.set_title("GIT 1 (EffNet MG + ViT US)\nValidation Threshold Sweep", fontsize=12, fontweight="bold")
    ax.set_xlabel("Decision Threshold (tau)", fontsize=11)
    ax.set_ylabel("Validation Metric Value", fontsize=11)
    ax.set_xlim(0.05, 0.95)
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="lower left", fontsize=9, framealpha=0.9)

    # Plot GIT 2
    ax = axes[1]
    ax.plot(df2["threshold"], df2["f1_score"], label="F1-Score", color="#2563EB", linewidth=2.2)
    ax.plot(df2["threshold"], df2["balanced_accuracy"], label="Balanced Accuracy", color="#059669", linewidth=2.2)
    ax.plot(df2["threshold"], df2["sensitivity"], label="Sensitivity", color="#DC2626", linestyle="--", linewidth=1.8)
    ax.plot(df2["threshold"], df2["specificity"], label="Specificity", color="#7C3AED", linestyle=":", linewidth=1.8)

    ax.axvline(0.50, color="#6B7280", linestyle="-.", label="Baseline (0.50)", alpha=0.8)
    ax.axvline(0.76, color="#2563EB", linestyle="--", label="Max F1/BalAcc (0.76)", alpha=0.9)
    ax.axvline(0.20, color="#DC2626", linestyle=":", label="High Sens >=90% (0.20)", alpha=0.9)

    ax.set_title("GIT 2 (ViT MG + EffNet US)\nValidation Threshold Sweep", fontsize=12, fontweight="bold")
    ax.set_xlabel("Decision Threshold (tau)", fontsize=11)
    ax.set_xlim(0.05, 0.95)
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="lower left", fontsize=9, framealpha=0.9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f"Saved threshold sweep curves to: {out_png}")

if __name__ == "__main__":
    plot_sweeps()
