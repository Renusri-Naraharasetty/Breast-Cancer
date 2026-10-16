"""Visualization module for Phase 5: Held-Out Test Evaluation.

Generates publication-quality plots:
- Confusion matrices (Image-level and Patient-level)
- ROC curve comparisons (Image-level and Patient-level)
- Precision-Recall curve comparisons (Image-level and Patient-level)
- Individual training and validation learning curves
- Validation loss and accuracy overlays
- Model comparison multi-metric bar charts
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, Any, List

# Set clean aesthetic style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8


def plot_confusion_matrix_pair(
    cm_image: List[List[int]],
    cm_patient: List[List[int]],
    exp_id: str,
    model_name: str,
    modality: str,
    save_path: str,
):
    """Plots side-by-side confusion matrices for Image-level and Patient-level evaluation."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=300)

    class_names = ["Low-Risk\n(BI-RADS 1-3)", "Suspicious\n(BI-RADS 4-6)"]

    for ax, cm, title in zip(
        axes,
        [cm_image, cm_patient],
        ["Image-Level Confusion Matrix", "Patient-Level Confusion Matrix (Max-Pooling)"],
    ):
        cm_arr = np.array(cm)
        total = np.sum(cm_arr)
        # Row-normalized percentages
        cm_norm = cm_arr.astype("float") / cm_arr.sum(axis=1)[:, np.newaxis]
        cm_norm = np.nan_to_num(cm_norm)

        # Annotations text
        annot = np.empty_like(cm_arr, dtype=object)
        for i in range(2):
            for j in range(2):
                annot[i, j] = f"{cm_arr[i, j]}\n({cm_norm[i, j]:.1%})"

        sns.heatmap(
            cm_arr,
            annot=annot,
            fmt="",
            cmap="Blues",
            cbar=False,
            ax=ax,
            xticklabels=class_names,
            yticklabels=class_names,
            annot_kws={"size": 11, "weight": "bold"},
        )
        ax.set_title(f"{title}\n(N={total})", fontsize=11, fontweight="bold", pad=10)
        ax.set_ylabel("True Ground Truth", fontsize=10)
        ax.set_xlabel("Predicted Label (Threshold = 0.50)", fontsize=10)

    fig.suptitle(
        f"Test Set Evaluation: {model_name} on {modality.capitalize()} ({exp_id})",
        fontsize=13,
        fontweight="bold",
        y=1.03,
    )
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_roc_comparison(results: List[Dict[str, Any]], level: str, save_path: str):
    """Plots combined ROC curves for all models on a single graph."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(7.5, 6), dpi=300)

    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    metric_key = "image_metrics" if level == "image" else "patient_metrics"
    level_title = "Image-Level" if level == "image" else "Patient-Level (Max-Pooling)"

    for i, res in enumerate(results):
        metrics = res[metric_key]
        curve = metrics["curve_data"]
        fpr = curve["fpr"]
        tpr = curve["tpr"]
        auc = metrics["roc_auc"]
        label = f"{res['model_name']} ({res['modality']}) — AUC: {auc:.4f}"
        plt.plot(fpr, tpr, label=label, color=palette[i % len(palette)], linewidth=2)

    plt.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Chance (AUC = 0.5000)")
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    plt.ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11)
    plt.title(f"ROC Curves Comparison — {level_title}\nHeld-Out Test Set", fontsize=12, fontweight="bold")
    plt.legend(loc="lower right", frameon=True, fontsize=9.5)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_pr_comparison(results: List[Dict[str, Any]], level: str, save_path: str):
    """Plots combined Precision-Recall curves for all models on a single graph."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(7.5, 6), dpi=300)

    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    metric_key = "image_metrics" if level == "image" else "patient_metrics"
    level_title = "Image-Level" if level == "image" else "Patient-Level (Max-Pooling)"

    for i, res in enumerate(results):
        metrics = res[metric_key]
        curve = metrics["curve_data"]
        precision = curve["precision_curve"]
        recall = curve["recall_curve"]
        pr_auc = metrics["pr_auc"]
        label = f"{res['model_name']} ({res['modality']}) — PR-AUC: {pr_auc:.4f}"
        plt.plot(recall, precision, label=label, color=palette[i % len(palette)], linewidth=2)

    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel("Recall (Sensitivity)", fontsize=11)
    plt.ylabel("Precision (Positive Predictive Value)", fontsize=11)
    plt.title(f"Precision-Recall Curves Comparison — {level_title}\nHeld-Out Test Set", fontsize=12, fontweight="bold")
    plt.legend(loc="upper right", frameon=True, fontsize=9.5)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_single_experiment_history(history_csv: str, best_epoch: int, exp_title: str, save_path: str):
    """Plots training vs. validation loss and accuracy curves for a single experiment."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df = pd.read_csv(history_csv)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=300)

    # Loss Curve
    ax1.plot(df["epoch"], df["train_loss"], "o-", label="Train Loss", color="#1f77b4", linewidth=1.8, markersize=4)
    ax1.plot(df["epoch"], df["val_loss"], "s--", label="Val Loss", color="#ff7f0e", linewidth=1.8, markersize=4)
    ax1.axvline(x=best_epoch, color="#2ca02c", linestyle=":", linewidth=2, label=f"Best Checkpoint (Ep {best_epoch})")
    ax1.set_xlabel("Epoch", fontsize=10)
    ax1.set_ylabel("Loss", fontsize=10)
    ax1.set_title("Cross-Entropy Loss vs. Epoch", fontsize=11, fontweight="bold")
    ax1.legend(loc="upper right", frameon=True, fontsize=9)

    # Accuracy Curve
    ax2.plot(df["epoch"], df["train_acc"] * 100, "o-", label="Train Accuracy", color="#1f77b4", linewidth=1.8, markersize=4)
    ax2.plot(df["epoch"], df["val_acc"] * 100, "s--", label="Val Accuracy", color="#ff7f0e", linewidth=1.8, markersize=4)
    ax2.axvline(x=best_epoch, color="#2ca02c", linestyle=":", linewidth=2, label=f"Best Checkpoint (Ep {best_epoch})")
    ax2.set_xlabel("Epoch", fontsize=10)
    ax2.set_ylabel("Accuracy (%)", fontsize=10)
    ax2.set_title("Classification Accuracy vs. Epoch", fontsize=11, fontweight="bold")
    ax2.legend(loc="lower right", frameon=True, fontsize=9)

    fig.suptitle(f"Training & Validation History: {exp_title}", fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_training_overlays(histories: Dict[str, pd.DataFrame], save_dir: str):
    """Plots comparative overlays of validation loss and validation accuracy across all four models."""
    os.makedirs(save_dir, exist_ok=True)
    colors = {
        "exp1_efficientnet_mammogram": "#1f77b4",
        "exp2_vit_ultrasound": "#ff7f0e",
        "exp3_vit_mammogram": "#2ca02c",
        "exp4_efficientnet_ultrasound": "#d62728",
    }
    labels = {
        "exp1_efficientnet_mammogram": "EfficientNet-B0 (Mammogram)",
        "exp2_vit_ultrasound": "ViT-B/16 (Ultrasound)",
        "exp3_vit_mammogram": "ViT-B/16 (Mammogram)",
        "exp4_efficientnet_ultrasound": "EfficientNet-B0 (Ultrasound)",
    }

    # 1. Loss Overlay
    plt.figure(figsize=(8, 5), dpi=300)
    for exp_id, df in histories.items():
        plt.plot(df["epoch"], df["val_loss"], "o-", label=labels.get(exp_id, exp_id), color=colors.get(exp_id, None), linewidth=1.8, markersize=4)
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Validation Loss", fontsize=11)
    plt.title("Validation Loss Comparison Across All 4 Experiments", fontsize=12, fontweight="bold")
    plt.legend(frameon=True, fontsize=9.5)
    plt.tight_layout()
    loss_path = os.path.join(save_dir, "val_loss_comparison_overlay.png")
    plt.savefig(loss_path, bbox_inches="tight")
    plt.close()

    # 2. Accuracy Overlay
    plt.figure(figsize=(8, 5), dpi=300)
    for exp_id, df in histories.items():
        plt.plot(df["epoch"], df["val_acc"] * 100, "s--", label=labels.get(exp_id, exp_id), color=colors.get(exp_id, None), linewidth=1.8, markersize=4)
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Validation Accuracy (%)", fontsize=11)
    plt.title("Validation Accuracy Comparison Across All 4 Experiments", fontsize=12, fontweight="bold")
    plt.legend(frameon=True, fontsize=9.5)
    plt.tight_layout()
    acc_path = os.path.join(save_dir, "val_accuracy_comparison_overlay.png")
    plt.savefig(acc_path, bbox_inches="tight")
    plt.close()


def plot_metrics_comparison_barchart(metrics_df: pd.DataFrame, save_path: str):
    """Plots combined model-comparison metric bar charts for Image and Patient levels."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    metrics_to_plot = ["accuracy", "recall", "specificity", "f1_score", "roc_auc", "pr_auc"]
    metric_labels = ["Accuracy", "Sensitivity\n(Recall)", "Specificity", "F1-Score", "ROC-AUC", "PR-AUC"]

    for ax, level in zip(axes, ["image", "patient"]):
        sub_df = metrics_df[metrics_df["level"] == level].copy()
        
        # Prepare data for plotting
        plot_rows = []
        for _, row in sub_df.iterrows():
            m_label = f"{row['model_name']}\n({row['modality'][:3].upper()})"
            for m, m_name in zip(metrics_to_plot, metric_labels):
                val = row[m] if pd.notnull(row[m]) else 0.0
                plot_rows.append({"Model": m_label, "Metric": m_name, "Value": val})
        
        pdf = pd.DataFrame(plot_rows)
        sns.barplot(data=pdf, x="Metric", y="Value", hue="Model", palette="tab10", ax=ax, edgecolor="#333333", linewidth=0.5)
        
        level_name = "Image-Level" if level == "image" else "Patient-Level (Max-Pooling)"
        ax.set_title(f"Test Set Performance: {level_name}", fontsize=11, fontweight="bold")
        ax.set_ylabel("Metric Value (0.0 - 1.0)", fontsize=10)
        ax.set_xlabel("", fontsize=10)
        ax.set_ylim([0, 1.05])
        ax.legend(title="Experiment", fontsize=8.5, title_fontsize=9)

    fig.suptitle("Independent Model Comparison on Held-Out Test Set (47 Patients)", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
