import os
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
import matplotlib
matplotlib.use('Agg') # Non-interactive headless backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix


def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    roc_auc: float,
    save_path: str,
    title: str = "ROC Curve"
) -> None:
    """Plots and saves the Receiver Operating Characteristic (ROC) curve."""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color="#2563EB", lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="#9CA3AF", lw=1.5, linestyle="--", label="Chance")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=12)
    plt.ylabel("True Positive Rate (Sensitivity)", fontsize=12)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_pr_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    pr_auc: float,
    save_path: str,
    title: str = "Precision-Recall Curve"
) -> None:
    """Plots and saves the Precision-Recall curve."""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    
    plt.figure(figsize=(7, 6))
    plt.plot(recall, precision, color="#059669", lw=2, label=f"PR curve (AUC = {pr_auc:.4f})")
    baseline = np.sum(y_true) / len(y_true) if len(y_true) > 0 else 0.0
    plt.plot([0, 1], [baseline, baseline], color="#9CA3AF", lw=1.5, linestyle="--", label=f"Baseline ({baseline:.3f})")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall (Sensitivity)", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.legend(loc="lower left", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_confusion_matrix(
    cm_dict: Dict[str, int],
    save_path: str,
    title: str = "Confusion Matrix"
) -> None:
    """Plots and saves an annotated confusion matrix heatmap."""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    cm = np.array([
        [cm_dict["tn"], cm_dict["fp"]],
        [cm_dict["fn"], cm_dict["tp"]]
    ])
    
    plt.figure(figsize=(6, 5))
    labels = ["Benign (0)", "Malignant (1)"]
    
    # Calculate percentages
    total = np.sum(cm)
    cm_text = np.array([
        [f"{cm[0,0]}\n({cm[0,0]/total*100:.1f}%)", f"{cm[0,1]}\n({cm[0,1]/total*100:.1f}%)"],
        [f"{cm[1,0]}\n({cm[1,0]/total*100:.1f}%)", f"{cm[1,1]}\n({cm[1,1]/total*100:.1f}%)"]
    ])
    
    sns.heatmap(cm, annot=cm_text, fmt="", cmap="Blues", cbar=True,
                xticklabels=labels, yticklabels=labels, annot_kws={"size": 13, "weight": "bold"})
    plt.xlabel("Predicted Class", fontsize=12)
    plt.ylabel("True Ground-Truth Class", fontsize=12)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_training_history(
    history: Dict[str, List[float]],
    save_path: str,
    title: str = "Training & Validation Curves"
) -> None:
    """Plots loss and metric progression across training epochs."""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    epochs = range(1, len(history["train_loss"]) + 1)
    
    plt.figure(figsize=(12, 5))
    
    # Loss subplot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], 'o-', color="#DC2626", label="Train Loss")
    if "val_loss" in history and history["val_loss"]:
        plt.plot(epochs, history["val_loss"], 's-', color="#2563EB", label="Val Loss")
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Loss", fontsize=11)
    plt.title("Loss vs. Epochs", fontsize=13, fontweight="bold")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Metric subplot (ROC-AUC or F1)
    plt.subplot(1, 2, 2)
    if "val_roc_auc" in history and history["val_roc_auc"]:
        plt.plot(epochs, history["val_roc_auc"], '^-', color="#059669", label="Val ROC-AUC")
    if "val_f1" in history and history["val_f1"]:
        plt.plot(epochs, history["val_f1"], 'd-', color="#7C3AED", label="Val F1")
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Score", fontsize=11)
    plt.title("Validation Metrics vs. Epochs", fontsize=13, fontweight="bold")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_model_comparison(
    comparison_data: Dict[str, Dict[str, float]],
    save_path: str
) -> None:
    """Generates comparative multi-metric bar chart comparing all evaluated models."""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    models = list(comparison_data.keys())
    metrics = ["accuracy", "sensitivity", "specificity", "f1", "roc_auc", "pr_auc"]
    metric_names = ["Accuracy", "Sensitivity", "Specificity", "F1-Score", "ROC-AUC", "PR-AUC"]
    
    x = np.arange(len(metric_names))
    width = 0.8 / len(models)
    
    plt.figure(figsize=(14, 7))
    colors = ["#93C5FD", "#FCA5A5", "#FDE047", "#3B82F6"]
    
    for idx, (m_name, m_metrics) in enumerate(comparison_data.items()):
        vals = [
            m_metrics.get("accuracy", 0.0),
            m_metrics.get("recall", m_metrics.get("sensitivity", 0.0)),
            m_metrics.get("specificity", 0.0),
            m_metrics.get("f1", 0.0),
            m_metrics.get("roc_auc", 0.0),
            m_metrics.get("pr_auc", 0.0)
        ]
        color = colors[idx % len(colors)]
        offset = (idx - len(models) / 2 + 0.5) * width
        plt.bar(x + offset, vals, width, label=m_name, color=color, edgecolor="black", alpha=0.85)
        
    plt.xlabel("Evaluation Metric", fontsize=12, fontweight="bold")
    plt.ylabel("Score", fontsize=12, fontweight="bold")
    plt.title("Model Performance Comparison (BCMID Test Cohort)", fontsize=15, fontweight="bold")
    plt.xticks(x, metric_names, fontsize=11)
    plt.ylim([0.0, 1.1])
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
