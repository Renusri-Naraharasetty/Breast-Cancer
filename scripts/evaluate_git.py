"""Final Evaluation and Comparative Analysis Script for Multimodal GIT.

Evaluates the saved best checkpoint on the frozen 47-patient test partition:
- Fixed decision threshold tau = 0.50
- Patient-level predictions
- Comprehensive metrics: Accuracy, Precision, Sensitivity, Specificity, F1, G-Mean, ROC-AUC, PR-AUC
- Patient-clustered bootstrap uncertainty intervals (95% CI)
- Saves:
  1. Patient predictions CSV
  2. Evaluation JSON
  3. Comparison table (CSV and JSON) comparing individual models (exp1, exp2) and combined GIT model
  4. Comparison figures (ROC, PR, confusion matrix, bar chart)
"""

import os
import sys
import json
import math
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Tuple, Optional
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.git import GraphInfusedTransformer
from src.data.multimodal_dataset import get_patient_multimodal_dataloader
from src.evaluation.evaluator import compute_classification_metrics


def run_patient_bootstrap_ci(
    df_patients: pd.DataFrame,
    n_bootstraps: int = 2000,
    seed: int = 42,
    threshold: float = 0.50,
) -> Dict[str, Dict[str, float]]:
    rng = np.random.RandomState(seed)
    n = len(df_patients)
    metrics_samples = {
        "accuracy": [],
        "sensitivity": [],
        "specificity": [],
        "precision": [],
        "f1_score": [],
        "roc_auc": [],
        "pr_auc": [],
    }

    y_true_all = df_patients["true_label"].values
    y_prob_all = df_patients["pred_prob"].values

    for _ in range(n_bootstraps):
        boot_idx = rng.choice(n, size=n, replace=True)
        y_t = y_true_all[boot_idx]
        y_p = y_prob_all[boot_idx]

        # Skip if single-class resample
        if len(np.unique(y_t)) < 2:
            continue

        y_pred = (y_p >= threshold).astype(int)

        acc = accuracy_score(y_t, y_pred)
        prec = precision_score(y_t, y_pred, zero_division=0)
        rec = recall_score(y_t, y_pred, zero_division=0)
        cm = confusion_matrix(y_t, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1 = f1_score(y_t, y_pred, zero_division=0)

        try:
            auc = roc_auc_score(y_t, y_p)
        except Exception:
            auc = float("nan")

        try:
            pr = average_precision_score(y_t, y_p)
        except Exception:
            pr = float("nan")

        metrics_samples["accuracy"].append(acc)
        metrics_samples["sensitivity"].append(rec)
        metrics_samples["specificity"].append(spec)
        metrics_samples["precision"].append(prec)
        metrics_samples["f1_score"].append(f1)
        if not math.isnan(auc):
            metrics_samples["roc_auc"].append(auc)
        if not math.isnan(pr):
            metrics_samples["pr_auc"].append(pr)

    ci_results = {}
    for m_name, vals in metrics_samples.items():
        if not vals:
            continue
        vals_arr = np.array(vals)
        ci_lower = float(np.percentile(vals_arr, 2.5))
        ci_upper = float(np.percentile(vals_arr, 97.5))
        mean_val = float(np.mean(vals_arr))
        std_val = float(np.std(vals_arr))
        ci_results[m_name] = {
            "mean": round(mean_val, 4),
            "std": round(std_val, 4),
            "ci_95_lower": round(ci_lower, 4),
            "ci_95_upper": round(ci_upper, 4),
        }

    return ci_results


def evaluate_git_model(
    checkpoint_path: str = "outputs/experiments/exp_git_multimodal/best_checkpoint.pth",
    device: Optional[torch.device] = None,
    threshold: float = 0.50,
) -> Dict[str, Any]:
    print("\n" + "=" * 80)
    print("PHASE 5 / GIT: EVALUATING MULTIMODAL GRAPH-INFUSED TRANSFORMER ON TEST COHORT")
    print("=" * 80)

    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 1. Load checkpoint
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = ckpt.get("config", {})
    selected_epoch = ckpt.get("epoch", -1)

    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Selected Epoch: {selected_epoch} | Best Val Loss: {ckpt.get('best_val_loss', 'N/A')} | Best Val AUC: {ckpt.get('best_val_roc_auc', 'N/A')}")

    # 2. Instantiate model architecture
    model = GraphInfusedTransformer(
        mg_feat_dim=1280,
        us_feat_dim=768,
        hidden_dim=config.get("hidden_dim", 256),
        num_gat_layers=config.get("num_gat_layers", 2),
        gat_heads=config.get("gat_heads", 4),
        transformer_layers=config.get("transformer_layers", 2),
        transformer_heads=config.get("transformer_heads", 8),
        dim_feedforward=config.get("dim_feedforward", 1024),
        num_classes=2,
        dropout_rate=config.get("dropout_rate", 0.3),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    # 3. Load frozen test dataloader
    test_loader = get_patient_multimodal_dataloader(
        split="test",
        batch_size=8,
        shuffle=False,
    )
    assert len(test_loader.dataset) == 47, f"Expected exactly 47 test patients, got {len(test_loader.dataset)}"
    print(f"Verified test cohort: exactly {len(test_loader.dataset)} patients.")

    # 4. Inference pass
    patient_records = []
    with torch.no_grad():
        for batch in test_loader:
            labels = batch["labels"].to(device, non_blocking=True)
            mg_feats = batch["mg_features"].to(device, non_blocking=True)
            us_feats = batch["us_features"].to(device, non_blocking=True)
            mg_sides = batch["mg_sides"].to(device, non_blocking=True)
            us_sides = batch["us_sides"].to(device, non_blocking=True)
            mg_mask = batch["mg_mask"].to(device, non_blocking=True)
            us_mask = batch["us_mask"].to(device, non_blocking=True)

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
            labels_np = labels.cpu().numpy()
            batch_size_cur = len(labels)

            for i in range(batch_size_cur):
                pid = batch["patient_ids"][i]
                root_id = batch["root_patient_ids"][i]
                birads = batch["highest_birads"][i]
                y_i = int(labels_np[i])
                p_i = float(probs_np[i])
                y_pred = int(p_i >= threshold)

                m_count = int(batch["mg_mask"][i].sum().item())
                u_count = int(batch["us_mask"][i].sum().item())

                patient_records.append({
                    "patient_id": pid,
                    "root_patient_id": root_id,
                    "highest_birads": birads,
                    "true_label": y_i,
                    "pred_prob": p_i,
                    "pred_label": y_pred,
                    "mg_image_count": m_count,
                    "us_image_count": u_count,
                })

    df_patients = pd.DataFrame(patient_records)

    # 5. Calculate comprehensive metrics
    metrics = compute_classification_metrics(
        y_true=df_patients["true_label"].values,
        y_prob=df_patients["pred_prob"].values,
        threshold=threshold,
    )

    # 6. Bootstrap uncertainty estimates
    print("Computing patient-clustered bootstrap confidence intervals (B=2,000)...")
    bootstrap_ci = run_patient_bootstrap_ci(df_patients, n_bootstraps=2000, seed=42, threshold=threshold)

    # 7. Save prediction CSV and evaluation JSON
    pred_csv_path = "outputs/phase5/predictions/exp_git_multimodal_patient_predictions.csv"
    os.makedirs(os.path.dirname(pred_csv_path), exist_ok=True)
    df_patients.to_csv(pred_csv_path, index=False)
    print(f"[OK] Patient predictions saved to: {pred_csv_path}")

    eval_json_path = "outputs/phase5/tables/exp_git_multimodal_evaluation_results.json"
    os.makedirs(os.path.dirname(eval_json_path), exist_ok=True)
    eval_payload = {
        "experiment_id": "exp_git_multimodal",
        "architecture": "Multimodal Graph-Infused Transformer (GIT)",
        "modalities": "Mammogram (EfficientNet-B0) + Ultrasound (ViT-B/16)",
        "selected_epoch": selected_epoch,
        "threshold": threshold,
        "n_samples": metrics["n_samples"],
        "n_positive": metrics["n_positive"],
        "n_negative": metrics["n_negative"],
        "metrics": metrics,
        "bootstrap_95_ci": bootstrap_ci,
        "checkpoint_path": checkpoint_path,
        "prediction_csv": pred_csv_path,
    }
    with open(eval_json_path, "w", encoding="utf-8") as f:
        json.dump(eval_payload, f, indent=2)
    print(f"[OK] Evaluation results JSON saved to: {eval_json_path}")

    return {
        "metrics": metrics,
        "bootstrap_ci": bootstrap_ci,
        "df_patients": df_patients,
        "eval_payload": eval_payload,
    }


def generate_comparison_table(
    git_metrics: Dict[str, Any],
    git_ci: Dict[str, Any],
    prior_results_csv: str = "outputs/phase5/tables/test_evaluation_results.csv",
    output_csv_path: str = "outputs/phase5/tables/multimodal_git_comparison_table.csv",
    output_json_path: str = "outputs/phase5/tables/multimodal_git_comparison_table.json",
) -> pd.DataFrame:
    print("\n" + "=" * 80)
    print("STEP: GENERATING COMPARISON TABLE FOR INDIVIDUAL MODELS AND COMBINED GIT MODEL")
    print("=" * 80)

    # Load prior test evaluation results
    df_prior = pd.read_csv(prior_results_csv)
    # Filter to patient level
    df_prior_pt = df_prior[df_prior["level"] == "patient"].copy()

    # Model 1: exp1_efficientnet_mammogram
    # Model 2: exp2_vit_ultrasound
    m1_row = df_prior_pt[df_prior_pt["experiment_id"] == "exp1_efficientnet_mammogram"].iloc[0]
    m2_row = df_prior_pt[df_prior_pt["experiment_id"] == "exp2_vit_ultrasound"].iloc[0]

    rows = [
        {
            "Model Designation": "Individual Model 1",
            "Experiment ID": "exp1_efficientnet_mammogram",
            "Architecture": "EfficientNet-B0",
            "Modality": "Mammogram",
            "Level": "Patient",
            "N (Pos / Neg)": f"{m1_row['n_samples']} ({m1_row['n_positive']} / {m1_row['n_negative']})",
            "Accuracy": f"{m1_row['accuracy'] * 100:.2f}%",
            "Precision": f"{m1_row['precision'] * 100:.2f}%",
            "Sensitivity": f"{m1_row['sensitivity'] * 100:.2f}%",
            "Specificity": f"{m1_row['specificity'] * 100:.2f}%",
            "F1-Score": f"{m1_row['f1_score']:.4f}",
            "G-Mean": f"{m1_row['g_mean']:.4f}",
            "ROC-AUC": f"{m1_row['roc_auc']:.4f}",
            "PR-AUC": f"{m1_row['pr_auc']:.4f}",
            "TP": int(m1_row["tp"]),
            "FP": int(m1_row["fp"]),
            "TN": int(m1_row["tn"]),
            "FN": int(m1_row["fn"]),
        },
        {
            "Model Designation": "Individual Model 2",
            "Experiment ID": "exp2_vit_ultrasound",
            "Architecture": "ViT-B/16",
            "Modality": "Ultrasound",
            "Level": "Patient",
            "N (Pos / Neg)": f"{m2_row['n_samples']} ({m2_row['n_positive']} / {m2_row['n_negative']})",
            "Accuracy": f"{m2_row['accuracy'] * 100:.2f}%",
            "Precision": f"{m2_row['precision'] * 100:.2f}%",
            "Sensitivity": f"{m2_row['sensitivity'] * 100:.2f}%",
            "Specificity": f"{m2_row['specificity'] * 100:.2f}%",
            "F1-Score": f"{m2_row['f1_score']:.4f}",
            "G-Mean": f"{m2_row['g_mean']:.4f}",
            "ROC-AUC": f"{m2_row['roc_auc']:.4f}",
            "PR-AUC": f"{m2_row['pr_auc']:.4f}",
            "TP": int(m2_row["tp"]),
            "FP": int(m2_row["fp"]),
            "TN": int(m2_row["tn"]),
            "FN": int(m2_row["fn"]),
        },
        {
            "Model Designation": "Combined GIT Model",
            "Experiment ID": "exp_git_multimodal",
            "Architecture": "Graph-Infused Transformer (GIT)",
            "Modality": "Mammogram + Ultrasound (Multimodal)",
            "Level": "Patient",
            "N (Pos / Neg)": f"{git_metrics['n_samples']} ({git_metrics['n_positive']} / {git_metrics['n_negative']})",
            "Accuracy": f"{git_metrics['accuracy'] * 100:.2f}%",
            "Precision": f"{git_metrics['precision'] * 100:.2f}%",
            "Sensitivity": f"{git_metrics['sensitivity'] * 100:.2f}%",
            "Specificity": f"{git_metrics['specificity'] * 100:.2f}%",
            "F1-Score": f"{git_metrics['f1_score']:.4f}",
            "G-Mean": f"{git_metrics['g_mean']:.4f}",
            "ROC-AUC": f"{git_metrics['roc_auc']:.4f}",
            "PR-AUC": f"{git_metrics['pr_auc']:.4f}",
            "TP": int(git_metrics["tp"]),
            "FP": int(git_metrics["fp"]),
            "TN": int(git_metrics["tn"]),
            "FN": int(git_metrics["fn"]),
        },
    ]

    df_comp = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    df_comp.to_csv(output_csv_path, index=False)
    print(f"[OK] Comparison table saved to: {output_csv_path}")

    # Also save JSON
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    print(f"[OK] Comparison JSON saved to: {output_json_path}")

    # Plot comparison bar chart
    fig_dir = "outputs/phase5/comparisons"
    os.makedirs(fig_dir, exist_ok=True)
    bar_path = os.path.join(fig_dir, "multimodal_git_comparison_barchart.png")

    metric_keys = ["Accuracy", "Sensitivity", "Specificity", "F1-Score", "ROC-AUC", "PR-AUC"]
    models = ["EffNet (MG)", "ViT (US)", "GIT (MG+US)"]

    eff_vals = [
        m1_row["accuracy"], m1_row["sensitivity"], m1_row["specificity"],
        m1_row["f1_score"], m1_row["roc_auc"], m1_row["pr_auc"]
    ]
    vit_vals = [
        m2_row["accuracy"], m2_row["sensitivity"], m2_row["specificity"],
        m2_row["f1_score"], m2_row["roc_auc"], m2_row["pr_auc"]
    ]
    git_vals = [
        git_metrics["accuracy"], git_metrics["sensitivity"], git_metrics["specificity"],
        git_metrics["f1_score"], git_metrics["roc_auc"], git_metrics["pr_auc"]
    ]

    x = np.arange(len(metric_keys))
    width = 0.25

    plt.figure(figsize=(12, 6))
    plt.bar(x - width, eff_vals, width, label="Individual: EfficientNet-B0 (MG)", color="#4A90E2", alpha=0.9)
    plt.bar(x, vit_vals, width, label="Individual: ViT-B/16 (US)", color="#50E3C2", alpha=0.9)
    plt.bar(x + width, git_vals, width, label="Combined: Multimodal GIT (MG+US)", color="#9013FE", alpha=0.9)

    plt.xlabel("Evaluation Metric (Patient-Level, tau=0.50)", fontsize=12, fontweight="bold")
    plt.ylabel("Score", fontsize=12, fontweight="bold")
    plt.title("Comparative Performance: Individual Modalities vs. Multimodal Graph-Infused Transformer", fontsize=14, fontweight="bold")
    plt.xticks(x, metric_keys, fontsize=11)
    plt.ylim(0, 1.1)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.legend(fontsize=11, loc="lower right")

    # Add numeric labels on top of bars
    for i in range(len(metric_keys)):
        plt.text(x[i] - width, eff_vals[i] + 0.02, f"{eff_vals[i]:.2f}", ha="center", fontsize=8)
        plt.text(x[i], vit_vals[i] + 0.02, f"{vit_vals[i]:.2f}", ha="center", fontsize=8)
        plt.text(x[i] + width, git_vals[i] + 0.02, f"{git_vals[i]:.2f}", ha="center", fontsize=8, fontweight="bold")

    plt.tight_layout()
    plt.savefig(bar_path, dpi=300)
    plt.close()
    print(f"[OK] Comparison bar chart saved to: {bar_path}")

    # Plot Confusion Matrix for GIT
    cm = git_metrics["confusion_matrix"]
    fig, ax = plt.subplots(figsize=(6, 5))
    cax = ax.matshow(cm, cmap=plt.cm.Purples)
    plt.title("Multimodal GIT - Patient Confusion Matrix (tau=0.50)", fontsize=12, pad=20)
    fig.colorbar(cax)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Low Risk (0)", "Suspicious (1)"])
    ax.set_yticklabels(["Low Risk (0)", "Suspicious (1)"])
    plt.xlabel("Predicted Label", fontsize=11)
    plt.ylabel("True Ground Truth", fontsize=11)

    for i in range(2):
        for j in range(2):
            val = cm[i][j]
            ax.text(j, i, str(val), va="center", ha="center", fontsize=14, color="white" if val > 20 else "black")

    cm_path = "outputs/phase5/confusion_matrices/exp_git_multimodal_confusion_matrix.png"
    os.makedirs(os.path.dirname(cm_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"[OK] GIT confusion matrix saved to: {cm_path}")

    return df_comp


if __name__ == "__main__":
    res = evaluate_git_model()
    df_comp = generate_comparison_table(res["metrics"], res["bootstrap_ci"])
    print("\n" + "=" * 80)
    print("FINAL COMPARISON TABLE:")
    print("=" * 80)
    print(df_comp.to_string(index=False))
