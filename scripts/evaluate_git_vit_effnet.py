"""Evaluation and Comparative Analysis Script for Second Multimodal GIT Experiment:

exp_git_vit_mammogram_effnet_ultrasound

Backbones:
- Mammogram: ViT-B/16 (exp3)
- Ultrasound: EfficientNet-B0 (exp4)

Evaluates on the frozen 47-patient test cohort:
- Fixed decision threshold tau = 0.50
- Patient-level predictions
- Comprehensive metrics: Accuracy, Precision, Sensitivity, Specificity, F1, G-Mean, ROC-AUC, PR-AUC
- Patient-clustered bootstrap uncertainty intervals (95% CI)
- Saves:
  1. Patient predictions CSV
  2. Evaluation JSON
  3. Comparison table (CSV and JSON) comparing Exp 3, Exp 4, GIT 1, and GIT 2
  4. Visualizations (confusion matrix, comparison bar chart)
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


def evaluate_git_vit_effnet_model(
    checkpoint_path: str = "outputs/experiments/exp_git_vit_mammogram_effnet_ultrasound/best_checkpoint.pth",
    features_path: str = "data/features/patient_multimodal_features_vit_effnet.pt",
    device: Optional[torch.device] = None,
    threshold: float = 0.50,
) -> Dict[str, Any]:
    print("\n" + "=" * 80, flush=True)
    print("PHASE 5: EVALUATING exp_git_vit_mammogram_effnet_ultrasound ON TEST COHORT", flush=True)
    print("=" * 80, flush=True)

    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = ckpt.get("config", {})
    selected_epoch = ckpt.get("epoch", -1)

    print(f"Loaded checkpoint: {checkpoint_path}", flush=True)
    print(f"Selected Epoch: {selected_epoch} | Best Val Loss: {ckpt.get('best_val_loss', 'N/A')} | Best Val AUC: {ckpt.get('best_val_roc_auc', 'N/A')}", flush=True)

    # Instantiate model with mg=768, us=1280
    model = GraphInfusedTransformer(
        mg_feat_dim=768,
        us_feat_dim=1280,
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

    # Load test dataloader
    test_loader = get_patient_multimodal_dataloader(
        features_path=features_path,
        split="test",
        batch_size=8,
        shuffle=False,
    )
    assert len(test_loader.dataset) == 47, f"Expected exactly 47 test patients, got {len(test_loader.dataset)}"
    print(f"Verified test cohort: exactly {len(test_loader.dataset)} patients.", flush=True)

    # Inference pass
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

    # Compute classification metrics
    metrics = compute_classification_metrics(
        y_true=df_patients["true_label"].values,
        y_prob=df_patients["pred_prob"].values,
        threshold=threshold,
    )

    # Bootstrap uncertainty estimates
    print("Computing patient-clustered bootstrap confidence intervals (B=2,000)...", flush=True)
    bootstrap_ci = run_patient_bootstrap_ci(df_patients, n_bootstraps=2000, seed=42, threshold=threshold)

    # Save prediction CSV and evaluation JSON
    pred_csv_path = "outputs/phase5/predictions/exp_git_vit_mammogram_effnet_ultrasound_patient_predictions.csv"
    os.makedirs(os.path.dirname(pred_csv_path), exist_ok=True)
    df_patients.to_csv(pred_csv_path, index=False)
    print(f"[OK] Patient predictions saved to: {pred_csv_path}", flush=True)

    eval_json_path = "outputs/phase5/tables/exp_git_vit_mammogram_effnet_ultrasound_evaluation_results.json"
    os.makedirs(os.path.dirname(eval_json_path), exist_ok=True)
    eval_payload = {
        "experiment_id": "exp_git_vit_mammogram_effnet_ultrasound",
        "architecture": "Multimodal Graph-Infused Transformer (GIT)",
        "modalities": "Mammogram (ViT-B/16, exp3) + Ultrasound (EfficientNet-B0, exp4)",
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
    print(f"[OK] Evaluation results JSON saved to: {eval_json_path}", flush=True)

    return {
        "metrics": metrics,
        "bootstrap_ci": bootstrap_ci,
        "df_patients": df_patients,
        "eval_payload": eval_payload,
    }


def generate_four_way_comparison_table(
    git2_metrics: Dict[str, Any],
    git2_ci: Dict[str, Any],
    prior_results_csv: str = "outputs/phase5/tables/test_evaluation_results.csv",
    git1_json_path: str = "outputs/phase5/tables/exp_git_multimodal_evaluation_results.json",
    output_csv_path: str = "outputs/phase5/tables/git_vit_effnet_comparison_table.csv",
    output_json_path: str = "outputs/phase5/tables/git_vit_effnet_comparison_table.json",
) -> pd.DataFrame:
    print("\n" + "=" * 80, flush=True)
    print("STEP: GENERATING FOUR-WAY COMPARISON TABLE (Exp 3, Exp 4, GIT 1, GIT 2)", flush=True)
    print("=" * 80, flush=True)

    df_prior = pd.read_csv(prior_results_csv)
    df_prior_pt = df_prior[df_prior["level"] == "patient"].copy()

    # Exp 3: ViT Mammogram
    m3_row = df_prior_pt[df_prior_pt["experiment_id"] == "exp3_vit_mammogram"].iloc[0]
    # Exp 4: EffNet Ultrasound
    m4_row = df_prior_pt[df_prior_pt["experiment_id"] == "exp4_efficientnet_ultrasound"].iloc[0]

    # Load GIT 1 metrics
    with open(git1_json_path, "r", encoding="utf-8") as f:
        git1_data = json.load(f)
    g1_metrics = git1_data["metrics"]

    rows = [
        {
            "Model Designation": "Individual Model: Exp 3",
            "Experiment ID": "exp3_vit_mammogram",
            "Architecture": "ViT-B/16",
            "Modality": "Mammogram",
            "Level": "Patient",
            "N (Pos / Neg)": f"{m3_row['n_samples']} ({m3_row['n_positive']} / {m3_row['n_negative']})",
            "Accuracy": f"{m3_row['accuracy'] * 100:.2f}%",
            "Precision": f"{m3_row['precision'] * 100:.2f}%",
            "Sensitivity": f"{m3_row['sensitivity'] * 100:.2f}%",
            "Specificity": f"{m3_row['specificity'] * 100:.2f}%",
            "F1-Score": f"{m3_row['f1_score']:.4f}",
            "G-Mean": f"{m3_row['g_mean']:.4f}",
            "ROC-AUC": f"{m3_row['roc_auc']:.4f}",
            "PR-AUC": f"{m3_row['pr_auc']:.4f}",
            "TP": int(m3_row["tp"]),
            "FP": int(m3_row["fp"]),
            "TN": int(m3_row["tn"]),
            "FN": int(m3_row["fn"]),
        },
        {
            "Model Designation": "Individual Model: Exp 4",
            "Experiment ID": "exp4_efficientnet_ultrasound",
            "Architecture": "EfficientNet-B0",
            "Modality": "Ultrasound",
            "Level": "Patient",
            "N (Pos / Neg)": f"{m4_row['n_samples']} ({m4_row['n_positive']} / {m4_row['n_negative']})",
            "Accuracy": f"{m4_row['accuracy'] * 100:.2f}%",
            "Precision": f"{m4_row['precision'] * 100:.2f}%",
            "Sensitivity": f"{m4_row['sensitivity'] * 100:.2f}%",
            "Specificity": f"{m4_row['specificity'] * 100:.2f}%",
            "F1-Score": f"{m4_row['f1_score']:.4f}",
            "G-Mean": f"{m4_row['g_mean']:.4f}",
            "ROC-AUC": f"{m4_row['roc_auc']:.4f}",
            "PR-AUC": f"{m4_row['pr_auc']:.4f}",
            "TP": int(m4_row["tp"]),
            "FP": int(m4_row["fp"]),
            "TN": int(m4_row["tn"]),
            "FN": int(m4_row["fn"]),
        },
        {
            "Model Designation": "Combined GIT Model 1",
            "Experiment ID": "exp_git_multimodal",
            "Architecture": "Graph-Infused Transformer",
            "Modality": "EffNet MG + ViT US",
            "Level": "Patient",
            "N (Pos / Neg)": f"{g1_metrics['n_samples']} ({g1_metrics['n_positive']} / {g1_metrics['n_negative']})",
            "Accuracy": f"{g1_metrics['accuracy'] * 100:.2f}%",
            "Precision": f"{g1_metrics['precision'] * 100:.2f}%",
            "Sensitivity": f"{g1_metrics['sensitivity'] * 100:.2f}%",
            "Specificity": f"{g1_metrics['specificity'] * 100:.2f}%",
            "F1-Score": f"{g1_metrics['f1_score']:.4f}",
            "G-Mean": f"{g1_metrics['g_mean']:.4f}",
            "ROC-AUC": f"{g1_metrics['roc_auc']:.4f}",
            "PR-AUC": f"{g1_metrics['pr_auc']:.4f}",
            "TP": int(g1_metrics["tp"]),
            "FP": int(g1_metrics["fp"]),
            "TN": int(g1_metrics["tn"]),
            "FN": int(g1_metrics["fn"]),
        },
        {
            "Model Designation": "Combined GIT Model 2",
            "Experiment ID": "exp_git_vit_mammogram_effnet_ultrasound",
            "Architecture": "Graph-Infused Transformer",
            "Modality": "ViT MG + EffNet US",
            "Level": "Patient",
            "N (Pos / Neg)": f"{git2_metrics['n_samples']} ({git2_metrics['n_positive']} / {git2_metrics['n_negative']})",
            "Accuracy": f"{git2_metrics['accuracy'] * 100:.2f}%",
            "Precision": f"{git2_metrics['precision'] * 100:.2f}%",
            "Sensitivity": f"{git2_metrics['sensitivity'] * 100:.2f}%",
            "Specificity": f"{git2_metrics['specificity'] * 100:.2f}%",
            "F1-Score": f"{git2_metrics['f1_score']:.4f}",
            "G-Mean": f"{git2_metrics['g_mean']:.4f}",
            "ROC-AUC": f"{git2_metrics['roc_auc']:.4f}",
            "PR-AUC": f"{git2_metrics['pr_auc']:.4f}",
            "TP": int(git2_metrics["tp"]),
            "FP": int(git2_metrics["fp"]),
            "TN": int(git2_metrics["tn"]),
            "FN": int(git2_metrics["fn"]),
        },
    ]

    df_comp = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    df_comp.to_csv(output_csv_path, index=False)
    print(f"[OK] Comparison table saved to: {output_csv_path}", flush=True)

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    print(f"[OK] Comparison JSON saved to: {output_json_path}", flush=True)

    # Plot Confusion Matrix for GIT Model 2
    cm = git2_metrics["confusion_matrix"]
    fig, ax = plt.subplots(figsize=(6, 5))
    cax = ax.matshow(cm, cmap=plt.cm.Blues)
    plt.title("GIT Model 2 (ViT MG + EffNet US) - Patient Confusion Matrix", fontsize=11, pad=20)
    fig.colorbar(cax)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Low Risk (0)", "Suspicious (1)"])
    ax.set_yticklabels(["Low Risk (0)", "Suspicious (1)"])
    plt.xlabel("Predicted Label (tau=0.50)", fontsize=11)
    plt.ylabel("True Ground Truth", fontsize=11)

    for i in range(2):
        for j in range(2):
            val = cm[i][j]
            ax.text(j, i, str(val), va="center", ha="center", fontsize=14, color="white" if val > 20 else "black")

    cm_path = "outputs/phase5/confusion_matrices/exp_git_vit_mammogram_effnet_ultrasound_confusion_matrix.png"
    os.makedirs(os.path.dirname(cm_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"[OK] GIT Model 2 confusion matrix saved to: {cm_path}", flush=True)

    # Plot comparison bar chart
    bar_path = "outputs/phase5/comparisons/git_vit_effnet_comparison_barchart.png"
    metric_keys = ["Accuracy", "Sensitivity", "Specificity", "F1-Score", "ROC-AUC", "PR-AUC"]

    vals_exp3 = [m3_row["accuracy"], m3_row["sensitivity"], m3_row["specificity"], m3_row["f1_score"], m3_row["roc_auc"], m3_row["pr_auc"]]
    vals_exp4 = [m4_row["accuracy"], m4_row["sensitivity"], m4_row["specificity"], m4_row["f1_score"], m4_row["roc_auc"], m4_row["pr_auc"]]
    vals_git1 = [g1_metrics["accuracy"], g1_metrics["sensitivity"], g1_metrics["specificity"], g1_metrics["f1_score"], g1_metrics["roc_auc"], g1_metrics["pr_auc"]]
    vals_git2 = [git2_metrics["accuracy"], git2_metrics["sensitivity"], git2_metrics["specificity"], git2_metrics["f1_score"], git2_metrics["roc_auc"], git2_metrics["pr_auc"]]

    x = np.arange(len(metric_keys))
    width = 0.20

    plt.figure(figsize=(14, 6))
    plt.bar(x - 1.5 * width, vals_exp3, width, label="Exp 3: ViT-B/16 (MG)", color="#4A90E2", alpha=0.9)
    plt.bar(x - 0.5 * width, vals_exp4, width, label="Exp 4: EffNet-B0 (US)", color="#E94E77", alpha=0.9)
    plt.bar(x + 0.5 * width, vals_git1, width, label="GIT 1: EffNet MG + ViT US", color="#9013FE", alpha=0.9)
    plt.bar(x + 1.5 * width, vals_git2, width, label="GIT 2: ViT MG + EffNet US", color="#F5A623", alpha=0.9)

    plt.xlabel("Evaluation Metric (Patient-Level, tau=0.50)", fontsize=12, fontweight="bold")
    plt.ylabel("Score", fontsize=12, fontweight="bold")
    plt.title("Four-Way Comparative Performance: Unimodal vs Multimodal GIT Architectures", fontsize=13, fontweight="bold")
    plt.xticks(x, metric_keys, fontsize=11)
    plt.ylim(0, 1.15)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.legend(fontsize=10, loc="lower right")

    plt.tight_layout()
    plt.savefig(bar_path, dpi=300)
    plt.close()
    print(f"[OK] Four-way comparison bar chart saved to: {bar_path}", flush=True)

    return df_comp


if __name__ == "__main__":
    res = evaluate_git_vit_effnet_model()
    df_comp = generate_four_way_comparison_table(res["metrics"], res["bootstrap_ci"])
    print("\n" + "=" * 80, flush=True)
    print("FINAL FOUR-WAY COMPARISON TABLE:", flush=True)
    print("=" * 80, flush=True)
    print(df_comp.to_string(index=False), flush=True)
