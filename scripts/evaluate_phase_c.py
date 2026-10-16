"""Evaluation and Comparative Analysis Script for Phase C Models on the Frozen Test Cohort.

Evaluates Candidate C3 (Primary) and Candidate C4 (Secondary) on the untouched 47-patient test cohort:
- Evaluates at canonical baseline threshold tau = 0.50
- Evaluates at validation-selected threshold tau* (tau=0.44 for C3, tau=0.21 for C4)
- Computes Accuracy, Precision, Recall/Sensitivity, Specificity, F1, Balanced Accuracy, G-Mean, ROC-AUC, PR-AUC
- Computes patient-clustered bootstrap 95% Confidence Intervals (B = 2,000 replicates)
- Exports patient predictions CSV, evaluation JSON, and final comparison tables.
"""

import os
import sys
import math
import json
import torch
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.git import GraphInfusedTransformer
from src.data.multimodal_dataset import get_patient_multimodal_dataloader


def run_patient_bootstrap_ci(
    df_patients: pd.DataFrame,
    threshold: float = 0.50,
    n_bootstraps: int = 2000,
    seed: int = 42,
) -> Dict[str, Dict[str, float]]:
    rng = np.random.RandomState(seed)
    n = len(df_patients)
    metrics_samples = {
        "accuracy": [],
        "sensitivity": [],
        "specificity": [],
        "precision": [],
        "f1_score": [],
        "balanced_accuracy": [],
        "gmean": [],
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
        bal_acc = 0.5 * (rec + spec)
        gmean = math.sqrt(rec * spec) if (rec * spec) > 0 else 0.0

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
        metrics_samples["balanced_accuracy"].append(bal_acc)
        metrics_samples["gmean"].append(gmean)
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


def run_test_inference(
    checkpoint_path: str,
    device: torch.device,
) -> pd.DataFrame:
    """Runs single forward pass on the frozen 47-patient test cohort."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = ckpt.get("config", {})

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

    test_loader = get_patient_multimodal_dataloader(
        features_path="data/features/patient_multimodal_features.pt",
        split="test",
        batch_size=8,
        shuffle=False,
    )
    assert len(test_loader.dataset) == 47, f"Expected 47 test patients, got {len(test_loader.dataset)}"

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

            for i in range(len(labels)):
                pid = batch["patient_ids"][i]
                root_id = batch["root_patient_ids"][i]
                birads = batch["highest_birads"][i]
                y_i = int(labels_np[i])
                p_i = float(probs_np[i])
                patient_records.append({
                    "patient_id": pid,
                    "root_patient_id": root_id,
                    "highest_birads": birads,
                    "true_label": y_i,
                    "pred_prob": p_i,
                })

    df = pd.DataFrame(patient_records)
    return df


def compute_metrics_at_threshold(
    df: pd.DataFrame,
    threshold: float,
) -> Dict[str, Any]:
    y_true = df["true_label"].values
    y_prob = df["pred_prob"].values
    y_pred = (y_prob >= threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = f1_score(y_true, y_pred, zero_division=0)
    bal_acc = 0.5 * (rec + spec)
    gmean = math.sqrt(rec * spec) if (rec * spec) > 0 else 0.0

    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except Exception:
        roc_auc = 0.5

    try:
        pr_auc = average_precision_score(y_true, y_prob)
    except Exception:
        pr_auc = float(np.mean(y_true))

    return {
        "threshold": threshold,
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "sensitivity": round(float(rec), 4),
        "specificity": round(float(spec), 4),
        "f1_score": round(float(f1), 4),
        "balanced_accuracy": round(float(bal_acc), 4),
        "gmean": round(float(gmean), 4),
        "roc_auc": round(float(roc_auc), 4),
        "pr_auc": round(float(pr_auc), 4),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Executing Phase C Evaluation on device: {device}")

    pred_dir = "outputs/experiments/phase_c/predictions"
    os.makedirs(pred_dir, exist_ok=True)

    # 1. Primary Candidate C3
    c3_ckpt = "outputs/experiments/phase_c/exp_c3_git1_focal_gamma1_5/best_checkpoint.pth"
    print("\nEvaluating Candidate C3 (Primary: Focal Loss gamma=1.5)...")
    df_c3 = run_test_inference(c3_ckpt, device)
    c3_pred_path = os.path.join(pred_dir, "exp_c3_test_predictions.csv")
    df_c3.to_csv(c3_pred_path, index=False)
    print(f"Saved C3 test predictions to {c3_pred_path}")

    # C3 metrics at tau=0.50 and tau=0.44
    m_c3_default = compute_metrics_at_threshold(df_c3, threshold=0.50)
    m_c3_val_opt = compute_metrics_at_threshold(df_c3, threshold=0.44)
    ci_c3_default = run_patient_bootstrap_ci(df_c3, threshold=0.50, seed=42)
    ci_c3_val_opt = run_patient_bootstrap_ci(df_c3, threshold=0.44, seed=42)

    # 2. Secondary Candidate C4
    c4_ckpt = "outputs/experiments/phase_c/exp_c4_git1_low_lr_reg/best_checkpoint.pth"
    print("\nEvaluating Candidate C4 (Secondary: Weighted BCE Low LR Reg)...")
    df_c4 = run_test_inference(c4_ckpt, device)
    c4_pred_path = os.path.join(pred_dir, "exp_c4_test_predictions.csv")
    df_c4.to_csv(c4_pred_path, index=False)
    print(f"Saved C4 test predictions to {c4_pred_path}")

    # C4 metrics at tau=0.50 and tau=0.21
    m_c4_default = compute_metrics_at_threshold(df_c4, threshold=0.50)
    m_c4_val_opt = compute_metrics_at_threshold(df_c4, threshold=0.21)
    ci_c4_default = run_patient_bootstrap_ci(df_c4, threshold=0.50, seed=42)
    ci_c4_val_opt = run_patient_bootstrap_ci(df_c4, threshold=0.21, seed=42)

    # 3. Assemble comparison rows
    comparison_rows = [
        {
            "Model": "C3 (Focal g=1.5, Primary)",
            "Evaluation Protocol": "Canonical Baseline (tau=0.50)",
            "Threshold": 0.50,
            "Accuracy": f"{m_c3_default['accuracy']*100:.2f}%",
            "Sensitivity": f"{m_c3_default['sensitivity']*100:.2f}%",
            "Specificity": f"{m_c3_default['specificity']*100:.2f}%",
            "Precision": f"{m_c3_default['precision']*100:.2f}%",
            "F1-Score": f"{m_c3_default['f1_score']:.4f}",
            "Balanced Acc": f"{m_c3_default['balanced_accuracy']*100:.2f}%",
            "G-Mean": f"{m_c3_default['gmean']:.4f}",
            "ROC-AUC": f"{m_c3_default['roc_auc']:.4f}",
            "PR-AUC": f"{m_c3_default['pr_auc']:.4f}",
            "CM [TN, FP, FN, TP]": f"[{m_c3_default['tn']}, {m_c3_default['fp']}, {m_c3_default['fn']}, {m_c3_default['tp']}]",
            "Sens 95% CI": f"[{ci_c3_default['sensitivity']['ci_95_lower']:.4f}, {ci_c3_default['sensitivity']['ci_95_upper']:.4f}]",
            "Spec 95% CI": f"[{ci_c3_default['specificity']['ci_95_lower']:.4f}, {ci_c3_default['specificity']['ci_95_upper']:.4f}]",
            "F1 95% CI": f"[{ci_c3_default['f1_score']['ci_95_lower']:.4f}, {ci_c3_default['f1_score']['ci_95_upper']:.4f}]",
            "AUC 95% CI": f"[{ci_c3_default['roc_auc']['ci_95_lower']:.4f}, {ci_c3_default['roc_auc']['ci_95_upper']:.4f}]",
        },
        {
            "Model": "C3 (Focal g=1.5, Primary)",
            "Evaluation Protocol": "Validation-Selected (tau=0.44)",
            "Threshold": 0.44,
            "Accuracy": f"{m_c3_val_opt['accuracy']*100:.2f}%",
            "Sensitivity": f"{m_c3_val_opt['sensitivity']*100:.2f}%",
            "Specificity": f"{m_c3_val_opt['specificity']*100:.2f}%",
            "Precision": f"{m_c3_val_opt['precision']*100:.2f}%",
            "F1-Score": f"{m_c3_val_opt['f1_score']:.4f}",
            "Balanced Acc": f"{m_c3_val_opt['balanced_accuracy']*100:.2f}%",
            "G-Mean": f"{m_c3_val_opt['gmean']:.4f}",
            "ROC-AUC": f"{m_c3_val_opt['roc_auc']:.4f}",
            "PR-AUC": f"{m_c3_val_opt['pr_auc']:.4f}",
            "CM [TN, FP, FN, TP]": f"[{m_c3_val_opt['tn']}, {m_c3_val_opt['fp']}, {m_c3_val_opt['fn']}, {m_c3_val_opt['tp']}]",
            "Sens 95% CI": f"[{ci_c3_val_opt['sensitivity']['ci_95_lower']:.4f}, {ci_c3_val_opt['sensitivity']['ci_95_upper']:.4f}]",
            "Spec 95% CI": f"[{ci_c3_val_opt['specificity']['ci_95_lower']:.4f}, {ci_c3_val_opt['specificity']['ci_95_upper']:.4f}]",
            "F1 95% CI": f"[{ci_c3_val_opt['f1_score']['ci_95_lower']:.4f}, {ci_c3_val_opt['f1_score']['ci_95_upper']:.4f}]",
            "AUC 95% CI": f"[{ci_c3_val_opt['roc_auc']['ci_95_lower']:.4f}, {ci_c3_val_opt['roc_auc']['ci_95_upper']:.4f}]",
        },
        {
            "Model": "C4 (W-BCE Reg, Secondary)",
            "Evaluation Protocol": "Canonical Baseline (tau=0.50)",
            "Threshold": 0.50,
            "Accuracy": f"{m_c4_default['accuracy']*100:.2f}%",
            "Sensitivity": f"{m_c4_default['sensitivity']*100:.2f}%",
            "Specificity": f"{m_c4_default['specificity']*100:.2f}%",
            "Precision": f"{m_c4_default['precision']*100:.2f}%",
            "F1-Score": f"{m_c4_default['f1_score']:.4f}",
            "Balanced Acc": f"{m_c4_default['balanced_accuracy']*100:.2f}%",
            "G-Mean": f"{m_c4_default['gmean']:.4f}",
            "ROC-AUC": f"{m_c4_default['roc_auc']:.4f}",
            "PR-AUC": f"{m_c4_default['pr_auc']:.4f}",
            "CM [TN, FP, FN, TP]": f"[{m_c4_default['tn']}, {m_c4_default['fp']}, {m_c4_default['fn']}, {m_c4_default['tp']}]",
            "Sens 95% CI": f"[{ci_c4_default['sensitivity']['ci_95_lower']:.4f}, {ci_c4_default['sensitivity']['ci_95_upper']:.4f}]",
            "Spec 95% CI": f"[{ci_c4_default['specificity']['ci_95_lower']:.4f}, {ci_c4_default['specificity']['ci_95_upper']:.4f}]",
            "F1 95% CI": f"[{ci_c4_default['f1_score']['ci_95_lower']:.4f}, {ci_c4_default['f1_score']['ci_95_upper']:.4f}]",
            "AUC 95% CI": f"[{ci_c4_default['roc_auc']['ci_95_lower']:.4f}, {ci_c4_default['roc_auc']['ci_95_upper']:.4f}]",
        },
        {
            "Model": "C4 (W-BCE Reg, Secondary)",
            "Evaluation Protocol": "Validation-Selected (tau=0.21)",
            "Threshold": 0.21,
            "Accuracy": f"{m_c4_val_opt['accuracy']*100:.2f}%",
            "Sensitivity": f"{m_c4_val_opt['sensitivity']*100:.2f}%",
            "Specificity": f"{m_c4_val_opt['specificity']*100:.2f}%",
            "Precision": f"{m_c4_val_opt['precision']*100:.2f}%",
            "F1-Score": f"{m_c4_val_opt['f1_score']:.4f}",
            "Balanced Acc": f"{m_c4_val_opt['balanced_accuracy']*100:.2f}%",
            "G-Mean": f"{m_c4_val_opt['gmean']:.4f}",
            "ROC-AUC": f"{m_c4_val_opt['roc_auc']:.4f}",
            "PR-AUC": f"{m_c4_val_opt['pr_auc']:.4f}",
            "CM [TN, FP, FN, TP]": f"[{m_c4_val_opt['tn']}, {m_c4_val_opt['fp']}, {m_c4_val_opt['fn']}, {m_c4_val_opt['tp']}]",
            "Sens 95% CI": f"[{ci_c4_val_opt['sensitivity']['ci_95_lower']:.4f}, {ci_c4_val_opt['sensitivity']['ci_95_upper']:.4f}]",
            "Spec 95% CI": f"[{ci_c4_val_opt['specificity']['ci_95_lower']:.4f}, {ci_c4_val_opt['specificity']['ci_95_upper']:.4f}]",
            "F1 95% CI": f"[{ci_c4_val_opt['f1_score']['ci_95_lower']:.4f}, {ci_c4_val_opt['f1_score']['ci_95_upper']:.4f}]",
            "AUC 95% CI": f"[{ci_c4_val_opt['roc_auc']['ci_95_lower']:.4f}, {ci_c4_val_opt['roc_auc']['ci_95_upper']:.4f}]",
        },
        # Historical Baselines from Phase 5 & Phase B
        {
            "Model": "Original GIT 1 Baseline",
            "Evaluation Protocol": "Canonical Baseline (tau=0.50)",
            "Threshold": 0.50,
            "Accuracy": "82.98%",
            "Sensitivity": "71.43%",
            "Specificity": "85.00%",
            "Precision": "45.45%",
            "F1-Score": "0.5556",
            "Balanced Acc": "78.21%",
            "G-Mean": "0.7792",
            "ROC-AUC": "0.8679",
            "PR-AUC": "0.6800",
            "CM [TN, FP, FN, TP]": "[34, 6, 2, 5]",
            "Sens 95% CI": "[0.2857, 1.0000]",
            "Spec 95% CI": "[0.7250, 0.9500]",
            "F1 95% CI": "[0.2857, 0.8000]",
            "AUC 95% CI": "[0.6696, 0.9856]",
        },
        {
            "Model": "Original GIT 1 Baseline",
            "Evaluation Protocol": "Validation-Selected (tau=0.42)",
            "Threshold": 0.42,
            "Accuracy": "78.72%",
            "Sensitivity": "85.71%",
            "Specificity": "77.50%",
            "Precision": "40.00%",
            "F1-Score": "0.5455",
            "Balanced Acc": "81.61%",
            "G-Mean": "0.8150",
            "ROC-AUC": "0.8679",
            "PR-AUC": "0.6800",
            "CM [TN, FP, FN, TP]": "[31, 9, 1, 6]",
            "Sens 95% CI": "[0.5000, 1.0000]",
            "Spec 95% CI": "[0.6389, 0.8974]",
            "F1 95% CI": "[0.3158, 0.7692]",
            "AUC 95% CI": "[0.6696, 0.9856]",
        },
        {
            "Model": "Original GIT 2 Baseline",
            "Evaluation Protocol": "Canonical Baseline (tau=0.50)",
            "Threshold": 0.50,
            "Accuracy": "63.83%",
            "Sensitivity": "100.00%",
            "Specificity": "57.50%",
            "Precision": "29.17%",
            "F1-Score": "0.4516",
            "Balanced Acc": "78.75%",
            "G-Mean": "0.7583",
            "ROC-AUC": "0.8429",
            "PR-AUC": "0.5739",
            "CM [TN, FP, FN, TP]": "[23, 17, 0, 7]",
            "Sens 95% CI": "[1.0000, 1.0000]",
            "Spec 95% CI": "[0.4250, 0.7250]",
            "F1 95% CI": "[0.2692, 0.6087]",
            "AUC 95% CI": "[0.7188, 0.9625]",
        },
        {
            "Model": "Original GIT 2 Baseline",
            "Evaluation Protocol": "Validation-Selected (tau=0.76)",
            "Threshold": 0.76,
            "Accuracy": "70.21%",
            "Sensitivity": "85.71%",
            "Specificity": "67.50%",
            "Precision": "31.58%",
            "F1-Score": "0.4615",
            "Balanced Acc": "76.61%",
            "G-Mean": "0.7606",
            "ROC-AUC": "0.8429",
            "PR-AUC": "0.5739",
            "CM [TN, FP, FN, TP]": "[27, 13, 1, 6]",
            "Sens 95% CI": "[0.5000, 1.0000]",
            "Spec 95% CI": "[0.5250, 0.8158]",
            "F1 95% CI": "[0.2400, 0.6957]",
            "AUC 95% CI": "[0.7188, 0.9625]",
        },
    ]

    df_comp = pd.DataFrame(comparison_rows)
    csv_path = "outputs/experiments/phase_c/final_test_evaluation_comparison.csv"
    json_path = "outputs/experiments/phase_c/final_test_evaluation_comparison.json"
    df_comp.to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(comparison_rows, f, indent=2)

    print("\n" + "=" * 100)
    print("PHASE C: FINAL TEST EVALUATION COMPARISON TABLE (N=47 Patients, 7 Positive / 40 Negative)")
    print("=" * 100)
    print(df_comp.to_string(index=False))

    payload_eval = {
        "candidate_c3": {
            "default_tau_0_50": m_c3_default,
            "default_bootstrap_ci": ci_c3_default,
            "val_selected_tau_0_44": m_c3_val_opt,
            "val_selected_bootstrap_ci": ci_c3_val_opt,
        },
        "candidate_c4": {
            "default_tau_0_50": m_c4_default,
            "default_bootstrap_ci": ci_c4_default,
            "val_selected_tau_0_21": m_c4_val_opt,
            "val_selected_bootstrap_ci": ci_c4_val_opt,
        },
    }
    payload_json = "outputs/experiments/phase_c/phase_c_final_evaluation_results.json"
    with open(payload_json, "w", encoding="utf-8") as f:
        json.dump(payload_eval, f, indent=2)
    print(f"\nDetailed evaluation results saved to: {payload_json}")
    print(f"Comparison CSV saved to: {csv_path}")


if __name__ == "__main__":
    main()
