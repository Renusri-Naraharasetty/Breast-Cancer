"""Phase B: Threshold Optimization for Multimodal GIT Models (GIT 1 and GIT 2).

Workflow:
1. Generate patient-level continuous validation predictions using saved checkpoints.
2. Verify patient IDs, labels, and cohort counts (48 validation patients: 8 pos, 40 neg).
3. Sweep thresholds [0.05, 0.95] in steps of 0.01 exclusively on validation predictions.
4. Select thresholds under 3 clinical criteria:
   - Max F1
   - Max Balanced Accuracy
   - High Sensitivity: Max Specificity subject to Sensitivity >= 90%
   Document tie-breaking rules clearly.
5. Save complete validation sweeps to CSV.
6. Apply validation-selected thresholds unchanged to held-out test predictions (N=47).
7. Save comparison tables and summary JSON.
"""

import os
import sys
import math
import json
import torch
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.git import GraphInfusedTransformer
from src.data.multimodal_dataset import get_patient_multimodal_dataloader


def generate_val_predictions(
    checkpoint_path: str,
    features_path: str,
    output_csv_path: str,
    mg_dim: int,
    us_dim: int,
    device: torch.device,
) -> pd.DataFrame:
    """Runs inference on validation split and saves predictions to CSV."""
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    config = checkpoint.get("config", {})
    model = GraphInfusedTransformer(
        mg_feat_dim=mg_dim,
        us_feat_dim=us_dim,
        hidden_dim=config.get("hidden_dim", 256),
        num_gat_layers=config.get("num_gat_layers", 2),
        gat_heads=config.get("gat_heads", 4),
        transformer_layers=config.get("transformer_layers", 2),
        transformer_heads=config.get("transformer_heads", 8),
        dim_feedforward=config.get("dim_feedforward", 1024),
        num_classes=2,
        dropout_rate=config.get("dropout_rate", 0.3),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    val_loader = get_patient_multimodal_dataloader(
        features_path=features_path,
        split="val",
        batch_size=8,
        shuffle=False,
        num_workers=0,
    )

    records = []
    with torch.no_grad():
        for batch in val_loader:
            mg_feats = batch["mg_features"].to(device)
            us_feats = batch["us_features"].to(device)
            mg_sides = batch["mg_sides"].to(device)
            us_sides = batch["us_sides"].to(device)
            mg_mask = batch["mg_mask"].to(device)
            us_mask = batch["us_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(
                mg_features=mg_feats,
                us_features=us_feats,
                mg_sides=mg_sides,
                us_sides=us_sides,
                mg_mask=mg_mask,
                us_mask=us_mask,
            )
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            labels_np = labels.cpu().numpy()

            for i in range(len(probs)):
                pid = batch["patient_ids"][i]
                root_id = batch["root_patient_ids"][i]
                birads = batch["highest_birads"][i]
                p_val = float(probs[i])
                y_val = int(labels_np[i])
                m_count = int(batch["mg_mask"][i].sum().item())
                u_count = int(batch["us_mask"][i].sum().item())

                records.append({
                    "patient_id": pid,
                    "root_patient_id": root_id,
                    "highest_birads": birads,
                    "true_label": y_val,
                    "pred_prob": p_val,
                    "pred_label": int(p_val >= 0.50),
                    "mg_image_count": m_count,
                    "us_image_count": u_count,
                })

    df_val = pd.DataFrame(records)
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    df_val.to_csv(output_csv_path, index=False)
    print(f"Saved {len(df_val)} validation predictions to: {output_csv_path}")
    return df_val


def compute_metrics_at_threshold(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict[str, Any]:
    """Calculates all triage metrics at a specific decision threshold."""
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = f1_score(y_true, y_pred, zero_division=0)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    gmean = math.sqrt(rec * spec) if (rec * spec) > 0 else 0.0

    return {
        "threshold": round(float(threshold), 4),
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "sensitivity": round(float(rec), 4),
        "recall": round(float(rec), 4),
        "specificity": round(float(spec), 4),
        "f1_score": round(float(f1), 4),
        "balanced_accuracy": round(float(bal_acc), 4),
        "g_mean": round(float(gmean), 4),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def sweep_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: np.ndarray,
    output_sweep_csv: str,
) -> pd.DataFrame:
    """Sweeps candidate thresholds and writes full sweep to CSV."""
    sweep_records = []
    for th in thresholds:
        m = compute_metrics_at_threshold(y_true, y_prob, th)
        sweep_records.append(m)

    df_sweep = pd.DataFrame(sweep_records)
    os.makedirs(os.path.dirname(output_sweep_csv), exist_ok=True)
    df_sweep.to_csv(output_sweep_csv, index=False)
    print(f"Saved complete threshold sweep ({len(df_sweep)} thresholds) to: {output_sweep_csv}")
    return df_sweep


def select_optimal_thresholds(df_sweep: pd.DataFrame) -> Dict[str, Any]:
    """Selects thresholds based on 3 distinct clinical criteria with explicit tie-breaking.

    Tie-breaking rules:
    - Max F1: If tied, pick highest balanced_accuracy, then highest specificity, then threshold closest to 0.50.
    - Max Balanced Accuracy: If tied, pick highest f1_score, then highest sensitivity, then threshold closest to 0.50.
    - High Sensitivity (Sens >= 0.90): Filter to sens >= 0.90, pick highest specificity, then highest precision, then closest to 0.50.
    """
    # 1. Max F1
    max_f1 = df_sweep["f1_score"].max()
    candidates_f1 = df_sweep[df_sweep["f1_score"] == max_f1].copy()
    candidates_f1["dist_to_half"] = (candidates_f1["threshold"] - 0.50).abs()
    best_f1_row = candidates_f1.sort_values(
        by=["balanced_accuracy", "specificity", "dist_to_half"],
        ascending=[False, False, True]
    ).iloc[0]

    # 2. Max Balanced Accuracy
    max_bal_acc = df_sweep["balanced_accuracy"].max()
    candidates_bal = df_sweep[df_sweep["balanced_accuracy"] == max_bal_acc].copy()
    candidates_bal["dist_to_half"] = (candidates_bal["threshold"] - 0.50).abs()
    best_bal_row = candidates_bal.sort_values(
        by=["f1_score", "sensitivity", "dist_to_half"],
        ascending=[False, False, True]
    ).iloc[0]

    # 3. High Sensitivity (Sens >= 0.90) with Max Specificity
    sens_candidates = df_sweep[df_sweep["sensitivity"] >= 0.90].copy()
    if len(sens_candidates) > 0:
        sens_candidates["dist_to_half"] = (sens_candidates["threshold"] - 0.50).abs()
        best_sens_row = sens_candidates.sort_values(
            by=["specificity", "precision", "dist_to_half"],
            ascending=[False, False, True]
        ).iloc[0]
        sens90_satisfied = True
    else:
        # Fallback to maximum achievable sensitivity
        max_sens = df_sweep["sensitivity"].max()
        fallback_candidates = df_sweep[df_sweep["sensitivity"] == max_sens].copy()
        fallback_candidates["dist_to_half"] = (fallback_candidates["threshold"] - 0.50).abs()
        best_sens_row = fallback_candidates.sort_values(
            by=["specificity", "dist_to_half"],
            ascending=[False, True]
        ).iloc[0]
        sens90_satisfied = False

    return {
        "max_f1": {
            "criterion": "Maximum Validation F1-Score",
            "threshold": float(best_f1_row["threshold"]),
            "tie_breaking_applied": len(candidates_f1) > 1,
            "val_metrics": best_f1_row.to_dict(),
        },
        "max_balanced_accuracy": {
            "criterion": "Maximum Validation Balanced Accuracy",
            "threshold": float(best_bal_row["threshold"]),
            "tie_breaking_applied": len(candidates_bal) > 1,
            "val_metrics": best_bal_row.to_dict(),
        },
        "high_sensitivity_90": {
            "criterion": "Maximum Specificity subject to Sensitivity >= 90%",
            "constraint_satisfied": sens90_satisfied,
            "threshold": float(best_sens_row["threshold"]),
            "tie_breaking_applied": len(sens_candidates) > 1 if sens90_satisfied else True,
            "val_metrics": best_sens_row.to_dict(),
        },
        "baseline_default": {
            "criterion": "Canonical Default (Threshold 0.50)",
            "threshold": 0.50,
            "val_metrics": df_sweep[df_sweep["threshold"] == 0.50].iloc[0].to_dict(),
        }
    }


def run_phase_b():
    print("=" * 80)
    print("PHASE B: RIGOROUS THRESHOLD OPTIMIZATION (VALIDATION ONLY)")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Compute device: {device}")

    # Threshold candidate range: 0.05 to 0.95 with step 0.01
    thresholds = np.round(np.arange(0.05, 0.951, 0.01), 2)
    print(f"Total candidate thresholds to evaluate: {len(thresholds)} (from {thresholds[0]} to {thresholds[-1]})")

    # 1. GIT 1 Validation Inference & Optimization
    print("\n--- 1. Evaluating GIT 1 (EffNet MG + ViT US) on Validation Set ---")
    git1_ckpt = "outputs/experiments/exp_git_multimodal/best_checkpoint.pth"
    git1_feat = "data/features/patient_multimodal_features.pt"
    git1_val_csv = "outputs/phase5/predictions/exp_git_multimodal_val_predictions.csv"
    git1_test_csv = "outputs/phase5/predictions/exp_git_multimodal_patient_predictions.csv"
    git1_sweep_csv = "outputs/phase5/tables/threshold_optimization_val_sweep_git1.csv"

    df_val_git1 = generate_val_predictions(
        checkpoint_path=git1_ckpt,
        features_path=git1_feat,
        output_csv_path=git1_val_csv,
        mg_dim=1280,
        us_dim=768,
        device=device,
    )

    # Verification of cohort
    assert len(df_val_git1) == 48, f"Expected 48 validation patients, got {len(df_val_git1)}"
    n_pos_val1 = (df_val_git1["true_label"] == 1).sum()
    n_neg_val1 = (df_val_git1["true_label"] == 0).sum()
    print(f"GIT 1 Validation Cohort: {len(df_val_git1)} patients ({n_pos_val1} positive, {n_neg_val1} negative)")
    assert n_pos_val1 == 8 and n_neg_val1 == 40, f"Expected 8 pos / 40 neg, got {n_pos_val1} / {n_neg_val1}"

    # Validation Continuous ROC-AUC and PR-AUC
    val_auc_git1 = roc_auc_score(df_val_git1["true_label"], df_val_git1["pred_prob"])
    val_pr_git1 = average_precision_score(df_val_git1["true_label"], df_val_git1["pred_prob"])
    print(f"GIT 1 Validation Continuous ROC-AUC: {val_auc_git1:.4f} | PR-AUC: {val_pr_git1:.4f}")

    # Sweep on validation
    df_sweep_git1 = sweep_thresholds(
        y_true=df_val_git1["true_label"].values,
        y_prob=df_val_git1["pred_prob"].values,
        thresholds=thresholds,
        output_sweep_csv=git1_sweep_csv,
    )
    selected_git1 = select_optimal_thresholds(df_sweep_git1)

    # 2. GIT 2 Validation Inference & Optimization
    print("\n--- 2. Evaluating GIT 2 (ViT MG + EffNet US) on Validation Set ---")
    git2_ckpt = "outputs/experiments/exp_git_vit_mammogram_effnet_ultrasound/best_checkpoint.pth"
    git2_feat = "data/features/patient_multimodal_features_vit_effnet.pt"
    git2_val_csv = "outputs/phase5/predictions/exp_git_vit_mammogram_effnet_ultrasound_val_predictions.csv"
    git2_test_csv = "outputs/phase5/predictions/exp_git_vit_mammogram_effnet_ultrasound_patient_predictions.csv"
    git2_sweep_csv = "outputs/phase5/tables/threshold_optimization_val_sweep_git2.csv"

    df_val_git2 = generate_val_predictions(
        checkpoint_path=git2_ckpt,
        features_path=git2_feat,
        output_csv_path=git2_val_csv,
        mg_dim=768,
        us_dim=1280,
        device=device,
    )

    # Verification of cohort
    assert len(df_val_git2) == 48, f"Expected 48 validation patients, got {len(df_val_git2)}"
    n_pos_val2 = (df_val_git2["true_label"] == 1).sum()
    n_neg_val2 = (df_val_git2["true_label"] == 0).sum()
    print(f"GIT 2 Validation Cohort: {len(df_val_git2)} patients ({n_pos_val2} positive, {n_neg_val2} negative)")
    assert n_pos_val2 == 8 and n_neg_val2 == 40, f"Expected 8 pos / 40 neg, got {n_pos_val2} / {n_neg_val2}"

    # Validation Continuous ROC-AUC and PR-AUC
    val_auc_git2 = roc_auc_score(df_val_git2["true_label"], df_val_git2["pred_prob"])
    val_pr_git2 = average_precision_score(df_val_git2["true_label"], df_val_git2["pred_prob"])
    print(f"GIT 2 Validation Continuous ROC-AUC: {val_auc_git2:.4f} | PR-AUC: {val_pr_git2:.4f}")

    # Sweep on validation
    df_sweep_git2 = sweep_thresholds(
        y_true=df_val_git2["true_label"].values,
        y_prob=df_val_git2["pred_prob"].values,
        thresholds=thresholds,
        output_sweep_csv=git2_sweep_csv,
    )
    selected_git2 = select_optimal_thresholds(df_sweep_git2)

    # 3. Apply Validation Selected Thresholds to Held-Out Test Set (N=47)
    print("\n--- 3. Applying Validation Selected Thresholds UNCHANGED to Held-Out Test Set ---")
    df_test_git1 = pd.read_csv(git1_test_csv)
    df_test_git2 = pd.read_csv(git2_test_csv)

    test_eval_summary = {
        "metadata": {
            "n_val_patients": 48,
            "n_val_pos": 8,
            "n_val_neg": 40,
            "n_test_patients": 47,
            "n_test_pos": 7,
            "n_test_neg": 40,
            "limitation_statement": (
                "The test set contains only 7 positive patients out of 47 total. "
                "Confidence intervals are broad, and thresholds selected on validation data "
                "are subject to finite-sample variability. These results do not constitute clinical validation."
            )
        },
        "GIT_1": {
            "model_name": "GIT 1 (EffNet MG + ViT US)",
            "val_roc_auc": round(float(val_auc_git1), 4),
            "val_pr_auc": round(float(val_pr_git1), 4),
            "operating_points": {},
        },
        "GIT_2": {
            "model_name": "GIT 2 (ViT MG + EffNet US)",
            "val_roc_auc": round(float(val_auc_git2), 4),
            "val_pr_auc": round(float(val_pr_git2), 4),
            "operating_points": {},
        }
    }

    comparison_rows = []

    for model_key, sel_dict, df_test in [
        ("GIT_1", selected_git1, df_test_git1),
        ("GIT_2", selected_git2, df_test_git2),
    ]:
        y_true_test = df_test["true_label"].values
        y_prob_test = df_test["pred_prob"].values

        for crit_key, crit_info in sel_dict.items():
            th = crit_info["threshold"]
            # Evaluate on test set
            test_metrics = compute_metrics_at_threshold(y_true_test, y_prob_test, th)
            val_metrics = crit_info["val_metrics"]

            test_eval_summary[model_key]["operating_points"][crit_key] = {
                "criterion": crit_info["criterion"],
                "threshold": th,
                "validation_performance": val_metrics,
                "test_performance": test_metrics,
            }

            comparison_rows.append({
                "Model": test_eval_summary[model_key]["model_name"],
                "Criterion": crit_info["criterion"],
                "Threshold": th,
                "Val Sens": f"{val_metrics['sensitivity']*100:.1f}%",
                "Val Spec": f"{val_metrics['specificity']*100:.1f}%",
                "Val F1": f"{val_metrics['f1_score']:.4f}",
                "Val BalAcc": f"{val_metrics['balanced_accuracy']*100:.1f}%",
                "Test Acc": f"{test_metrics['accuracy']*100:.2f}%",
                "Test Sens": f"{test_metrics['sensitivity']*100:.2f}%",
                "Test Spec": f"{test_metrics['specificity']*100:.2f}%",
                "Test Prec": f"{test_metrics['precision']*100:.2f}%",
                "Test F1": f"{test_metrics['f1_score']:.4f}",
                "Test BalAcc": f"{test_metrics['balanced_accuracy']*100:.2f}%",
                "Test G-Mean": f"{test_metrics['g_mean']:.4f}",
                "Test TP": test_metrics["tp"],
                "Test FP": test_metrics["fp"],
                "Test TN": test_metrics["tn"],
                "Test FN": test_metrics["fn"],
            })

    # Save summary JSON and CSV
    summary_json_path = "outputs/phase5/tables/threshold_optimization_summary.json"
    summary_csv_path = "outputs/phase5/tables/threshold_optimization_summary.csv"

    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(test_eval_summary, f, indent=2)

    df_comp = pd.DataFrame(comparison_rows)
    df_comp.to_csv(summary_csv_path, index=False)

    print(f"\nSaved threshold optimization summary JSON to: {summary_json_path}")
    print(f"Saved threshold optimization summary CSV to:  {summary_csv_path}")

    print("\n" + "=" * 80)
    print("PHASE B OPERATING POINTS COMPARISON TABLE")
    print("=" * 80)
    print(df_comp.to_string(index=False))

    print("\n" + "=" * 80)
    print("PHASE B COMPLETED SUCCESSFULLY")
    print("=" * 80)

if __name__ == "__main__":
    run_phase_b()
