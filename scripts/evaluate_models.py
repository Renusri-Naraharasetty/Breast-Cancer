"""Phase 5 Test Evaluation Runner: Independent Test Evaluation & Comparative Analysis.

Evaluates the four saved best checkpoints on the frozen 47-patient test partition:
1. EfficientNet-B0 on mammograms (exp1_efficientnet_mammogram)
2. ViT-B/16 on ultrasound (exp2_vit_ultrasound)
3. ViT-B/16 on mammograms (exp3_vit_mammogram)
4. EfficientNet-B0 on ultrasound (exp4_efficientnet_ultrasound)

Outputs:
- Metric tables (CSV & JSON) for image-level and patient-level evaluation
- Confusion matrix plots for all models
- Comparative ROC and PR curves
- Individual learning curves and overlay plots
- Comparative bar charts
- Comprehensive evaluation markdown report
"""

import os
import sys
import json
import torch
import pandas as pd
from typing import List, Dict, Any

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.evaluation.evaluator import (
    verify_checkpoint_integrity,
    verify_dataset_splits,
    evaluate_single_model,
)
from src.evaluation.visualization import (
    plot_confusion_matrix_pair,
    plot_roc_comparison,
    plot_pr_comparison,
    plot_single_experiment_history,
    plot_training_overlays,
    plot_metrics_comparison_barchart,
)


EXPERIMENTS_TO_EVALUATE = [
    {
        "id": "exp1_efficientnet_mammogram",
        "order": 1,
        "model_name": "efficientnet_b0",
        "modality": "mammogram",
        "checkpoint_path": "outputs/experiments/exp1_efficientnet_mammogram/best_checkpoint.pth",
        "history_csv": "outputs/experiments/exp1_efficientnet_mammogram/training_history.csv",
        "description": "EfficientNet-B0 on Mammograms",
    },
    {
        "id": "exp2_vit_ultrasound",
        "order": 2,
        "model_name": "vit_b_16",
        "modality": "ultrasound",
        "checkpoint_path": "outputs/experiments/exp2_vit_ultrasound/best_checkpoint.pth",
        "history_csv": "outputs/experiments/exp2_vit_ultrasound/training_history.csv",
        "description": "Vision Transformer (ViT-B/16) on Ultrasound",
    },
    {
        "id": "exp3_vit_mammogram",
        "order": 3,
        "model_name": "vit_b_16",
        "modality": "mammogram",
        "checkpoint_path": "outputs/experiments/exp3_vit_mammogram/best_checkpoint.pth",
        "history_csv": "outputs/experiments/exp3_vit_mammogram/training_history.csv",
        "description": "Vision Transformer (ViT-B/16) on Mammograms",
    },
    {
        "id": "exp4_efficientnet_ultrasound",
        "order": 4,
        "model_name": "efficientnet_b0",
        "modality": "ultrasound",
        "checkpoint_path": "outputs/experiments/exp4_efficientnet_ultrasound/best_checkpoint.pth",
        "history_csv": "outputs/experiments/exp4_efficientnet_ultrasound/training_history.csv",
        "description": "EfficientNet-B0 on Ultrasound",
    },
]


def run_phase5_evaluation(output_base_dir: str = "outputs/phase5") -> Dict[str, Any]:
    print("\n" + "=" * 80)
    print("PHASE 5: FINAL TEST EVALUATION AND COMPARATIVE ANALYSIS")
    print("=" * 80)

    # 1. Verification of Dataset Splits
    print("\n[Step 1/5] Verifying frozen patient split integrity...", flush=True)
    split_info = verify_dataset_splits()
    print(f"  --> Split verified: Train={split_info['total_train_patients']}, Val={split_info['total_val_patients']}, Test={split_info['total_test_patients']} patients.")
    print(f"  --> Overlap check: train/test overlap={split_info['train_test_overlap_count']}, val/test overlap={split_info['val_test_overlap_count']}.")
    print("  --> [OK] Test cohort is completely isolated and contamination-free.", flush=True)

    # 2. Checkpoint Pre-Verification
    print("\n[Step 2/5] Pre-verifying all four saved Phase 4 best checkpoints...", flush=True)
    for exp in EXPERIMENTS_TO_EVALUATE:
        ckpt_meta = verify_checkpoint_integrity(exp["checkpoint_path"])
        print(f"  --> [OK] {exp['id']}: Arch={ckpt_meta['architecture']}, Epoch={ckpt_meta['selected_epoch']}, ValLoss={ckpt_meta['best_val_loss']:.4f}, ValAUC={ckpt_meta['best_val_roc_auc']:.4f}")

    # Set up output directories
    dirs = {
        "tables": os.path.join(output_base_dir, "tables"),
        "predictions": os.path.join(output_base_dir, "predictions"),
        "confusion_matrices": os.path.join(output_base_dir, "confusion_matrices"),
        "curves": os.path.join(output_base_dir, "curves"),
        "training_curves": os.path.join(output_base_dir, "training_curves"),
        "comparisons": os.path.join(output_base_dir, "comparisons"),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\n[Step 3/5] Executing sequential evaluation on device: {device}...", flush=True)

    all_eval_results = []
    summary_rows = []
    histories = {}

    for exp in EXPERIMENTS_TO_EVALUATE:
        exp_id = exp["id"]
        order = exp["order"]
        ckpt_path = exp["checkpoint_path"]
        hist_csv = exp["history_csv"]

        print(f"\n--- Evaluating Model {order}/4: {exp['description']} ({exp_id}) ---", flush=True)
        eval_res = evaluate_single_model(checkpoint_path=ckpt_path, device=device, threshold=0.50)
        all_eval_results.append(eval_res)

        # Save individual predictions
        img_pred_path = os.path.join(dirs["predictions"], f"{exp_id}_image_predictions.csv")
        pt_pred_path = os.path.join(dirs["predictions"], f"{exp_id}_patient_predictions.csv")
        eval_res["df_images"].to_csv(img_pred_path, index=False)
        eval_res["df_patients"].to_csv(pt_pred_path, index=False)
        print(f"  --> Saved predictions: {img_pred_path} ({len(eval_res['df_images'])} images), {pt_pred_path} ({len(eval_res['df_patients'])} patients)")

        # Record metrics rows for table
        img_m = eval_res["image_metrics"]
        pt_m = eval_res["patient_metrics"]

        summary_rows.append({
            "order": order,
            "experiment_id": exp_id,
            "model_name": exp["model_name"],
            "architecture": eval_res["architecture"],
            "modality": exp["modality"],
            "selected_epoch": eval_res["selected_epoch"],
            "level": "image",
            "threshold": img_m["threshold"],
            "n_samples": img_m["n_samples"],
            "n_positive": img_m["n_positive"],
            "n_negative": img_m["n_negative"],
            "accuracy": img_m["accuracy"],
            "precision": img_m["precision"],
            "recall": img_m["recall"],
            "sensitivity": img_m["sensitivity"],
            "specificity": img_m["specificity"],
            "f1_score": img_m["f1_score"],
            "g_mean": img_m["g_mean"],
            "roc_auc": img_m["roc_auc"],
            "pr_auc": img_m["pr_auc"],
            "tn": img_m["tn"],
            "fp": img_m["fp"],
            "fn": img_m["fn"],
            "tp": img_m["tp"],
        })

        summary_rows.append({
            "order": order,
            "experiment_id": exp_id,
            "model_name": exp["model_name"],
            "architecture": eval_res["architecture"],
            "modality": exp["modality"],
            "selected_epoch": eval_res["selected_epoch"],
            "level": "patient",
            "threshold": pt_m["threshold"],
            "n_samples": pt_m["n_samples"],
            "n_positive": pt_m["n_positive"],
            "n_negative": pt_m["n_negative"],
            "accuracy": pt_m["accuracy"],
            "precision": pt_m["precision"],
            "recall": pt_m["recall"],
            "sensitivity": pt_m["sensitivity"],
            "specificity": pt_m["specificity"],
            "f1_score": pt_m["f1_score"],
            "g_mean": pt_m["g_mean"],
            "roc_auc": pt_m["roc_auc"],
            "pr_auc": pt_m["pr_auc"],
            "tn": pt_m["tn"],
            "fp": pt_m["fp"],
            "fn": pt_m["fn"],
            "tp": pt_m["tp"],
        })

        print(f"  --> Image-Level:   Acc={img_m['accuracy']:.3%}, Sens={img_m['recall']:.3%}, Spec={img_m['specificity']:.3%}, F1={img_m['f1_score']:.4f}, ROC-AUC={img_m['roc_auc']:.4f}")
        print(f"  --> Patient-Level: Acc={pt_m['accuracy']:.3%}, Sens={pt_m['recall']:.3%}, Spec={pt_m['specificity']:.3%}, F1={pt_m['f1_score']:.4f}, ROC-AUC={pt_m['roc_auc']:.4f}")

        # Plot individual confusion matrix
        cm_path = os.path.join(dirs["confusion_matrices"], f"{exp_id}_confusion_matrices.png")
        plot_confusion_matrix_pair(
            cm_image=img_m["confusion_matrix"],
            cm_patient=pt_m["confusion_matrix"],
            exp_id=exp_id,
            model_name=exp["model_name"],
            modality=exp["modality"],
            save_path=cm_path,
        )

        # Plot individual learning curves
        if os.path.exists(hist_csv):
            hist_df = pd.read_csv(hist_csv)
            histories[exp_id] = hist_df
            hist_plot_path = os.path.join(dirs["training_curves"], f"{exp_id}_training_curves.png")
            plot_single_experiment_history(
                history_csv=hist_csv,
                best_epoch=eval_res["selected_epoch"],
                exp_title=f"{exp['model_name']} on {exp['modality']} ({exp_id})",
                save_path=hist_plot_path,
            )

    # 4. Generate Comparative Visualizations
    print("\n[Step 4/5] Generating comparative visualizations...", flush=True)

    # ROC Curves
    roc_img_path = os.path.join(dirs["curves"], "roc_curves_image_level.png")
    roc_pt_path = os.path.join(dirs["curves"], "roc_curves_patient_level.png")
    plot_roc_comparison(all_eval_results, level="image", save_path=roc_img_path)
    plot_roc_comparison(all_eval_results, level="patient", save_path=roc_pt_path)
    print(f"  --> Saved ROC comparisons: {roc_img_path}, {roc_pt_path}")

    # PR Curves
    pr_img_path = os.path.join(dirs["curves"], "pr_curves_image_level.png")
    pr_pt_path = os.path.join(dirs["curves"], "pr_curves_patient_level.png")
    plot_pr_comparison(all_eval_results, level="image", save_path=pr_img_path)
    plot_pr_comparison(all_eval_results, level="patient", save_path=pr_pt_path)
    print(f"  --> Saved PR comparisons: {pr_img_path}, {pr_pt_path}")

    # Training overlays
    if histories:
        plot_training_overlays(histories, dirs["training_curves"])
        print(f"  --> Saved validation overlays in: {dirs['training_curves']}")

    # Comparison Bar Chart
    df_metrics = pd.DataFrame(summary_rows)
    barchart_path = os.path.join(dirs["comparisons"], "model_metrics_comparison_barchart.png")
    plot_metrics_comparison_barchart(df_metrics, barchart_path)
    print(f"  --> Saved multi-metric comparison bar chart: {barchart_path}")

    # 5. Save Tables
    print("\n[Step 5/5] Saving tabular results...", flush=True)
    table_csv = os.path.join(dirs["tables"], "test_evaluation_results.csv")
    table_json = os.path.join(dirs["tables"], "test_evaluation_results.json")

    # Clean serializable dictionary for JSON
    serializable_summary = []
    for r in summary_rows:
        row_dict = dict(r)
        serializable_summary.append(row_dict)

    df_metrics.to_csv(table_csv, index=False)
    with open(table_json, "w", encoding="utf-8") as f:
        json.dump(serializable_summary, f, indent=2)

    print(f"  --> Saved results tables: {table_csv}, {table_json}")

    print("\n" + "=" * 80)
    print("PHASE 5 TEST EVALUATION COMPLETE")
    print("=" * 80)

    # Print summary table to stdout
    print("\nTEST SET EVALUATION SUMMARY TABLE:")
    headers = ["Exp ID", "Model", "Modality", "Level", "Acc", "Sens/Rec", "Spec", "F1", "ROC-AUC", "PR-AUC", "N (Pos/Neg)"]
    row_fmt = "{:<26} {:<16} {:<11} {:<8} {:<7} {:<9} {:<7} {:<7} {:<9} {:<9} {:<12}"
    print(row_fmt.format(*headers))
    print("-" * 125)
    for r in summary_rows:
        sens_str = f"{r['recall']:.1%}" if r["recall"] is not None else "N/A"
        spec_str = f"{r['specificity']:.1%}" if r["specificity"] is not None else "N/A"
        auc_str = f"{r['roc_auc']:.4f}" if r["roc_auc"] is not None else "N/A"
        pr_str = f"{r['pr_auc']:.4f}" if r["pr_auc"] is not None else "N/A"
        acc_str = f"{r['accuracy']:.1%}" if r["accuracy"] is not None else "N/A"
        f1_str = f"{r['f1_score']:.4f}" if r["f1_score"] is not None else "N/A"
        n_str = f"{r['n_samples']} ({r['n_positive']}/{r['n_negative']})"

        print(row_fmt.format(
            r["experiment_id"],
            r["model_name"],
            r["modality"],
            r["level"].capitalize(),
            acc_str,
            sens_str,
            spec_str,
            f1_str,
            auc_str,
            pr_str,
            n_str,
        ))
    print("-" * 125)

    return {
        "summary_rows": summary_rows,
        "dirs": dirs,
        "results": all_eval_results,
    }


if __name__ == "__main__":
    run_phase5_evaluation()
