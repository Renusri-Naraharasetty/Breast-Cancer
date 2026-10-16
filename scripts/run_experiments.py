"""Phase 4 Training Runner: Independent Training and Checkpointing of Four Models.

Order of Execution:
1. EfficientNet-B0 on mammograms (exp1_efficientnet_mammogram)
2. ViT-B/16 on ultrasound (exp2_vit_ultrasound)
3. ViT-B/16 on mammograms (exp3_vit_mammogram)
4. EfficientNet-B0 on ultrasound (exp4_efficientnet_ultrasound)

Hardware Safety & Integrity:
- Mixed precision (torch.amp) enabled
- Small batch size (8) with gradient accumulation (2) -> effective batch size 16
- Automatic CUDA OOM fallback recovery
- Strict sequential execution with GPU memory purge between models
- Training data only for updates; validation data for early stopping / best checkpoint
- Held-out test set is NEVER touched during Phase 4
"""

import os
import sys
import gc
import json
import time
import argparse
import torch
import pandas as pd
from typing import List, Dict, Any

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.training.trainer import train_single_experiment


EXPERIMENTS = [
    {
        "id": "exp1_efficientnet_mammogram",
        "order": 1,
        "model_name": "efficientnet_b0",
        "modality": "mammogram",
        "lr": 1e-4,
        "batch_size": 8,
        "gradient_accumulation_steps": 2,
        "epochs": 12,
        "patience": 4,
        "weight_decay": 1e-2,
        "description": "EfficientNet-B0 on Mammograms",
    },
    {
        "id": "exp2_vit_ultrasound",
        "order": 2,
        "model_name": "vit_b_16",
        "modality": "ultrasound",
        "lr": 5e-5,
        "batch_size": 4,
        "gradient_accumulation_steps": 4,
        "epochs": 12,
        "patience": 4,
        "weight_decay": 1e-2,
        "description": "Vision Transformer (ViT-B/16) on Ultrasound",
    },
    {
        "id": "exp3_vit_mammogram",
        "order": 3,
        "model_name": "vit_b_16",
        "modality": "mammogram",
        "lr": 5e-5,
        "batch_size": 4,
        "gradient_accumulation_steps": 4,
        "epochs": 12,
        "patience": 4,
        "weight_decay": 1e-2,
        "description": "Vision Transformer (ViT-B/16) on Mammograms",
    },
    {
        "id": "exp4_efficientnet_ultrasound",
        "order": 4,
        "model_name": "efficientnet_b0",
        "modality": "ultrasound",
        "lr": 1e-4,
        "batch_size": 8,
        "gradient_accumulation_steps": 2,
        "epochs": 12,
        "patience": 4,
        "weight_decay": 1e-2,
        "description": "EfficientNet-B0 on Ultrasound",
    },
]


def run_all_experiments(
    output_base_dir: str = "outputs/experiments",
    selected_indices: List[int] = None,
) -> List[Dict[str, Any]]:
    os.makedirs(output_base_dir, exist_ok=True)
    results = []

    print("\n" + "=" * 80, flush=True)
    print("PHASE 4: INDEPENDENT TRAINING AND CHECKPOINTING PIPELINE", flush=True)
    print(f"Total experiments planned: {len(EXPERIMENTS)}", flush=True)
    print(f"Device: {'CUDA (' + torch.cuda.get_device_name(0) + ')' if torch.cuda.is_available() else 'CPU'}", flush=True)
    print("=" * 80, flush=True)

    for exp_cfg in EXPERIMENTS:
        idx = exp_cfg["order"]
        if selected_indices and idx not in selected_indices:
            print(f"Skipping experiment {idx} ({exp_cfg['id']}) as requested.", flush=True)
            continue

        print(f"\n>>> Running Experiment {idx}/4: {exp_cfg['description']} ({exp_cfg['id']})", flush=True)

        # Memory cleanup before experiment starts
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        t0 = time.time()
        try:
            exp_result = train_single_experiment(
                experiment_id=exp_cfg["id"],
                model_name=exp_cfg["model_name"],
                modality=exp_cfg["modality"],
                output_base_dir=output_base_dir,
                epochs=exp_cfg["epochs"],
                batch_size=exp_cfg["batch_size"],
                gradient_accumulation_steps=exp_cfg["gradient_accumulation_steps"],
                lr=exp_cfg["lr"],
                weight_decay=exp_cfg["weight_decay"],
                patience=exp_cfg["patience"],
                seed=42,
                use_amp=True,
            )
            exp_result["status"] = "COMPLETED"
            results.append(exp_result)
        except Exception as e:
            print(f"\n[ERROR] Experiment {exp_cfg['id']} failed with error: {str(e)}")
            import traceback
            traceback.print_exc()
            results.append({
                "experiment_id": exp_cfg["id"],
                "model_name": exp_cfg["model_name"],
                "modality": exp_cfg["modality"],
                "status": f"FAILED: {str(e)}",
                "checkpoint_path": None,
                "best_epoch": None,
                "best_val_loss": None,
                "best_val_roc_auc": None,
                "total_time_mins": round((time.time() - t0) / 60, 2),
            })

        # Memory cleanup after experiment completes
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save summary manifest
    summary_path = os.path.join(output_base_dir, "phase4_training_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        # Avoid serializing full history lists into top-level summary
        serializable_results = []
        for r in results:
            item = dict(r)
            if "history" in item:
                del item["history"]
            serializable_results.append(item)
        json.dump(serializable_results, f, indent=2)

    print("\n" + "=" * 80)
    print("PHASE 4 TRAINING COMPLETE - SUMMARY TABLE")
    print("=" * 80)
    headers = ["Exp ID", "Model", "Modality", "Status", "Best Ep", "Val Loss", "Val AUC", "Time (m)"]
    row_fmt = "{:<28} {:<16} {:<12} {:<10} {:<8} {:<10} {:<10} {:<8}"
    print(row_fmt.format(*headers))
    print("-" * 105)
    for r in results:
        best_ep = str(r.get("best_epoch", "N/A"))
        val_loss = f"{r.get('best_val_loss', 0.0):.4f}" if r.get("best_val_loss") is not None else "N/A"
        val_auc = f"{r.get('best_val_roc_auc', 0.0):.4f}" if r.get("best_val_roc_auc") is not None else "N/A"
        time_m = f"{r.get('total_time_mins', 0.0):.1f}"
        print(row_fmt.format(
            r["experiment_id"],
            r["model_name"],
            r["modality"],
            r["status"],
            best_ep,
            val_loss,
            val_auc,
            time_m,
        ))
    print("-" * 105)
    print(f"Summary JSON saved to: {summary_path}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 4 Training Runner")
    parser.add_argument("--exp", type=int, nargs="+", help="Specific experiment numbers to run (1, 2, 3, 4)")
    args = parser.parse_args()

    run_all_experiments(selected_indices=args.exp)
