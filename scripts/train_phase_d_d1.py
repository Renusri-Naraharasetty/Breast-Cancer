"""Phase D2 - Experiment D1: Reproduction of Frozen C3 Benchmark.

Strictly adheres to:
1. Exact C3 configuration:
   - Architecture: GraphInfusedTransformer (mg_dim=1280, us_dim=768, hidden_dim=256,
                   gat_layers=2, gat_heads=4, tf_layers=2, tf_heads=8, d_ff=1024,
                   dropout=0.3, num_classes=2)
   - Seed: 42
   - Loss: ClassBalancedFocalLoss(gamma=1.5, alpha=[1.0, 5.1111], reduction='mean')
   - Optimizer: AdamW(lr=1e-4, weight_decay=1e-2)
   - Scheduler: CosineAnnealingLR(T_max=20, eta_min=1e-6)
   - Effective batch size: 16 (batch_size=8, gradient_accumulation_steps=2)
   - Mixed precision: torch.amp.autocast
   - Checkpoint selection: Predeclared Composite Score (0.5*ROC-AUC + 0.3*PR-AUC + 0.2*F1)
   - Early stopping: patience=8
2. Evaluation strictly on Train (N=220) and Val (N=48) cohorts.
3. Test cohort (N=47) is strictly QUARANTINED and NEVER loaded.
4. Preserves all Phase C files (read-only baseline).
5. Writes outputs exclusively under outputs/phase_d/experiments/exp_d1_c3_repro/.
6. Tracks runtime, peak VRAM, epoch metrics, and compares directly with C3 reference.
"""

import os
import sys
import time
import math
import json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, Any, List
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
from src.models.focal_loss import ClassBalancedFocalLoss
from src.data.multimodal_dataset import get_patient_multimodal_dataloader


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_val_metrics(
    model: nn.Module,
    val_loader,
    loss_fn: nn.Module,
    device: torch.device,
    use_amp: bool = True,
) -> Dict[str, Any]:
    """Evaluates model on validation cohort and calculates all metrics + composite score."""
    model.eval()
    total_loss = 0.0
    all_targets = []
    all_probs = []
    patient_records = []

    with torch.no_grad():
        for batch in val_loader:
            labels = batch["labels"].to(device, non_blocking=True)
            mg_feats = batch["mg_features"].to(device, non_blocking=True)
            us_feats = batch["us_features"].to(device, non_blocking=True)
            mg_sides = batch["mg_sides"].to(device, non_blocking=True)
            us_sides = batch["us_sides"].to(device, non_blocking=True)
            mg_mask = batch["mg_mask"].to(device, non_blocking=True)
            us_mask = batch["us_mask"].to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
                logits = model(
                    mg_features=mg_feats,
                    us_features=us_feats,
                    mg_sides=mg_sides,
                    us_sides=us_sides,
                    mg_mask=mg_mask,
                    us_mask=us_mask,
                )
                loss = loss_fn(logits, labels)

            total_loss += loss.item() * len(labels)
            probs = torch.softmax(logits, dim=1)[:, 1]

            probs_np = probs.cpu().numpy()
            labels_np = labels.cpu().numpy()

            all_targets.extend(labels_np.tolist())
            all_probs.extend(probs_np.tolist())

            for i in range(len(labels)):
                patient_records.append({
                    "patient_id": batch["patient_ids"][i],
                    "root_patient_id": batch["root_patient_ids"][i],
                    "highest_birads": batch["highest_birads"][i],
                    "true_label": int(labels_np[i]),
                    "pred_prob": float(probs_np[i]),
                    "pred_label": int(probs_np[i] >= 0.50),
                })

    y_true = np.array(all_targets)
    y_prob = np.array(all_probs)
    y_pred = (y_prob >= 0.50).astype(int)

    n_samples = len(y_true)
    val_loss = total_loss / max(n_samples, 1)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    bal_acc = balanced_accuracy_score(y_true, y_pred)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    gmean = math.sqrt(rec * spec) if (rec * spec) > 0 else 0.0

    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except Exception:
        roc_auc = 0.5

    try:
        pr_auc = average_precision_score(y_true, y_prob)
    except Exception:
        pr_auc = float(np.mean(y_true))

    # Predeclared composite score: 0.5 * ROC-AUC + 0.3 * PR-AUC + 0.2 * F1
    composite_score = 0.5 * roc_auc + 0.3 * pr_auc + 0.2 * f1

    return {
        "val_loss": round(float(val_loss), 4),
        "val_acc": round(float(acc), 4),
        "val_precision": round(float(prec), 4),
        "val_recall": round(float(rec), 4),
        "val_sensitivity": round(float(rec), 4),
        "val_specificity": round(float(spec), 4),
        "val_f1": round(float(f1), 4),
        "val_balanced_acc": round(float(bal_acc), 4),
        "val_gmean": round(float(gmean), 4),
        "val_roc_auc": round(float(roc_auc), 4),
        "val_pr_auc": round(float(pr_auc), 4),
        "composite_score": round(float(composite_score), 4),
        "val_tn": int(tn),
        "val_fp": int(fp),
        "val_fn": int(fn),
        "val_tp": int(tp),
        "patient_records": patient_records,
    }


def run_experiment_d1():
    print("=" * 80)
    print("PHASE D2 - EXPERIMENT D1: REPRODUCE FROZEN C3 BENCHMARK")
    print("=" * 80)

    output_dir = "outputs/phase_d/experiments/exp_d1_c3_repro"
    os.makedirs(output_dir, exist_ok=True)

    seed = 42
    set_seed(seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    use_amp = (device.type == "cuda")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        gpu_name = torch.cuda.get_device_name(0)
    else:
        gpu_name = "CPU"

    print(f"Device:            {device} ({gpu_name})")
    print(f"Mixed Precision:   {use_amp}")
    print(f"Output Directory:  {output_dir}")
    print(f"Random Seed:       {seed}")

    # Checkpoint selection & hyperparameters matching C3 exactly
    lr = 1e-4
    weight_decay = 1e-2
    dropout_rate = 0.3
    epochs = 20
    patience = 8
    batch_size = 8
    gradient_accumulation_steps = 2
    focal_gamma = 1.5
    features_path = "data/features/patient_multimodal_features.pt"

    # Verify feature cache exists
    if not os.path.exists(features_path):
        raise FileNotFoundError(f"Feature cache missing: {features_path}")

    # 1. Dataloaders: Train and Validation ONLY. Test is strictly forbidden.
    train_loader = get_patient_multimodal_dataloader(
        features_path=features_path,
        split="train",
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = get_patient_multimodal_dataloader(
        features_path=features_path,
        split="val",
        batch_size=batch_size,
        shuffle=False,
    )

    print(f"Train Cohort:      {len(train_loader.dataset)} patients ({len(train_loader)} batches, bs={batch_size})")
    print(f"Validation Cohort: {len(val_loader.dataset)} patients ({len(val_loader)} batches, bs={batch_size})")
    print("Test Cohort:       [QUARANTINED - NEVER LOADED]")

    # 2. Loss Function Definition
    class_weights = torch.tensor([1.0, 5.1111], dtype=torch.float32).to(device)
    loss_fn = ClassBalancedFocalLoss(gamma=focal_gamma, alpha=class_weights)
    loss_formula_str = f"FL(p_t) = -alpha_t * (1 - p_t)^{focal_gamma} * log(p_t), normalized by sum(alpha_t)"

    # 3. Model Architecture
    model = GraphInfusedTransformer(
        mg_feat_dim=1280,
        us_feat_dim=768,
        hidden_dim=256,
        num_gat_layers=2,
        gat_heads=4,
        transformer_layers=2,
        transformer_heads=8,
        dim_feedforward=1024,
        num_classes=2,
        dropout_rate=dropout_rate,
    ).to(device)

    # 4. Optimizer, Scheduler, Scaler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_composite_score = -1.0
    best_val_loss = float("inf")
    best_epoch = -1
    best_val_metrics = {}
    best_model_state = None
    best_val_records = []
    epochs_without_improvement = 0

    history_records = []

    total_t0 = time.perf_counter()

    for epoch in range(1, epochs + 1):
        ep_t0 = time.perf_counter()
        model.train()
        total_train_loss = 0.0
        train_correct = 0
        train_total = 0

        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(train_loader):
            labels = batch["labels"].to(device, non_blocking=True)
            mg_feats = batch["mg_features"].to(device, non_blocking=True)
            us_feats = batch["us_features"].to(device, non_blocking=True)
            mg_sides = batch["mg_sides"].to(device, non_blocking=True)
            us_sides = batch["us_sides"].to(device, non_blocking=True)
            mg_mask = batch["mg_mask"].to(device, non_blocking=True)
            us_mask = batch["us_mask"].to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(
                    mg_features=mg_feats,
                    us_features=us_feats,
                    mg_sides=mg_sides,
                    us_sides=us_sides,
                    mg_mask=mg_mask,
                    us_mask=us_mask,
                )
                loss = loss_fn(logits, labels)
                loss_scaled = loss / gradient_accumulation_steps

            scaler.scale(loss_scaled).backward()

            if (step + 1) % gradient_accumulation_steps == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            total_train_loss += loss.item() * len(labels)
            preds = logits.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += len(labels)

        scheduler.step()

        train_loss = total_train_loss / max(train_total, 1)
        train_acc = train_correct / max(train_total, 1)

        # Validation evaluation
        val_res = compute_val_metrics(
            model=model,
            val_loader=val_loader,
            loss_fn=loss_fn,
            device=device,
            use_amp=use_amp,
        )

        current_lr = scheduler.get_last_lr()[0]
        epoch_time = time.perf_counter() - ep_t0

        comp_score = val_res["composite_score"]
        record = {
            "epoch": epoch,
            "train_loss": round(float(train_loss), 4),
            "train_acc": round(float(train_acc), 4),
            "val_loss": val_res["val_loss"],
            "val_acc": val_res["val_acc"],
            "val_sensitivity": val_res["val_sensitivity"],
            "val_specificity": val_res["val_specificity"],
            "val_precision": val_res["val_precision"],
            "val_f1": val_res["val_f1"],
            "val_balanced_acc": val_res["val_balanced_acc"],
            "val_roc_auc": val_res["val_roc_auc"],
            "val_pr_auc": val_res["val_pr_auc"],
            "composite_score": comp_score,
            "lr": round(float(current_lr), 7),
            "epoch_time_sec": round(float(epoch_time), 2),
        }
        history_records.append(record)

        # Predeclared selection protocol: highest composite score, then lower val loss
        is_best = False
        if comp_score > best_composite_score + 1e-4:
            is_best = True
        elif abs(comp_score - best_composite_score) <= 1e-4 and val_res["val_loss"] < best_val_loss:
            is_best = True

        if is_best:
            best_composite_score = comp_score
            best_val_loss = val_res["val_loss"]
            best_epoch = epoch
            best_val_metrics = val_res
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_val_records = val_res["patient_records"]
            epochs_without_improvement = 0
            mark = "(*) BEST"
        else:
            epochs_without_improvement += 1
            mark = ""

        print(
            f"Epoch {epoch:2d}/{epochs:2d} | TrainLoss: {train_loss:.4f} Acc: {train_acc:.3f} | "
            f"ValLoss: {val_res['val_loss']:.4f} AUC: {val_res['val_roc_auc']:.4f} PR-AUC: {val_res['val_pr_auc']:.4f} "
            f"F1: {val_res['val_f1']:.4f} Sens: {val_res['val_sensitivity']:.2f} Spec: {val_res['val_specificity']:.2f} "
            f"BalAcc: {val_res['val_balanced_acc']:.4f} | Comp: {comp_score:.4f} {mark}"
        )

        if epochs_without_improvement >= patience:
            print(f"Early stopping triggered at epoch {epoch} (patience={patience}).")
            break

    total_runtime_sec = time.perf_counter() - total_t0

    # VRAM measurement
    if device.type == "cuda":
        peak_vram_bytes = torch.cuda.max_memory_allocated()
        peak_vram_mb = peak_vram_bytes / (1024 * 1024)
    else:
        peak_vram_mb = 0.0

    print("-" * 80)
    print(f"Training completed in {total_runtime_sec:.2f}s ({total_runtime_sec/60:.2f} min).")
    print(f"Peak VRAM: {peak_vram_mb:.2f} MB")
    print(f"Selected Best Epoch: {best_epoch} with Composite Score: {best_composite_score:.4f}")
    print("-" * 80)

    # Overfitting indicators
    selected_record = next((r for r in history_records if r["epoch"] == best_epoch), history_records[-1])
    train_loss_at_best = selected_record["train_loss"]
    train_acc_at_best = selected_record["train_acc"]
    final_record = history_records[-1]
    final_train_loss = final_record["train_loss"]
    final_val_loss = final_record["val_loss"]

    overfitting_indicators = {
        "train_loss_selected_epoch": train_loss_at_best,
        "val_loss_selected_epoch": best_val_metrics["val_loss"],
        "loss_gap_selected_epoch": round(float(best_val_metrics["val_loss"] - train_loss_at_best), 4),
        "train_acc_selected_epoch": train_acc_at_best,
        "val_acc_selected_epoch": best_val_metrics["val_acc"],
        "final_epoch": final_record["epoch"],
        "final_train_loss": final_train_loss,
        "final_val_loss": final_val_loss,
        "final_loss_gap": round(float(final_val_loss - final_train_loss), 4),
    }

    config_payload = {
        "exp_id": "exp_d1_c3_repro",
        "description": "Phase D2 Reproduction of Frozen Candidate C3 Benchmark",
        "architecture": "GraphInfusedTransformer",
        "loss_type": "focal",
        "focal_gamma": focal_gamma,
        "loss_formula": loss_formula_str,
        "alpha_weights": [1.0, 5.1111],
        "reduction": "mean (divided by sum of class weights across batch)",
        "training_distribution": {"class_0_neg": 184, "class_1_pos": 36, "ratio_neg_to_pos": 5.1111},
        "validation_distribution": {"class_0_neg": 40, "class_1_pos": 8, "ratio_neg_to_pos": 5.0},
        "checkpoint_selection_protocol": "Predeclared Composite Score = 0.5 * ROC-AUC + 0.3 * PR-AUC + 0.2 * F1 (tie-breaker: lower val_loss)",
        "lr": lr,
        "weight_decay": weight_decay,
        "dropout_rate": dropout_rate,
        "hidden_dim": 256,
        "num_gat_layers": 2,
        "gat_heads": 4,
        "transformer_layers": 2,
        "transformer_heads": 8,
        "dim_feedforward": 1024,
        "batch_size": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "effective_batch_size": batch_size * gradient_accumulation_steps,
        "seed": seed,
        "device": str(device),
        "gpu_name": gpu_name,
        "total_runtime_seconds": round(total_runtime_sec, 2),
        "peak_vram_mb": round(peak_vram_mb, 2),
    }

    # Save outputs under outputs/phase_d/experiments/exp_d1_c3_repro/
    # 1. Config JSON
    with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config_payload, f, indent=2)

    # 2. Best Checkpoint
    ckpt_path = os.path.join(output_dir, "best_checkpoint.pth")
    torch.save({
        "epoch": best_epoch,
        "model_state_dict": best_model_state,
        "composite_score": best_composite_score,
        "val_metrics": best_val_metrics,
        "overfitting_indicators": overfitting_indicators,
        "config": config_payload,
    }, ckpt_path)
    print(f">> Saved D1 best checkpoint (Epoch {best_epoch}, CompScore {best_composite_score:.4f}) to {ckpt_path}")

    # 3. Training History CSV & JSON
    df_hist = pd.DataFrame(history_records)
    df_hist.to_csv(os.path.join(output_dir, "training_history.csv"), index=False)
    with open(os.path.join(output_dir, "training_history.json"), "w", encoding="utf-8") as f:
        json.dump(history_records, f, indent=2)

    # 4. Validation Predictions CSV
    df_val_preds = pd.DataFrame(best_val_records)
    val_pred_path = os.path.join(output_dir, "val_predictions.csv")
    df_val_preds.to_csv(val_pred_path, index=False)

    # 5. Validation Metrics JSON
    summary_path = os.path.join(output_dir, "validation_metrics.json")
    val_clean = {k: v for k, v in best_val_metrics.items() if k != "patient_records"}
    val_clean["selected_epoch"] = best_epoch
    val_clean["exp_id"] = "exp_d1_c3_repro"
    val_clean["total_runtime_seconds"] = round(total_runtime_sec, 2)
    val_clean["peak_vram_mb"] = round(peak_vram_mb, 2)
    val_clean["overfitting_indicators"] = overfitting_indicators
    val_clean["metadata"] = config_payload
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(val_clean, f, indent=2)

    print(f">> Saved D1 validation metrics to {summary_path}")

    # Reference C3 metrics for instant comparison
    c3_ref = {
        "selected_epoch": 4,
        "val_loss": 0.1679,
        "val_roc_auc": 0.9281,
        "val_pr_auc": 0.7661,
        "val_f1": 0.7059,
        "val_sensitivity": 0.7500,
        "val_specificity": 0.9250,
        "val_balanced_acc": 0.8375,
        "composite_score": 0.8351,
        "val_tn": 37,
        "val_fp": 3,
        "val_fn": 2,
        "val_tp": 6,
    }

    print("\n" + "=" * 80)
    print("REPRODUCTION COMPARISON: D1 vs ORIGINAL FROZEN C3")
    print("=" * 80)
    row_fmt = "{:<25} {:<15} {:<15} {:<15}"
    print(row_fmt.format("Metric", "Frozen C3", "D1 Repro", "Difference"))
    print("-" * 70)
    for k in ["selected_epoch", "composite_score", "val_roc_auc", "val_pr_auc", "val_f1",
              "val_sensitivity", "val_specificity", "val_balanced_acc", "val_loss"]:
        v_c3 = c3_ref[k]
        v_d1 = val_clean[k]
        diff = v_d1 - v_c3
        diff_str = f"{diff:+.4f}" if isinstance(diff, float) else f"{diff:+d}"
        val_c3_str = f"{v_c3:.4f}" if isinstance(v_c3, float) else str(v_c3)
        val_d1_str = f"{v_d1:.4f}" if isinstance(v_d1, float) else str(v_d1)
        print(row_fmt.format(k, val_c3_str, val_d1_str, diff_str))

    print("-" * 70)
    cm_c3 = f"[{c3_ref['val_tn']}, {c3_ref['val_fp']}, {c3_ref['val_fn']}, {c3_ref['val_tp']}]"
    cm_d1 = f"[{val_clean['val_tn']}, {val_clean['val_fp']}, {val_clean['val_fn']}, {val_clean['val_tp']}]"
    print(row_fmt.format("Confusion Matrix", cm_c3, cm_d1, "Identical" if cm_c3 == cm_d1 else "Different"))
    print("=" * 80)


if __name__ == "__main__":
    run_experiment_d1()
