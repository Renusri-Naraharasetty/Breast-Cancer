"""Training Pipeline for Second Multimodal Graph-Infused Transformer Experiment:

exp_git_vit_mammogram_effnet_ultrasound

Backbones:
- Mammogram: ViT-B/16 (768-dim) from Experiment 3
- Ultrasound: EfficientNet-B0 (1280-dim) from Experiment 4

Hardware Safety & Optimization:
- Tuned for NVIDIA GeForce MX550 (2 GB VRAM).
- Uses memory-efficient feature tensors (VRAM footprint < 100 MB).
- Mixed precision (torch.amp) enabled.
- Effective batch size 16 (8 x 2).
- Patient-level stratified split (train=220, val=48).
- Model selection governed strictly by validation performance.
- Held-out test set is NEVER touched during training or hyperparameter selection.
"""

import os
import sys
import gc
import json
import time
import math
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.git import GraphInfusedTransformer
from src.data.multimodal_dataset import get_patient_multimodal_dataloader


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def evaluate_git_validation(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    use_amp: bool = True,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    all_targets = []
    all_preds = []
    all_probs = []

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
            preds = (probs >= 0.50).long()

            all_targets.extend(labels.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    n_samples = len(y_true)
    val_loss = total_loss / max(n_samples, 1)

    val_acc = accuracy_score(y_true, y_pred)
    val_prec = precision_score(y_true, y_pred, zero_division=0)
    val_rec = recall_score(y_true, y_pred, zero_division=0)
    val_f1 = f1_score(y_true, y_pred, zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    val_spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    val_gmean = math.sqrt(val_rec * val_spec) if (val_rec * val_spec) > 0 else 0.0

    try:
        val_roc_auc = roc_auc_score(y_true, y_prob)
    except Exception:
        val_roc_auc = 0.5

    try:
        val_pr_auc = average_precision_score(y_true, y_prob)
    except Exception:
        val_pr_auc = float(np.mean(y_true))

    return {
        "val_loss": round(float(val_loss), 4),
        "val_acc": round(float(val_acc), 4),
        "val_precision": round(float(val_prec), 4),
        "val_recall": round(float(val_rec), 4),
        "val_sensitivity": round(float(val_rec), 4),
        "val_specificity": round(float(val_spec), 4),
        "val_f1": round(float(val_f1), 4),
        "val_gmean": round(float(val_gmean), 4),
        "val_roc_auc": round(float(val_roc_auc), 4),
        "val_pr_auc": round(float(val_pr_auc), 4),
        "val_tn": int(tn),
        "val_fp": int(fp),
        "val_fn": int(fn),
        "val_tp": int(tp),
    }


def train_git_vit_effnet_experiment(
    features_path: str = "data/features/patient_multimodal_features_vit_effnet.pt",
    output_dir: str = "outputs/experiments/exp_git_vit_mammogram_effnet_ultrasound",
    epochs: int = 30,
    lr: float = 1e-4,
    weight_decay: float = 1e-2,
    batch_size: int = 8,
    gradient_accumulation_steps: int = 2,
    patience: int = 8,
    hidden_dim: int = 256,
    num_gat_layers: int = 2,
    gat_heads: int = 4,
    transformer_layers: int = 2,
    transformer_heads: int = 8,
    dim_feedforward: int = 1024,
    dropout_rate: float = 0.3,
    seed: int = 42,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    set_seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    use_amp = (device.type == "cuda")
    print("\n" + "=" * 80, flush=True)
    print("TRAINING MULTIMODAL GIT (ViT-B/16 MG + EfficientNet-B0 US)", flush=True)
    print(f"Experiment ID: exp_git_vit_mammogram_effnet_ultrasound", flush=True)
    print(f"Device: {device} | Mixed Precision (AMP): {use_amp}", flush=True)
    print(f"Effective Batch Size: {batch_size * gradient_accumulation_steps} ({batch_size} x {gradient_accumulation_steps})", flush=True)
    print(f"Feature Dims: MG={768} (ViT-B/16), US={1280} (EfficientNet-B0)", flush=True)
    print("=" * 80, flush=True)

    # 1. Dataloaders
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

    print(f"Loaded Train: {len(train_loader.dataset)} patients ({len(train_loader)} batches)", flush=True)
    print(f"Loaded Val:   {len(val_loader.dataset)} patients ({len(val_loader)} batches)", flush=True)

    # 2. Class weights for balanced triage
    class_weights = torch.tensor([1.0, 5.1111], dtype=torch.float32).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    # 3. Model instantiation with adapted feature dimensions (mg=768, us=1280)
    model = GraphInfusedTransformer(
        mg_feat_dim=768,
        us_feat_dim=1280,
        hidden_dim=hidden_dim,
        num_gat_layers=num_gat_layers,
        gat_heads=gat_heads,
        transformer_layers=transformer_layers,
        transformer_heads=transformer_heads,
        dim_feedforward=dim_feedforward,
        num_classes=2,
        dropout_rate=dropout_rate,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters: {total_params:,} (Trainable: {trainable_params:,})", flush=True)

    # 4. Optimizer, Scheduler, Scaler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_val_loss = float("inf")
    best_val_roc_auc = 0.0
    best_epoch = -1
    best_val_metrics = {}
    best_model_state = None
    epochs_without_improvement = 0

    history_records = []

    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        model.train()
        train_loss_accum = 0.0
        train_targets = []
        train_preds = []

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

            train_loss_accum += loss.item() * len(labels)
            preds = torch.argmax(logits, dim=1)
            train_targets.extend(labels.cpu().numpy().tolist())
            train_preds.extend(preds.cpu().numpy().tolist())

            if (step + 1) % gradient_accumulation_steps == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

        scheduler.step()

        train_loss = train_loss_accum / len(train_loader.dataset)
        train_acc = accuracy_score(train_targets, train_preds)

        # Validation evaluation
        val_res = evaluate_git_validation(
            model=model,
            val_loader=val_loader,
            loss_fn=loss_fn,
            device=device,
            use_amp=use_amp,
        )

        current_lr = scheduler.get_last_lr()[0]
        epoch_time = time.time() - epoch_start_time

        record = {
            "epoch": epoch,
            "train_loss": round(float(train_loss), 4),
            "train_acc": round(float(train_acc), 4),
            "val_loss": val_res["val_loss"],
            "val_acc": val_res["val_acc"],
            "val_precision": val_res["val_precision"],
            "val_recall": val_res["val_recall"],
            "val_sensitivity": val_res["val_sensitivity"],
            "val_specificity": val_res["val_specificity"],
            "val_f1": val_res["val_f1"],
            "val_roc_auc": val_res["val_roc_auc"],
            "val_pr_auc": val_res["val_pr_auc"],
            "lr": round(float(current_lr), 7),
            "epoch_time_sec": round(float(epoch_time), 2),
        }
        history_records.append(record)

        is_best = False
        if val_res["val_loss"] < best_val_loss - 1e-4:
            is_best = True
        elif abs(val_res["val_loss"] - best_val_loss) < 0.02 and val_res["val_roc_auc"] > best_val_roc_auc:
            is_best = True

        if is_best:
            best_val_loss = val_res["val_loss"]
            best_val_roc_auc = val_res["val_roc_auc"]
            best_epoch = epoch
            best_val_metrics = val_res
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
            mark = "(*) BEST"
        else:
            epochs_without_improvement += 1
            mark = ""

        print(
            f"Epoch {epoch:2d}/{epochs:2d} | TrainLoss: {train_loss:.4f} Acc: {train_acc:.3f} | "
            f"ValLoss: {val_res['val_loss']:.4f} Acc: {val_res['val_acc']:.3f} "
            f"Sens: {val_res['val_sensitivity']:.3f} Spec: {val_res['val_specificity']:.3f} "
            f"F1: {val_res['val_f1']:.3f} AUC: {val_res['val_roc_auc']:.3f} | "
            f"{epoch_time:.1f}s {mark}",
            flush=True,
        )

        if epochs_without_improvement >= patience:
            print(f"\nEarly stopping triggered after {patience} epochs without validation improvement.", flush=True)
            break

    # Save best checkpoint
    checkpoint_path = os.path.join(output_dir, "best_checkpoint.pth")
    checkpoint_payload = {
        "experiment_id": "exp_git_vit_mammogram_effnet_ultrasound",
        "model_name": "git_vit_effnet",
        "architecture": "GraphInfusedTransformer",
        "backbones": "ViT-B/16 (Mammogram, exp3) + EfficientNet-B0 (Ultrasound, exp4)",
        "epoch": best_epoch,
        "model_state_dict": best_model_state,
        "class_weights": class_weights.cpu().tolist(),
        "best_val_loss": best_val_loss,
        "best_val_roc_auc": best_val_roc_auc,
        "val_metrics": best_val_metrics,
        "config": {
            "mg_feat_dim": 768,
            "us_feat_dim": 1280,
            "hidden_dim": hidden_dim,
            "num_gat_layers": num_gat_layers,
            "gat_heads": gat_heads,
            "transformer_layers": transformer_layers,
            "transformer_heads": transformer_heads,
            "dim_feedforward": dim_feedforward,
            "dropout_rate": dropout_rate,
            "batch_size": batch_size,
            "effective_batch_size": batch_size * gradient_accumulation_steps,
            "lr": lr,
            "weight_decay": weight_decay,
            "seed": seed,
        },
    }
    torch.save(checkpoint_payload, checkpoint_path)
    print(f"\n[OK] Best checkpoint saved to: {checkpoint_path}", flush=True)
    print(f"     Selected Epoch: {best_epoch} | Best Val Loss: {best_val_loss:.4f} | Best Val ROC-AUC: {best_val_roc_auc:.4f}", flush=True)

    # Save training history
    history_csv = os.path.join(output_dir, "training_history.csv")
    pd.DataFrame(history_records).to_csv(history_csv, index=False)

    summary_json = os.path.join(output_dir, "training_summary.json")
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump({
            "experiment_id": "exp_git_vit_mammogram_effnet_ultrasound",
            "selected_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "best_val_roc_auc": best_val_roc_auc,
            "val_metrics": best_val_metrics,
            "history_csv": history_csv,
            "checkpoint_path": checkpoint_path,
        }, f, indent=2)

    return checkpoint_payload


if __name__ == "__main__":
    train_git_vit_effnet_experiment()
