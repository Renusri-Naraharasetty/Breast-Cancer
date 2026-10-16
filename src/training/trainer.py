import os
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

from ..models.factory import create_model, get_model_info
from ..data.dataset import get_dataloader
from .losses import get_loss_function, compute_image_level_class_weights


def set_seed(seed: int = 42):
    """Sets deterministic random seed across Python, NumPy, and PyTorch."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def evaluate_validation(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    use_amp: bool = True,
) -> Dict[str, float]:
    """Evaluates model performance on the validation partition."""
    model.eval()
    total_loss = 0.0
    all_targets = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for images, labels, _ in val_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
                logits = model(images)
                loss = loss_fn(logits, labels)

            total_loss += loss.item() * len(labels)
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = torch.argmax(logits, dim=1)

            all_targets.extend(labels.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    n_samples = len(y_true)
    val_loss = total_loss / max(n_samples, 1)

    # Standard metrics
    val_acc = accuracy_score(y_true, y_pred)
    val_prec = precision_score(y_true, y_pred, zero_division=0)
    val_rec = recall_score(y_true, y_pred, zero_division=0)
    val_f1 = f1_score(y_true, y_pred, zero_division=0)

    # Specificity
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    val_spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # Geometric Mean (G-Mean)
    val_gmean = math.sqrt(val_rec * val_spec) if (val_rec * val_spec) > 0 else 0.0

    # ROC-AUC and PR-AUC
    try:
        val_roc_auc = roc_auc_score(y_true, y_prob)
    except Exception:
        val_roc_auc = 0.5

    try:
        val_pr_auc = average_precision_score(y_true, y_prob)
    except Exception:
        val_pr_auc = float(np.mean(y_true))

    return {
        "val_loss": round(val_loss, 4),
        "val_acc": round(float(val_acc), 4),
        "val_precision": round(float(val_prec), 4),
        "val_recall": round(float(val_rec), 4),
        "val_specificity": round(float(val_spec), 4),
        "val_f1": round(float(val_f1), 4),
        "val_gmean": round(float(val_gmean), 4),
        "val_roc_auc": round(float(val_roc_auc), 4),
        "val_pr_auc": round(float(val_pr_auc), 4),
    }


def _train_single_experiment_core(
    experiment_id: str,
    model_name: str,
    modality: str,
    output_base_dir: str,
    epochs: int,
    batch_size: int,
    gradient_accumulation_steps: int,
    lr: float,
    weight_decay: float,
    patience: int,
    seed: int,
    use_amp: bool,
) -> Dict[str, Any]:
    """Internal core routine for training a single experiment."""
    set_seed(seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    exp_dir = os.path.join(output_base_dir, experiment_id)
    os.makedirs(exp_dir, exist_ok=True)

    print(f"\n{'='*70}", flush=True)
    print(f"STARTING EXPERIMENT: {experiment_id}", flush=True)
    print(f"Model: {model_name} | Modality: {modality} | Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})", flush=True)
    print(f"Config: Epochs={epochs}, BatchSize={batch_size}, GradAccum={gradient_accumulation_steps} (Effective Batch={batch_size*gradient_accumulation_steps}), LR={lr}, Patience={patience}", flush=True)
    print(f"{'='*70}", flush=True)

    # 1. Data Loaders (Train & Val only - Held-out test set is strictly untouched)
    train_loader = get_dataloader(
        modality=modality,
        split="train",
        batch_size=batch_size,
        side_aware_labeling=True,
        shuffle=True,
        num_workers=0,
    )
    val_loader = get_dataloader(
        modality=modality,
        split="val",
        batch_size=batch_size,
        side_aware_labeling=True,
        shuffle=False,
        num_workers=0,
    )

    # 2. Compute exact empirical image-level class weights
    class_weights = compute_image_level_class_weights(
        modality=modality,
        side_aware_labeling=True,
        normalize_w0_to_one=True,
    )
    print(f"Empirical Image Class Weights for {modality.upper()}: [LOW_RISK: {class_weights[0]:.4f}, SUSPICIOUS: {class_weights[1]:.4f}]", flush=True)

    loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(device))

    # 3. Model Instantiation with Pretrained Weights
    model = create_model(model_name=model_name, pretrained=True, num_classes=2, dropout_rate=0.3)
    model.to(device)

    arch_name = model.__class__.__name__
    model_info = get_model_info(model)
    print(f"Model: {arch_name} | Params: Total={model_info['total_parameters']:,}, Trainable={model_info['trainable_parameters']:,} ({model_info['model_size_mb']} MB)", flush=True)

    # 4. Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and device.type == "cuda")

    # 5. Training Loop State
    history = []
    best_val_loss = float("inf")
    best_val_roc_auc = 0.0
    best_epoch = 0
    epochs_no_improve = 0
    early_stopped = False
    best_checkpoint_path = os.path.join(exp_dir, "best_checkpoint.pth")
    last_checkpoint_path = os.path.join(exp_dir, "last_checkpoint.pth")

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_accum = 0.0
        train_correct = 0
        train_total = 0

        optimizer.zero_grad()

        for step, (images, labels, _) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
                logits = model(images)
                loss = loss_fn(logits, labels)
                # Scale loss for gradient accumulation
                loss = loss / gradient_accumulation_steps

            scaler.scale(loss).backward()

            if (step + 1) % gradient_accumulation_steps == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            train_loss_accum += (loss.item() * gradient_accumulation_steps) * len(labels)
            preds = torch.argmax(logits, dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += len(labels)

        train_loss = train_loss_accum / max(train_total, 1)
        train_acc = train_correct / max(train_total, 1)

        # Validation evaluation
        val_metrics = evaluate_validation(model, val_loader, loss_fn, device, use_amp)
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        epoch_record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 4),
            "lr": round(current_lr, 6),
            **val_metrics,
        }
        history.append(epoch_record)

        val_loss = val_metrics["val_loss"]
        val_auc = val_metrics["val_roc_auc"]
        val_f1 = val_metrics["val_f1"]

        print(
            f"Epoch {epoch:02d}/{epochs:02d} | "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.3%} | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_metrics['val_acc']:.3%}, Val AUC: {val_auc:.3f}, Val F1: {val_f1:.3f} | "
            f"LR: {current_lr:.6f}",
            flush=True,
        )

        # Checkpoint payload
        checkpoint_payload = {
            "experiment_id": experiment_id,
            "model_name": model_name,
            "architecture": model.__class__.__name__,
            "pretrained_initialization": "ImageNet-1K (torchvision default)",
            "modality": modality,
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "class_weights": class_weights.cpu().numpy().tolist(),
            "best_val_loss": best_val_loss,
            "best_val_roc_auc": best_val_roc_auc,
            "val_metrics": val_metrics,
            "config": {
                "epochs": epochs,
                "batch_size": batch_size,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "effective_batch_size": batch_size * gradient_accumulation_steps,
                "lr": lr,
                "weight_decay": weight_decay,
                "seed": seed,
                "pretrained": True,
            },
        }

        # Save last checkpoint
        torch.save(checkpoint_payload, last_checkpoint_path)

        # Checkpoint criterion: Lowest validation loss with tiebreaker on ROC-AUC
        is_best = False
        if val_loss < best_val_loss - 1e-4:
            is_best = True
        elif abs(val_loss - best_val_loss) <= 1e-4 and val_auc > best_val_roc_auc:
            is_best = True

        if is_best:
            best_val_loss = val_loss
            best_val_roc_auc = val_auc
            best_epoch = epoch
            epochs_no_improve = 0

            checkpoint_payload["best_val_loss"] = best_val_loss
            checkpoint_payload["best_val_roc_auc"] = best_val_roc_auc
            torch.save(checkpoint_payload, best_checkpoint_path)
            print(f"  --> [*] New best model saved to: {best_checkpoint_path}", flush=True)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"  [!] Early stopping triggered at epoch {epoch} (no improvement in {patience} epochs).", flush=True)
                early_stopped = True
                break

    total_time = time.time() - start_time
    print(f"Completed {experiment_id} in {total_time/60:.2f} mins. Best Epoch: {best_epoch} (Val Loss: {best_val_loss:.4f}, Val AUC: {best_val_roc_auc:.3f})", flush=True)

    # Save training history files
    df_hist = pd.DataFrame(history)
    df_hist.to_csv(os.path.join(exp_dir, "training_history.csv"), index=False)
    with open(os.path.join(exp_dir, "training_history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    # Clean up GPU memory
    del model, optimizer, scheduler, scaler
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "experiment_id": experiment_id,
        "model_name": model_name,
        "modality": modality,
        "architecture": arch_name,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_roc_auc": best_val_roc_auc,
        "early_stopped": early_stopped,
        "total_time_mins": round(total_time / 60, 2),
        "checkpoint_path": best_checkpoint_path,
        "history": history,
    }


def train_single_experiment(
    experiment_id: str,
    model_name: str,
    modality: str,
    output_base_dir: str = "outputs/experiments",
    epochs: int = 12,
    batch_size: int = 8,
    gradient_accumulation_steps: int = 2,
    lr: float = 1e-4,
    weight_decay: float = 1e-2,
    patience: int = 4,
    seed: int = 42,
    use_amp: bool = True,
) -> Dict[str, Any]:
    """Executes independent training and checkpointing with automatic OOM fallback recovery."""
    curr_batch_size = batch_size
    curr_grad_accum = gradient_accumulation_steps

    while curr_batch_size >= 2:
        try:
            return _train_single_experiment_core(
                experiment_id=experiment_id,
                model_name=model_name,
                modality=modality,
                output_base_dir=output_base_dir,
                epochs=epochs,
                batch_size=curr_batch_size,
                gradient_accumulation_steps=curr_grad_accum,
                lr=lr,
                weight_decay=weight_decay,
                patience=patience,
                seed=seed,
                use_amp=use_amp,
            )
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if "out of memory" in str(e).lower() and curr_batch_size > 2:
                print(f"\n[CUDA OOM DETECTED in {experiment_id}]")
                print(f"Halving batch_size ({curr_batch_size} -> {curr_batch_size // 2}) and doubling grad_accum ({curr_grad_accum} -> {curr_grad_accum * 2}). Retrying...")
                curr_batch_size //= 2
                curr_grad_accum *= 2
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                import gc
                gc.collect()
            else:
                raise e

