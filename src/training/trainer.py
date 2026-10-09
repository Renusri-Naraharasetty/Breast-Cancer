import os
import time
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.logging import get_logger
from src.utils.checkpoint import save_checkpoint
from src.training.losses import get_loss_function
from src.training.scheduler import get_lr_scheduler
from src.evaluation.metrics import compute_metrics
from src.evaluation.plots import plot_training_history


class ModelTrainer:
    """Production-grade PyTorch trainer with mixed-precision, gradient accumulation,
    backbone progressive unfreezing, class imbalance compensation, and checkpointing.
    """
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict[str, Any],
        output_dir: str,
        class_weights: Optional[torch.Tensor] = None,
        model_name: str = "model"
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.output_dir = output_dir
        self.model_name = model_name
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        self.logger = get_logger(f"Trainer-{model_name}", log_file=os.path.join(output_dir, "train.log"))

        # Device setup
        dev_cfg = config.get("training", {}).get("device", "auto")
        if dev_cfg == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(dev_cfg)
            
        self.logger.info(f"Using device: {self.device} ({torch.cuda.get_device_name(0) if self.device.type == 'cuda' else 'CPU'})")
        self.model.to(self.device)

        # Training hyperparameters
        t_cfg = config.get("training", {})
        self.num_epochs = t_cfg.get("num_epochs", 15)
        self.lr = t_cfg.get("learning_rate", 1e-4)
        self.backbone_lr = t_cfg.get("backbone_lr", 1e-5)
        self.weight_decay = t_cfg.get("weight_decay", 1e-4)
        self.grad_accum_steps = max(1, t_cfg.get("grad_accum_steps", 1))
        self.early_stopping_patience = t_cfg.get("early_stopping_patience", 5)
        self.freeze_backbones = t_cfg.get("freeze_backbones", True)
        self.unfreeze_epoch = t_cfg.get("unfreeze_epoch", 5)
        self.use_amp = t_cfg.get("use_amp", True) and (self.device.type == "cuda")

        # Mixed precision scaler
        self.scaler = torch.amp.GradScaler('cuda') if self.use_amp else None
        if self.use_amp:
            self.logger.info("AMP (Automatic Mixed Precision) enabled for memory-efficient training.")

        # Loss function
        self.criterion = get_loss_function(class_weights=class_weights, device=self.device)

        # Optimizer setup with parameter groups
        if hasattr(self.model, "get_parameter_groups"):
            param_groups = self.model.get_parameter_groups(
                lr=self.lr,
                backbone_lr=self.backbone_lr,
                weight_decay=self.weight_decay
            )
            self.optimizer = torch.optim.AdamW(param_groups)
        else:
            self.optimizer = torch.optim.AdamW(
                [p for p in self.model.parameters() if p.requires_grad],
                lr=self.lr,
                weight_decay=self.weight_decay
            )

        # Scheduler setup
        self.scheduler = get_lr_scheduler(self.optimizer, scheduler_type="cosine", num_epochs=self.num_epochs)

        # Tracking state
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "val_accuracy": [],
            "val_f1": [],
            "val_roc_auc": []
        }
        self.best_metric = -1.0
        self.patience_counter = 0

    def _forward_batch(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Helper to run forward pass based on model architecture."""
        batch_size = batch["batch_size"]
        
        if hasattr(self.model, "gat") or "SimpleFusion" in self.model.__class__.__name__:
            m_imgs = batch["mammo_tensors"].to(self.device)
            m_idx = batch["mammo_batch_indices"].to(self.device)
            u_imgs = batch["us_tensors"].to(self.device)
            u_idx = batch["us_batch_indices"].to(self.device)
            
            if hasattr(self.model, "gat"):
                logits, _ = self.model(m_imgs, m_idx, u_imgs, u_idx, batch_size)
            else:
                logits = self.model(m_imgs, m_idx, u_imgs, u_idx, batch_size)
                
        elif "MammogramDenseNet" in self.model.__class__.__name__:
            m_imgs = batch["mammo_tensors"].to(self.device)
            m_idx = batch["mammo_batch_indices"].to(self.device)
            logits, _ = self.model(m_imgs, m_idx, batch_size)
            
        elif "UltrasoundResNet" in self.model.__class__.__name__:
            u_imgs = batch["us_tensors"].to(self.device)
            u_idx = batch["us_batch_indices"].to(self.device)
            logits, _ = self.model(u_imgs, u_idx, batch_size)
            
        else:
            raise ValueError(f"Unknown model class: {self.model.__class__.__name__}")

        return logits

    def train_epoch(self, epoch: int) -> float:
        """Executes one training epoch with gradient accumulation and AMP."""
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)
        
        # Check unfreezing schedule
        if self.freeze_backbones and epoch == self.unfreeze_epoch:
            self.logger.info(f"Epoch {epoch}: Unfreezing backbone layers for fine-tuning!")
            if hasattr(self.model, "unfreeze_backbones"):
                self.model.unfreeze_backbones()
            elif hasattr(self.model, "unfreeze_backbone"):
                self.model.unfreeze_backbone()

        self.optimizer.zero_grad()
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.num_epochs} [Train]", leave=False)

        for step, batch in enumerate(progress_bar):
            labels = batch["labels"].to(self.device)

            if self.use_amp:
                with torch.amp.autocast('cuda'):
                    logits = self._forward_batch(batch)
                    loss = self.criterion(logits, labels)
                    loss = loss / self.grad_accum_steps
                    
                self.scaler.scale(loss).backward()

                if (step + 1) % self.grad_accum_steps == 0 or (step + 1) == num_batches:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                logits = self._forward_batch(batch)
                loss = self.criterion(logits, labels)
                loss = loss / self.grad_accum_steps
                loss.backward()

                if (step + 1) % self.grad_accum_steps == 0 or (step + 1) == num_batches:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            total_loss += loss.item() * self.grad_accum_steps
            progress_bar.set_postfix({"loss": f"{loss.item() * self.grad_accum_steps:.4f}"})

        return total_loss / max(1, num_batches)

    def validate(self, epoch: int) -> Dict[str, Any]:
        """Runs validation and computes evaluation metrics."""
        self.model.eval()
        total_loss = 0.0
        all_targets = []
        all_probs = []

        with torch.no_grad():
            for batch in self.val_loader:
                labels = batch["labels"].to(self.device)
                
                if self.use_amp:
                    with torch.amp.autocast('cuda'):
                        logits = self._forward_batch(batch)
                        loss = self.criterion(logits, labels)
                else:
                    logits = self._forward_batch(batch)
                    loss = self.criterion(logits, labels)

                total_loss += loss.item()
                probs = torch.softmax(logits, dim=1)[:, 1] # Probability of malignant
                
                all_targets.extend(labels.cpu().numpy().tolist())
                all_probs.extend(probs.cpu().numpy().tolist())

        avg_loss = total_loss / max(1, len(self.val_loader))
        metrics = compute_metrics(
            y_true=torch.tensor(all_targets).numpy(),
            y_prob=torch.tensor(all_probs).numpy()
        )
        metrics["loss"] = avg_loss
        return metrics

    def train(self) -> Dict[str, Any]:
        """Runs complete training pipeline with early stopping and checkpointing."""
        self.logger.info(f"Starting training: {self.model_name} for {self.num_epochs} epochs.")
        start_time = time.time()

        for epoch in range(1, self.num_epochs + 1):
            train_loss = self.train_epoch(epoch)
            val_metrics = self.validate(epoch)

            # Update scheduler
            if self.scheduler:
                self.scheduler.step()

            # Record history
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["val_accuracy"].append(val_metrics["accuracy"])
            self.history["val_f1"].append(val_metrics["f1"])
            self.history["val_roc_auc"].append(val_metrics["roc_auc"])

            self.logger.info(
                f"Epoch {epoch:02d}/{self.num_epochs:02d} - "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Val Acc: {val_metrics['accuracy']:.4f} | "
                f"Val F1: {val_metrics['f1']:.4f} | "
                f"Val ROC-AUC: {val_metrics['roc_auc']:.4f}"
            )

            # Check if best model
            current_metric = val_metrics["roc_auc"]
            is_best = current_metric > self.best_metric

            if is_best:
                self.best_metric = current_metric
                self.patience_counter = 0
                save_checkpoint(
                    state={
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "val_metrics": val_metrics,
                        "config": self.config,
                        "is_smoke_test": self.config.get("training", {}).get("is_smoke_test", False)
                    },
                    checkpoint_dir=os.path.join(self.output_dir, "checkpoints"),
                    filename=f"checkpoint_epoch_{epoch}.pth",
                    is_best=True
                )
                self.logger.info(f"Saved new best model (ROC-AUC: {current_metric:.4f})")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.early_stopping_patience:
                    self.logger.info(f"Early stopping triggered after {epoch} epochs (no improvement for {self.early_stopping_patience} epochs).")
                    break

        elapsed = time.time() - start_time
        self.logger.info(f"Training completed in {elapsed/60:.2f} minutes.")

        # Save training history and plot curves
        with open(os.path.join(self.output_dir, "training_history.json"), "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)

        plot_training_history(
            history=self.history,
            save_path=os.path.join(self.output_dir, "training_curves.png"),
            title=f"{self.model_name} Training History"
        )

        return self.history
