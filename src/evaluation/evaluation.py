import os
import json
from pathlib import Path
from typing import Dict, Any, Optional
import torch
import torch.nn.functional as F
import numpy as np
from .metrics import compute_metrics
from .plots import plot_roc_curve, plot_pr_curve, plot_confusion_matrix


def evaluate_model(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: str = "cpu",
    output_dir: Optional[str] = None,
    model_name: str = "Model",
    threshold: float = 0.5
) -> Dict[str, Any]:
    """Evaluates a trained model on a test or validation dataset.
    
    Args:
        model: Trained PyTorch model
        data_loader: DataLoader yielding patient batches
        device: 'cuda' or 'cpu'
        output_dir: Directory to save plots, metrics.json, and predictions.json
        model_name: Name used in plot titles
        threshold: Decision threshold for binary classification (default: 0.5)
        
    Returns:
        Dict containing computed metrics, arrays of ground-truth, probabilities, and predictions.
    """
    model.eval()
    all_targets = []
    all_probs = []
    all_preds = []
    all_pids = []

    with torch.no_grad():
        for batch in data_loader:
            labels = batch["labels"].to(device)
            pids = batch["patient_ids"]
            batch_size = batch["batch_size"]
            
            # Identify model input format
            if hasattr(model, "gat") or (hasattr(model, "classifier") and "SimpleFusion" in model.__class__.__name__):
                # Multimodal or Fusion model
                m_imgs = batch["mammo_tensors"].to(device)
                m_idx = batch["mammo_batch_indices"].to(device)
                u_imgs = batch["us_tensors"].to(device)
                u_idx = batch["us_batch_indices"].to(device)
                
                if hasattr(model, "gat"):
                    logits, _ = model(m_imgs, m_idx, u_imgs, u_idx, batch_size)
                else:
                    logits = model(m_imgs, m_idx, u_imgs, u_idx, batch_size)
                    
            elif "MammogramDenseNet" in model.__class__.__name__:
                m_imgs = batch["mammo_tensors"].to(device)
                m_idx = batch["mammo_batch_indices"].to(device)
                logits, _ = model(m_imgs, m_idx, batch_size)
                
            elif "UltrasoundResNet" in model.__class__.__name__:
                u_imgs = batch["us_tensors"].to(device)
                u_idx = batch["us_batch_indices"].to(device)
                logits, _ = model(u_imgs, u_idx, batch_size)
            else:
                raise ValueError(f"Unrecognized model type: {model.__class__.__name__}")

            probs = F.softmax(logits, dim=1)[:, 1] # Probability of Malignant (class 1)
            preds = (probs >= threshold).long()

            all_targets.extend(labels.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())
            all_pids.extend(pids)

    y_true = np.array(all_targets)
    y_prob = np.array(all_probs)
    y_pred = np.array(all_preds)

    metrics = compute_metrics(y_true, y_prob, threshold=threshold)

    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Save metrics.json
        with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        # Save predictions.json
        preds_record = [
            {"patient_id": pid, "true_label": int(t), "pred_label": int(p), "prob_malignant": float(prob)}
            for pid, t, p, prob in zip(all_pids, y_true, y_pred, y_prob)
        ]
        with open(os.path.join(output_dir, "predictions.json"), "w", encoding="utf-8") as f:
            json.dump(preds_record, f, indent=2)

        # Generate plots
        plot_roc_curve(
            y_true=y_true,
            y_prob=y_prob,
            roc_auc=metrics["roc_auc"],
            save_path=os.path.join(output_dir, "roc_curve.png"),
            title=f"{model_name} - ROC Curve"
        )
        plot_pr_curve(
            y_true=y_true,
            y_prob=y_prob,
            pr_auc=metrics["pr_auc"],
            save_path=os.path.join(output_dir, "pr_curve.png"),
            title=f"{model_name} - PR Curve"
        )
        plot_confusion_matrix(
            cm_dict=metrics["confusion_matrix"],
            save_path=os.path.join(output_dir, "confusion_matrix.png"),
            title=f"{model_name} - Confusion Matrix"
        )

    return {
        "metrics": metrics,
        "y_true": y_true,
        "y_prob": y_prob,
        "y_pred": y_pred,
        "patient_ids": all_pids
    }

