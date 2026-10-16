"""Core Evaluator for Phase 5: Held-Out Test Evaluation.

Performs rigorous verification of checkpoints and splits, executes model inference
on the frozen 47-patient test cohort, computes comprehensive image-level and
patient-level metrics (using max-pooling aggregation), and validates all results.
"""

import os
import json
import math
import torch
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List, Optional
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

from ..models.factory import create_model
from ..data.dataset import get_dataloader


def verify_checkpoint_integrity(checkpoint_path: str) -> Dict[str, Any]:
    """Verifies that a saved model checkpoint exists and contains valid state dicts and metadata."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    # Load checkpoint safely on CPU for metadata inspection
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    required_keys = [
        "experiment_id",
        "model_name",
        "architecture",
        "pretrained_initialization",
        "modality",
        "epoch",
        "model_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "class_weights",
        "best_val_loss",
        "best_val_roc_auc",
        "config",
    ]

    missing_keys = [k for k in required_keys if k not in ckpt]
    if missing_keys:
        raise KeyError(f"Checkpoint at {checkpoint_path} is missing required keys: {missing_keys}")

    # Verify tensor integrity
    state_dict = ckpt["model_state_dict"]
    n_params = sum(p.numel() for p in state_dict.values())
    if n_params == 0:
        raise ValueError(f"Checkpoint at {checkpoint_path} contains 0 parameters!")

    return {
        "checkpoint_path": checkpoint_path,
        "experiment_id": ckpt["experiment_id"],
        "model_name": ckpt["model_name"],
        "architecture": ckpt["architecture"],
        "pretrained_initialization": ckpt["pretrained_initialization"],
        "modality": ckpt["modality"],
        "selected_epoch": ckpt["epoch"],
        "total_parameters": n_params,
        "best_val_loss": ckpt["best_val_loss"],
        "best_val_roc_auc": ckpt["best_val_roc_auc"],
        "config": ckpt["config"],
        "class_weights": ckpt["class_weights"],
    }


def verify_dataset_splits(
    splits_path: str = "data/splits/patient_stratified_split.json",
) -> Dict[str, Any]:
    """Verifies that the frozen patient splits exist, are mutually disjoint, and test set has 47 patients."""
    if not os.path.exists(splits_path):
        raise FileNotFoundError(f"Split file not found at: {splits_path}")

    with open(splits_path, "r", encoding="utf-8") as f:
        splits = json.load(f)

    folder_splits = splits.get("folder_patient_splits", {})
    train_pts = set(folder_splits.get("train", []))
    val_pts = set(folder_splits.get("val", []))
    test_pts = set(folder_splits.get("test", []))

    # Assert mutual exclusivity
    train_val_overlap = train_pts.intersection(val_pts)
    train_test_overlap = train_pts.intersection(test_pts)
    val_test_overlap = val_pts.intersection(test_pts)

    if train_val_overlap:
        raise ValueError(f"Contamination between train and val splits: {train_val_overlap}")
    if train_test_overlap:
        raise ValueError(f"Contamination between train and test splits: {train_test_overlap}")
    if val_test_overlap:
        raise ValueError(f"Contamination between val and test splits: {val_test_overlap}")

    if len(test_pts) != 47:
        raise ValueError(f"Expected 47 test patients, got {len(test_pts)}")

    return {
        "total_test_patients": len(test_pts),
        "total_val_patients": len(val_pts),
        "total_train_patients": len(train_pts),
        "train_test_overlap_count": 0,
        "val_test_overlap_count": 0,
        "verified_isolated": True,
    }


def compute_classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.50,
) -> Dict[str, Any]:
    """Calculates all binary classification metrics transparently.

    Positive class: 1 (SUSPICIOUS_ACTIONABLE)
    Negative class: 0 (LOW_RISK)
    Default classification threshold: 0.50
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    n_samples = len(y_true)
    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))

    # Confusion matrix: [[TN, FP], [FN, TP]]
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    # Accuracy
    acc = accuracy_score(y_true, y_pred)

    # Precision, Recall, Specificity, F1
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # Geometric Mean (G-Mean): sqrt(Sensitivity * Specificity)
    gmean = math.sqrt(rec * spec) if (rec * spec) > 0 else 0.0

    # ROC-AUC
    try:
        if len(np.unique(y_true)) > 1:
            roc_auc = float(roc_auc_score(y_true, y_prob))
        else:
            roc_auc = float("nan")
    except Exception:
        roc_auc = float("nan")

    # PR-AUC (Average Precision)
    try:
        if len(np.unique(y_true)) > 1:
            pr_auc = float(average_precision_score(y_true, y_prob))
        else:
            pr_auc = float("nan")
    except Exception:
        pr_auc = float("nan")

    # Compute ROC and PR curve coordinates
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_prob)
    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_true, y_prob)

    return {
        "n_samples": n_samples,
        "n_positive": n_pos,
        "n_negative": n_neg,
        "threshold": threshold,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "sensitivity": round(float(rec), 4),
        "specificity": round(float(spec), 4),
        "f1_score": round(float(f1), 4),
        "g_mean": round(float(gmean), 4),
        "roc_auc": round(float(roc_auc), 4) if not math.isnan(roc_auc) else None,
        "pr_auc": round(float(pr_auc), 4) if not math.isnan(pr_auc) else None,
        "confusion_matrix": cm.tolist(),
        "curve_data": {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "roc_thresholds": roc_thresholds.tolist(),
            "precision_curve": precision_curve.tolist(),
            "recall_curve": recall_curve.tolist(),
            "pr_thresholds": pr_thresholds.tolist(),
        },
    }


def evaluate_single_model(
    checkpoint_path: str,
    device: Optional[torch.device] = None,
    threshold: float = 0.50,
) -> Dict[str, Any]:
    """Loads a verified checkpoint and evaluates both image-level and patient-level test performance.

    Patient-level prediction uses the documented max-pooling aggregation rule:
        P_patient = max_{i in images(patient)} P_image(i)
    """
    # 1. Verify checkpoint integrity
    ckpt_meta = verify_checkpoint_integrity(checkpoint_path)
    model_name = ckpt_meta["model_name"]
    modality = ckpt_meta["modality"]
    exp_id = ckpt_meta["experiment_id"]

    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 2. Instantiate architecture and load state dict
    model = create_model(model_name=model_name, pretrained=False, num_classes=2, dropout_rate=0.3)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    # 3. Load frozen test dataloader
    test_loader = get_dataloader(
        modality=modality,
        split="test",
        batch_size=8,
        side_aware_labeling=True,
        shuffle=False,
        num_workers=0,
    )

    # 4. Inference pass
    image_records = []
    with torch.no_grad():
        for images, labels, metadatas in test_loader:
            images = images.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(images)
                probs = torch.softmax(logits, dim=1)[:, 1]

            probs_np = probs.cpu().numpy()
            labels_np = labels.cpu().numpy()

            batch_size = len(labels)
            for i in range(batch_size):
                p_id = metadatas["patient_id"][i]
                fn = metadatas["filename"][i]
                side = metadatas["breast_side"][i]
                y_i = int(labels_np[i])
                p_i = float(probs_np[i])

                image_records.append({
                    "patient_id": p_id,
                    "filename": fn,
                    "breast_side": side,
                    "modality": modality,
                    "true_label": y_i,
                    "pred_prob": p_i,
                    "pred_label": int(p_i >= threshold),
                })

    df_images = pd.DataFrame(image_records)

    # 5. Image-level metrics
    image_metrics = compute_classification_metrics(
        y_true=df_images["true_label"].values,
        y_prob=df_images["pred_prob"].values,
        threshold=threshold,
    )

    # 6. Patient-level max-pooling aggregation
    # Max probability across all images of the same patient for this modality
    patient_records = []
    for p_id, group in df_images.groupby("patient_id"):
        # Ground truth: 1 if any image has positive label, else 0
        p_true = int(group["true_label"].max())
        # Max-pooling rule:
        p_prob = float(group["pred_prob"].max())
        p_pred = int(p_prob >= threshold)
        n_imgs = len(group)

        patient_records.append({
            "patient_id": p_id,
            "modality": modality,
            "image_count": n_imgs,
            "true_label": p_true,
            "pred_prob": p_prob,
            "pred_label": p_pred,
        })

    df_patients = pd.DataFrame(patient_records)

    # Patient-level metrics
    patient_metrics = compute_classification_metrics(
        y_true=df_patients["true_label"].values,
        y_prob=df_patients["pred_prob"].values,
        threshold=threshold,
    )

    return {
        "experiment_id": exp_id,
        "model_name": model_name,
        "modality": modality,
        "architecture": ckpt_meta["architecture"],
        "selected_epoch": ckpt_meta["selected_epoch"],
        "checkpoint_path": checkpoint_path,
        "image_metrics": image_metrics,
        "patient_metrics": patient_metrics,
        "df_images": df_images,
        "df_patients": df_patients,
    }
