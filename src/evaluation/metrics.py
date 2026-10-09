import numpy as np
from typing import Dict, Any, List, Tuple
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, precision_recall_curve, roc_curve, auc, confusion_matrix
)


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]:
    """Computes comprehensive clinical evaluation metrics for binary classification.
    
    Args:
        y_true: Ground truth binary labels (0 or 1), shape [N]
        y_prob: Predicted probability for class 1 (Malignant), shape [N]
        threshold: Decision threshold for converting probabilities to binary predictions
        
    Returns:
        Dict of computed metrics including Sensitivity, Specificity, ROC-AUC, PR-AUC, etc.
    """
    y_prob = np.nan_to_num(y_prob, nan=0.5)
    y_prob = np.clip(y_prob, 0.0, 1.0)
    y_pred = (y_prob >= threshold).astype(int)

    # Confusion matrix elements: tn, fp, fn, tp
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    # Sensitivity (Recall) and Specificity
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    acc = accuracy_score(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # ROC-AUC (requires continuous malignant probabilities and both classes present in y_true)
    try:
        if len(np.unique(y_true)) > 1:
            roc_auc = float(roc_auc_score(y_true, y_prob))
        else:
            roc_auc = 0.5
    except Exception:
        roc_auc = 0.5

    # PR-AUC (requires continuous malignant probabilities)
    try:
        if len(np.unique(y_true)) > 1:
            precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_prob)
            pr_auc = float(auc(recall_vals, precision_vals))
        else:
            pr_auc = 0.0
    except Exception:
        pr_auc = 0.0

    return {
        "threshold": float(threshold),
        "accuracy": float(acc),
        "balanced_accuracy": float(bacc),
        "precision": float(prec),
        "recall": float(sensitivity),        # Sensitivity
        "sensitivity": float(sensitivity),   # Explicit alias
        "specificity": float(specificity),   # Specificity
        "f1": float(f1),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp)
        }
    }


def find_youden_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, float]:
    """Selects the classification threshold that maximizes Youden's J statistic on validation data.
    
    J = sensitivity + specificity - 1 = TPR - FPR
    
    Args:
        y_true: Validation ground truth binary labels
        y_prob: Validation continuous malignant probabilities
        
    Returns:
        (optimal_threshold, max_j_score)
    """
    y_prob = np.nan_to_num(y_prob, nan=0.5)
    y_prob = np.clip(y_prob, 0.0, 1.0)
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j_scores = tpr - fpr
    best_idx = int(np.argmax(j_scores))
    best_thresh = float(thresholds[best_idx])
    if best_thresh > 1.0:
        best_thresh = float(np.max(y_prob))
    return float(best_thresh), float(j_scores[best_idx])


def find_f1_threshold(y_true: np.ndarray, y_prob: np.ndarray, num_steps: int = 1000) -> Tuple[float, float]:
    """Selects the classification threshold that maximizes F1-score on validation data.
    
    Args:
        y_true: Validation ground truth binary labels
        y_prob: Validation continuous malignant probabilities
        num_steps: Grid resolution
        
    Returns:
        (optimal_threshold, max_f1_score)
    """
    y_prob = np.nan_to_num(y_prob, nan=0.5)
    y_prob = np.clip(y_prob, 0.0, 1.0)
    candidates = np.linspace(max(0.001, float(y_prob.min()) * 0.9), min(0.999, float(y_prob.max()) * 1.1), num_steps)
    candidates = np.unique(np.concatenate([candidates, y_prob]))
    candidates = np.sort(candidates)
    best_f1 = -1.0
    best_thresh = 0.5
    for t in candidates:
        pred = (y_prob >= t).astype(int)
        score = f1_score(y_true, pred, zero_division=0)
        if score > best_f1:
            best_f1 = float(score)
            best_thresh = float(t)
    return float(best_thresh), float(best_f1)

