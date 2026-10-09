"""Evaluation metrics, plotting utilities, and model evaluation pipeline."""
from .metrics import compute_metrics, find_youden_threshold, find_f1_threshold
from .plots import (
    plot_roc_curve, plot_pr_curve, plot_confusion_matrix,
    plot_training_history, plot_model_comparison
)
from .evaluation import evaluate_model

