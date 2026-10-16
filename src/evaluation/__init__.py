"""Evaluation and metrics module for Phase 5."""

from .evaluator import (
    verify_checkpoint_integrity,
    verify_dataset_splits,
    evaluate_single_model,
    compute_classification_metrics,
)

__all__ = [
    "verify_checkpoint_integrity",
    "verify_dataset_splits",
    "evaluate_single_model",
    "compute_classification_metrics",
]
