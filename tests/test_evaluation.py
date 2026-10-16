"""Unit tests for Phase 5 evaluation pipeline."""

import pytest
import numpy as np
from src.evaluation.evaluator import (
    compute_classification_metrics,
    verify_checkpoint_integrity,
    verify_dataset_splits,
)


def test_metric_computation_standard():
    y_true = np.array([0, 0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.8, 0.9, 0.4])  # at 0.5: pred = [0, 0, 1, 1, 0] -> TN=2, FP=1, FN=1, TP=1

    metrics = compute_classification_metrics(y_true, y_prob, threshold=0.5)
    assert metrics["n_samples"] == 5
    assert metrics["tn"] == 2
    assert metrics["fp"] == 1
    assert metrics["fn"] == 1
    assert metrics["tp"] == 1
    assert metrics["accuracy"] == 0.6
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["specificity"] == 2.0 / 3.0 or round(metrics["specificity"], 4) == 0.6667
    assert metrics["f1_score"] == 0.5


def test_metric_computation_edge_cases():
    # All negative, model predicts all negative
    y_true = np.array([0, 0, 0, 0])
    y_prob = np.array([0.1, 0.2, 0.3, 0.4])

    metrics = compute_classification_metrics(y_true, y_prob, threshold=0.5)
    assert metrics["tp"] == 0
    assert metrics["fp"] == 0
    assert metrics["fn"] == 0
    assert metrics["tn"] == 4
    assert metrics["precision"] == 0.0  # zero division handled
    assert metrics["recall"] == 0.0     # zero division handled
    assert metrics["f1_score"] == 0.0
    assert metrics["specificity"] == 1.0


def test_splits_verification():
    res = verify_dataset_splits()
    assert res["total_test_patients"] == 47
    assert res["train_test_overlap_count"] == 0
    assert res["val_test_overlap_count"] == 0
    assert res["verified_isolated"] is True


def test_checkpoints_verification():
    for exp in ["exp1_efficientnet_mammogram", "exp2_vit_ultrasound", "exp3_vit_mammogram", "exp4_efficientnet_ultrasound"]:
        ckpt_path = f"outputs/experiments/{exp}/best_checkpoint.pth"
        meta = verify_checkpoint_integrity(ckpt_path)
        assert meta["selected_epoch"] >= 1
        assert meta["total_parameters"] > 0
        assert meta["best_val_loss"] is not None
