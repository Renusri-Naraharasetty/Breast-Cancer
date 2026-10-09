import os
import sys
import json
import yaml
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logging import get_logger
from src.data.dataset import BCMIDDataset, collate_patient_batch
from src.data.label_extractor import PatientLabel
from src.evaluation.metrics import compute_metrics, find_youden_threshold, find_f1_threshold
from src.evaluation.plots import (
    plot_roc_curve, plot_pr_curve, plot_confusion_matrix, plot_model_comparison
)
from src.models.densenet_mammogram import MammogramDenseNet
from src.models.resnet_ultrasound import UltrasoundResNet
from src.models.fusion_baseline import SimpleFusionBaseline
from src.models.multimodal_model import MultimodalGraphTransformer
from src.explainability.gradcam import visualize_multimodal_attention


MODEL_REGISTRY = {
    "densenet": {
        "name": "DenseNet-121 Mammogram",
        "dir_name": "densenet_mammogram",
        "modality": "mammogram_only",
    },
    "resnet": {
        "name": "ResNet-50 Ultrasound",
        "dir_name": "resnet_ultrasound",
        "modality": "ultrasound_only",
    },
    "fusion": {
        "name": "Simple Fusion Baseline",
        "dir_name": "fusion_baseline",
        "modality": "multimodal",
    },
    "graph_transformer": {
        "name": "Multimodal Graph Transformer",
        "dir_name": "graph_transformer",
        "modality": "multimodal",
    }
}


def build_model(model_type: str, config: Dict[str, Any]) -> torch.nn.Module:
    """Instantiates the exact model architecture specified in config."""
    m_cfg = config.get("model", {})
    t_cfg = config.get("training", {})
    dropout = t_cfg.get("dropout", 0.3)
    num_classes = m_cfg.get("num_classes", 2)
    mammo_dim = m_cfg.get("mammo_feature_dim", 1024)
    us_dim = m_cfg.get("us_feature_dim", 2048)
    hidden_dim = m_cfg.get("hidden_dim", 256)

    if model_type == "densenet":
        return MammogramDenseNet(
            pretrained=False,
            feature_dim=mammo_dim,
            num_classes=num_classes,
            dropout=dropout,
            freeze_backbone=False
        )
    elif model_type == "resnet":
        return UltrasoundResNet(
            pretrained=False,
            feature_dim=us_dim,
            num_classes=num_classes,
            dropout=dropout,
            freeze_backbone=False
        )
    elif model_type == "fusion":
        return SimpleFusionBaseline(
            pretrained=False,
            mammo_dim=mammo_dim,
            us_dim=us_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            dropout=dropout,
            freeze_backbones=False
        )
    elif model_type == "graph_transformer":
        return MultimodalGraphTransformer(
            pretrained=False,
            mammo_dim=mammo_dim,
            us_dim=us_dim,
            hidden_dim=hidden_dim,
            gat_heads=m_cfg.get("gat_heads", 4),
            gat_layers=m_cfg.get("gat_layers", 2),
            gat_dropout=m_cfg.get("gat_dropout", 0.2),
            transformer_heads=m_cfg.get("transformer_heads", 4),
            transformer_layers=m_cfg.get("transformer_layers", 2),
            transformer_dim_feedforward=m_cfg.get("transformer_dim_feedforward", 512),
            transformer_dropout=m_cfg.get("transformer_dropout", 0.2),
            num_classes=num_classes,
            dropout=dropout,
            freeze_backbones=False
        )
    else:
        raise ValueError(f"Unknown model type '{model_type}'. Supported: {list(MODEL_REGISTRY.keys())}")


def run_inference_on_loader(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Runs forward pass over a DataLoader and extracts true labels, continuous malignant probabilities, and patient IDs."""
    model.eval()
    all_targets = []
    all_probs = []
    all_pids = []

    with torch.no_grad():
        for batch in data_loader:
            labels = batch["labels"].to(device)
            pids = batch["patient_ids"]
            batch_size = batch["batch_size"]

            if hasattr(model, "gat") or (hasattr(model, "classifier") and "SimpleFusion" in model.__class__.__name__):
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

            all_targets.extend(labels.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())
            all_pids.extend(pids)

    return np.array(all_targets), np.array(all_probs), all_pids


def evaluate_single_model(
    model_type: str,
    config_path: str = "configs/config.yaml",
    checkpoint_path: Optional[str] = None,
    allow_smoke_checkpoint: bool = False,
    base_dir: Optional[str] = None
) -> Dict[str, Any]:
    """Performs rigorous evaluation of a trained model checkpoint.
    
    1. Loads checkpoint weights and validates metadata.
    2. Runs inference on the validation cohort to obtain validation-set probabilities.
    3. Calibrates the classification threshold using Youden's J statistic strictly on validation data.
    4. Runs inference on the test cohort.
    5. Evaluates test performance at both Default (0.5) and Calibrated operating points.
    6. Writes fresh predictions, metrics, validation diagnostics, and clinical plots.
    """
    logger = get_logger("Model-Evaluation")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{model_type}'. Choose from: {list(MODEL_REGISTRY.keys())}")

    info = MODEL_REGISTRY[model_type]
    if base_dir is None:
        base_dir = config["output"]["base_dir"]

    # 1. Resolve checkpoint path
    if checkpoint_path is None:
        checkpoint_path = os.path.join(base_dir, info["dir_name"], "checkpoints", "best_model.pth")

    # 2. Checkpoint existence validation
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"No trained checkpoint found at: '{checkpoint_path}'.\n"
            f"Run full training first: python main.py --action train --model {model_type}"
        )

    # 3. Detect device & load checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() and config.get("training", {}).get("device", "auto") != "cpu" else "cpu")
    logger.info(f"Loading checkpoint for {info['name']} on {device}: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # 4. Checkpoint validation & metadata inspection
    epoch = checkpoint.get("epoch", "Unknown")
    ckpt_val_metrics = checkpoint.get("val_metrics", {})
    ckpt_val_roc_auc = ckpt_val_metrics.get("roc_auc", 0.0)
    ckpt_config = checkpoint.get("config", {})
    num_epochs_trained = ckpt_config.get("training", {}).get("num_epochs", None)

    # Inspect for smoke-test indicators
    cm = ckpt_val_metrics.get("confusion_matrix", {})
    val_samples = cm.get("tn", 0) + cm.get("fp", 0) + cm.get("fn", 0) + cm.get("tp", 0)
    is_smoke = (
        checkpoint.get("is_smoke_test", False) or
        ckpt_config.get("training", {}).get("is_smoke_test", False) or
        (num_epochs_trained == 1) or
        (epoch == 1 and val_samples <= 2)
    )

    if is_smoke:
        print(f"\n>>> [WARNING] Checkpoint at '{checkpoint_path}' was produced by a 1-epoch SMOKE TEST "
              f"(epoch={epoch}, num_epochs={num_epochs_trained}, val_samples={val_samples})! <<<")
        if not allow_smoke_checkpoint:
            raise ValueError(
                f"Refusing to evaluate smoke-test checkpoint '{checkpoint_path}' as a real experiment checkpoint.\n"
                f"Run full training first: python main.py --action train --model {model_type}\n"
                f"(If you explicitly intend to evaluate this smoke-test checkpoint, pass --allow-smoke-checkpoint)"
            )

    # 5. Load dataset manifest & splits
    manifest_path = config["data"]["manifest_path"]
    splits_path = config["data"]["splits_path"]

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest_raw = json.load(f)
    patient_labels = {pid: PatientLabel(**data) for pid, data in manifest_raw.items()}

    with open(splits_path, "r", encoding="utf-8") as f:
        splits = json.load(f)

    val_pids = splits.get("val", [])
    test_pids = splits.get("test", [])
    if not val_pids:
        raise ValueError(f"No validation patients found in splits file: {splits_path}")
    if not test_pids:
        raise ValueError(f"No test patients found in splits file: {splits_path}")

    # 6. Construct Model and Load Trained Weights
    model = build_model(model_type, config)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    img_size = tuple(config["data"].get("image_size", [224, 224]))
    batch_size = config["training"].get("batch_size", 4)
    num_workers = config["data"].get("num_workers", 0)

    # 7. Run Validation-Set Inference & Threshold Calibration
    logger.info(f"Running inference on validation cohort ({len(val_pids)} patients) for threshold calibration...")
    val_dataset = BCMIDDataset(
        dataset_dir=config["data"]["dataset_dir"],
        patient_ids=val_pids,
        patient_labels=patient_labels,
        image_size=img_size,
        is_training=False,
        modality=info["modality"]
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_patient_batch
    )

    y_true_val, y_prob_val, val_pids_out = run_inference_on_loader(model, val_loader, device)

    # Calibrate decision thresholds strictly on validation data
    val_youden_threshold, max_youden_j = find_youden_threshold(y_true_val, y_prob_val)
    val_f1_threshold, max_val_f1 = find_f1_threshold(y_true_val, y_prob_val)
    val_metrics = compute_metrics(y_true_val, y_prob_val, threshold=val_youden_threshold)
    val_roc_auc = val_metrics["roc_auc"]

    # 8. Print Validation Calibration Info
    print("\n" + "=" * 65)
    print("CHECKPOINT VALIDATION & THRESHOLD CALIBRATION")
    print("=" * 65)
    print(f"Model:                         {info['name']}")
    print(f"Checkpoint:                    {checkpoint_path}")
    print(f"Epoch:                         {epoch}")
    print(f"Validation Patients:           {len(val_pids)}")
    print(f"Validation ROC-AUC:            {val_roc_auc:.4f}")
    print(f"Selected Youden Threshold:     {val_youden_threshold:.4f} (Max Youden J = {max_youden_j:.4f})")
    print(f"Reference F1-Optimal Threshold:{val_f1_threshold:.4f} (Max Val F1 = {max_val_f1:.4f})")
    print(f"Number of Test Patients:       {len(test_pids)}")
    print("=" * 65 + "\n")

    # 9. Run Test-Set Inference
    logger.info(f"Running inference on test cohort ({len(test_pids)} patients)...")
    test_dataset = BCMIDDataset(
        dataset_dir=config["data"]["dataset_dir"],
        patient_ids=test_pids,
        patient_labels=patient_labels,
        image_size=img_size,
        is_training=False,
        modality=info["modality"]
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_patient_batch
    )

    y_true_test, y_prob_test, test_pids_out = run_inference_on_loader(model, test_loader, device)

    # 10. Compute Metrics at Both Operating Points
    default_metrics = compute_metrics(y_true_test, y_prob_test, threshold=0.5)
    calibrated_metrics = compute_metrics(y_true_test, y_prob_test, threshold=val_youden_threshold)

    output_eval_dir = os.path.join(base_dir, info["dir_name"], "test_evaluation")
    Path(output_eval_dir).mkdir(parents=True, exist_ok=True)

    # 11. Save Validation Diagnostics
    val_diag_record = {
        "model_name": info["name"],
        "checkpoint_epoch": epoch,
        "validation_patient_ids": val_pids_out,
        "validation_labels": [int(y) for y in y_true_val],
        "validation_probabilities": [float(p) for p in y_prob_val],
        "validation_roc_auc": float(val_roc_auc),
        "selected_threshold": float(val_youden_threshold),
        "threshold_method": "youden_j",
        "max_youden_j": float(max_youden_j),
        "reference_f1_threshold": float(val_f1_threshold),
        "max_validation_f1": float(max_val_f1)
    }
    with open(os.path.join(output_eval_dir, "validation_diagnostics.json"), "w", encoding="utf-8") as f:
        json.dump(val_diag_record, f, indent=2)

    # 12. Save Detailed metrics.json
    metrics_record = {
        "default_threshold": 0.5,
        "validation_threshold_method": "youden_j",
        "validation_selected_threshold": float(val_youden_threshold),
        "validation_f1_threshold": float(val_f1_threshold),
        "validation_roc_auc": float(val_roc_auc),
        "default_metrics": default_metrics,
        "calibrated_metrics": calibrated_metrics,
        # Top-level aliases for backward compatibility
        "accuracy": calibrated_metrics["accuracy"],
        "balanced_accuracy": calibrated_metrics["balanced_accuracy"],
        "precision": calibrated_metrics["precision"],
        "recall": calibrated_metrics["recall"],
        "sensitivity": calibrated_metrics["sensitivity"],
        "specificity": calibrated_metrics["specificity"],
        "f1": calibrated_metrics["f1"],
        "roc_auc": calibrated_metrics["roc_auc"],
        "pr_auc": calibrated_metrics["pr_auc"],
        "confusion_matrix": calibrated_metrics["confusion_matrix"]
    }
    with open(os.path.join(output_eval_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics_record, f, indent=2)

    # 13. Save predictions.json
    preds_record = [
        {
            "patient_id": pid,
            "true_label": int(t),
            "pred_label": int(prob >= val_youden_threshold),
            "pred_label_default_05": int(prob >= 0.5),
            "prob_malignant": float(prob)
        }
        for pid, t, prob in zip(test_pids_out, y_true_test, y_prob_test)
    ]
    with open(os.path.join(output_eval_dir, "predictions.json"), "w", encoding="utf-8") as f:
        json.dump(preds_record, f, indent=2)

    # 14. Generate Plots
    plot_roc_curve(
        y_true=y_true_test,
        y_prob=y_prob_test,
        roc_auc=calibrated_metrics["roc_auc"],
        save_path=os.path.join(output_eval_dir, "roc_curve.png"),
        title=f"{info['name']} - ROC Curve"
    )
    plot_pr_curve(
        y_true=y_true_test,
        y_prob=y_prob_test,
        pr_auc=calibrated_metrics["pr_auc"],
        save_path=os.path.join(output_eval_dir, "pr_curve.png"),
        title=f"{info['name']} - PR Curve"
    )
    plot_confusion_matrix(
        cm_dict=calibrated_metrics["confusion_matrix"],
        save_path=os.path.join(output_eval_dir, "confusion_matrix.png"),
        title=f"{info['name']} - Confusion Matrix (Calibrated Threshold: {val_youden_threshold:.4f})"
    )

    # 15. Generate multimodal attention visualization for Graph Transformer
    if model_type == "graph_transformer":
        try:
            with torch.no_grad():
                sample_batch = next(iter(test_loader))
                m_imgs = sample_batch["mammo_tensors"].to(device)
                m_idx = sample_batch["mammo_batch_indices"].to(device)
                u_imgs = sample_batch["us_tensors"].to(device)
                u_idx = sample_batch["us_batch_indices"].to(device)
                bs = sample_batch["batch_size"]
                pid = sample_batch["patient_ids"][0]

                _, explanations = model(m_imgs, m_idx, u_imgs, u_idx, bs, return_explanations=True)
                if explanations:
                    vis_path = os.path.join(output_eval_dir, "attention_visualization.png")
                    visualize_multimodal_attention(
                        gat_attention=explanations["gat_attention"],
                        transformer_attention=explanations["transformer_attention"],
                        save_path=vis_path,
                        patient_id=pid
                    )
                    logger.info(f"Saved multimodal attention visualization to: {vis_path}")
        except Exception as e:
            logger.warning(f"Could not generate attention visualization: {e}")

    # 16. Print Comprehensive Clinical Summary Reporting Both Operating Points
    print("-" * 65)
    print(f"EVALUATION RESULTS: {info['name']}")
    print("-" * 65)
    print(f"Validation ROC-AUC:            {val_roc_auc:.4f}")
    print(f"Selected Validation Threshold: {val_youden_threshold:.4f} (Youden's J)")
    print(f"Test ROC-AUC:                  {calibrated_metrics['roc_auc']:.4f}")
    print(f"Test PR-AUC:                   {calibrated_metrics['pr_auc']:.4f}")
    print("\n  [Operating Point 1: Default Threshold = 0.5000]")
    print(f"  Accuracy:          {default_metrics['accuracy']:.4f}")
    print(f"  Balanced Accuracy: {default_metrics['balanced_accuracy']:.4f}")
    print(f"  Precision:         {default_metrics['precision']:.4f}")
    print(f"  Recall/Sensitivity:{default_metrics['recall']:.4f}")
    print(f"  Specificity:       {default_metrics['specificity']:.4f}")
    print(f"  F1-Score:          {default_metrics['f1']:.4f}")
    cm_def = default_metrics['confusion_matrix']
    print(f"  Confusion Matrix:  TN={cm_def['tn']}, FP={cm_def['fp']}, FN={cm_def['fn']}, TP={cm_def['tp']}")
    print(f"\n  [Operating Point 2: Calibrated Youden Threshold = {val_youden_threshold:.4f}]")
    print(f"  Accuracy:          {calibrated_metrics['accuracy']:.4f}")
    print(f"  Balanced Accuracy: {calibrated_metrics['balanced_accuracy']:.4f}")
    print(f"  Precision:         {calibrated_metrics['precision']:.4f}")
    print(f"  Recall/Sensitivity:{calibrated_metrics['recall']:.4f}")
    print(f"  Specificity:       {calibrated_metrics['specificity']:.4f}")
    print(f"  F1-Score:          {calibrated_metrics['f1']:.4f}")
    cm_cal = calibrated_metrics['confusion_matrix']
    print(f"  Confusion Matrix:  TN={cm_cal['tn']}, FP={cm_cal['fp']}, FN={cm_cal['fn']}, TP={cm_cal['tp']}")
    print("-" * 65 + "\n")

    return {
        "metrics": metrics_record,
        "default_metrics": default_metrics,
        "calibrated_metrics": calibrated_metrics,
        "y_true": y_true_test,
        "y_prob": y_prob_test,
        "patient_ids": test_pids_out,
        "selected_threshold": val_youden_threshold
    }


def evaluate_all_models(
    config_path: str = "configs/config.yaml",
    allow_smoke_checkpoint: bool = False,
    base_dir: Optional[str] = None
) -> Dict[str, Any]:
    """Evaluates checkpoints for all four models in sequence, then generates comparative reports."""
    all_results = {}
    for model_type in ["densenet", "resnet", "fusion", "graph_transformer"]:
        res = evaluate_single_model(
            model_type=model_type,
            config_path=config_path,
            allow_smoke_checkpoint=allow_smoke_checkpoint,
            base_dir=base_dir
        )
        all_results[model_type] = res

    # Generate combined comparative report and plots
    generate_overall_comparison(config_path=config_path, base_dir=base_dir)
    return all_results


def generate_overall_comparison(
    config_path: str = "configs/config.yaml",
    base_dir: Optional[str] = None
):
    """Aggregates test evaluation metrics across all four models, prints comparison table,
    and generates combined ROC curves and comparison bar chart.
    """
    logger = get_logger("Model-Comparison")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    splits_path = config["data"]["splits_path"]
    if base_dir is None:
        base_dir = config["output"]["base_dir"]

    # Load splits metadata
    total_patients = 0
    n_train = 0
    n_val = 0
    n_test = 0
    if os.path.exists(splits_path):
        with open(splits_path, "r", encoding="utf-8") as f:
            splits = json.load(f)
            n_train = len(splits.get("train", []))
            n_val = len(splits.get("val", []))
            n_test = len(splits.get("test", []))
            total_patients = n_train + n_val + n_test + len(splits.get("unknown", []))

    models_info = [
        ("DenseNet-121 Mammogram", os.path.join(base_dir, "densenet_mammogram", "test_evaluation")),
        ("ResNet-50 Ultrasound", os.path.join(base_dir, "resnet_ultrasound", "test_evaluation")),
        ("Simple Fusion Baseline", os.path.join(base_dir, "fusion_baseline", "test_evaluation")),
        ("Multimodal Graph Transformer", os.path.join(base_dir, "graph_transformer", "test_evaluation"))
    ]

    results_data = {}
    curves_data = {}

    for name, eval_dir in models_info:
        metrics_file = os.path.join(eval_dir, "metrics.json")
        preds_file = os.path.join(eval_dir, "predictions.json")

        if os.path.exists(metrics_file):
            with open(metrics_file, "r", encoding="utf-8") as f:
                results_data[name] = json.load(f)

        if os.path.exists(preds_file):
            with open(preds_file, "r", encoding="utf-8") as f:
                preds = json.load(f)
                y_true = np.array([p["true_label"] for p in preds])
                y_prob = np.array([p["prob_malignant"] for p in preds])
                curves_data[name] = (y_true, y_prob)

    # Print formatted clinical report
    print("\n" + "=" * 80)
    print("BREAST CANCER MULTIMODAL CLASSIFICATION: EVALUATION COMPARISON")
    print("=" * 80)
    print(f"Dataset: BCMID | Total Patients: {total_patients} | Train: {n_train} | Val: {n_val} | Test: {n_test}\n")

    print("-" * 110)
    print("OPERATING POINT A: DEFAULT THRESHOLD = 0.5000")
    print("-" * 110)
    print(f"{'Model':<30} | {'Accuracy':<8} | {'Sens/Rec':<8} | {'Spec':<8} | {'F1':<6} | {'ROC-AUC':<7} | {'PR-AUC':<6} | {'CM (TN/FP/FN/TP)':<18}")
    print("-" * 110)
    for name, _ in models_info:
        rec = results_data.get(name, {})
        def_m = rec.get("default_metrics", rec)
        cm = def_m.get("confusion_matrix", {})
        cm_str = f"{cm.get('tn', 0)}/{cm.get('fp', 0)}/{cm.get('fn', 0)}/{cm.get('tp', 0)}"
        print(
            f"{name:<30} | "
            f"{def_m.get('accuracy', 0.0):<8.4f} | "
            f"{def_m.get('recall', def_m.get('sensitivity', 0.0)):<8.4f} | "
            f"{def_m.get('specificity', 0.0):<8.4f} | "
            f"{def_m.get('f1', 0.0):<6.4f} | "
            f"{def_m.get('roc_auc', 0.0):<7.4f} | "
            f"{def_m.get('pr_auc', 0.0):<6.4f} | "
            f"{cm_str:<18}"
        )
    print("-" * 110)

    print("\n" + "=" * 110)
    print("OPERATING POINT B: VALIDATION-CALIBRATED YOUDEN THRESHOLD")
    print("=" * 110)
    print(f"{'Model':<30} | {'Thresh':<6} | {'Accuracy':<8} | {'Sens/Rec':<8} | {'Spec':<8} | {'F1':<6} | {'ROC-AUC':<7} | {'PR-AUC':<6} | {'CM (TN/FP/FN/TP)':<18}")
    print("-" * 110)
    for name, _ in models_info:
        rec = results_data.get(name, {})
        cal_m = rec.get("calibrated_metrics", rec)
        thresh = rec.get("validation_selected_threshold", 0.5)
        cm = cal_m.get("confusion_matrix", {})
        cm_str = f"{cm.get('tn', 0)}/{cm.get('fp', 0)}/{cm.get('fn', 0)}/{cm.get('tp', 0)}"
        print(
            f"{name:<30} | "
            f"{thresh:<6.4f} | "
            f"{cal_m.get('accuracy', 0.0):<8.4f} | "
            f"{cal_m.get('recall', cal_m.get('sensitivity', 0.0)):<8.4f} | "
            f"{cal_m.get('specificity', 0.0):<8.4f} | "
            f"{cal_m.get('f1', 0.0):<6.4f} | "
            f"{cal_m.get('roc_auc', 0.0):<7.4f} | "
            f"{cal_m.get('pr_auc', 0.0):<6.4f} | "
            f"{cm_str:<18}"
        )
    print("=" * 110 + "\n")

    # Generate combined comparison plots
    comparison_dir = os.path.join(base_dir, "comparison")
    Path(comparison_dir).mkdir(parents=True, exist_ok=True)

    if results_data:
        plot_model_comparison(
            comparison_data=results_data,
            save_path=os.path.join(comparison_dir, "model_comparison_bar.png")
        )

        # Combined ROC Plot
        plt.figure(figsize=(8, 7))
        colors = {
            "DenseNet-121 Mammogram": "#2563EB",
            "ResNet-50 Ultrasound": "#DC2626",
            "Simple Fusion Baseline": "#D97706",
            "Multimodal Graph Transformer": "#059669"
        }

        for name, (yt, yp) in curves_data.items():
            if len(np.unique(yt)) > 1:
                fpr, tpr, _ = roc_curve(yt, yp)
                auc_score = results_data.get(name, {}).get("roc_auc", 0.5)
                lw = 2.5 if "Graph" in name else 1.8
                plt.plot(fpr, tpr, label=f"{name} (AUC = {auc_score:.4f})", color=colors.get(name, "gray"), lw=lw)

        plt.plot([0, 1], [0, 1], 'k--', lw=1.2, label="Chance")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=12)
        plt.ylabel("True Positive Rate (Sensitivity)", fontsize=12)
        plt.title("Comparative ROC Curves (BCMID Test Cohort)", fontsize=14, fontweight="bold")
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(comparison_dir, "combined_roc_curves.png"), dpi=300)
        plt.close()

        # Save comparison json
        with open(os.path.join(comparison_dir, "model_comparison.json"), "w", encoding="utf-8") as f:
            json.dump(results_data, f, indent=2)

    logger.info(f"Comparison report and artifacts generated in: {comparison_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BCMID Model Evaluation")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config YAML")
    parser.add_argument("--model", type=str, default="all", choices=["densenet", "resnet", "fusion", "graph_transformer", "all"])
    parser.add_argument("--checkpoint", type=str, default=None, help="Custom checkpoint path to evaluate")
    parser.add_argument("--allow-smoke-checkpoint", action="store_true", help="Allow evaluating 1-epoch smoke-test checkpoints")
    parser.add_argument("--base-dir", type=str, default=None, help="Base directory containing model runs")
    args = parser.parse_args()

    if args.model == "all":
        evaluate_all_models(
            config_path=args.config,
            allow_smoke_checkpoint=args.allow_smoke_checkpoint,
            base_dir=args.base_dir
        )
    else:
        evaluate_single_model(
            model_type=args.model,
            config_path=args.config,
            checkpoint_path=args.checkpoint,
            allow_smoke_checkpoint=args.allow_smoke_checkpoint,
            base_dir=args.base_dir
        )
