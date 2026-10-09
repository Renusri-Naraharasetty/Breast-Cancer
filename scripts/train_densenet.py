import os
import sys
import json
import yaml
from pathlib import Path
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.seed import set_seed
from src.utils.logging import get_logger
from src.data.dataset import BCMIDDataset, collate_patient_batch
from src.data.label_extractor import PatientLabel
from src.models.densenet_mammogram import MammogramDenseNet
from src.training.trainer import ModelTrainer
from src.evaluation.evaluation import evaluate_model


def train_densenet(config_path: str = "configs/config.yaml", quick_smoke_test: bool = False):
    logger = get_logger("Train-DenseNet")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    set_seed(config["training"].get("seed", 42))

    dataset_dir = config["data"]["dataset_dir"]
    manifest_path = config["data"]["manifest_path"]
    splits_path = config["data"]["splits_path"]

    # Load manifest and splits
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest_raw = json.load(f)
    patient_labels = {pid: PatientLabel(**data) for pid, data in manifest_raw.items()}

    with open(splits_path, "r", encoding="utf-8") as f:
        splits = json.load(f)

    train_pids = splits["train"]
    val_pids = splits["val"]
    test_pids = splits["test"]

    if quick_smoke_test:
        train_pids = train_pids[:4]
        val_pids = val_pids[:2]
        test_pids = test_pids[:2]
        config["training"]["num_epochs"] = 1
        config["training"]["is_smoke_test"] = True
        output_dir = os.path.join("outputs", "smoke_test", "densenet_mammogram")
    else:
        config["training"]["is_smoke_test"] = False
        output_dir = os.path.join(config["output"]["base_dir"], "densenet_mammogram")

    img_size = tuple(config["data"].get("image_size", [224, 224]))
    batch_size = config["training"].get("batch_size", 4)
    num_workers = config["data"].get("num_workers", 0)

    # Create Datasets and DataLoaders
    train_dataset = BCMIDDataset(
        dataset_dir=dataset_dir,
        patient_ids=train_pids,
        patient_labels=patient_labels,
        image_size=img_size,
        is_training=True,
        modality="mammogram_only"
    )
    val_dataset = BCMIDDataset(
        dataset_dir=dataset_dir,
        patient_ids=val_pids,
        patient_labels=patient_labels,
        image_size=img_size,
        is_training=False,
        modality="mammogram_only"
    )
    test_dataset = BCMIDDataset(
        dataset_dir=dataset_dir,
        patient_ids=test_pids,
        patient_labels=patient_labels,
        image_size=img_size,
        is_training=False,
        modality="mammogram_only"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_patient_batch
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_patient_batch
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_patient_batch
    )

    # Class weights
    weights_path = os.path.join(Path(splits_path).parent, "class_weights.json")
    class_weights = None
    if os.path.exists(weights_path) and config["training"].get("use_class_weights", True):
        with open(weights_path, "r", encoding="utf-8") as f:
            w_data = json.load(f)
            class_weights = torch.tensor(w_data["weights"], dtype=torch.float)

    # Initialize model
    m_cfg = config["model"]
    model = MammogramDenseNet(
        pretrained=m_cfg.get("densenet_pretrained", True),
        feature_dim=m_cfg.get("mammo_feature_dim", 1024),
        num_classes=m_cfg.get("num_classes", 2),
        dropout=config["training"].get("dropout", 0.3),
        freeze_backbone=config["training"].get("freeze_backbones", True)
    )

    trainer = ModelTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        output_dir=output_dir,
        class_weights=class_weights,
        model_name="DenseNet-121 (Mammogram)"
    )

    trainer.train()

    # Final test evaluation using best model
    best_model_path = os.path.join(output_dir, "checkpoints", "best_model.pth")
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path, map_location=trainer.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"Loaded best model checkpoint from epoch {checkpoint.get('epoch')}")

    test_results = evaluate_model(
        model=model,
        data_loader=test_loader,
        device=trainer.device,
        output_dir=os.path.join(output_dir, "test_evaluation"),
        model_name="DenseNet-121 Mammogram"
    )

    logger.info(f"Test Accuracy: {test_results['metrics']['accuracy']:.4f}, ROC-AUC: {test_results['metrics']['roc_auc']:.4f}")
    return test_results


if __name__ == "__main__":
    train_densenet()
