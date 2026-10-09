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
from src.models.multimodal_model import MultimodalGraphTransformer
from src.training.trainer import ModelTrainer
from src.evaluation.evaluation import evaluate_model
from src.explainability.gradcam import visualize_multimodal_attention


def train_graph_transformer(config_path: str = "configs/config.yaml", quick_smoke_test: bool = False):
    logger = get_logger("Train-GraphTransformer")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    set_seed(config["training"].get("seed", 42))

    dataset_dir = config["data"]["dataset_dir"]
    manifest_path = config["data"]["manifest_path"]
    splits_path = config["data"]["splits_path"]

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
        output_dir = os.path.join("outputs", "smoke_test", "graph_transformer")
    else:
        config["training"]["is_smoke_test"] = False
        output_dir = os.path.join(config["output"]["base_dir"], "graph_transformer")

    img_size = tuple(config["data"].get("image_size", [224, 224]))
    batch_size = config["training"].get("batch_size", 4)
    num_workers = config["data"].get("num_workers", 0)

    train_dataset = BCMIDDataset(
        dataset_dir=dataset_dir,
        patient_ids=train_pids,
        patient_labels=patient_labels,
        image_size=img_size,
        is_training=True,
        modality="multimodal"
    )
    val_dataset = BCMIDDataset(
        dataset_dir=dataset_dir,
        patient_ids=val_pids,
        patient_labels=patient_labels,
        image_size=img_size,
        is_training=False,
        modality="multimodal"
    )
    test_dataset = BCMIDDataset(
        dataset_dir=dataset_dir,
        patient_ids=test_pids,
        patient_labels=patient_labels,
        image_size=img_size,
        is_training=False,
        modality="multimodal"
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

    weights_path = os.path.join(Path(splits_path).parent, "class_weights.json")
    class_weights = None
    if os.path.exists(weights_path) and config["training"].get("use_class_weights", True):
        with open(weights_path, "r", encoding="utf-8") as f:
            w_data = json.load(f)
            class_weights = torch.tensor(w_data["weights"], dtype=torch.float)

    m_cfg = config["model"]
    model = MultimodalGraphTransformer(
        pretrained=True,
        mammo_dim=m_cfg.get("mammo_feature_dim", 1024),
        us_dim=m_cfg.get("us_feature_dim", 2048),
        hidden_dim=m_cfg.get("hidden_dim", 256),
        gat_heads=m_cfg.get("gat_heads", 4),
        gat_layers=m_cfg.get("gat_layers", 2),
        gat_dropout=m_cfg.get("gat_dropout", 0.2),
        transformer_heads=m_cfg.get("transformer_heads", 4),
        transformer_layers=m_cfg.get("transformer_layers", 2),
        transformer_dim_feedforward=m_cfg.get("transformer_dim_feedforward", 512),
        transformer_dropout=m_cfg.get("transformer_dropout", 0.2),
        num_classes=m_cfg.get("num_classes", 2),
        dropout=config["training"].get("dropout", 0.3),
        freeze_backbones=config["training"].get("freeze_backbones", True)
    )

    trainer = ModelTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        output_dir=output_dir,
        class_weights=class_weights,
        model_name="Multimodal Graph-Infused Transformer"
    )

    trainer.train()

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
        model_name="Multimodal Graph-Infused Transformer"
    )

    # Generate multimodal attention visualization on a sample test patient
    try:
        model.eval()
        with torch.no_grad():
            sample_batch = next(iter(test_loader))
            m_imgs = sample_batch["mammo_tensors"].to(trainer.device)
            m_idx = sample_batch["mammo_batch_indices"].to(trainer.device)
            u_imgs = sample_batch["us_tensors"].to(trainer.device)
            u_idx = sample_batch["us_batch_indices"].to(trainer.device)
            bs = sample_batch["batch_size"]
            pid = sample_batch["patient_ids"][0]

            _, explanations = model(m_imgs, m_idx, u_imgs, u_idx, bs, return_explanations=True)
            if explanations:
                vis_path = os.path.join(output_dir, "test_evaluation", "attention_visualization.png")
                visualize_multimodal_attention(
                    gat_attention=explanations["gat_attention"],
                    transformer_attention=explanations["transformer_attention"],
                    save_path=vis_path,
                    patient_id=pid
                )
                logger.info(f"Saved multimodal attention visualization to: {vis_path}")
    except Exception as e:
        logger.warning(f"Could not generate attention visualization: {e}")

    logger.info(f"Test Accuracy: {test_results['metrics']['accuracy']:.4f}, ROC-AUC: {test_results['metrics']['roc_auc']:.4f}")
    return test_results


if __name__ == "__main__":
    train_graph_transformer()
