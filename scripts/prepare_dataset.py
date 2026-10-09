import os
import sys
import json
import yaml
from pathlib import Path
from dataclasses import asdict
from collections import Counter
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.label_extractor import extract_patient_labels
from src.data.split import create_patient_stratified_split
from src.utils.logging import get_logger


def prepare_dataset(config_path: str = "configs/config.yaml"):
    logger = get_logger("PrepareDataset")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    dataset_dir = config["data"]["dataset_dir"]
    manifest_path = config["data"]["manifest_path"]
    splits_path = config["data"]["splits_path"]
    seed = config["data"].get("seed", 42)
    train_ratio = config["data"].get("train_split", 0.70)
    val_ratio = config["data"].get("val_split", 0.15)
    test_ratio = config["data"].get("test_split", 0.15)

    Path(manifest_path).parent.mkdir(parents=True, exist_ok=True)
    Path(splits_path).parent.mkdir(parents=True, exist_ok=True)

    # 1. Extract labels
    logger.info("Extracting ground-truth patient labels from PDF clinical reports...")
    patient_labels = extract_patient_labels(dataset_dir)

    # 2. Save manifest
    manifest_dict = {pid: asdict(pl) for pid, pl in patient_labels.items()}
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_dict, f, indent=2)
    logger.info(f"Dataset manifest saved to: {manifest_path}")

    # 3. Create patient-level stratified splits
    logger.info(f"Creating patient-level stratified splits (Train: {train_ratio}, Val: {val_ratio}, Test: {test_ratio})...")
    splits = create_patient_stratified_split(
        patient_labels=patient_labels,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
        save_path=splits_path
    )
    logger.info(f"Splits saved to: {splits_path}")

    # 4. Compute class distribution and class weights in training set
    train_pids = splits["train"]
    train_labels = [patient_labels[pid].label_id for pid in train_pids]
    label_counts = Counter(train_labels)
    n_total = len(train_labels)
    n_benign = label_counts[0]
    n_malignant = label_counts[1]

    w_benign = n_total / (2.0 * n_benign) if n_benign > 0 else 1.0
    w_malignant = n_total / (2.0 * n_malignant) if n_malignant > 0 else 1.0

    class_weights = {
        "class_counts": {
            "0_benign": n_benign,
            "1_malignant": n_malignant
        },
        "weights": [float(w_benign), float(w_malignant)]
    }

    weights_path = os.path.join(Path(splits_path).parent, "class_weights.json")
    with open(weights_path, "w", encoding="utf-8") as f:
        json.dump(class_weights, f, indent=2)
    logger.info(f"Computed training class weights: Benign={w_benign:.3f}, Malignant={w_malignant:.3f} (saved to {weights_path})")

    # Print summary
    print("\n" + "=" * 60)
    print("           PATIENT-LEVEL DATASET PREPARATION SUMMARY")
    print("=" * 60)
    print(f"Total Evaluated Patients:        {len(patient_labels)}")
    print(f"Train Cohort (Patients):         {len(splits['train'])} (Benign: {n_benign}, Malignant: {n_malignant})")
    
    val_labels = Counter([patient_labels[pid].label_id for pid in splits['val']])
    test_labels = Counter([patient_labels[pid].label_id for pid in splits['test']])
    print(f"Validation Cohort (Patients):    {len(splits['val'])} (Benign: {val_labels[0]}, Malignant: {val_labels[1]})")
    print(f"Test Cohort (Patients):          {len(splits['test'])} (Benign: {test_labels[0]}, Malignant: {test_labels[1]})")
    print(f"Excluded Unknown (Patients):     {len(splits['unknown'])}")
    print("-" * 60)
    print(f"Class Weights Applied in Loss:   [0: {w_benign:.3f}, 1: {w_malignant:.3f}]")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    prepare_dataset()
