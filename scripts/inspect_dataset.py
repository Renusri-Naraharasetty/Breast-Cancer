import os
import sys
import json
import yaml
from pathlib import Path
from collections import Counter
import numpy as np

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.label_extractor import extract_patient_labels
from src.utils.logging import get_logger


def inspect_dataset(config_path: str = "configs/config.yaml"):
    logger = get_logger("InspectDataset")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    dataset_dir = config["data"]["dataset_dir"]
    logger.info(f"Inspecting dataset at: {dataset_dir}")

    if not os.path.exists(dataset_dir):
        logger.error(f"Dataset directory not found: {dataset_dir}")
        return

    patient_dirs = sorted([d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))])
    num_patients = len(patient_dirs)

    has_both = 0
    missing_mammo = []
    missing_us = []
    mammo_counts = []
    us_counts = []

    for p in patient_dirs:
        p_dir = os.path.join(dataset_dir, p)
        m_dir = os.path.join(p_dir, "Mammogram")
        u_dir = os.path.join(p_dir, "Ultrasound")

        m_imgs = [f for f in os.listdir(m_dir) if not f.startswith('.')] if os.path.exists(m_dir) else []
        u_imgs = [f for f in os.listdir(u_dir) if not f.startswith('.')] if os.path.exists(u_dir) else []

        mammo_counts.append(len(m_imgs))
        us_counts.append(len(u_imgs))

        if len(m_imgs) == 0:
            missing_mammo.append(p)
        if len(u_imgs) == 0:
            missing_us.append(p)
        if len(m_imgs) > 0 and len(u_imgs) > 0:
            has_both += 1

    # Extract labels from reports
    logger.info("Extracting and validating labels from PDF clinical reports...")
    patient_labels = extract_patient_labels(dataset_dir)

    label_counts = Counter([pl.label_str for pl in patient_labels.values()])
    status_counts = Counter([pl.status for pl in patient_labels.values()])
    missing_reports = [pl.patient_id for pl in patient_labels.values() if pl.reason == "Missing PDF report file"]
    unreadable_reports = [pl.patient_id for pl in patient_labels.values() if pl.status == "UNKNOWN" and pl.patient_id not in missing_reports]

    print("\n" + "=" * 65)
    print("      BCMID DATASET INSPECTION & VALIDATION REPORT")
    print("=" * 65)
    print(f"Total Patient Folders:             {num_patients}")
    print(f"Patients with Both Modalities:     {has_both} ({has_both/num_patients*100:.1f}%)")
    print(f"Patients Missing Mammograms:       {len(missing_mammo)}")
    print(f"Patients Missing Ultrasounds:      {len(missing_us)}")
    print(f"Total Mammogram Images:            {sum(mammo_counts)}")
    print(f"  Mammograms per Patient:          min={min(mammo_counts)}, max={max(mammo_counts)}, mean={np.mean(mammo_counts):.2f}, median={np.median(mammo_counts):.1f}")
    print(f"Total Ultrasound Images:           {sum(us_counts)}")
    print(f"  Ultrasounds per Patient:         min={min(us_counts)}, max={max(us_counts)}, mean={np.mean(us_counts):.2f}, median={np.median(us_counts):.1f}")
    print("-" * 65)
    print("CLINICAL LABELS (Ground Truth from PDF Reports):")
    print(f"  Benign Patients (0):             {label_counts.get('BENIGN', 0)} ({label_counts.get('BENIGN', 0)/num_patients*100:.1f}%)")
    print(f"  Malignant Patients (1):          {label_counts.get('MALIGNANT', 0)} ({label_counts.get('MALIGNANT', 0)/num_patients*100:.1f}%)")
    print(f"  Unknown / Excluded:              {label_counts.get('UNKNOWN', 0)} ({label_counts.get('UNKNOWN', 0)/num_patients*100:.1f}%)")
    print(f"    - Missing PDF reports:         {len(missing_reports)}")
    print(f"    - Unreadable / Non-diagnostic: {len(unreadable_reports)}")
    print("-" * 65)
    print("EXTRACTION CONFIDENCE STATUS:")
    for s, count in status_counts.items():
        print(f"  {s}: {count}")
    print("=" * 65 + "\n")

    logger.info("Dataset inspection completed successfully.")


if __name__ == "__main__":
    inspect_dataset()
