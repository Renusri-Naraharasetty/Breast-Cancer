import os
import json
import torch
import pytest
import numpy as np
from src.data.dataset import BCMIDModalityDataset, get_dataloader, detect_breast_side


def test_zero_patient_leakage():
    """Verify that root patients do not cross any split boundaries."""
    with open("data/splits/patient_stratified_split.json", "r") as f:
        splits = json.load(f)

    train_roots = set(splits["root_patient_splits"]["train"])
    val_roots = set(splits["root_patient_splits"]["val"])
    test_roots = set(splits["root_patient_splits"]["test"])

    # Disjointness checks
    assert len(train_roots & val_roots) == 0, f"Leakage Train/Val: {train_roots & val_roots}"
    assert len(train_roots & test_roots) == 0, f"Leakage Train/Test: {train_roots & test_roots}"
    assert len(val_roots & test_roots) == 0, f"Leakage Val/Test: {val_roots & test_roots}"

    # Folder-level disjointness
    train_folders = set(splits["folder_patient_splits"]["train"])
    val_folders = set(splits["folder_patient_splits"]["val"])
    test_folders = set(splits["folder_patient_splits"]["test"])

    assert len(train_folders & val_folders) == 0
    assert len(train_folders & test_folders) == 0
    assert len(val_folders & test_folders) == 0


def test_verified_alias_clustering():
    """Verify that verified aliases remain in the exact same partition."""
    with open("data/splits/patient_stratified_split.json", "r") as f:
        splits = json.load(f)

    folder_splits = splits["folder_patient_splits"]

    alias_pairs = [
        ("01566y21", "01566y211"),
        ("03128y21", "03128y211"),
        ("04412y22", "04412y221"),
        ("05128y22", "1000y16"),
    ]

    for p1, p2 in alias_pairs:
        found_in = []
        for sname in ["train", "val", "test"]:
            if p1 in folder_splits[sname] and p2 in folder_splits[sname]:
                found_in.append(sname)
        assert len(found_in) == 1, f"Pair ({p1}, {p2}) separated or missing across splits: {found_in}"


def test_frozen_test_cohort_shared():
    """Verify that both modalities use the identical test patient cohort."""
    with open("data/splits/patient_stratified_split.json", "r") as f:
        splits = json.load(f)

    test_roots = set(splits["root_patient_splits"]["test"])
    assert len(test_roots) == 47

    mg_test = BCMIDModalityDataset(split="test", modality="mammogram")
    us_test = BCMIDModalityDataset(split="test", modality="ultrasound")

    mg_pids = set(s["root_patient_id"] for s in mg_test.samples)
    us_pids = set(s["root_patient_id"] for s in us_test.samples)

    assert mg_pids == test_roots, "Mammogram test cohort does not match frozen test cohort"
    assert us_pids == test_roots, "Ultrasound test cohort does not match frozen test cohort"
    assert mg_pids == us_pids, "Mammogram and Ultrasound test patient cohorts differ!"


def test_side_aware_label_resolution_05439():
    """Verify that discordant patient 05439y20 assigns side-accurate labels (no contralateral label contamination)."""
    mg_train = BCMIDModalityDataset(split="train", modality="mammogram", side_aware_labeling=True)
    us_train = BCMIDModalityDataset(split="train", modality="ultrasound", side_aware_labeling=True)

    # Mammogram checks
    p_mg = [s for s in mg_train.samples if s["patient_id"] == "05439y20"]
    assert len(p_mg) == 4
    for s in p_mg:
        if s["breast_side"] == "L":
            assert s["label_id"] == 1
            assert s["label_name"] == "SUSPICIOUS_ACTIONABLE"
            assert s["highest_birads"] == "6"
        elif s["breast_side"] == "R":
            assert s["label_id"] == 0
            assert s["label_name"] == "LOW_RISK"
            assert s["highest_birads"] == "3"

    # Ultrasound checks
    p_us = [s for s in us_train.samples if s["patient_id"] == "05439y20"]
    assert len(p_us) == 6
    for s in p_us:
        if s["breast_side"] == "L":
            assert s["label_id"] == 1
            assert s["label_name"] == "SUSPICIOUS_ACTIONABLE"
            assert s["highest_birads"] == "6"
        elif s["breast_side"] == "R":
            assert s["label_id"] == 0
            assert s["label_name"] == "LOW_RISK"
            assert s["highest_birads"] == "3"


def test_corrupted_image_excluded():
    """Verify that the 0-byte corrupt image is excluded from all datasets."""
    for mod in ["mammogram", "ultrasound"]:
        for split in ["train", "val", "test"]:
            ds = BCMIDModalityDataset(split=split, modality=mod)
            for s in ds.samples:
                assert "0165y09_US_1.png" not in s["image_path"], "Corrupted image was included!"


def test_dataset_tensor_shapes_and_types():
    """Verify that images load as (3, 224, 224) float32 tensors."""
    mg_ds = BCMIDModalityDataset(split="val", modality="mammogram")
    us_ds = BCMIDModalityDataset(split="val", modality="ultrasound")

    img_mg, label_mg, meta_mg = mg_ds[0]
    img_us, label_us, meta_us = us_ds[0]

    assert isinstance(img_mg, torch.Tensor)
    assert img_mg.shape == (3, 224, 224)
    assert img_mg.dtype == torch.float32

    assert isinstance(img_us, torch.Tensor)
    assert img_us.shape == (3, 224, 224)
    assert img_us.dtype == torch.float32

    assert label_mg.item() in [0, 1]
    assert label_us.item() in [0, 1]


def test_deterministic_val_test_transforms():
    """Verify that val and test sets are strictly deterministic (zero random augmentation)."""
    val_ds = BCMIDModalityDataset(split="val", modality="mammogram")

    img1, _, _ = val_ds[0]
    img2, _, _ = val_ds[0]

    # Deterministic transforms must produce identical tensors
    diff = torch.abs(img1 - img2).max().item()
    assert diff == 0.0, f"Val transform is not deterministic (max diff: {diff})"


def test_dataloader_batching():
    """Verify that DataLoader loads and collates batches properly."""
    loader = get_dataloader(modality="mammogram", split="val", batch_size=8)
    images, labels, metadata = next(iter(loader))

    assert images.shape == (8, 3, 224, 224)
    assert labels.shape == (8,)
    assert len(metadata["patient_id"]) == 8
    assert len(metadata["breast_side"]) == 8
