import os
import json
import torch
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms

from .dataset import detect_breast_side

SIDE_TO_INT = {"L": 0, "R": 1, "BIL": 2, "UNK": 3}


class BCMIDPatientMultimodalDataset(Dataset):
    """Patient-level multimodal dataset for BCMID breast cancer triage.

    Aligns mammogram and ultrasound modalities at the patient level.
    Supports either pre-extracted features (for memory-efficient fast training)
    or raw image loading (for end-to-end evaluation).
    """

    def __init__(
        self,
        bcmid_dir: str = "BCMID",
        tsv_path: str = "BCMID_annotations.tsv",
        splits_path: str = "data/splits/patient_stratified_split.json",
        features_path: Optional[str] = "data/features/patient_multimodal_features.pt",
        split: str = "train",
        load_images: bool = False,
        img_size: int = 224,
    ):
        super().__init__()
        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        self.split = split
        self.bcmid_dir = bcmid_dir
        self.load_images = load_images
        self.img_size = img_size

        # Load annotation metadata
        df = pd.read_csv(tsv_path, sep="\t")
        self.patient_info = df.set_index("patient_id").to_dict(orient="index")

        # Load split definitions
        with open(splits_path, "r", encoding="utf-8") as f:
            splits_data = json.load(f)
        allowed_patients = sorted(list(set(splits_data["folder_patient_splits"][split])))

        # Check for pre-extracted feature cache
        self.cached_features = None
        if features_path and os.path.exists(features_path) and not load_images:
            all_features = torch.load(features_path, map_location="cpu", weights_only=False)
            self.cached_features = all_features.get(split, {})

        # Build list of valid patient samples
        self.patients = []
        for pid in allowed_patients:
            info = self.patient_info.get(pid, {})
            label_id = info.get("label_id", -1)
            if label_id == -1:
                continue

            p_mg_dir = os.path.join(bcmid_dir, pid, "Mammogram")
            p_us_dir = os.path.join(bcmid_dir, pid, "Ultrasound")

            mg_files = []
            if os.path.exists(p_mg_dir):
                for f in sorted(os.listdir(p_mg_dir)):
                    full_p = os.path.join(p_mg_dir, f)
                    if f.lower().endswith(".png") and os.path.getsize(full_p) > 0:
                        mg_files.append(f)

            us_files = []
            if os.path.exists(p_us_dir):
                for f in sorted(os.listdir(p_us_dir)):
                    full_p = os.path.join(p_us_dir, f)
                    if f.lower().endswith(".png") and os.path.getsize(full_p) > 0:
                        us_files.append(f)

            # Both modalities must be present for multimodal alignment
            if len(mg_files) == 0 or len(us_files) == 0:
                continue

            self.patients.append({
                "patient_id": pid,
                "root_patient_id": info.get("root_patient_id", pid),
                "label_id": int(label_id),
                "label_name": str(info.get("label_name", "UNKNOWN")),
                "highest_birads": str(info.get("highest_birads", "UNKNOWN")),
                "is_pathology_confirmed": bool(info.get("is_pathology_confirmed", False)),
                "mg_files": mg_files,
                "us_files": us_files,
                "mg_sides": [detect_breast_side(f) for f in mg_files],
                "us_sides": [detect_breast_side(f) for f in us_files],
            })

        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self) -> int:
        return len(self.patients)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        patient = self.patients[idx]
        pid = patient["patient_id"]

        mg_sides_int = torch.tensor([SIDE_TO_INT.get(s, 3) for s in patient["mg_sides"]], dtype=torch.long)
        us_sides_int = torch.tensor([SIDE_TO_INT.get(s, 3) for s in patient["us_sides"]], dtype=torch.long)
        label = torch.tensor(patient["label_id"], dtype=torch.long)

        record = {
            "patient_id": pid,
            "root_patient_id": patient["root_patient_id"],
            "label_id": label,
            "label_name": patient["label_name"],
            "highest_birads": patient["highest_birads"],
            "is_pathology_confirmed": patient["is_pathology_confirmed"],
            "mg_sides": mg_sides_int,
            "us_sides": us_sides_int,
            "mg_files": patient["mg_files"],
            "us_files": patient["us_files"],
        }

        # 1. Return cached features if available
        if self.cached_features is not None and pid in self.cached_features:
            feat_data = self.cached_features[pid]
            record["mg_features"] = feat_data["mg_features"].clone()
            record["us_features"] = feat_data["us_features"].clone()
            return record

        # 2. Or load raw images
        if self.load_images:
            mg_tensors = []
            for f in patient["mg_files"]:
                c_path = os.path.join("data", "cached_224", "mammogram", f)
                img_path = c_path if os.path.exists(c_path) else os.path.join(self.bcmid_dir, pid, "Mammogram", f)
                with Image.open(img_path) as img:
                    mg_tensors.append(self.transform(img.convert("RGB")))

            us_tensors = []
            for f in patient["us_files"]:
                c_path = os.path.join("data", "cached_224", "ultrasound", f)
                img_path = c_path if os.path.exists(c_path) else os.path.join(self.bcmid_dir, pid, "Ultrasound", f)
                with Image.open(img_path) as img:
                    us_tensors.append(self.transform(img.convert("RGB")))

            record["mg_images"] = torch.stack(mg_tensors, dim=0)  # (M, 3, 224, 224)
            record["us_images"] = torch.stack(us_tensors, dim=0)  # (U, 3, 224, 224)

        return record


def collate_patient_multimodal(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate function that dynamically pads variable number of MG and US views per patient."""
    batch_size = len(batch)

    # Determine maximum number of MG and US views across the batch
    max_mg = max(len(item["mg_sides"]) for item in batch)
    max_us = max(len(item["us_sides"]) for item in batch)

    labels = torch.stack([item["label_id"] for item in batch], dim=0)

    # Padding masks (True = valid, False = padded)
    mg_mask = torch.zeros(batch_size, max_mg, dtype=torch.bool)
    us_mask = torch.zeros(batch_size, max_us, dtype=torch.bool)

    # Side tensors initialized with UNK (3)
    mg_sides = torch.full((batch_size, max_mg), fill_value=3, dtype=torch.long)
    us_sides = torch.full((batch_size, max_us), fill_value=3, dtype=torch.long)

    has_features = "mg_features" in batch[0]
    has_images = "mg_images" in batch[0]

    mg_features = None
    us_features = None
    mg_images = None
    us_images = None

    if has_features:
        mg_feat_dim = batch[0]["mg_features"].size(1)
        us_feat_dim = batch[0]["us_features"].size(1)
        mg_features = torch.zeros(batch_size, max_mg, mg_feat_dim, dtype=torch.float32)
        us_features = torch.zeros(batch_size, max_us, us_feat_dim, dtype=torch.float32)

    if has_images:
        mg_images = torch.zeros(batch_size, max_mg, 3, 224, 224, dtype=torch.float32)
        us_images = torch.zeros(batch_size, max_us, 3, 224, 224, dtype=torch.float32)

    patient_ids = []
    root_ids = []
    highest_birads = []

    for b, item in enumerate(batch):
        m_len = len(item["mg_sides"])
        u_len = len(item["us_sides"])

        mg_mask[b, :m_len] = True
        us_mask[b, :u_len] = True

        mg_sides[b, :m_len] = item["mg_sides"]
        us_sides[b, :u_len] = item["us_sides"]

        if has_features:
            mg_features[b, :m_len, :] = item["mg_features"]
            us_features[b, :u_len, :] = item["us_features"]

        if has_images:
            mg_images[b, :m_len, :, :, :] = item["mg_images"]
            us_images[b, :u_len, :, :, :] = item["us_images"]

        patient_ids.append(item["patient_id"])
        root_ids.append(item["root_patient_id"])
        highest_birads.append(item["highest_birads"])

    collated = {
        "labels": labels,
        "mg_sides": mg_sides,
        "us_sides": us_sides,
        "mg_mask": mg_mask,
        "us_mask": us_mask,
        "patient_ids": patient_ids,
        "root_patient_ids": root_ids,
        "highest_birads": highest_birads,
    }

    if has_features:
        collated["mg_features"] = mg_features
        collated["us_features"] = us_features

    if has_images:
        collated["mg_images"] = mg_images
        collated["us_images"] = us_images

    return collated


def get_patient_multimodal_dataloader(
    split: str,
    batch_size: int = 8,
    bcmid_dir: str = "BCMID",
    tsv_path: str = "BCMID_annotations.tsv",
    splits_path: str = "data/splits/patient_stratified_split.json",
    features_path: Optional[str] = "data/features/patient_multimodal_features.pt",
    load_images: bool = False,
    shuffle: Optional[bool] = None,
    num_workers: int = 0,
) -> DataLoader:
    """Helper to instantiate DataLoader for patient multimodal training/evaluation."""
    if shuffle is None:
        shuffle = (split == "train")

    dataset = BCMIDPatientMultimodalDataset(
        bcmid_dir=bcmid_dir,
        tsv_path=tsv_path,
        splits_path=splits_path,
        features_path=features_path,
        split=split,
        load_images=load_images,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_patient_multimodal,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
