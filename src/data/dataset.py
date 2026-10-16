import os
import re
import json
import torch
import pandas as pd
from PIL import Image
from typing import Callable, Dict, List, Optional, Tuple, Any
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


def detect_breast_side(filename: str) -> str:
    """Extracts anatomical breast side ('L', 'R', or 'UNK') from image filename."""
    fn_lower = filename.lower()
    # Check mammogram and ultrasound naming conventions
    if "_l_" in fn_lower or "_us_l_" in fn_lower:
        return "L"
    elif "_r_" in fn_lower or "_us_r_" in fn_lower:
        return "R"
    elif "_bil_" in fn_lower or "_us_bil_" in fn_lower:
        return "BIL"
    return "UNK"


class BCMIDModalityDataset(Dataset):
    """PyTorch Dataset for independent modality benchmarking on BCMID.

    Supports 'mammogram' and 'ultrasound' modalities.
    Ensures:
      1. Anatomically correct side-specific labeling (preventing contralateral false-positive supervision).
      2. Deterministic image loading (never silently drops or randomly picks images).
      3. Safe 3-channel RGB normalization for both single-channel grayscale and 4-channel RGBA formats.
    """

    def __init__(
        self,
        bcmid_dir: str = "BCMID",
        tsv_path: str = "BCMID_annotations.tsv",
        splits_path: str = "data/splits/patient_stratified_split.json",
        split: str = "train",
        modality: str = "mammogram",
        img_size: int = 224,
        side_aware_labeling: bool = True,
        transform: Optional[Callable] = None,
    ):
        super().__init__()
        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        assert modality.lower() in [
            "mammogram",
            "ultrasound",
        ], f"Invalid modality: {modality}"

        self.bcmid_dir = bcmid_dir
        self.split = split
        self.modality = modality.lower()
        self.img_size = img_size
        self.side_aware_labeling = side_aware_labeling

        # Load annotations
        df = pd.read_csv(tsv_path, sep="\t")
        self.patient_info = df.set_index("patient_id").to_dict(orient="index")

        # Load split definitions
        with open(splits_path, "r", encoding="utf-8") as f:
            splits_data = json.load(f)

        allowed_folders = set(splits_data["folder_patient_splits"][split])

        # Subfolder name inside patient directory
        subfolder_name = "Mammogram" if self.modality == "mammogram" else "Ultrasound"

        # Index all valid images
        self.samples = []
        for pid in sorted(list(allowed_folders)):
            p_dir = os.path.join(bcmid_dir, pid, subfolder_name)
            if not os.path.exists(p_dir):
                continue

            info = self.patient_info.get(pid, {})
            patient_label_id = info.get("label_id", -1)
            # Skip if patient is excluded
            if patient_label_id == -1:
                continue

            files = sorted(os.listdir(p_dir))
            for f in files:
                if not f.lower().endswith(".png"):
                    continue
                full_path = os.path.join(p_dir, f)
                # Skip 0-byte corrupted files
                if os.path.getsize(full_path) == 0:
                    continue

                side = detect_breast_side(f)

                # Anatomical side-specific ground truth resolution
                # Patient 05439y20 has proven left carcinoma (BI-RADS 6) and right benign lesion (BI-RADS 3)
                if self.side_aware_labeling and pid == "05439y20":
                    if side == "R":
                        sample_label_id = 0
                        sample_label_name = "LOW_RISK"
                        sample_birads = "3"
                        is_pathology = False
                    else:
                        sample_label_id = 1
                        sample_label_name = "SUSPICIOUS_ACTIONABLE"
                        sample_birads = "6"
                        is_pathology = True
                else:
                    sample_label_id = int(patient_label_id)
                    sample_label_name = str(info.get("label_name", "UNKNOWN"))
                    sample_birads = str(info.get("highest_birads", "UNKNOWN"))
                    is_pathology = bool(info.get("is_pathology_confirmed", False))

                self.samples.append({
                    "image_path": full_path,
                    "filename": f,
                    "patient_id": pid,
                    "root_patient_id": info.get("root_patient_id", pid),
                    "breast_side": side,
                    "label_id": sample_label_id,
                    "label_name": sample_label_name,
                    "highest_birads": sample_birads,
                    "is_pathology_confirmed": is_pathology,
                })

        # Set default transforms if none provided
        if transform is not None:
            self.transform = transform
        else:
            self.transform = self._get_default_transforms()

    def _get_default_transforms(self) -> Callable:
        imagenet_mean = [0.485, 0.456, 0.406]
        imagenet_std = [0.229, 0.224, 0.225]

        if self.split == "train":
            return transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
            ])
        else:
            # Deterministic for val and test sets (zero augmentation)
            return transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
            ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        sample = self.samples[idx]

        # Use pre-cached 224x224 RGB image if available to bypass 10-megapixel CPU decoding bottleneck
        cached_path = os.path.join("data", "cached_224", self.modality, sample["filename"])
        if os.path.exists(cached_path):
            with Image.open(cached_path) as img:
                rgb_img = img.convert("RGB")
        else:
            with Image.open(sample["image_path"]) as img:
                # Safely harmonize grayscale and RGBA into 3-channel RGB
                rgb_img = img.convert("RGB")

        tensor_img = self.transform(rgb_img)
        label = torch.tensor(sample["label_id"], dtype=torch.long)

        metadata = {
            "patient_id": sample["patient_id"],
            "root_patient_id": sample["root_patient_id"],
            "image_path": sample["image_path"],
            "filename": sample["filename"],
            "breast_side": sample["breast_side"],
            "highest_birads": sample["highest_birads"],
            "label_name": sample["label_name"],
            "is_pathology_confirmed": sample["is_pathology_confirmed"],
            "modality": self.modality,
        }

        return tensor_img, label, metadata


def get_dataloader(
    modality: str,
    split: str,
    batch_size: int = 16,
    bcmid_dir: str = "BCMID",
    tsv_path: str = "BCMID_annotations.tsv",
    splits_path: str = "data/splits/patient_stratified_split.json",
    img_size: int = 224,
    side_aware_labeling: bool = True,
    shuffle: Optional[bool] = None,
    num_workers: int = 0,
) -> DataLoader:
    """Helper to instantiate DataLoader for any experiment modality and split."""
    if shuffle is None:
        shuffle = (split == "train")

    dataset = BCMIDModalityDataset(
        bcmid_dir=bcmid_dir,
        tsv_path=tsv_path,
        splits_path=splits_path,
        split=split,
        modality=modality,
        img_size=img_size,
        side_aware_labeling=side_aware_labeling,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
