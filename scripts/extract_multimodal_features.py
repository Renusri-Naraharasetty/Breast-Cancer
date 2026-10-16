"""Feature Extraction Pipeline for Multimodal Graph-Infused Transformer (GIT).

Extracts:
1. 1280-dimensional mammogram features using EfficientNet-B0 (from exp1 checkpoint).
2. 768-dimensional ultrasound features using ViT-B/16 (from exp2 checkpoint).

Saves patient-aligned multimodal feature tensors to:
data/features/patient_multimodal_features.pt
"""

import os
import sys
import gc
import json
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms
from typing import Dict, Any

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.factory import create_model
from src.data.dataset import detect_breast_side


def get_feature_extraction_transforms(img_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def load_image_rgb(image_path: str, cached_path: str, transform: transforms.Compose) -> torch.Tensor:
    target_path = cached_path if (os.path.exists(cached_path) and os.path.getsize(cached_path) > 0) else image_path
    with Image.open(target_path) as img:
        return transform(img.convert("RGB"))


def extract_mammogram_features(
    checkpoint_path: str = "outputs/experiments/exp1_efficientnet_mammogram/best_checkpoint.pth",
    device: torch.device = None,
    batch_size: int = 16,
) -> Dict[str, torch.Tensor]:
    print("\n" + "=" * 70)
    print("STEP 1: Extracting 1280-dim Mammogram features with EfficientNet-B0")
    print("=" * 70)

    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = create_model("efficientnet_b0", pretrained=False)
    if os.path.exists(checkpoint_path):
        print(f"Loading weights from verified checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        print("Warning: Checkpoint not found. Instantiating ImageNet pretrained model.")
        model = create_model("efficientnet_b0", pretrained=True)

    model.to(device)
    model.eval()

    transform = get_feature_extraction_transforms(224)
    cached_dir = os.path.join("data", "cached_224", "mammogram")

    # Discover all mammogram files
    mg_files = sorted(os.listdir(cached_dir)) if os.path.exists(cached_dir) else []
    print(f"Found {len(mg_files)} cached mammogram images.")

    feature_dict = {}
    batches = [mg_files[i : i + batch_size] for i in range(0, len(mg_files), batch_size)]

    with torch.no_grad():
        for b_idx, batch_fns in enumerate(batches):
            tensors = []
            valid_fns = []
            for fn in batch_fns:
                full_p = os.path.join(cached_dir, fn)
                try:
                    t = load_image_rgb(full_p, full_p, transform)
                    tensors.append(t)
                    valid_fns.append(fn)
                except Exception as e:
                    print(f"Error loading {fn}: {e}")

            if not tensors:
                continue

            batch_tensor = torch.stack(tensors, dim=0).to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                feats = model.extract_features(batch_tensor)  # (B, 1280)

            feats_cpu = feats.cpu().float()
            for fn, f_vec in zip(valid_fns, feats_cpu):
                feature_dict[fn] = f_vec

            if (b_idx + 1) % 15 == 0 or (b_idx + 1) == len(batches):
                print(f"  Processed {len(feature_dict)} / {len(mg_files)} mammograms...")

    print(f"Extracted {len(feature_dict)} mammogram feature vectors (dim={next(iter(feature_dict.values())).shape}).")

    # Purge model from GPU VRAM
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    return feature_dict


def extract_ultrasound_features(
    checkpoint_path: str = "outputs/experiments/exp2_vit_ultrasound/best_checkpoint.pth",
    device: torch.device = None,
    batch_size: int = 8,
) -> Dict[str, torch.Tensor]:
    print("\n" + "=" * 70)
    print("STEP 2: Extracting 768-dim Ultrasound features with ViT-B/16")
    print("=" * 70)

    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = create_model("vit_b_16", pretrained=False)
    if os.path.exists(checkpoint_path):
        print(f"Loading weights from verified checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        print("Warning: Checkpoint not found. Instantiating ImageNet pretrained model.")
        model = create_model("vit_b_16", pretrained=True)

    model.to(device)
    model.eval()

    transform = get_feature_extraction_transforms(224)
    cached_dir = os.path.join("data", "cached_224", "ultrasound")

    us_files = sorted(os.listdir(cached_dir)) if os.path.exists(cached_dir) else []
    print(f"Found {len(us_files)} cached ultrasound images.")

    feature_dict = {}
    batches = [us_files[i : i + batch_size] for i in range(0, len(us_files), batch_size)]

    with torch.no_grad():
        for b_idx, batch_fns in enumerate(batches):
            tensors = []
            valid_fns = []
            for fn in batch_fns:
                full_p = os.path.join(cached_dir, fn)
                try:
                    t = load_image_rgb(full_p, full_p, transform)
                    tensors.append(t)
                    valid_fns.append(fn)
                except Exception as e:
                    print(f"Error loading {fn}: {e}")

            if not tensors:
                continue

            batch_tensor = torch.stack(tensors, dim=0).to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                feats = model.extract_features(batch_tensor)  # (B, 768)

            feats_cpu = feats.cpu().float()
            for fn, f_vec in zip(valid_fns, feats_cpu):
                feature_dict[fn] = f_vec

            if (b_idx + 1) % 20 == 0 or (b_idx + 1) == len(batches):
                print(f"  Processed {len(feature_dict)} / {len(us_files)} ultrasound images...")

    print(f"Extracted {len(feature_dict)} ultrasound feature vectors (dim={next(iter(feature_dict.values())).shape}).")

    # Purge model from GPU VRAM
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    return feature_dict


def assemble_patient_multimodal_dataset(
    mg_feat_dict: Dict[str, torch.Tensor],
    us_feat_dict: Dict[str, torch.Tensor],
    bcmid_dir: str = "BCMID",
    tsv_path: str = "BCMID_annotations.tsv",
    splits_path: str = "data/splits/patient_stratified_split.json",
    output_path: str = "data/features/patient_multimodal_features.pt",
) -> Dict[str, Any]:
    print("\n" + "=" * 70)
    print("STEP 3: Assembling patient-level aligned multimodal feature dataset")
    print("=" * 70)

    df = pd.read_csv(tsv_path, sep="\t")
    patient_info = df.set_index("patient_id").to_dict(orient="index")

    with open(splits_path, "r", encoding="utf-8") as f:
        splits = json.load(f)

    assembled_data = {"train": {}, "val": {}, "test": {}, "metadata": {}}

    for split in ["train", "val", "test"]:
        pids = sorted(list(set(splits["folder_patient_splits"][split])))
        for pid in pids:
            info = patient_info.get(pid, {})
            label_id = info.get("label_id", -1)
            if label_id == -1:
                continue

            p_mg_dir = os.path.join(bcmid_dir, pid, "Mammogram")
            p_us_dir = os.path.join(bcmid_dir, pid, "Ultrasound")

            mg_files = sorted([f for f in os.listdir(p_mg_dir) if f.lower().endswith(".png") and os.path.getsize(os.path.join(p_mg_dir, f)) > 0]) if os.path.exists(p_mg_dir) else []
            us_files = sorted([f for f in os.listdir(p_us_dir) if f.lower().endswith(".png") and os.path.getsize(os.path.join(p_us_dir, f)) > 0]) if os.path.exists(p_us_dir) else []

            if not mg_files or not us_files:
                print(f"Warning: Patient {pid} missing one modality (MG: {len(mg_files)}, US: {len(us_files)})")
                continue

            # Gather mammogram features
            mg_feats_list = [mg_feat_dict[f] for f in mg_files if f in mg_feat_dict]
            us_feats_list = [us_feat_dict[f] for f in us_files if f in us_feat_dict]

            if len(mg_feats_list) == 0 or len(us_feats_list) == 0:
                print(f"Warning: Missing features for patient {pid}")
                continue

            mg_tensor = torch.stack(mg_feats_list, dim=0)  # (M, 1280)
            us_tensor = torch.stack(us_feats_list, dim=0)  # (U, 768)

            assembled_data[split][pid] = {
                "patient_id": pid,
                "root_patient_id": info.get("root_patient_id", pid),
                "label_id": int(label_id),
                "label_name": str(info.get("label_name", "UNKNOWN")),
                "highest_birads": str(info.get("highest_birads", "UNKNOWN")),
                "is_pathology_confirmed": bool(info.get("is_pathology_confirmed", False)),
                "mg_features": mg_tensor,
                "us_features": us_tensor,
                "mg_files": mg_files,
                "us_files": us_files,
                "mg_sides": [detect_breast_side(f) for f in mg_files],
                "us_sides": [detect_breast_side(f) for f in us_files],
            }

        print(f"  {split.upper()}: Assembled {len(assembled_data[split])} patients.")

    assembled_data["metadata"] = {
        "train_patients": len(assembled_data["train"]),
        "val_patients": len(assembled_data["val"]),
        "test_patients": len(assembled_data["test"]),
        "mg_feat_dim": 1280,
        "us_feat_dim": 768,
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(assembled_data, output_path)
    print(f"\n[OK] Patient multimodal features successfully saved to: {output_path}")
    print(f"     Train: {assembled_data['metadata']['train_patients']}, Val: {assembled_data['metadata']['val_patients']}, Test: {assembled_data['metadata']['test_patients']}")

    return assembled_data


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Running feature extraction on: {device}")

    mg_feats = extract_mammogram_features(device=device, batch_size=16)
    us_feats = extract_ultrasound_features(device=device, batch_size=8)
    assemble_patient_multimodal_dataset(mg_feats, us_feats)
    print("\nMultimodal feature extraction completed successfully!")
