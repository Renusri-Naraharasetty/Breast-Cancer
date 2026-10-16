"""Multimodal Patient-Level Prediction Script for Graph-Infused Transformer (GIT).

Runs multimodal patient triage using:
1. Mammogram views (processed via EfficientNet-B0).
2. Ultrasound views (processed via ViT-B/16).
3. Patient-level alignment, cross-modal attention, GAT graph message passing,
   and Graph-Infused Transformer classification head.

Default Decision Threshold: tau = 0.50
"""

import os
import sys
import argparse
import json
import torch
from PIL import Image
from torchvision import transforms
from typing import Dict, Any, List, Optional, Tuple

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.models.git import GraphInfusedTransformer
from src.models.factory import create_model
from src.data.dataset import detect_breast_side
from src.data.multimodal_dataset import SIDE_TO_INT


def get_inference_transforms(img_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def load_image_tensor(image_path: str, img_size: int = 224) -> torch.Tensor:
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    with Image.open(image_path) as img:
        rgb_img = img.convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(rgb_img)


def predict_multimodal_patient(
    mg_image_paths: List[str],
    us_image_paths: List[str],
    mg_sides: Optional[List[str]] = None,
    us_sides: Optional[List[str]] = None,
    git_checkpoint: str = "outputs/experiments/exp_git_multimodal/best_checkpoint.pth",
    effnet_checkpoint: str = "outputs/experiments/exp1_efficientnet_mammogram/best_checkpoint.pth",
    vit_checkpoint: str = "outputs/experiments/exp2_vit_ultrasound/best_checkpoint.pth",
    threshold: float = 0.50,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if not mg_image_paths:
        raise ValueError("At least one mammogram image must be provided.")
    if not us_image_paths:
        raise ValueError("At least one ultrasound image must be provided.")

    # 1. Load Backbones for Feature Extraction
    effnet = create_model("efficientnet_b0", pretrained=False)
    if os.path.exists(effnet_checkpoint):
        ckpt_eff = torch.load(effnet_checkpoint, map_location="cpu", weights_only=False)
        effnet.load_state_dict(ckpt_eff["model_state_dict"])
    effnet.to(device).eval()

    vit = create_model("vit_b_16", pretrained=False)
    if os.path.exists(vit_checkpoint):
        ckpt_vit = torch.load(vit_checkpoint, map_location="cpu", weights_only=False)
        vit.load_state_dict(ckpt_vit["model_state_dict"])
    vit.to(device).eval()

    # 2. Extract Mammogram Features
    mg_tensors = [load_image_tensor(p).to(device) for p in mg_image_paths]
    mg_batch = torch.stack(mg_tensors, dim=0)  # (M, 3, 224, 224)
    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            mg_feats = effnet.extract_features(mg_batch)  # (M, 1280)
    mg_feats = mg_feats.unsqueeze(0)  # (1, M, 1280)

    # 3. Extract Ultrasound Features
    us_tensors = [load_image_tensor(p).to(device) for p in us_image_paths]
    us_batch = torch.stack(us_tensors, dim=0)  # (U, 3, 224, 224)
    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            us_feats = vit.extract_features(us_batch)  # (U, 768)
    us_feats = us_feats.unsqueeze(0)  # (1, U, 768)

    # Clean backbones from memory
    del effnet, vit
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # 4. Resolve Sides
    if mg_sides is None:
        mg_sides = [detect_breast_side(os.path.basename(p)) for p in mg_image_paths]
    if us_sides is None:
        us_sides = [detect_breast_side(os.path.basename(p)) for p in us_image_paths]

    mg_sides_t = torch.tensor([[SIDE_TO_INT.get(s, 3) for s in mg_sides]], dtype=torch.long, device=device)
    us_sides_t = torch.tensor([[SIDE_TO_INT.get(s, 3) for s in us_sides]], dtype=torch.long, device=device)

    # 5. Load GIT Model
    if not os.path.exists(git_checkpoint):
        raise FileNotFoundError(f"GIT checkpoint missing: {git_checkpoint}")

    ckpt_git = torch.load(git_checkpoint, map_location="cpu", weights_only=False)
    config = ckpt_git.get("config", {})

    git_model = GraphInfusedTransformer(
        mg_feat_dim=1280,
        us_feat_dim=768,
        hidden_dim=config.get("hidden_dim", 256),
        num_gat_layers=config.get("num_gat_layers", 2),
        gat_heads=config.get("gat_heads", 4),
        transformer_layers=config.get("transformer_layers", 2),
        transformer_heads=config.get("transformer_heads", 8),
        dim_feedforward=config.get("dim_feedforward", 1024),
        num_classes=2,
        dropout_rate=0.0,
    )
    git_model.load_state_dict(ckpt_git["model_state_dict"])
    git_model.to(device).eval()

    # 6. Forward Pass
    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            logits = git_model(
                mg_features=mg_feats,
                us_features=us_feats,
                mg_sides=mg_sides_t,
                us_sides=us_sides_t,
            )
            probs = torch.softmax(logits, dim=1)[0]

    p_low_risk = float(probs[0].item())
    p_suspicious = float(probs[1].item())

    if p_suspicious >= threshold:
        pred_class = "SUSPICIOUS_ACTIONABLE"
        pred_birads = "BI-RADS 4-6 (Suspicious / Highly Suggestive / Malignant)"
        action = "Urgent diagnostic correlation, expedited workup, or biopsy recommended."
    else:
        pred_class = "LOW_RISK"
        pred_birads = "BI-RADS 1-3 (Negative / Benign / Probably Benign)"
        action = "Routine screening interval or short-term clinical follow-up."

    return {
        "model": "Graph-Infused Transformer (GIT)",
        "decision_threshold": threshold,
        "predicted_class": pred_class,
        "predicted_category": pred_birads,
        "clinical_action_triage": action,
        "probability_low_risk": round(p_low_risk, 4),
        "probability_suspicious": round(p_suspicious, 4),
        "probability_low_risk_percent": f"{p_low_risk * 100:.2f}%",
        "probability_suspicious_percent": f"{p_suspicious * 100:.2f}%",
        "mammogram_views_analyzed": len(mg_image_paths),
        "ultrasound_views_analyzed": len(us_image_paths),
        "mg_sides": mg_sides,
        "us_sides": us_sides,
        "selected_epoch": ckpt_git.get("epoch", "N/A"),
        "best_val_roc_auc": ckpt_git.get("best_val_roc_auc", "N/A"),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multimodal GIT Patient Inference")
    parser.add_argument("--mg_images", nargs="+", required=True, help="List of mammogram image paths")
    parser.add_argument("--us_images", nargs="+", required=True, help="List of ultrasound image paths")
    parser.add_argument("--threshold", type=float, default=0.50, help="Decision threshold (default 0.50)")
    args = parser.parse_args()

    result = predict_multimodal_patient(
        mg_image_paths=args.mg_images,
        us_image_paths=args.us_images,
        threshold=args.threshold,
    )
    print("\n" + "=" * 70)
    print("MULTIMODAL GRAPH-INFUSED TRANSFORMER (GIT) PREDICTION RESULT")
    print("=" * 70)
    for k, v in result.items():
        print(f"  {k:30s}: {v}")
    print("=" * 70)
