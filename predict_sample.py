"""Sample-Image Prediction Script for BCMID Breast Imaging Classification Models.

Supports single-image inference on any of the four independently trained Phase 4 checkpoints:
  - exp1: EfficientNet-B0 on Mammograms
  - exp2: Vision Transformer (ViT-B/16) on Ultrasound
  - exp3: Vision Transformer (ViT-B/16) on Mammograms
  - exp4: EfficientNet-B0 on Ultrasound

Requirements:
- Input image: Any standard mammogram or ultrasound image (.png, .jpg, .jpeg, etc.)
- Model: One of ['exp1', 'exp2', 'exp3', 'exp4']
- Threshold: 0.50 (documented standard triage threshold)
"""

import os
import sys
import argparse
import json
import torch
from PIL import Image
from torchvision import transforms
from typing import Dict, Any, Tuple

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.models.factory import create_model


# Registry of trained experiment checkpoints
MODEL_REGISTRY = {
    "exp1": {
        "id": "exp1_efficientnet_mammogram",
        "name": "EfficientNet-B0 (Mammogram)",
        "model_name": "efficientnet_b0",
        "modality": "mammogram",
        "checkpoint_path": os.path.join("outputs", "experiments", "exp1_efficientnet_mammogram", "best_checkpoint.pth"),
        "description": "EfficientNet-B0 trained on full-field digital mammograms",
    },
    "exp2": {
        "id": "exp2_vit_ultrasound",
        "name": "ViT-B/16 (Ultrasound)",
        "model_name": "vit_b_16",
        "modality": "ultrasound",
        "checkpoint_path": os.path.join("outputs", "experiments", "exp2_vit_ultrasound", "best_checkpoint.pth"),
        "description": "Vision Transformer (ViT-B/16) trained on breast ultrasound images",
    },
    "exp3": {
        "id": "exp3_vit_mammogram",
        "name": "ViT-B/16 (Mammogram)",
        "model_name": "vit_b_16",
        "modality": "mammogram",
        "checkpoint_path": os.path.join("outputs", "experiments", "exp3_vit_mammogram", "best_checkpoint.pth"),
        "description": "Vision Transformer (ViT-B/16) trained on full-field digital mammograms",
    },
    "exp4": {
        "id": "exp4_efficientnet_ultrasound",
        "name": "EfficientNet-B0 (Ultrasound)",
        "model_name": "efficientnet_b0",
        "modality": "ultrasound",
        "checkpoint_path": os.path.join("outputs", "experiments", "exp4_efficientnet_ultrasound", "best_checkpoint.pth"),
        "description": "EfficientNet-B0 trained on breast ultrasound images",
    },
}


def get_inference_transforms(img_size: int = 224) -> transforms.Compose:
    """Returns exact deterministic preprocessing transforms used during validation and testing."""
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    return transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])


def load_image_tensor(image_path: str, img_size: int = 224) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """Loads an image file, safely harmonizes to 3-channel RGB, and applies transforms."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Input image not found: {image_path}")

    try:
        with Image.open(image_path) as img:
            orig_size = img.size  # (width, height)
            rgb_img = img.convert("RGB")
    except Exception as e:
        raise ValueError(f"Failed to load image file '{image_path}': {str(e)}")

    transform = get_inference_transforms(img_size)
    tensor = transform(rgb_img)  # Shape: (3, img_size, img_size)
    tensor = tensor.unsqueeze(0)  # Shape: (1, 3, img_size, img_size)
    return tensor, orig_size


def load_model_from_checkpoint(
    model_key: str,
    checkpoint_path: str = None,
    device: torch.device = None,
) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    """Loads model architecture and state dictionary from verified checkpoint."""
    if model_key not in MODEL_REGISTRY:
        valid_keys = list(MODEL_REGISTRY.keys())
        raise ValueError(f"Invalid model choice '{model_key}'. Supported choices: {valid_keys}")

    meta = MODEL_REGISTRY[model_key]
    ckpt_path = checkpoint_path or meta["checkpoint_path"]

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Required checkpoint not found at: '{ckpt_path}'. "
            f"Ensure Phase 4 training completed successfully."
        )

    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load checkpoint safely on CPU first
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if "model_state_dict" not in ckpt:
        raise KeyError(f"Checkpoint at '{ckpt_path}' is corrupted: missing 'model_state_dict'.")

    # Instantiate architecture
    model_name = ckpt.get("model_name", meta["model_name"])
    model = create_model(model_name=model_name, pretrained=False, num_classes=2, dropout_rate=0.3)

    # Load weights
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    ckpt_info = {
        "experiment_id": ckpt.get("experiment_id", meta["id"]),
        "model_name": model_name,
        "architecture": model.__class__.__name__,
        "modality": ckpt.get("modality", meta["modality"]),
        "selected_epoch": ckpt.get("epoch", "N/A"),
        "best_val_loss": ckpt.get("best_val_loss", None),
        "best_val_roc_auc": ckpt.get("best_val_roc_auc", None),
        "checkpoint_path": ckpt_path,
        "device": str(device),
    }
    return model, ckpt_info


def predict_sample(
    image_path: str,
    model_key: str = "exp1",
    checkpoint_path: str = None,
    threshold: float = 0.50,
    device: torch.device = None,
) -> Dict[str, Any]:
    """Runs single-image inference and returns structured prediction results."""
    # 1. Load model
    model, ckpt_info = load_model_from_checkpoint(model_key, checkpoint_path, device)
    dev = next(model.parameters()).device

    # 2. Load and transform image
    tensor, orig_size = load_image_tensor(image_path)
    tensor = tensor.to(dev)

    # 3. Model forward pass
    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=dev.type == "cuda"):
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1)[0]

    p_low_risk = float(probs[0].item())
    p_suspicious = float(probs[1].item())

    # 4. Classification decision at threshold 0.50
    if p_suspicious >= threshold:
        predicted_class = "SUSPICIOUS_ACTIONABLE"
        predicted_birads = "BI-RADS 4-6 (Suspicious / Highly Suggestive / Malignant)"
        action_recommendation = "Expedited clinical workup / diagnostic mammography / biopsy recommended."
    else:
        predicted_class = "LOW_RISK"
        predicted_birads = "BI-RADS 1-3 (Negative / Benign / Probably Benign)"
        action_recommendation = "Routine screening interval or short-term follow-up."

    result = {
        "image_path": os.path.abspath(image_path),
        "image_filename": os.path.basename(image_path),
        "original_resolution": f"{orig_size[0]}x{orig_size[1]}",
        "model_choice": model_key,
        "model_name": ckpt_info["model_name"],
        "architecture": ckpt_info["architecture"],
        "modality": ckpt_info["modality"],
        "checkpoint_path": ckpt_info["checkpoint_path"],
        "selected_epoch": ckpt_info["selected_epoch"],
        "decision_threshold": threshold,
        "predicted_class": predicted_class,
        "predicted_category": predicted_birads,
        "clinical_action_triage": action_recommendation,
        "probability_low_risk": round(p_low_risk, 4),
        "probability_suspicious": round(p_suspicious, 4),
        "probability_low_risk_percent": f"{p_low_risk * 100:.2f}%",
        "probability_suspicious_percent": f"{p_suspicious * 100:.2f}%",
        "disclaimer": (
            "RESEARCH USE ONLY: This result is an automated radiologic triage estimate "
            "(LOW_RISK vs. SUSPICIOUS_ACTIONABLE) based on BI-RADS criteria. It is NOT a confirmed "
            "pathological cancer diagnosis and must not replace professional clinical judgment."
        ),
    }
    return result


def print_prediction_report(res: Dict[str, Any]):
    """Prints a clean, formatted clinical-style prediction card in the terminal."""
    pred_cls = res["predicted_class"]
    prob_susp = res["probability_suspicious_percent"]
    prob_low = res["probability_low_risk_percent"]

    print("\n" + "=" * 78)
    print("      BREAST CANCER RADIOLOGIC TRIAGE PREDICTION (SAMPLE INFERENCE)")
    print("=" * 78)

    print("\n[INPUT DETAILS]")
    print(f"  - Image Filename   : {res['image_filename']}")
    print(f"  - Full Image Path  : {res['image_path']}")
    print(f"  - Original Size    : {res['original_resolution']} (Resized to 224x224 for inference)")

    print("\n[MODEL SPECIFICATION]")
    print(f"  - Model Choice     : {res['model_choice'].upper()} ({res['model_name']})")
    print(f"  - Architecture     : {res['architecture']}")
    print(f"  - Target Modality  : {res['modality'].upper()}")
    print(f"  - Loaded Checkpoint: {res['checkpoint_path']}")
    print(f"  - Checkpoint Epoch : Epoch {res['selected_epoch']}")

    print("\n" + "-" * 78)
    print("                       TRIAGE CLASSIFICATION RESULT")
    print("-" * 78)

    status_tag = "[ ! ] SUSPICIOUS / ACTIONABLE" if pred_cls == "SUSPICIOUS_ACTIONABLE" else "[ * ] LOW RISK"
    print(f"\n  Status: {status_tag}")
    print(f"  Category: {res['predicted_category']}")
    print(f"  Threshold Applied: {res['decision_threshold']:.2f} (Canonical Documented Threshold)")

    print("\n  Class Probabilities:")
    print(f"    |-- P(LOW_RISK)              [BI-RADS 1-3] : {prob_low:>8}  ({res['probability_low_risk']:.4f})")
    print(f"    \\-- P(SUSPICIOUS_ACTIONABLE) [BI-RADS 4-6] : {prob_susp:>8}  ({res['probability_suspicious']:.4f})")

    print(f"\n  Triage Guidance: {res['clinical_action_triage']}")

    print("\n" + "=" * 78)
    print("                              IMPORTANT NOTICE")
    print("=" * 78)
    print(f"  {res['disclaimer']}")
    print("=" * 78 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Run inference on a single breast imaging sample using trained BCMID models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Supported Models:
  exp1 : EfficientNet-B0 on Mammograms (outputs/experiments/exp1_efficientnet_mammogram/best_checkpoint.pth)
  exp2 : ViT-B/16 on Ultrasound       (outputs/experiments/exp2_vit_ultrasound/best_checkpoint.pth)
  exp3 : ViT-B/16 on Mammograms       (outputs/experiments/exp3_vit_mammogram/best_checkpoint.pth)
  exp4 : EfficientNet-B0 on Ultrasound (outputs/experiments/exp4_efficientnet_ultrasound/best_checkpoint.pth)

Examples:
  python predict_sample.py --image path/to/mammogram.png --model exp1
  python predict_sample.py --image path/to/ultrasound.png --model exp2
  python predict_sample.py --image path/to/mammogram.png --model exp3
  python predict_sample.py --image path/to/ultrasound.png --model exp4
        """,
    )
    parser.add_argument(
        "--image",
        "-i",
        type=str,
        required=True,
        help="Path to the input mammogram or ultrasound image file (.png, .jpg, etc.)",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="exp1",
        choices=["exp1", "exp2", "exp3", "exp4"],
        help="Model experiment to use for prediction (exp1, exp2, exp3, exp4). Default: exp1.",
    )
    parser.add_argument(
        "--checkpoint",
        "-c",
        type=str,
        default=None,
        help="Optional custom path to best_checkpoint.pth. If not provided, uses default registry path.",
    )
    parser.add_argument(
        "--threshold",
        "-t",
        type=float,
        default=0.50,
        help="Classification decision threshold for SUSPICIOUS_ACTIONABLE class. Default: 0.50.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON results instead of human-readable text card.",
    )

    args = parser.parse_args()

    try:
        res = predict_sample(
            image_path=args.image,
            model_key=args.model.lower(),
            checkpoint_path=args.checkpoint,
            threshold=args.threshold,
        )

        if args.json:
            print(json.dumps(res, indent=2))
        else:
            print_prediction_report(res)

    except Exception as e:
        print(f"\n[ERROR] Inference failed: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
