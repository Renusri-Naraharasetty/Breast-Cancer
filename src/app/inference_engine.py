"""Inference engine for Multimodal and Single-Modality Breast Cancer Classification.

Supports:
1. Multimodal (Mammogram + Ultrasound) -> Graph Transformer (threshold: 0.2565)
2. Mammogram only -> DenseNet-121 (threshold: 0.2932)
3. Ultrasound only -> ResNet-50 (threshold: 0.3844)

Applies exact evaluation preprocessing, runs forward inference, applies validation-selected
decision thresholds, and generates explainability visualizations (Grad-CAM or cross-attention).

DISCLAIMER: Research prototype — not for clinical diagnosis.
"""

import io
import os
import base64
from pathlib import Path
import yaml
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from typing import Dict, Any, Tuple, Optional
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data.preprocessing import get_mammogram_transforms, get_ultrasound_transforms
from scripts.evaluate import build_model
from src.explainability.gradcam import GradCAM


def resolve_checkpoint_path(path: str) -> str:
    """Resolves checkpoint path supporting alternative directory names."""
    p = Path(path)
    if p.exists():
        return str(p)

    path_str = str(path).replace("\\", "/")
    replacements = [
        ("densenet/checkpoints", "densenet_mammogram/checkpoints"),
        ("densenet_mammogram/checkpoints", "densenet/checkpoints"),
        ("resnet/checkpoints", "resnet_ultrasound/checkpoints"),
        ("resnet_ultrasound/checkpoints", "resnet/checkpoints"),
    ]
    for orig, repl in replacements:
        if orig in path_str:
            alt = path_str.replace(orig, repl)
            if Path(alt).exists():
                return alt

    return str(path)


class MultimodalInferenceEngine:
    """Production-grade inference engine supporting both single-modality and multimodal models."""

    # Validation-calibrated decision thresholds
    THRESHOLDS = {
        "graph_transformer": 0.2565,
        "densenet": 0.2932,
        "resnet": 0.3844
    }
    DEFAULT_THRESHOLD = 0.2565  # Default multimodal threshold
    TARGET_IMAGE_SIZE = (224, 224)

    def __init__(
        self,
        config_path: str = "configs/config.yaml",
        gt_checkpoint_path: str = "outputs/experiments/graph_transformer/checkpoints/best_model.pth",
        densenet_checkpoint_path: str = "outputs/experiments/densenet/checkpoints/best_model.pth",
        resnet_checkpoint_path: str = "outputs/experiments/resnet/checkpoints/best_model.pth",
        device: Optional[str] = None,
        # Backwards-compatibility alias:
        checkpoint_path: Optional[str] = None
    ):
        if checkpoint_path is not None:
            gt_checkpoint_path = checkpoint_path

        self.config_path = config_path
        self.gt_checkpoint_path = resolve_checkpoint_path(gt_checkpoint_path)
        self.densenet_checkpoint_path = resolve_checkpoint_path(densenet_checkpoint_path)
        self.resnet_checkpoint_path = resolve_checkpoint_path(resnet_checkpoint_path)

        # 1. Load configuration
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        # 2. Select device
        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() and self.config.get("training", {}).get("device") != "cpu" else "cpu"
            )
        else:
            self.device = torch.device(device)

        # 3. Preprocessing transforms (exact evaluation transforms)
        self.mammo_transform = get_mammogram_transforms(
            image_size=self.TARGET_IMAGE_SIZE,
            is_training=False
        )
        self.us_transform = get_ultrasound_transforms(
            image_size=self.TARGET_IMAGE_SIZE,
            is_training=False
        )

        # 4. Load models and checkpoints
        self.models_meta: Dict[str, Dict[str, Any]] = {}

        # Graph Transformer
        self.gt_model, gt_meta = self._load_model_checkpoint(
            model_type="graph_transformer",
            checkpoint_path=self.gt_checkpoint_path,
            arch_name="Multimodal Graph-Infused Transformer",
            calibrated_threshold=self.THRESHOLDS["graph_transformer"]
        )
        self.models_meta["graph_transformer"] = gt_meta
        self.model = self.gt_model  # Backwards compatibility
        self.checkpoint_meta = gt_meta  # Backwards compatibility

        # DenseNet-121 (Mammogram unimodal)
        self.densenet_model, densenet_meta = self._load_model_checkpoint(
            model_type="densenet",
            checkpoint_path=self.densenet_checkpoint_path,
            arch_name="DenseNet-121",
            calibrated_threshold=self.THRESHOLDS["densenet"]
        )
        self.models_meta["densenet"] = densenet_meta

        # ResNet-50 (Ultrasound unimodal)
        self.resnet_model, resnet_meta = self._load_model_checkpoint(
            model_type="resnet",
            checkpoint_path=self.resnet_checkpoint_path,
            arch_name="ResNet-50",
            calibrated_threshold=self.THRESHOLDS["resnet"]
        )
        self.models_meta["resnet"] = resnet_meta

    def _load_model_checkpoint(
        self,
        model_type: str,
        checkpoint_path: str,
        arch_name: str,
        calibrated_threshold: float
    ) -> Tuple[torch.nn.Module, Dict[str, Any]]:
        resolved_path = resolve_checkpoint_path(checkpoint_path)
        if not os.path.exists(resolved_path):
            raise FileNotFoundError(f"Checkpoint for {arch_name} not found at: {resolved_path}")

        checkpoint = torch.load(resolved_path, map_location=self.device)

        epoch = checkpoint.get("epoch", "Unknown")
        val_metrics = checkpoint.get("val_metrics", {})
        is_smoke = (
            checkpoint.get("is_smoke_test", False)
            or checkpoint.get("config", {}).get("training", {}).get("is_smoke_test", False)
        )

        if is_smoke or (epoch == 1 and val_metrics.get("confusion_matrix", {}).get("tn", 0) <= 2):
            raise ValueError(
                f"Safety Violation: Checkpoint at {resolved_path} is from a smoke test. "
                f"Expected full training checkpoint for {arch_name}."
            )

        model = build_model(model_type, self.config)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)

        model.to(self.device)
        model.eval()

        meta = {
            "epoch": epoch,
            "architecture": arch_name,
            "model_type": model_type,
            "checkpoint_path": resolved_path,
            "calibrated_threshold": calibrated_threshold,
            "device": str(self.device),
            "val_roc_auc": float(val_metrics.get("roc_auc", 0.0)) if val_metrics else None
        }

        return model, meta

    def preprocess_image(self, image_input: Any, modality: str = "mammogram") -> torch.Tensor:
        """Loads and applies exact evaluation preprocessing to an image input."""
        if isinstance(image_input, bytes):
            image = Image.open(io.BytesIO(image_input))
        elif hasattr(image_input, "read"):
            image = Image.open(image_input)
        elif isinstance(image_input, Image.Image):
            image = image_input
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")

        rgb_image = image.convert("RGB")

        if modality.lower().startswith("mammo"):
            tensor = self.mammo_transform(rgb_image)
        elif modality.lower().startswith("us") or modality.lower().startswith("ultra"):
            tensor = self.us_transform(rgb_image)
        else:
            raise ValueError(f"Unknown modality: {modality}")

        return tensor.unsqueeze(0)  # [1, 3, H, W]

    def predict(
        self,
        mammo_input: Optional[Any] = None,
        us_input: Optional[Any] = None,
        threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """Performs classification inference for Mammogram only, Ultrasound only, or both.

        Args:
            mammo_input: Optional Mammogram image bytes or PIL Image
            us_input: Optional Ultrasound image bytes or PIL Image
            threshold: Optional custom classification threshold

        Returns:
            Dict containing prediction, probabilities, threshold, metadata, and explainability.
        """
        has_mammo = mammo_input is not None
        has_us = us_input is not None

        if not has_mammo and not has_us:
            raise ValueError("No input image provided. Please upload at least one scan (Mammogram or Ultrasound).")

        # Route to appropriate model
        if has_mammo and has_us:
            return self._predict_multimodal(mammo_input, us_input, threshold)
        elif has_mammo:
            return self._predict_mammogram_only(mammo_input, threshold)
        else:
            return self._predict_ultrasound_only(us_input, threshold)

    def _predict_multimodal(
        self,
        mammo_input: Any,
        us_input: Any,
        threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """Runs true multimodal inference using the Graph Transformer."""
        thresh = threshold if threshold is not None else self.THRESHOLDS["graph_transformer"]

        mammo_tensor = self.preprocess_image(mammo_input, modality="mammogram").to(self.device)
        us_tensor = self.preprocess_image(us_input, modality="ultrasound").to(self.device)

        batch_size = 1
        m_idx = torch.zeros(1, dtype=torch.long, device=self.device)
        u_idx = torch.zeros(1, dtype=torch.long, device=self.device)

        self.gt_model.eval()
        with torch.no_grad():
            logits, explanations = self.gt_model(
                mammo_tensor, m_idx, us_tensor, u_idx, batch_size=batch_size, return_explanations=True
            )
            probs = F.softmax(logits, dim=1)[0]
            prob_benign = float(probs[0].item())
            prob_malignant = float(probs[1].item())

        is_suspicious = prob_malignant >= thresh
        pred_class_id = 1 if is_suspicious else 0
        pred_class_name = (
            "BI-RADS-derived positive/suspicious class" if is_suspicious else "BI-RADS-derived negative/benign class"
        )

        explainability_dict = self._generate_multimodal_explainability(
            mammo_tensor=mammo_tensor,
            us_tensor=us_tensor,
            explanations=explanations,
            prob_malignant=prob_malignant,
            threshold=thresh
        )

        return {
            "prediction": {
                "class_id": pred_class_id,
                "class_name": pred_class_name,
                "is_positive_suspicious": bool(is_suspicious),
                "prob_malignant": float(prob_malignant),
                "prob_benign": float(prob_benign),
                "calibrated_threshold": float(thresh),
                "confidence_level": "High" if abs(prob_malignant - thresh) > 0.08 else "Moderate",
                "clinical_guidance": (
                    "Model prediction exceeds the validation-calibrated operating threshold. "
                    "Multi-disciplinary review recommended."
                    if is_suspicious else
                    "Model prediction is below the validation-calibrated operating threshold."
                )
            },
            "analysis_mode": "Multimodal Analysis",
            "model_name": "Graph Transformer",
            "model_metadata": {
                "architecture": "Multimodal Graph-Infused Transformer",
                "model_name": "Graph Transformer",
                "checkpoint_epoch": self.models_meta["graph_transformer"]["epoch"],
                "calibrated_threshold": float(thresh),
                "device": str(self.device),
                "val_roc_auc": self.models_meta["graph_transformer"].get("val_roc_auc")
            },
            "explainability": explainability_dict,
            "disclaimer": "Research prototype — not for clinical diagnosis."
        }

    def _predict_mammogram_only(
        self,
        mammo_input: Any,
        threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """Runs unimodal mammogram inference using DenseNet-121."""
        thresh = threshold if threshold is not None else self.THRESHOLDS["densenet"]

        mammo_tensor = self.preprocess_image(mammo_input, modality="mammogram").to(self.device)

        self.densenet_model.eval()
        with torch.no_grad():
            batch_indices = torch.zeros(1, dtype=torch.long, device=self.device)
            logits, _ = self.densenet_model(mammo_tensor, batch_indices=batch_indices, batch_size=1)
            probs = F.softmax(logits, dim=1)[0]
            prob_benign = float(probs[0].item())
            prob_malignant = float(probs[1].item())

        is_suspicious = prob_malignant >= thresh
        pred_class_id = 1 if is_suspicious else 0
        pred_class_name = (
            "BI-RADS-derived positive/suspicious class" if is_suspicious else "BI-RADS-derived negative/benign class"
        )

        explainability_dict = self._generate_unimodal_explainability(
            model=self.densenet_model,
            image_tensor=mammo_tensor,
            modality="mammogram",
            target_class=pred_class_id
        )

        return {
            "prediction": {
                "class_id": pred_class_id,
                "class_name": pred_class_name,
                "is_positive_suspicious": bool(is_suspicious),
                "prob_malignant": float(prob_malignant),
                "prob_benign": float(prob_benign),
                "calibrated_threshold": float(thresh),
                "confidence_level": "High" if abs(prob_malignant - thresh) > 0.08 else "Moderate",
                "clinical_guidance": (
                    "Model prediction exceeds the validation-calibrated operating threshold. "
                    "Multi-disciplinary review recommended."
                    if is_suspicious else
                    "Model prediction is below the validation-calibrated operating threshold."
                )
            },
            "analysis_mode": "Single-Modality Analysis — Mammogram",
            "model_name": "DenseNet-121",
            "model_metadata": {
                "architecture": "DenseNet-121",
                "model_name": "DenseNet-121",
                "checkpoint_epoch": self.models_meta["densenet"]["epoch"],
                "calibrated_threshold": float(thresh),
                "device": str(self.device),
                "val_roc_auc": self.models_meta["densenet"].get("val_roc_auc")
            },
            "explainability": explainability_dict,
            "disclaimer": "Research prototype — not for clinical diagnosis."
        }

    def _predict_ultrasound_only(
        self,
        us_input: Any,
        threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """Runs unimodal ultrasound inference using ResNet-50."""
        thresh = threshold if threshold is not None else self.THRESHOLDS["resnet"]

        us_tensor = self.preprocess_image(us_input, modality="ultrasound").to(self.device)

        self.resnet_model.eval()
        with torch.no_grad():
            batch_indices = torch.zeros(1, dtype=torch.long, device=self.device)
            logits, _ = self.resnet_model(us_tensor, batch_indices=batch_indices, batch_size=1)
            probs = F.softmax(logits, dim=1)[0]
            prob_benign = float(probs[0].item())
            prob_malignant = float(probs[1].item())

        is_suspicious = prob_malignant >= thresh
        pred_class_id = 1 if is_suspicious else 0
        pred_class_name = (
            "BI-RADS-derived positive/suspicious class" if is_suspicious else "BI-RADS-derived negative/benign class"
        )

        explainability_dict = self._generate_unimodal_explainability(
            model=self.resnet_model,
            image_tensor=us_tensor,
            modality="ultrasound",
            target_class=pred_class_id
        )

        return {
            "prediction": {
                "class_id": pred_class_id,
                "class_name": pred_class_name,
                "is_positive_suspicious": bool(is_suspicious),
                "prob_malignant": float(prob_malignant),
                "prob_benign": float(prob_benign),
                "calibrated_threshold": float(thresh),
                "confidence_level": "High" if abs(prob_malignant - thresh) > 0.08 else "Moderate",
                "clinical_guidance": (
                    "Model prediction exceeds the validation-calibrated operating threshold. "
                    "Multi-disciplinary review recommended."
                    if is_suspicious else
                    "Model prediction is below the validation-calibrated operating threshold."
                )
            },
            "analysis_mode": "Single-Modality Analysis — Ultrasound",
            "model_name": "ResNet-50",
            "model_metadata": {
                "architecture": "ResNet-50",
                "model_name": "ResNet-50",
                "checkpoint_epoch": self.models_meta["resnet"]["epoch"],
                "calibrated_threshold": float(thresh),
                "device": str(self.device),
                "val_roc_auc": self.models_meta["resnet"].get("val_roc_auc")
            },
            "explainability": explainability_dict,
            "disclaimer": "Research prototype — not for clinical diagnosis."
        }

    def _generate_unimodal_explainability(
        self,
        model: torch.nn.Module,
        image_tensor: torch.Tensor,
        modality: str,
        target_class: int
    ) -> Dict[str, Any]:
        """Generates Grad-CAM saliency map for single-modality models."""
        if modality == "mammogram":
            target_layer = model.features.denseblock4
            modality_title = "Mammogram (DenseNet-121)"
        else:
            target_layer = model.layer4
            modality_title = "Ultrasound (ResNet-50)"

        cam_generator = GradCAM(model=model, target_layer=target_layer)
        try:
            cam_heatmap = cam_generator.generate_cam(image_tensor, target_class=target_class)
        finally:
            cam_generator.remove_hooks()

        # Un-normalize image for visualization: [3, H, W] -> [H, W, 3]
        img = image_tensor.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = np.clip(std * img + mean, 0, 1)

        fig = plt.figure(figsize=(12, 4), facecolor="#0F172A")
        plt.rcParams["text.color"] = "#E2E8F0"
        plt.rcParams["axes.labelcolor"] = "#94A3B8"
        plt.rcParams["xtick.color"] = "#94A3B8"
        plt.rcParams["ytick.color"] = "#94A3B8"

        # Panel 1: Input Image
        ax1 = fig.add_subplot(1, 3, 1)
        ax1.imshow(img)
        ax1.set_title(f"Input: {modality_title}", fontsize=11, fontweight="bold", color="#38BDF8", pad=8)
        ax1.axis("off")

        # Panel 2: Grad-CAM Activation Heatmap
        ax2 = fig.add_subplot(1, 3, 2)
        im2 = ax2.imshow(cam_heatmap, cmap="magma", vmin=0, vmax=1)
        ax2.set_title("Grad-CAM Activation Map", fontsize=11, fontweight="bold", color="#F43F5E", pad=8)
        ax2.axis("off")
        cbar2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        cbar2.ax.tick_params(colors="#94A3B8")

        # Panel 3: Grad-CAM Saliency Overlay
        ax3 = fig.add_subplot(1, 3, 3)
        ax3.imshow(img)
        ax3.imshow(cam_heatmap, cmap="magma", alpha=0.5, vmin=0, vmax=1)
        ax3.set_title("Saliency Overlay", fontsize=11, fontweight="bold", color="#10B981", pad=8)
        ax3.axis("off")

        fig.suptitle(
            f"{modality_title} Grad-CAM Saliency Analysis\n"
            "[Research Prototype — Not for Clinical Diagnosis]",
            fontsize=12, fontweight="bold", color="#F1F5F9", y=1.05
        )
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor(), edgecolor="none")
        plt.close(fig)
        buf.seek(0)
        b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")

        return {
            "type": "gradcam",
            "plot_base64": f"data:image/png;base64,{b64_str}",
            "attention_weights": {},
            "details": f"Grad-CAM activation mapping targeting discriminative regions for {modality_title}."
        }

    def _generate_multimodal_explainability(
        self,
        mammo_tensor: torch.Tensor,
        us_tensor: torch.Tensor,
        explanations: Optional[Dict[str, Any]],
        prob_malignant: float,
        threshold: float
    ) -> Dict[str, Any]:
        """Generates cross-modality attention heatmap plot and encodes as base64 PNG."""
        if explanations is None:
            return {"type": "multimodal_attention", "plot_base64": None, "attention_weights": {}}

        gat_attn = explanations["gat_attention"].cpu().numpy()  # [1, heads, 2, 2]
        trans_attn = explanations["transformer_attention"].cpu().numpy()  # [1, 2, 2]

        gat_attn_0 = gat_attn[0]  # [heads, 2, 2]
        trans_attn_0 = trans_attn[0]  # [2, 2]

        # Extract un-normalized images for visual overlay
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])

        m_img = mammo_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        m_img = np.clip(std * m_img + mean, 0, 1)

        u_img = us_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        u_img = np.clip(std * u_img + mean, 0, 1)

        # Plot 4 panels:
        # 1. Mammogram Scan
        # 2. Ultrasound Scan
        # 3. Average GAT Cross-Attention Matrix
        # 4. Transformer Self-Attention Matrix
        fig = plt.figure(figsize=(14, 4), facecolor="#0F172A")
        plt.rcParams["text.color"] = "#E2E8F0"
        plt.rcParams["axes.labelcolor"] = "#94A3B8"
        plt.rcParams["xtick.color"] = "#94A3B8"
        plt.rcParams["ytick.color"] = "#94A3B8"

        labels = ["Mammogram", "Ultrasound"]

        # Panel 1: Input Mammogram
        ax1 = fig.add_subplot(1, 4, 1)
        ax1.imshow(m_img)
        ax1.set_title("Input: Mammogram (X-ray)", fontsize=11, fontweight="bold", color="#38BDF8", pad=8)
        ax1.axis("off")

        # Panel 2: Input Ultrasound
        ax2 = fig.add_subplot(1, 4, 2)
        ax2.imshow(u_img)
        ax2.set_title("Input: Ultrasound (US)", fontsize=11, fontweight="bold", color="#38BDF8", pad=8)
        ax2.axis("off")

        # Panel 3: GAT Multi-Head Attention (Averaged across heads)
        avg_gat = np.mean(gat_attn_0, axis=0)  # [2, 2]
        ax3 = fig.add_subplot(1, 4, 3)
        im3 = ax3.imshow(avg_gat, cmap="magma", vmin=0, vmax=1)
        ax3.set_xticks([0, 1])
        ax3.set_yticks([0, 1])
        ax3.set_xticklabels(labels, fontsize=9)
        ax3.set_yticklabels(labels, fontsize=9)
        ax3.set_title("GAT Cross-Attention (Avg)", fontsize=11, fontweight="bold", color="#F43F5E", pad=8)
        for i in range(2):
            for j in range(2):
                ax3.text(
                    j, i, f"{avg_gat[i, j]:.2f}",
                    ha="center", va="center", color="#FFFFFF", fontweight="bold", fontsize=10
                )
        cbar3 = fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
        cbar3.ax.tick_params(colors="#94A3B8")

        # Panel 4: Transformer Self-Attention Matrix
        ax4 = fig.add_subplot(1, 4, 4)
        im4 = ax4.imshow(trans_attn_0, cmap="viridis", vmin=0, vmax=1)
        ax4.set_xticks([0, 1])
        ax4.set_yticks([0, 1])
        ax4.set_xticklabels(labels, fontsize=9)
        ax4.set_yticklabels(labels, fontsize=9)
        ax4.set_title("Transformer Fusion Attention", fontsize=11, fontweight="bold", color="#10B981", pad=8)
        for i in range(2):
            for j in range(2):
                ax4.text(
                    j, i, f"{trans_attn_0[i, j]:.2f}",
                    ha="center", va="center", color="#FFFFFF", fontweight="bold", fontsize=10
                )
        cbar4 = fig.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
        cbar4.ax.tick_params(colors="#94A3B8")

        fig.suptitle(
            "Multimodal Attention & Token Interaction Visualization\n"
            "[Research Prototype — Not for Clinical Diagnosis]",
            fontsize=12, fontweight="bold", color="#F1F5F9", y=1.05
        )
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor(), edgecolor="none")
        plt.close(fig)
        buf.seek(0)
        b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")

        return {
            "type": "multimodal_attention",
            "plot_base64": f"data:image/png;base64,{b64_str}",
            "attention_weights": {
                "gat_mammogram_self": float(avg_gat[0, 0]),
                "gat_mammo_to_us": float(avg_gat[0, 1]),
                "gat_us_to_mammo": float(avg_gat[1, 0]),
                "gat_ultrasound_self": float(avg_gat[1, 1]),
                "transformer_mammo_self": float(trans_attn_0[0, 0]),
                "transformer_mammo_to_us": float(trans_attn_0[0, 1]),
                "transformer_us_to_mammo": float(trans_attn_0[1, 0]),
                "transformer_us_self": float(trans_attn_0[1, 1])
            }
        }
