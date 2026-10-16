import torch.nn as nn
from typing import Dict, Any
from .efficientnet import EfficientNetB0Binary
from .vit import ViTB16Binary
from .git import GraphInfusedTransformer


def create_model(
    model_name: str,
    pretrained: bool = False,
    num_classes: int = 2,
    dropout_rate: float = 0.3,
    **kwargs,
) -> nn.Module:
    """Factory function to build vision backbones and multimodal GIT models for BCMID experiments.

    Supported model names:
      - 'efficientnet_b0' or 'efficientnet'
      - 'vit_b_16' or 'vit'
      - 'git' or 'graph_infused_transformer' or 'multimodal_git'
    """
    normalized_name = model_name.lower().replace("-", "_")

    if normalized_name in ["efficientnet_b0", "efficientnet"]:
        return EfficientNetB0Binary(
            pretrained=pretrained,
            num_classes=num_classes,
            dropout_rate=dropout_rate,
        )
    elif normalized_name in ["vit_b_16", "vit", "vit_base_patch16_224"]:
        return ViTB16Binary(
            pretrained=pretrained,
            num_classes=num_classes,
            dropout_rate=dropout_rate,
        )
    elif normalized_name in ["git", "graph_infused_transformer", "multimodal_git"]:
        return GraphInfusedTransformer(
            num_classes=num_classes,
            dropout_rate=dropout_rate,
            **kwargs,
        )
    else:
        raise ValueError(
            f"Unknown model_name: '{model_name}'. Supported models: ['efficientnet_b0', 'vit_b_16', 'git']."
        )


def get_model_info(model: nn.Module) -> Dict[str, Any]:
    """Computes parameter counts and model footprint in megabytes."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)

    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "model_size_mb": round(model_size_mb, 2),
    }
