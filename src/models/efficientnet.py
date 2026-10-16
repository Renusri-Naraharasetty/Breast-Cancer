import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from typing import Optional, Tuple


class EfficientNetB0Binary(nn.Module):
    """EfficientNet-B0 architecture adapted for binary breast cancer triage.

    Replaces the 1000-class ImageNet head with a dropout-regularized binary classifier.
    Exposes feature extraction for Grad-CAM explainability and latent representations.
    """

    def __init__(
        self,
        pretrained: bool = False,
        num_classes: int = 2,
        dropout_rate: float = 0.3,
    ):
        super().__init__()
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        self.backbone = efficientnet_b0(weights=weights)

        in_features = self.backbone.classifier[1].in_features  # 1280
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate, inplace=True),
            nn.Linear(in_features, num_classes),
        )

        self.num_classes = num_classes
        self.dropout_rate = dropout_rate

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts the 1280-dimensional pooled feature vector before the classifier."""
        features = self.backbone.features(x)
        pooled = self.backbone.avgpool(features)
        return torch.flatten(pooled, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning raw class logits of shape (batch_size, num_classes)."""
        return self.backbone(x)
