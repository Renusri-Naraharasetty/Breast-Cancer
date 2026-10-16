import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights
from typing import Optional


class ViTB16Binary(nn.Module):
    """Vision Transformer (ViT-B/16) architecture adapted for binary breast cancer triage.

    Input resolution: 224x224.
    Patch size: 16x16 (yielding 14x14 = 196 image patches + 1 CLS token = 197 tokens).
    Hidden representation: 768 dimensions across 12 Transformer encoder blocks.
    Replaces 1000-class ImageNet head with a dropout-regularized binary classifier.
    """

    def __init__(
        self,
        pretrained: bool = False,
        num_classes: int = 2,
        dropout_rate: float = 0.3,
    ):
        super().__init__()
        weights = ViT_B_16_Weights.DEFAULT if pretrained else None
        self.backbone = vit_b_16(weights=weights)

        in_features = self.backbone.heads.head.in_features  # 768
        self.backbone.heads = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, num_classes),
        )

        self.num_classes = num_classes
        self.dropout_rate = dropout_rate

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts the 768-dimensional CLS token representation before classification head."""
        x = self.backbone._process_input(x)
        n = x.shape[0]

        batch_class_token = self.backbone.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.backbone.encoder(x)

        # Return CLS token (index 0)
        return x[:, 0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning raw class logits of shape (batch_size, num_classes)."""
        return self.backbone(x)
