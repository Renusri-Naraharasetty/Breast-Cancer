import torch
import torch.nn as nn
from torchvision.models import densenet121, DenseNet121_Weights
from .pooling import AttentionPooling


class MammogramDenseNet(nn.Module):
    """DenseNet-121 model for Mammography classification with intermediate feature extraction.
    
    Uses ImageNet-pretrained weights, exposes 1024-dimensional feature embeddings,
    and pools multiple views per patient using learnable attention pooling.
    """
    def __init__(
        self,
        pretrained: bool = True,
        feature_dim: int = 1024,
        num_classes: int = 2,
        dropout: float = 0.3,
        freeze_backbone: bool = False
    ):
        super().__init__()
        weights = DenseNet121_Weights.DEFAULT if pretrained else None
        base_model = densenet121(weights=weights)
        
        # DenseNet feature extractor (convolutional + pooling layers)
        self.features = base_model.features
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.feature_dim = feature_dim

        # Attention pooling for multiple mammogram views per patient
        self.pooler = AttentionPooling(input_dim=feature_dim)

        # Standalone classification head
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes)
        )

        self.is_frozen = False
        if freeze_backbone:
            self.freeze_backbone()

    def freeze_backbone(self) -> None:
        """Freezes DenseNet-121 convolutional backbone parameters."""
        self.is_frozen = True
        self.features.eval()
        for param in self.features.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Unfreezes DenseNet-121 convolutional backbone parameters."""
        self.is_frozen = False
        for param in self.features.parameters():
            param.requires_grad = True

    def train(self, mode: bool = True):
        super().train(mode)
        if mode and self.is_frozen:
            self.features.eval()
        return self

    def extract_image_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts 1024-dimensional feature embeddings from a batch of images.
        
        Args:
            x: [N, 3, H, W]
        Returns:
            embeddings: [N, 1024]
        """
        out = self.features(x)
        out = self.relu(out)
        out = self.avgpool(out)
        embeddings = torch.flatten(out, 1)
        return embeddings

    def forward(
        self,
        images: torch.Tensor,
        batch_indices: torch.Tensor,
        batch_size: int
    ) -> torch.Tensor:
        """Forward pass for patient-level mammogram classification.
        
        Args:
            images: [Total_Images, 3, H, W]
            batch_indices: [Total_Images] mapping each image to patient index (0 to batch_size - 1)
            batch_size: Number of patients
            
        Returns:
            logits: [batch_size, num_classes]
        """
        embeddings = self.extract_image_features(images)
        patient_features = self.pooler(embeddings, batch_indices, batch_size)
        logits = self.classifier(patient_features)
        return logits, patient_features
