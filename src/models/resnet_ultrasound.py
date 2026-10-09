import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from .pooling import AttentionPooling


class UltrasoundResNet(nn.Module):
    """ResNet-50 model for Ultrasound classification with intermediate feature extraction.
    
    Uses ImageNet-pretrained weights, exposes 2048-dimensional feature embeddings,
    and pools multiple frames per patient using learnable attention pooling.
    """
    def __init__(
        self,
        pretrained: bool = True,
        feature_dim: int = 2048,
        num_classes: int = 2,
        dropout: float = 0.3,
        freeze_backbone: bool = False
    ):
        super().__init__()
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        base_model = resnet50(weights=weights)

        # ResNet feature extractor: layers up to avgpool
        self.conv1 = base_model.conv1
        self.bn1 = base_model.bn1
        self.relu = base_model.relu
        self.maxpool = base_model.maxpool

        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4

        self.avgpool = base_model.avgpool
        self.feature_dim = feature_dim

        # Attention pooling for multiple ultrasound frames per patient
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
        """Freezes ResNet-50 convolutional backbone parameters."""
        self.is_frozen = True
        layers = [self.conv1, self.bn1, self.layer1, self.layer2, self.layer3, self.layer4]
        for layer in layers:
            layer.eval()
            for param in layer.parameters():
                param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Unfreezes ResNet-50 convolutional backbone parameters."""
        self.is_frozen = False
        layers = [self.conv1, self.bn1, self.layer1, self.layer2, self.layer3, self.layer4]
        for layer in layers:
            for param in layer.parameters():
                param.requires_grad = True

    def train(self, mode: bool = True):
        super().train(mode)
        if mode and self.is_frozen:
            layers = [self.conv1, self.bn1, self.layer1, self.layer2, self.layer3, self.layer4]
            for layer in layers:
                layer.eval()
        return self

    def extract_image_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts 2048-dimensional feature embeddings from a batch of ultrasound images.
        
        Args:
            x: [N, 3, H, W]
        Returns:
            embeddings: [N, 2048]
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        embeddings = torch.flatten(x, 1)
        return embeddings

    def forward(
        self,
        images: torch.Tensor,
        batch_indices: torch.Tensor,
        batch_size: int
    ) -> torch.Tensor:
        """Forward pass for patient-level ultrasound classification.
        
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
