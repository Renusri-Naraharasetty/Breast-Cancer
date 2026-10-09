import torch
import torch.nn as nn
from typing import Dict, Any
from .densenet_mammogram import MammogramDenseNet
from .resnet_ultrasound import UltrasoundResNet


class SimpleFusionBaseline(nn.Module):
    """Early/Late Concatenation Fusion Baseline for comparison against Graph-Infused Transformer.
    
    Concatenates attention-pooled Mammogram (DenseNet-121) and Ultrasound (ResNet-50) features
    without graph modeling or cross-modal attention, followed by an MLP classifier.
    """
    def __init__(
        self,
        pretrained: bool = True,
        mammo_dim: int = 1024,
        us_dim: int = 2048,
        hidden_dim: int = 256,
        num_classes: int = 2,
        dropout: float = 0.3,
        freeze_backbones: bool = False
    ):
        super().__init__()
        self.mammo_net = MammogramDenseNet(
            pretrained=pretrained,
            feature_dim=mammo_dim,
            num_classes=num_classes,
            dropout=dropout,
            freeze_backbone=freeze_backbones
        )
        self.us_net = UltrasoundResNet(
            pretrained=pretrained,
            feature_dim=us_dim,
            num_classes=num_classes,
            dropout=dropout,
            freeze_backbone=freeze_backbones
        )
        
        fused_dim = mammo_dim + us_dim
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(fused_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def freeze_backbones(self) -> None:
        self.mammo_net.freeze_backbone()
        self.us_net.freeze_backbone()

    def unfreeze_backbones(self) -> None:
        self.mammo_net.unfreeze_backbone()
        self.us_net.unfreeze_backbone()

    def train(self, mode: bool = True):
        super().train(mode)
        if mode:
            if getattr(self.mammo_net, "is_frozen", False):
                self.mammo_net.features.eval()
            if getattr(self.us_net, "is_frozen", False):
                layers = [self.us_net.conv1, self.us_net.bn1, self.us_net.layer1, self.us_net.layer2, self.us_net.layer3, self.us_net.layer4]
                for layer in layers:
                    layer.eval()
        return self

    def forward(
        self,
        mammo_images: torch.Tensor,
        mammo_batch_indices: torch.Tensor,
        us_images: torch.Tensor,
        us_batch_indices: torch.Tensor,
        batch_size: int
    ) -> torch.Tensor:
        """Args:
            mammo_images: [Total_M, 3, H, W]
            mammo_batch_indices: [Total_M]
            us_images: [Total_U, 3, H, W]
            us_batch_indices: [Total_U]
            batch_size: Number of patients
            
        Returns:
            logits: [batch_size, num_classes]
        """
        device = next(self.parameters()).device
        
        # 1. Extract and pool Mammogram features
        if mammo_images.numel() > 0:
            m_embeds = self.mammo_net.extract_image_features(mammo_images)
            m_pooled = self.mammo_net.pooler(m_embeds, mammo_batch_indices, batch_size)
        else:
            m_pooled = torch.zeros(batch_size, self.mammo_net.feature_dim, device=device)

        # 2. Extract and pool Ultrasound features
        if us_images.numel() > 0:
            u_embeds = self.us_net.extract_image_features(us_images)
            u_pooled = self.us_net.pooler(u_embeds, us_batch_indices, batch_size)
        else:
            u_pooled = torch.zeros(batch_size, self.us_net.feature_dim, device=device)

        # 3. Concatenation fusion
        fused = torch.cat([m_pooled, u_pooled], dim=1)

        # 4. Classification
        logits = self.classifier(fused)
        return logits
