import torch
import torch.nn as nn
from typing import Tuple, Dict, Any, List, Optional
from .densenet_mammogram import MammogramDenseNet
from .resnet_ultrasound import UltrasoundResNet
from .graph import MultimodalGraphBuilder
from .gat import MultimodalGAT
from .transformer import GraphInfusedTransformer


class MultimodalGraphTransformer(nn.Module):
    """PRIMARY PROPOSED MODEL: Multimodal Graph-Infused Transformer for Breast Cancer Classification.
    
    Architecture:
        1. Mammogram images -> DenseNet-121 -> Feature embeddings (1024-d)
        2. Ultrasound images -> ResNet-50 -> Feature embeddings (2048-d)
        3. Learnable Attention Pooling across multiple patient images -> Patient representations
        4. Multimodal Graph Construction (Mammogram node + Ultrasound node + bidirectional edges)
        5. Multi-Head GATv2 information propagation -> Graph-enhanced modality tokens
        6. Graph-Infused Transformer (Multi-Head Self-Attention over graph tokens)
        7. Multimodal Fusion with residual connections
        8. Binary Classification Head -> Benign vs Malignant
    """
    def __init__(
        self,
        pretrained: bool = True,
        mammo_dim: int = 1024,
        us_dim: int = 2048,
        hidden_dim: int = 256,
        gat_heads: int = 4,
        gat_layers: int = 2,
        gat_dropout: float = 0.2,
        transformer_heads: int = 4,
        transformer_layers: int = 2,
        transformer_dim_feedforward: int = 512,
        transformer_dropout: float = 0.2,
        num_classes: int = 2,
        dropout: float = 0.3,
        freeze_backbones: bool = False
    ):
        super().__init__()
        self.mammo_dim = mammo_dim
        self.us_dim = us_dim
        self.hidden_dim = hidden_dim

        # 1. Backbones with attention poolers
        self.mammo_backbone = MammogramDenseNet(
            pretrained=pretrained,
            feature_dim=mammo_dim,
            num_classes=num_classes,
            dropout=dropout,
            freeze_backbone=freeze_backbones
        )
        self.us_backbone = UltrasoundResNet(
            pretrained=pretrained,
            feature_dim=us_dim,
            num_classes=num_classes,
            dropout=dropout,
            freeze_backbone=freeze_backbones
        )

        # 2. Graph builder (projects to hidden_dim & adds modality type embeddings)
        self.graph_builder = MultimodalGraphBuilder(
            mammo_dim=mammo_dim,
            us_dim=us_dim,
            hidden_dim=hidden_dim
        )

        # 3. Graph Attention Network (GATv2)
        self.gat = MultimodalGAT(
            hidden_dim=hidden_dim,
            num_layers=gat_layers,
            num_heads=gat_heads,
            dropout=gat_dropout
        )

        # 4. Graph-Infused Transformer
        self.transformer = GraphInfusedTransformer(
            d_model=hidden_dim,
            nhead=transformer_heads,
            num_layers=transformer_layers,
            dim_feedforward=transformer_dim_feedforward,
            dropout=transformer_dropout
        )

        # 5. Multimodal Fusion & Classification Head
        # Concatenate Mammogram & Ultrasound graph-transformer tokens [B, 2 * hidden_dim]
        fused_dim = hidden_dim * 2
        self.fusion_norm = nn.LayerNorm(fused_dim)
        
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(fused_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def freeze_backbones(self) -> None:
        """Freezes DenseNet-121 and ResNet-50 backbones."""
        self.mammo_backbone.freeze_backbone()
        self.us_backbone.freeze_backbone()

    def unfreeze_backbones(self) -> None:
        """Unfreezes DenseNet-121 and ResNet-50 backbones."""
        self.mammo_backbone.unfreeze_backbone()
        self.us_backbone.unfreeze_backbone()

    def train(self, mode: bool = True):
        super().train(mode)
        if mode:
            if getattr(self.mammo_backbone, "is_frozen", False):
                self.mammo_backbone.features.eval()
            if getattr(self.us_backbone, "is_frozen", False):
                layers = [self.us_backbone.conv1, self.us_backbone.bn1, self.us_backbone.layer1, self.us_backbone.layer2, self.us_backbone.layer3, self.us_backbone.layer4]
                for layer in layers:
                    layer.eval()
        return self

    def get_parameter_groups(self, lr: float, backbone_lr: float, weight_decay: float) -> List[Dict[str, Any]]:
        """Returns separate parameter groups with distinct learning rates for backbones vs fusion/graph."""
        backbone_params = list(self.mammo_backbone.features.parameters()) + \
                          list(self.us_backbone.conv1.parameters()) + \
                          list(self.us_backbone.bn1.parameters()) + \
                          list(self.us_backbone.layer1.parameters()) + \
                          list(self.us_backbone.layer2.parameters()) + \
                          list(self.us_backbone.layer3.parameters()) + \
                          list(self.us_backbone.layer4.parameters())
                          
        backbone_param_ids = set(id(p) for p in backbone_params)
        head_params = [p for p in self.parameters() if id(p) not in backbone_param_ids and p.requires_grad]

        return [
            {"params": [p for p in backbone_params if p.requires_grad], "lr": backbone_lr, "weight_decay": weight_decay},
            {"params": head_params, "lr": lr, "weight_decay": weight_decay}
        ]

    def forward(
        self,
        mammo_images: torch.Tensor,
        mammo_batch_indices: torch.Tensor,
        us_images: torch.Tensor,
        us_batch_indices: torch.Tensor,
        batch_size: int,
        return_explanations: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        """Full end-to-end forward pass.
        
        Args:
            mammo_images: [Total_M, 3, H, W]
            mammo_batch_indices: [Total_M]
            us_images: [Total_U, 3, H, W]
            us_batch_indices: [Total_U]
            batch_size: Number of patients in batch
            return_explanations: If True, returns attention weights for GAT and Transformer
            
        Returns:
            logits: [batch_size, num_classes]
            explanations: Dict containing attention weights if requested
        """
        device = next(self.parameters()).device

        # Step 1: Mammogram feature extraction & learnable attention pooling
        if mammo_images.numel() > 0:
            m_embeds = self.mammo_backbone.extract_image_features(mammo_images)
            m_patient = self.mammo_backbone.pooler(m_embeds, mammo_batch_indices, batch_size)
        else:
            m_patient = torch.zeros(batch_size, self.mammo_dim, device=device)

        # Step 2: Ultrasound feature extraction & learnable attention pooling
        if us_images.numel() > 0:
            u_embeds = self.us_backbone.extract_image_features(us_images)
            u_patient = self.us_backbone.pooler(u_embeds, us_batch_indices, batch_size)
        else:
            u_patient = torch.zeros(batch_size, self.us_dim, device=device)

        # Step 3: Multimodal Graph Construction
        initial_tokens, adj_mask = self.graph_builder(m_patient, u_patient) # [B, 2, hidden_dim]

        # Step 4: Multi-Head Graph Attention Network (GATv2)
        graph_tokens, gat_attn = self.gat(initial_tokens, adj_mask) # [B, 2, hidden_dim]

        # Step 5: Graph-Infused Transformer (Self-Attention over graph tokens)
        transformer_tokens, trans_attns = self.transformer(graph_tokens) # [B, 2, hidden_dim]

        # Step 6: Multimodal Fusion with residual from initial graph projection
        # Residual fusion: (Transformer output + initial projection)
        fused_tokens = transformer_tokens + initial_tokens
        
        # Flatten modality tokens: [B, 2 * hidden_dim]
        fused_representation = fused_tokens.view(batch_size, -1)
        fused_representation = self.fusion_norm(fused_representation)

        # Step 7: Classification Head
        logits = self.classifier(fused_representation)

        if return_explanations:
            explanations = {
                "gat_attention": gat_attn,                 # [B, heads, 2, 2]
                "transformer_attention": trans_attns[-1],  # [B, 2, 2]
                "mammo_patient_features": m_patient,
                "us_patient_features": u_patient
            }
            return logits, explanations

        return logits, None
