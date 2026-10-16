import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple, Any

from .gat import MultimodalGAT, GraphAttentionLayer
from .efficientnet import EfficientNetB0Binary
from .vit import ViTB16Binary


SIDE_MAP = {"L": 0, "R": 1, "BIL": 2, "UNK": 3}


class AttentivePooling(nn.Module):
    """Attention-weighted pooling over a variable number of view tokens."""

    def __init__(self, hidden_dim: int, dropout_rate: float = 0.1):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Args:

        x: (batch_size, num_views, hidden_dim) mask: (batch_size, num_views)
        boolean mask (True=valid, False=padding)
        Returns:
            pooled: (batch_size, hidden_dim)
        """
        # scores: (batch_size, num_views, 1)
        scores = self.attn(x)
        if mask is not None:
            scores = scores.masked_fill(~mask.unsqueeze(-1), -1e4)
        weights = F.softmax(scores, dim=1)
        weights = torch.nan_to_num(weights, nan=0.0)
        pooled = (x * weights).sum(dim=1)
        return pooled


class CrossModalAttention(nn.Module):
    """Bi-directional cross-attention between two modalities."""

    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout_rate: float = 0.1):
        super().__init__()
        self.mha_mg_to_us = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout_rate, batch_first=True
        )
        self.mha_us_to_mg = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout_rate, batch_first=True
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        mg_summary: torch.Tensor,  # (B, 1, D)
        us_summary: torch.Tensor,  # (B, 1, D)
        mg_views: torch.Tensor,    # (B, M, D)
        us_views: torch.Tensor,    # (B, U, D)
        mg_mask: Optional[torch.Tensor] = None,  # (B, M)
        us_mask: Optional[torch.Tensor] = None,  # (B, U)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Mammogram summary queries ultrasound views
        us_key_padding = ~us_mask if us_mask is not None else None
        cross_mg, _ = self.mha_mg_to_us(
            query=mg_summary, key=us_views, value=us_views, key_padding_mask=us_key_padding
        )
        cross_mg = self.norm1(mg_summary + cross_mg)

        # Ultrasound summary queries mammogram views
        mg_key_padding = ~mg_mask if mg_mask is not None else None
        cross_us, _ = self.mha_us_to_mg(
            query=us_summary, key=mg_views, value=mg_views, key_padding_mask=mg_key_padding
        )
        cross_us = self.norm2(us_summary + cross_us)

        return cross_mg.squeeze(1), cross_us.squeeze(1)


class GraphInfusedTransformer(nn.Module):
    """Multimodal Graph-Infused Transformer (GIT) for BCMID Breast Imaging Triage.

    Architecture pipeline:
      1. Mammogram features (1280-dim, EfficientNet-B0) & Ultrasound features (768-dim, ViT-B/16)
         projected to shared latent dimension D.
      2. Learnable modality embeddings and anatomical breast side embeddings (L, R, BIL, UNK).
      3. Attentive pooling modality summaries and bi-directional cross-modal fusion.
      4. Dynamic patient graph construction linking views, summaries, cross-modal fused nodes,
         and ipsilateral (same breast side) cross-modality correspondences.
      5. Multi-Head Graph Attention Network (GAT) to model structured anatomical relationships.
      6. Graph-Infused Transformer Encoder blocks providing unconstrained global self-attention.
      7. Dropout-regularized binary classification head for patient triage.
    """

    def __init__(
        self,
        mg_feat_dim: int = 1280,
        us_feat_dim: int = 768,
        hidden_dim: int = 256,
        num_gat_layers: int = 2,
        gat_heads: int = 4,
        transformer_layers: int = 2,
        transformer_heads: int = 8,
        dim_feedforward: int = 1024,
        num_classes: int = 2,
        dropout_rate: float = 0.2,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate

        # 1. Feature Projections
        self.mg_proj = nn.Sequential(
            nn.Linear(mg_feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )
        self.us_proj = nn.Sequential(
            nn.Linear(us_feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )

        # 2. Modality & Side Embeddings
        self.mg_modality_emb = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.us_modality_emb = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.side_embedding = nn.Embedding(4, hidden_dim)  # L, R, BIL, UNK

        # Patient Global CLS Token
        self.global_cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)

        # 3. Patient Summaries & Fusion
        self.mg_pooling = AttentivePooling(hidden_dim, dropout_rate=dropout_rate)
        self.us_pooling = AttentivePooling(hidden_dim, dropout_rate=dropout_rate)
        self.cross_attention = CrossModalAttention(
            hidden_dim, num_heads=gat_heads, dropout_rate=dropout_rate
        )

        # Non-linear Fusion MLP
        # Combines [z_mg; z_us; z_mg * z_us; cross_mg; cross_us] -> hidden_dim
        self.fusion_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 5, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        # 4. Graph Attention Network (GAT)
        self.gat = MultimodalGAT(
            hidden_dim=hidden_dim,
            num_layers=num_gat_layers,
            num_heads=gat_heads,
            dropout_rate=dropout_rate,
        )

        # 5. Graph-Infused Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=transformer_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout_rate,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=transformer_layers,
        )
        self.post_norm = nn.LayerNorm(hidden_dim)

        # 6. Binary Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def _build_patient_graph(
        self,
        z_global: torch.Tensor,
        z_mg: torch.Tensor,
        z_us: torch.Tensor,
        z_fused: torch.Tensor,
        mg_tokens: torch.Tensor,
        us_tokens: torch.Tensor,
        mg_sides: torch.Tensor,
        us_sides: torch.Tensor,
        mg_mask: torch.Tensor,
        us_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Constructs the batched node feature tensor, adjacency matrix, and node mask.

        Node layout for each patient:
          Index 0: Patient Global Token [CLS]
          Index 1: Mammogram Modality Summary
          Index 2: Ultrasound Modality Summary
          Index 3: Cross-Modal Fused Representation
          Indices 4 .. 3+M: Mammogram views
          Indices 4+M .. 3+M+U: Ultrasound views
        """
        batch_size = z_global.size(0)
        M = mg_tokens.size(1)
        U = us_tokens.size(1)
        total_nodes = 4 + M + U

        device = z_global.device

        # Concatenate all node tokens along dim 1: (B, total_nodes, D)
        nodes = torch.cat(
            [
                z_global.unsqueeze(1),
                z_mg.unsqueeze(1),
                z_us.unsqueeze(1),
                z_fused.unsqueeze(1),
                mg_tokens,
                us_tokens,
            ],
            dim=1,
        )

        # Construct boolean node_mask: (B, total_nodes)
        # First 4 meta nodes are always valid
        meta_mask = torch.ones(batch_size, 4, dtype=torch.bool, device=device)
        node_mask = torch.cat([meta_mask, mg_mask, us_mask], dim=1)

        # Initialize adjacency matrix with zeros: (B, total_nodes, total_nodes)
        adj = torch.zeros(batch_size, total_nodes, total_nodes, device=device)

        # 1. Self-loops for all nodes
        eye = torch.eye(total_nodes, device=device).unsqueeze(0)
        adj = adj + eye

        # 2. Global CLS (0) connects to Fused (3), MG summary (1), US summary (2)
        adj[:, 0, 1] = 1.0
        adj[:, 1, 0] = 1.0
        adj[:, 0, 2] = 1.0
        adj[:, 2, 0] = 1.0
        adj[:, 0, 3] = 1.0
        adj[:, 3, 0] = 1.0

        # 3. Fused token (3) connects to MG summary (1), US summary (2)
        adj[:, 3, 1] = 1.0
        adj[:, 1, 3] = 1.0
        adj[:, 3, 2] = 1.0
        adj[:, 2, 3] = 1.0

        # 4. MG summary (1) connects to all valid MG views
        for m in range(M):
            idx = 4 + m
            adj[:, 1, idx] = 1.0
            adj[:, idx, 1] = 1.0
            # Also connect Fused token to views for multi-resolution fusion
            adj[:, 3, idx] = 1.0
            adj[:, idx, 3] = 1.0

        # 5. US summary (2) connects to all valid US views
        for u in range(U):
            idx = 4 + M + u
            adj[:, 2, idx] = 1.0
            adj[:, idx, 2] = 1.0
            adj[:, 3, idx] = 1.0
            adj[:, idx, 3] = 1.0

        # 6. Intra-modality view connections (MG views to MG views, US views to US views)
        for m1 in range(M):
            for m2 in range(m1 + 1, M):
                adj[:, 4 + m1, 4 + m2] = 1.0
                adj[:, 4 + m2, 4 + m1] = 1.0

        for u1 in range(U):
            for u2 in range(u1 + 1, U):
                adj[:, 4 + M + u1, 4 + M + u2] = 1.0
                adj[:, 4 + M + u2, 4 + M + u1] = 1.0

        # 7. Cross-Modality Anatomical Breast Side Connections (Ipsilateral Matching)
        # If mammogram view side matches ultrasound view side, or either is BIL/UNK
        # mg_sides: (B, M), us_sides: (B, U)
        for m in range(M):
            idx_m = 4 + m
            for u in range(U):
                idx_u = 4 + M + u
                # Match condition: same side or bilateral/unknown
                same_side = (mg_sides[:, m] == us_sides[:, u])
                bil_unk = (
                    (mg_sides[:, m] >= 2) | (us_sides[:, u] >= 2)
                )
                match = (same_side | bil_unk).float()
                adj[:, idx_m, idx_u] = match
                adj[:, idx_u, idx_m] = match

        # Mask out any edges connected to invalid padded nodes
        pair_mask = (node_mask.unsqueeze(2) & node_mask.unsqueeze(1)).float()
        adj = adj * pair_mask

        return nodes, adj, node_mask

    def forward_features(
        self,
        mg_features: torch.Tensor,  # (B, M, 1280)
        us_features: torch.Tensor,  # (B, U, 768)
        mg_sides: torch.Tensor,     # (B, M) int in [0, 3]
        us_sides: torch.Tensor,     # (B, U) int in [0, 3]
        mg_mask: Optional[torch.Tensor] = None,  # (B, M) bool
        us_mask: Optional[torch.Tensor] = None,  # (B, U) bool
    ) -> torch.Tensor:
        """Forward pass using pre-extracted or backbone-extracted feature tensors."""
        batch_size = mg_features.size(0)
        M = mg_features.size(1)
        U = us_features.size(1)
        device = mg_features.device

        if mg_mask is None:
            mg_mask = torch.ones(batch_size, M, dtype=torch.bool, device=device)
        if us_mask is None:
            us_mask = torch.ones(batch_size, U, dtype=torch.bool, device=device)

        # 1. Project features to hidden_dim
        mg_tokens = self.mg_proj(mg_features)  # (B, M, D)
        us_tokens = self.us_proj(us_features)  # (B, U, D)

        # 2. Add modality embeddings & anatomical side embeddings
        mg_tokens = mg_tokens + self.mg_modality_emb + self.side_embedding(mg_sides)
        us_tokens = us_tokens + self.us_modality_emb + self.side_embedding(us_sides)

        # 3. Attentive pooling for modality summaries
        z_mg = self.mg_pooling(mg_tokens, mask=mg_mask)  # (B, D)
        z_us = self.us_pooling(us_tokens, mask=us_mask)  # (B, D)

        # 4. Cross-modal attention
        cross_mg, cross_us = self.cross_attention(
            mg_summary=z_mg.unsqueeze(1),
            us_summary=z_us.unsqueeze(1),
            mg_views=mg_tokens,
            us_views=us_tokens,
            mg_mask=mg_mask,
            us_mask=us_mask,
        )

        # Non-linear fusion: [z_mg, z_us, elementwise_prod, cross_mg, cross_us]
        fusion_input = torch.cat(
            [z_mg, z_us, z_mg * z_us, cross_mg, cross_us], dim=-1
        )
        z_fused = self.fusion_mlp(fusion_input)  # (B, D)

        # Global CLS token
        z_global = self.global_cls_token.expand(batch_size, -1, -1).squeeze(1) + z_fused

        # 5. Build Graph Representation
        nodes, adj, node_mask = self._build_patient_graph(
            z_global=z_global,
            z_mg=z_mg,
            z_us=z_us,
            z_fused=z_fused,
            mg_tokens=mg_tokens,
            us_tokens=us_tokens,
            mg_sides=mg_sides,
            us_sides=us_sides,
            mg_mask=mg_mask,
            us_mask=us_mask,
        )

        # 6. GAT Message Passing
        gat_nodes = self.gat(nodes, adj, node_mask=node_mask)

        # Residual graph infusion: combine original nodes and GAT-refined nodes
        infused_nodes = F.layer_norm(nodes + gat_nodes, (self.hidden_dim,))

        # 7. Graph-Infused Transformer Encoder
        # Transformer expects src_key_padding_mask where True = ignore/padding
        key_padding_mask = ~node_mask
        transformed = self.transformer_encoder(
            infused_nodes, src_key_padding_mask=key_padding_mask
        )
        transformed = self.post_norm(transformed)

        # 8. Patient Global Representation (Token at index 0)
        patient_repr = transformed[:, 0, :]

        # 9. Binary Classification Logits
        logits = self.classifier(patient_repr)
        return logits

    def forward(
        self,
        mg_features: torch.Tensor,
        us_features: torch.Tensor,
        mg_sides: torch.Tensor,
        us_sides: torch.Tensor,
        mg_mask: Optional[torch.Tensor] = None,
        us_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Alias for forward_features returning raw logits of shape (batch_size, 2)."""
        return self.forward_features(
            mg_features=mg_features,
            us_features=us_features,
            mg_sides=mg_sides,
            us_sides=us_sides,
            mg_mask=mg_mask,
            us_mask=us_mask,
        )


class EndToEndMultimodalGIT(nn.Module):
    """End-to-End Multimodal GIT Pipeline wrapping frozen/fine-tunable vision backbones

    and Graph-Infused Transformer for end-to-end inference from raw image tensors.
    """

    def __init__(
        self,
        git_model: GraphInfusedTransformer,
        effnet_checkpoint: Optional[str] = None,
        vit_checkpoint: Optional[str] = None,
        freeze_backbones: bool = True,
    ):
        super().__init__()
        self.git = git_model

        # Load EfficientNet-B0 backbone for Mammogram
        self.effnet = EfficientNetB0Binary(pretrained=False)
        if effnet_checkpoint:
            ckpt = torch.load(effnet_checkpoint, map_location="cpu", weights_only=False)
            self.effnet.load_state_dict(ckpt["model_state_dict"])

        # Load ViT-B/16 backbone for Ultrasound
        self.vit = ViTB16Binary(pretrained=False)
        if vit_checkpoint:
            ckpt = torch.load(vit_checkpoint, map_location="cpu", weights_only=False)
            self.vit.load_state_dict(ckpt["model_state_dict"])

        if freeze_backbones:
            for p in self.effnet.parameters():
                p.requires_grad = False
            for p in self.vit.parameters():
                p.requires_grad = False
            self.effnet.eval()
            self.vit.eval()

        self.freeze_backbones = freeze_backbones

    def extract_image_features(
        self,
        mg_images: torch.Tensor,  # (B, M, 3, 224, 224)
        us_images: torch.Tensor,  # (B, U, 3, 224, 224)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, M, C, H, W = mg_images.shape
        _, U, _, _, _ = us_images.shape

        mg_flat = mg_images.view(B * M, C, H, W)
        us_flat = us_images.view(B * U, C, H, W)

        if self.freeze_backbones:
            with torch.no_grad():
                mg_feats = self.effnet.extract_features(mg_flat)
                us_feats = self.vit.extract_features(us_flat)
        else:
            mg_feats = self.effnet.extract_features(mg_flat)
            us_feats = self.vit.extract_features(us_flat)

        mg_feats = mg_feats.view(B, M, 1280)
        us_feats = us_feats.view(B, U, 768)
        return mg_feats, us_feats

    def forward(
        self,
        mg_images: torch.Tensor,
        us_images: torch.Tensor,
        mg_sides: torch.Tensor,
        us_sides: torch.Tensor,
        mg_mask: Optional[torch.Tensor] = None,
        us_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        mg_feats, us_feats = self.extract_image_features(mg_images, us_images)
        return self.git(
            mg_features=mg_feats,
            us_features=us_feats,
            mg_sides=mg_sides,
            us_sides=us_sides,
            mg_mask=mg_mask,
            us_mask=us_mask,
        )
