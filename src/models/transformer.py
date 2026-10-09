import torch
import torch.nn as nn
from typing import Tuple, Optional, List


class GraphInfusedTransformerLayer(nn.Module):
    """Transformer Encoder Layer with Multi-Head Self-Attention for graph-enhanced tokens."""
    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.2
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, src: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Args:
            src: [B, N, d_model]
            
        Returns:
            out: [B, N, d_model]
            attn_weights: [B, N, N]
        """
        # Multi-Head Self-Attention with residual connection (Pre-LN)
        src2 = self.norm1(src)
        src2, attn_weights = self.self_attn(src2, src2, src2, need_weights=True, average_attn_weights=True)
        src = src + self.dropout1(src2)

        # Feedforward network with residual connection
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)

        return src, attn_weights


class GraphInfusedTransformer(nn.Module):
    """Configurable Graph-Infused Transformer encoder operating on graph-enhanced multimodal tokens."""
    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 512,
        dropout: float = 0.2
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            GraphInfusedTransformerLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Passes graph-enhanced tokens through sequential Transformer layers.
        
        Args:
            x: [B, N, d_model]
            
        Returns:
            out: [B, N, d_model]
            all_attn_weights: List of [B, N, N] attention matrices per layer
        """
        all_attn = []
        for layer in self.layers:
            x, attn = layer(x)
            all_attn.append(attn)
            
        out = self.final_norm(x)
        return out, all_attn
