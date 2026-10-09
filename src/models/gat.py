import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class GATv2Layer(nn.Module):
    """Multi-Head Graph Attention Network v2 (GATv2) layer.
    
    Implements dynamic graph attention between modality nodes.
    Computes cross-modality and self-attention weights and produces
    graph-enhanced representations with residual connections and layer norm.
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_heads: int = 4,
        dropout: float = 0.2,
        concat: bool = False
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.concat = concat
        
        assert out_dim % num_heads == 0, "out_dim must be divisible by num_heads"
        self.head_dim = out_dim // num_heads

        # Linear projections for source and target nodes
        self.lin_src = nn.Linear(in_dim, out_dim, bias=False)
        self.lin_dst = nn.Linear(in_dim, out_dim, bias=False)
        
        # Attention vector for each head
        self.attn = nn.Parameter(torch.Tensor(1, num_heads, 1, self.head_dim))
        
        # Value projection
        self.lin_val = nn.Linear(in_dim, out_dim, bias=False)
        
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)
        
        self.residual = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        self.norm = nn.LayerNorm(out_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.lin_src.weight)
        nn.init.xavier_uniform_(self.lin_dst.weight)
        nn.init.xavier_uniform_(self.lin_val.weight)
        nn.init.xavier_uniform_(self.attn)

    def forward(
        self,
        node_feats: torch.Tensor,
        adj_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Args:
            node_feats: [B, N, in_dim] where B is batch size, N is number of nodes (e.g. 2: Mammo, US)
            adj_mask: Optional [N, N] adjacency boolean mask (True for existing edges)
            
        Returns:
            out_feats: [B, N, out_dim] graph-enhanced node representations
            attn_weights: [B, num_heads, N, N] attention coefficients
        """
        B, N, _ = node_feats.shape
        
        # Project features: [B, N, num_heads, head_dim]
        h_src = self.lin_src(node_feats).view(B, N, self.num_heads, self.head_dim)
        h_dst = self.lin_dst(node_feats).view(B, N, self.num_heads, self.head_dim)
        h_val = self.lin_val(node_feats).view(B, N, self.num_heads, self.head_dim)

        # Transpose to [B, num_heads, N, head_dim]
        h_src = h_src.permute(0, 2, 1, 3)
        h_dst = h_dst.permute(0, 2, 1, 3)
        h_val = h_val.permute(0, 2, 1, 3)

        # Compute pairwise sum: h_src[:, :, :, None, :] + h_dst[:, :, None, :, :] -> [B, num_heads, N, N, head_dim]
        pairwise = h_src.unsqueeze(3) + h_dst.unsqueeze(2)
        scores = self.leaky_relu(pairwise)
        
        # Dot product with attention vector: [B, num_heads, N, N]
        attn_logits = (scores * self.attn.unsqueeze(2)).sum(dim=-1)

        if adj_mask is not None:
            # Mask out non-edges: [N, N] -> [1, 1, N, N]
            mask = adj_mask.unsqueeze(0).unsqueeze(0)
            attn_logits = attn_logits.masked_fill(~mask, float('-inf'))

        # Softmax over source nodes (dim=-1)
        attn_weights = F.softmax(attn_logits, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Aggregate values: [B, num_heads, N, N] x [B, num_heads, N, head_dim] -> [B, num_heads, N, head_dim]
        out = torch.matmul(attn_weights, h_val)

        # Reshape to [B, N, out_dim]
        out = out.permute(0, 2, 1, 3).contiguous().view(B, N, self.out_dim)

        # Residual connection + LayerNorm
        res = self.residual(node_feats)
        out = self.norm(out + res)

        return out, attn_weights


class MultimodalGAT(nn.Module):
    """Stack of GATv2 layers for multimodal message passing between Mammogram and Ultrasound."""
    def __init__(
        self,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.2
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            GATv2Layer(
                in_dim=hidden_dim,
                out_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])

    def forward(
        self,
        node_feats: torch.Tensor,
        adj_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Passes multimodal tokens through sequential GAT layers.
        
        Returns:
            enhanced_feats: [B, N, hidden_dim]
            last_attn_weights: [B, num_heads, N, N]
        """
        x = node_feats
        last_attn = None
        for layer in self.layers:
            x, last_attn = layer(x, adj_mask)
        return x, last_attn
