import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class GraphAttentionLayer(nn.Module):
    """Multi-Head Graph Attention Layer (GAT) implemented in native PyTorch.

    Computes masked multi-head attention over graph adjacency structures.
    Supports batched graph tensors with variable node counts via node masks.
    Includes residual connection, dropout, and layer normalization.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_heads: int = 4,
        dropout_rate: float = 0.2,
        alpha: float = 0.2,
        concat: bool = False,
        residual: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.alpha = alpha
        self.concat = concat
        self.residual = residual

        if concat:
            assert out_features % num_heads == 0, "out_features must be divisible by num_heads when concat=True"
            self.head_dim = out_features // num_heads
        else:
            self.head_dim = out_features

        # Projection weight matrix W: (in_features, num_heads * head_dim)
        self.W = nn.Linear(in_features, num_heads * self.head_dim, bias=False)

        # Attention vectors for source and destination nodes
        self.a_src = nn.Parameter(torch.empty(size=(1, 1, num_heads, self.head_dim)))
        self.a_dst = nn.Parameter(torch.empty(size=(1, 1, num_heads, self.head_dim)))

        self.leaky_relu = nn.LeakyReLU(self.alpha)
        self.dropout = nn.Dropout(dropout_rate)

        # Output projection when concat is False, or to align residual dimension
        if concat:
            self.out_proj = nn.Identity()
        else:
            self.out_proj = nn.Linear(num_heads * self.head_dim, out_features)

        if self.residual:
            if in_features != out_features:
                self.res_proj = nn.Linear(in_features, out_features)
            else:
                self.res_proj = nn.Identity()
            self.layer_norm = nn.LayerNorm(out_features)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.W.weight, gain=1.414)
        nn.init.xavier_uniform_(self.a_src, gain=1.414)
        nn.init.xavier_uniform_(self.a_dst, gain=1.414)
        if hasattr(self, "out_proj") and isinstance(self.out_proj, nn.Linear):
            nn.init.xavier_uniform_(self.out_proj.weight, gain=1.414)
            nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        node_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass for batched graph attention.

        Args:
            x: Node feature tensor of shape (batch_size, num_nodes, in_features).
            adj: Adjacency matrix of shape (batch_size, num_nodes, num_nodes) where
                 1 indicates an edge and 0 indicates no edge.
            node_mask: Optional boolean mask of shape (batch_size, num_nodes) where
                       True indicates a valid node and False indicates a padded node.

        Returns:
            Output node features of shape (batch_size, num_nodes, out_features).
        """
        batch_size, num_nodes, _ = x.shape

        # Linear projection: (B, N, num_heads * head_dim)
        h = self.W(x)
        # Reshape to (B, N, num_heads, head_dim)
        h = h.view(batch_size, num_nodes, self.num_heads, self.head_dim)

        # Compute source and destination attention components
        # (B, N, num_heads, 1)
        score_src = (h * self.a_src).sum(dim=-1, keepdim=True)
        score_dst = (h * self.a_dst).sum(dim=-1, keepdim=True)

        # Pairwise attention scores: e_{ij} = LeakyReLU(score_src_i + score_dst_j)
        # score_src: (B, N, 1, num_heads) -> broadcast over dst
        # score_dst: (B, 1, N, num_heads) -> broadcast over src
        score_src = score_src.squeeze(-1).unsqueeze(2)  # (B, N, 1, num_heads)
        score_dst = score_dst.squeeze(-1).unsqueeze(1)  # (B, 1, N, num_heads)
        scores = self.leaky_relu(score_src + score_dst)  # (B, N, N, num_heads)

        # Mask out non-edges
        # adj is (B, N, N) -> expand to (B, N, N, 1)
        adj_mask = (adj > 0).unsqueeze(-1)  # (B, N, N, 1)

        if node_mask is not None:
            # node_mask is (B, N), valid pairs have both nodes valid
            pair_mask = (node_mask.unsqueeze(2) & node_mask.unsqueeze(1)).unsqueeze(-1)
            adj_mask = adj_mask & pair_mask

        # Apply large negative number to non-edges so softmax weight is 0
        scores = scores.masked_fill(~adj_mask, -1e4)

        # Softmax over target neighbors (dim=2)
        attn_weights = F.softmax(scores, dim=2)
        attn_weights = self.dropout(attn_weights)

        # Handle nodes that have zero valid neighbors (prevent NaN in softmax)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

        # Weighted message aggregation:
        # attn_weights: (B, N_i, N_j, H), h: (B, N_j, H, D_head)
        # aggregated: (B, N_i, H, D_head)
        out = torch.einsum("bijh,bjhd->bihd", attn_weights, h)

        if self.concat:
            # Concatenate heads: (B, N, num_heads * head_dim) = (B, N, out_features)
            out = out.reshape(batch_size, num_nodes, self.num_heads * self.head_dim)
            out = self.out_proj(out)
        else:
            # Average heads or project
            out = out.reshape(batch_size, num_nodes, self.num_heads * self.head_dim)
            out = self.out_proj(out)

        # Residual connection and Layer Normalization
        if self.residual:
            res = self.res_proj(x)
            out = self.layer_norm(res + self.dropout(out))

        if node_mask is not None:
            out = out * node_mask.unsqueeze(-1).float()

        return out


class MultimodalGAT(nn.Module):
    """Multi-layer Graph Attention Network for modeling relationships between fused features."""

    def __init__(
        self,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout_rate: float = 0.2,
        alpha: float = 0.2,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        for _ in range(num_layers):
            self.layers.append(
                GraphAttentionLayer(
                    in_features=hidden_dim,
                    out_features=hidden_dim,
                    num_heads=num_heads,
                    dropout_rate=dropout_rate,
                    alpha=alpha,
                    concat=False,
                    residual=True,
                )
            )

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        node_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Passes node features through stacked GAT layers with residual connections."""
        for layer in self.layers:
            x = layer(x, adj, node_mask=node_mask)
        return x
