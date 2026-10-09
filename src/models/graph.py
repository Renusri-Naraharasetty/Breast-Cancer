import torch
import torch.nn as nn
from typing import Tuple


class MultimodalGraphBuilder(nn.Module):
    """Constructs a multimodal graph representation for each patient.
    
    Nodes:
        Node 0: Mammogram representation (projected to hidden_dim)
        Node 1: Ultrasound representation (projected to hidden_dim)
        
    Edges:
        Cross-modality bidirectional edges (0 <-> 1)
        Self-loops (0 -> 0, 1 -> 1)
        
    Modality type embeddings are added to distinguish imaging modalities.
    """
    def __init__(
        self,
        mammo_dim: int = 1024,
        us_dim: int = 2048,
        hidden_dim: int = 256
    ):
        super().__init__()
        self.proj_mammo = nn.Sequential(
            nn.Linear(mammo_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.proj_us = nn.Sequential(
            nn.Linear(us_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        # Learnable modality type embeddings (0: Mammo, 1: US)
        self.modality_embeddings = nn.Parameter(torch.randn(2, hidden_dim) * 0.02)
        
        # Fully connected 2x2 adjacency with self-loops
        self.register_buffer("adj_mask", torch.ones(2, 2, dtype=torch.bool))

    def forward(self, mammo_feats: torch.Tensor, us_feats: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Args:
            mammo_feats: [B, mammo_dim]
            us_feats: [B, us_dim]
            
        Returns:
            node_tokens: [B, 2, hidden_dim]
            adj_mask: [2, 2]
        """
        B = mammo_feats.shape[0]
        
        h_m = self.proj_mammo(mammo_feats) # [B, hidden_dim]
        h_u = self.proj_us(us_feats)       # [B, hidden_dim]

        # Stack into [B, 2, hidden_dim]
        tokens = torch.stack([h_m, h_u], dim=1)

        # Add modality identity embeddings
        tokens = tokens + self.modality_embeddings.unsqueeze(0)

        return tokens, self.adj_mask
