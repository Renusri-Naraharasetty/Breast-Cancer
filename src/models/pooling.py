import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionPooling(nn.Module):
    """Learnable Attention-based pooling for aggregating variable numbers of images per patient.
    
    Given a set of image embeddings {h_1, h_2, ..., h_k} belonging to patient i,
    computes an attention weight a_j = exp(w^T tanh(W h_j + b)) / sum_l exp(w^T tanh(W h_l + b))
    and outputs patient representation h_patient = sum_j a_j * h_j.
    
    Preserves all images while learning to attend to the most diagnostically significant views.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.attention_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, features: torch.Tensor, batch_indices: torch.Tensor, batch_size: int) -> torch.Tensor:
        """Args:
            features: [Total_Images, Input_Dim]
            batch_indices: [Total_Images] specifying which patient each image belongs to (0 to batch_size - 1)
            batch_size: Number of patients in the batch
            
        Returns:
            pooled_features: [batch_size, Input_Dim]
        """
        device = features.device
        if features.numel() == 0:
            return torch.zeros(batch_size, features.shape[-1], device=device)

        # Compute unnormalized attention scores: [Total_Images, 1]
        scores = self.attention_net(features)
        scores = torch.clamp(scores, min=-20.0, max=20.0)

        pooled = torch.zeros(batch_size, features.shape[-1], device=device)

        # Aggregate per patient
        for b in range(batch_size):
            mask = (batch_indices == b)
            if mask.any():
                patient_feats = features[mask]    # [K, Input_Dim]
                patient_scores = scores[mask]     # [K, 1]
                weights = F.softmax(patient_scores, dim=0) # [K, 1]
                pooled[b] = (patient_feats * weights).sum(dim=0)
            else:
                # Modality missing for this patient; keep zero vector
                pass

        return pooled
