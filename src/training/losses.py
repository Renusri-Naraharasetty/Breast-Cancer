import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List


def get_loss_function(
    class_weights: Optional[torch.Tensor] = None,
    device: str = "cpu"
) -> nn.Module:
    """Returns class-weighted CrossEntropyLoss to address benign/malignant imbalance."""
    if class_weights is not None:
        class_weights = class_weights.to(device)
    return nn.CrossEntropyLoss(weight=class_weights)


class FocalLoss(nn.Module):
    """Focal Loss for addressing extreme class imbalance in medical imaging.
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()
