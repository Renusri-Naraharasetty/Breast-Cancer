import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ClassBalancedFocalLoss(nn.Module):
    """Class-Balanced Focal Loss for imbalanced multi-class / binary classification.

    FL(p_t) = - alpha_t * (1 - p_t)^gamma * log(p_t)

    When gamma=0, this identically equals nn.CrossEntropyLoss(weight=alpha).
    When gamma > 0, easy examples with p_t close to 1 are down-weighted,
    forcing model gradients to focus on hard, ambiguous diagnostic cases.

    Args:
        gamma: Focusing parameter (e.g. 0.0 for standard CE, 1.5 or 2.0 for Focal Loss).
        alpha: Class weighting tensor [w_0, w_1, ...] matching class frequencies.
        reduction: 'mean', 'sum', or 'none'.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[torch.Tensor] = None,
        reduction: str = "mean",
        eps: float = 1e-8,
    ):
        super().__init__()
        self.gamma = float(gamma)
        self.reduction = reduction
        self.eps = eps
        if alpha is not None:
            self.register_buffer("alpha", alpha.clone().detach().float())
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Args:

        logits: (batch_size, num_classes) raw unnormalized scores targets:
        (batch_size,) integer class labels in [0, num_classes-1]
        """
        log_probs = F.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs).clamp(min=self.eps, max=1.0 - self.eps)

        # Gather log_prob and prob for true target
        target_log_probs = log_probs.gather(dim=-1, index=targets.unsqueeze(1)).squeeze(1)
        target_probs = probs.gather(dim=-1, index=targets.unsqueeze(1)).squeeze(1)

        # Compute focal modulation factor: (1 - p_t)^gamma
        if self.gamma > 0.0:
            focal_weight = torch.pow(1.0 - target_probs, self.gamma)
        else:
            focal_weight = torch.ones_like(target_probs)

        # Compute alpha class weight
        if self.alpha is not None:
            alpha_weight = self.alpha.gather(dim=0, index=targets)
        else:
            alpha_weight = torch.ones_like(target_probs)

        # Element-wise loss
        loss = -alpha_weight * focal_weight * target_log_probs

        if self.reduction == "mean":
            # Normalize by sum of alpha weights to maintain stable gradient scale across batch
            return loss.sum() / alpha_weight.sum().clamp(min=self.eps)
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
