import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from collections import Counter
from ..data.dataset import BCMIDModalityDataset


def compute_image_level_class_weights(
    modality: str,
    bcmid_dir: str = "BCMID",
    tsv_path: str = "BCMID_annotations.tsv",
    splits_path: str = "data/splits/patient_stratified_split.json",
    side_aware_labeling: bool = True,
    normalize_w0_to_one: bool = True,
) -> torch.Tensor:
    """Computes exact class weights directly from actual training images.

    Adheres strictly to actual training sample counts after applying side-aware
    anatomical labeling (never assuming patient-level ratios equal image ratios).

    For Mammography: 570 Class 0 vs 74 Class 1 (Ratio: 7.70 : 1).
    For Ultrasound:  541 Class 0 vs 130 Class 1 (Ratio: 4.16 : 1).
    """
    train_dataset = BCMIDModalityDataset(
        bcmid_dir=bcmid_dir,
        tsv_path=tsv_path,
        splits_path=splits_path,
        split="train",
        modality=modality,
        side_aware_labeling=side_aware_labeling,
    )

    labels = [s["label_id"] for s in train_dataset.samples]
    counts = Counter(labels)
    n0 = counts.get(0, 0)
    n1 = counts.get(1, 0)
    total = len(labels)

    assert n0 > 0 and n1 > 0, f"Cannot compute class weights with 0 samples for a class: {counts}"

    if normalize_w0_to_one:
        # Normalized so class 0 weight is 1.0, class 1 weight is n0 / n1
        w0 = 1.0
        w1 = float(n0) / float(n1)
    else:
        # Standard balanced inverse frequency: total / (2.0 * n_class)
        w0 = total / (2.0 * n0)
        w1 = total / (2.0 * n1)

    return torch.tensor([w0, w1], dtype=torch.float32)


class FocalLoss(nn.Module):
    """Focal Loss for imbalanced classification tasks (Lin et al., 2017).

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(
        self,
        weight: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


def get_loss_function(
    modality: str,
    loss_type: str = "weighted_ce",
    bcmid_dir: str = "BCMID",
    tsv_path: str = "BCMID_annotations.tsv",
    splits_path: str = "data/splits/patient_stratified_split.json",
    side_aware_labeling: bool = True,
    device: Optional[torch.device] = None,
) -> nn.Module:
    """Factory to construct loss function weighted by actual training image counts."""
    weights = compute_image_level_class_weights(
        modality=modality,
        bcmid_dir=bcmid_dir,
        tsv_path=tsv_path,
        splits_path=splits_path,
        side_aware_labeling=side_aware_labeling,
    )

    if device is not None:
        weights = weights.to(device)

    if loss_type.lower() in ["weighted_ce", "cross_entropy", "ce"]:
        return nn.CrossEntropyLoss(weight=weights)
    elif loss_type.lower() in ["focal", "focal_loss"]:
        return FocalLoss(weight=weights, gamma=2.0)
    else:
        raise ValueError(f"Unknown loss_type: '{loss_type}'. Supported: ['weighted_ce', 'focal'].")
