import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR


def get_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str = "cosine",
    num_epochs: int = 20,
    min_lr: float = 1.0e-6
):
    """Returns configured learning rate scheduler."""
    if scheduler_type == "cosine":
        return CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=min_lr)
    elif scheduler_type == "plateau":
        return ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2, min_lr=min_lr)
    elif scheduler_type == "step":
        return StepLR(optimizer, step_size=5, gamma=0.5)
    else:
        return None
