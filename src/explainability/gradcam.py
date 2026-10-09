import os
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class GradCAM:
    """Gradient-weighted Class Activation Mapping (Grad-CAM) for CNN backbones.
    
    Identifies discriminative regions in Mammogram (DenseNet-121) and Ultrasound (ResNet-50) images.
    
    DISCLAIMER: For research/experimental use only. Not a clinical diagnostic system.
    """
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.hook_fwd = target_layer.register_forward_hook(self._forward_hook)
        self.hook_bwd = target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_cam(
        self,
        image_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> np.ndarray:
        """Args:
            image_tensor: Single image tensor [1, 3, H, W]
            target_class: Index of class to explain (default: argmax class)
            
        Returns:
            cam_heatmap: [H, W] normalized float numpy array in [0, 1]
        """
        self.model.eval()
        self.model.zero_grad()
        
        # Forward pass through feature extractor
        if hasattr(self.model, "extract_image_features"):
            feats = self.model.extract_image_features(image_tensor)
            logits = self.model.classifier(feats)
        else:
            logits = self.model(image_tensor)

        if target_class is None:
            target_class = torch.argmax(logits, dim=1).item()

        score = logits[0, target_class]
        score.backward()

        # Global average pooling of gradients: [1, C, 1, 1]
        weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)
        
        # Weighted combination of forward activation maps: [1, 1, H', W']
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam) # Only features with positive influence

        # Resize to original image dimensions
        H, W = image_tensor.shape[2], image_tensor.shape[3]
        cam = F.interpolate(cam, size=(H, W), mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        # Normalize to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam

    def remove_hooks(self):
        self.hook_fwd.remove()
        self.hook_bwd.remove()


def overlay_cam_on_image(
    image_tensor: torch.Tensor,
    cam_heatmap: np.ndarray,
    save_path: str,
    title: str = "Grad-CAM Heatmap"
) -> None:
    """Overlays Grad-CAM heatmap on the input image and saves to disk."""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Un-normalize image tensor: [3, H, W]
    img = image_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean
    img = np.clip(img, 0, 1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(img)
    axes[0].set_title("Input Image", fontsize=12)
    axes[0].axis("off")
    
    # CAM heatmap
    im1 = axes[1].imshow(cam_heatmap, cmap="jet")
    axes[1].set_title("Class Activation Map (Grad-CAM)", fontsize=12)
    axes[1].axis("off")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Overlay
    axes[2].imshow(img)
    axes[2].imshow(cam_heatmap, cmap="jet", alpha=0.45)
    axes[2].set_title("Grad-CAM Overlay", fontsize=12)
    axes[2].axis("off")
    
    plt.suptitle(f"{title}\n[RESEARCH USE ONLY - NOT CLINICALLY VALIDATED]", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def visualize_multimodal_attention(
    gat_attention: torch.Tensor,
    transformer_attention: torch.Tensor,
    save_path: str,
    patient_id: str = "Sample"
) -> None:
    """Visualizes cross-modality GAT attention weights and Transformer self-attention."""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    # gat_attention: [heads, 2, 2]
    # transformer_attention: [2, 2]
    gat_attn_np = gat_attention.cpu().numpy()
    trans_attn_np = transformer_attention.cpu().numpy()
    
    if gat_attn_np.ndim == 4:
        gat_attn_np = gat_attn_np[0] # Take first sample in batch
    if trans_attn_np.ndim == 3:
        trans_attn_np = trans_attn_np[0]

    num_heads = gat_attn_np.shape[0]
    fig, axes = plt.subplots(1, num_heads + 1, figsize=(4 * (num_heads + 1), 4))
    
    labels = ["Mammogram", "Ultrasound"]

    # GAT heads
    for h in range(num_heads):
        im = axes[h].imshow(gat_attn_np[h], cmap="viridis", vmin=0, vmax=1)
        axes[h].set_xticks([0, 1])
        axes[h].set_yticks([0, 1])
        axes[h].set_xticklabels(labels, fontsize=9)
        axes[h].set_yticklabels(labels, fontsize=9)
        axes[h].set_title(f"GATv2 Head {h+1}", fontsize=11, fontweight="bold")
        for i in range(2):
            for j in range(2):
                axes[h].text(j, i, f"{gat_attn_np[h, i, j]:.3f}", ha="center", va="center", color="white", fontweight="bold")

    # Transformer attention
    axes[num_heads].imshow(trans_attn_np, cmap="plasma", vmin=0, vmax=1)
    axes[num_heads].set_xticks([0, 1])
    axes[num_heads].set_yticks([0, 1])
    axes[num_heads].set_xticklabels(labels, fontsize=9)
    axes[num_heads].set_yticklabels(labels, fontsize=9)
    axes[num_heads].set_title("Transformer Self-Attention", fontsize=11, fontweight="bold")
    for i in range(2):
        for j in range(2):
            axes[num_heads].text(j, i, f"{trans_attn_np[i, j]:.3f}", ha="center", va="center", color="white", fontweight="bold")

    plt.suptitle(
        f"Multimodal Cross-Attention Weights (Patient: {patient_id})\n"
        f"[RESEARCH PROTOTYPE ONLY - NOT A CLINICAL DIAGNOSTIC TOOL]",
        fontsize=13, fontweight="bold", y=1.05
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
