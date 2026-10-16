"""Pilot GPU memory benchmark script for Phase 4.

Validates forward and backward passes on actual hardware (NVIDIA GeForce MX550 2GB)
using pretrained EfficientNet-B0 and ViT-B/16 models with mixed precision (AMP)
and gradient accumulation.
"""

import os
import sys
import gc
import torch
import torch.nn as nn

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.factory import create_model, get_model_info
from src.data.dataset import get_dataloader
from src.training.losses import compute_image_level_class_weights


def benchmark_model(model_name: str, modality: str, batch_size: int, grad_accum: int = 2):
    print(f"\n{'='*60}")
    print(f"PILOT BENCHMARK: {model_name} on {modality.upper()}")
    print(f"Batch size: {batch_size}, Grad accum: {grad_accum}")
    print(f"{'='*60}")

    if not torch.cuda.is_available():
        print("CUDA is not available. Running on CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    # 1. Load data batch
    loader = get_dataloader(
        modality=modality,
        split="train",
        batch_size=batch_size,
        side_aware_labeling=True,
        shuffle=True,
        num_workers=0,
    )
    images, labels, paths = next(iter(loader))
    images = images.to(device)
    labels = labels.to(device)
    print(f"Input batch shape: {images.shape}, Labels shape: {labels.shape}")

    # 2. Compute class weights
    weights = compute_image_level_class_weights(modality=modality, side_aware_labeling=True)
    loss_fn = nn.CrossEntropyLoss(weight=weights.to(device))

    # 3. Model creation with pretrained weights
    print(f"Instantiating pretrained {model_name}...")
    model = create_model(model_name=model_name, pretrained=True, num_classes=2, dropout_rate=0.3)
    model.to(device)
    model_info = get_model_info(model)
    print(f"Params: {model_info['total_parameters']:,} ({model_info['model_size_mb']} MB)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    # 4. Perform 3 forward + backward passes
    model.train()
    for step in range(3):
        optimizer.zero_grad()
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            outputs = model(images)
            loss = loss_fn(outputs, labels) / grad_accum

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        print(f"  Step {step + 1}: Loss = {loss.item() * grad_accum:.4f}, Output shape = {outputs.shape}")

    if device.type == "cuda":
        max_mem_bytes = torch.cuda.max_memory_allocated()
        max_mem_mb = max_mem_bytes / (1024 * 1024)
        total_mem_mb = torch.cuda.get_device_properties(device).total_memory / (1024 * 1024)
        print(f"Peak VRAM allocated: {max_mem_mb:.2f} MB / {total_mem_mb:.2f} MB ({max_mem_mb/total_mem_mb*100:.1f}%)")
        assert max_mem_mb < total_mem_mb * 0.9, f"VRAM exceeds safe headroom: {max_mem_mb:.2f} MB"

    # Cleanup
    del model, optimizer, scaler, images, labels, loss_fn
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"[SUCCESS] Pilot benchmark passed for {model_name} on {modality}!")


if __name__ == "__main__":
    # Test EfficientNet-B0 (batch_size=8)
    benchmark_model("efficientnet_b0", "mammogram", batch_size=8, grad_accum=2)
    # Test ViT-B/16 (batch_size=8)
    benchmark_model("vit_b_16", "ultrasound", batch_size=8, grad_accum=2)
