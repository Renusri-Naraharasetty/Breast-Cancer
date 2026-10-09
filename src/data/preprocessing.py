import os
from typing import Tuple, List, Optional
import torch
from torchvision import transforms
from PIL import Image


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_mammogram_transforms(image_size: Tuple[int, int] = (224, 224), is_training: bool = True) -> transforms.Compose:
    """Returns modality-specific transformations for Mammography images.
    
    Mammograms require high anatomical fidelity:
    - Conservative horizontal flip (left/right symmetry)
    - Gentle rotation (±7 degrees)
    - Mild brightness/contrast variation
    - ImageNet normalization
    """
    if is_training:
        return transforms.Compose([
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=7),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
    else:
        return transforms.Compose([
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])


def get_ultrasound_transforms(image_size: Tuple[int, int] = (224, 224), is_training: bool = True) -> transforms.Compose:
    """Returns modality-specific transformations for Ultrasound images.
    
    Ultrasound images have speckle and acoustic characteristics:
    - Conservative horizontal flip
    - Gentle rotation (±10 degrees)
    - Slight brightness/contrast variation
    - ImageNet normalization
    """
    if is_training:
        return transforms.Compose([
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
    else:
        return transforms.Compose([
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])


def load_image_tensor(image_path: str, transform: transforms.Compose) -> Optional[torch.Tensor]:
    """Safely loads an image file, converts to 3-channel RGB, and applies transform.
    
    Handles corrupted images gracefully by returning None rather than crashing.
    """
    if not os.path.exists(image_path):
        return None
        
    try:
        with Image.open(image_path) as img:
            # Convert to RGB (handles grayscale 'L', RGBA, CMYK, etc.)
            rgb_img = img.convert("RGB")
            tensor = transform(rgb_img)
            return tensor
    except Exception:
        # Gracefully handle corrupted image files
        return None
