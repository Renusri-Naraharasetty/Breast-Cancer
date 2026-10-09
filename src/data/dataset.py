import os
import glob
from typing import Dict, List, Optional, Tuple, Any
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from .preprocessing import get_mammogram_transforms, get_ultrasound_transforms, load_image_tensor
from .label_extractor import PatientLabel


class BCMIDDataset(Dataset):
    """PyTorch Dataset for the Breast Cancer Multimodal Imaging Dataset (BCMID).
    
    Organized strictly patient-wise. Preserves all images belonging to each patient.
    """
    def __init__(
        self,
        dataset_dir: str,
        patient_ids: List[str],
        patient_labels: Dict[str, PatientLabel],
        image_size: Tuple[int, int] = (224, 224),
        is_training: bool = False,
        modality: str = "multimodal"  # "multimodal", "mammogram_only", "ultrasound_only"
    ):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.patient_ids = [pid for pid in patient_ids if pid in patient_labels]
        self.patient_labels = patient_labels
        self.image_size = image_size
        self.is_training = is_training
        self.modality = modality

        self.mammo_transform = get_mammogram_transforms(image_size=image_size, is_training=is_training)
        self.us_transform = get_ultrasound_transforms(image_size=image_size, is_training=is_training)

        # Pre-scan image paths for each patient to make __getitem__ fast
        self.patient_images = {}
        for pid in self.patient_ids:
            p_dir = os.path.join(dataset_dir, pid)
            m_dir = os.path.join(p_dir, "Mammogram")
            u_dir = os.path.join(p_dir, "Ultrasound")
            
            m_paths = []
            if os.path.exists(m_dir):
                m_paths = sorted([
                    os.path.join(m_dir, f) for f in os.listdir(m_dir)
                    if f.lower().endswith(('.png', '.jpg', '.jpeg')) and not f.startswith('.')
                ])
                
            u_paths = []
            if os.path.exists(u_dir):
                u_paths = sorted([
                    os.path.join(u_dir, f) for f in os.listdir(u_dir)
                    if f.lower().endswith(('.png', '.jpg', '.jpeg')) and not f.startswith('.')
                ])
                
            self.patient_images[pid] = {
                "mammo": m_paths,
                "us": u_paths
            }

    def __len__(self) -> int:
        return len(self.patient_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        pid = self.patient_ids[idx]
        plabel = self.patient_labels[pid]
        label_id = plabel.label_id

        # 1. Load all mammogram images for this patient
        mammo_tensors = []
        if self.modality in ["multimodal", "mammogram_only"]:
            for m_path in self.patient_images[pid]["mammo"]:
                t = load_image_tensor(m_path, self.mammo_transform)
                if t is not None:
                    mammo_tensors.append(t)
                    
        # Fallback if corrupted or missing
        if len(mammo_tensors) == 0 and self.modality in ["multimodal", "mammogram_only"]:
            # Dummy tensor to prevent pipeline crash
            mammo_tensors.append(torch.zeros(3, self.image_size[0], self.image_size[1]))

        # 2. Load all ultrasound images for this patient
        us_tensors = []
        if self.modality in ["multimodal", "ultrasound_only"]:
            for u_path in self.patient_images[pid]["us"]:
                t = load_image_tensor(u_path, self.us_transform)
                if t is not None:
                    us_tensors.append(t)
                    
        # Fallback if corrupted or missing
        if len(us_tensors) == 0 and self.modality in ["multimodal", "ultrasound_only"]:
            us_tensors.append(torch.zeros(3, self.image_size[0], self.image_size[1]))

        return {
            "patient_id": pid,
            "mammo_tensors": torch.stack(mammo_tensors) if mammo_tensors else torch.empty(0),
            "us_tensors": torch.stack(us_tensors) if us_tensors else torch.empty(0),
            "label": torch.tensor(label_id, dtype=torch.long)
        }


def collate_patient_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Custom collate function for patient batches.
    
    Concatenates variable numbers of mammograms and ultrasounds per patient into
    flat tensors while tracking patient indices for downstream attention pooling.
    """
    patient_ids = [item["patient_id"] for item in batch]
    labels = torch.stack([item["label"] for item in batch])

    mammo_list = []
    mammo_batch_indices = []
    
    us_list = []
    us_batch_indices = []

    for b_idx, item in enumerate(batch):
        m_imgs = item["mammo_tensors"]  # Shape: [N_i, 3, H, W]
        if m_imgs.numel() > 0:
            mammo_list.append(m_imgs)
            mammo_batch_indices.extend([b_idx] * m_imgs.shape[0])

        u_imgs = item["us_tensors"]    # Shape: [M_i, 3, H, W]
        if u_imgs.numel() > 0:
            us_list.append(u_imgs)
            us_batch_indices.extend([b_idx] * u_imgs.shape[0])

    mammo_tensors = torch.cat(mammo_list, dim=0) if mammo_list else torch.empty(0)
    us_tensors = torch.cat(us_list, dim=0) if us_list else torch.empty(0)

    return {
        "patient_ids": patient_ids,
        "labels": labels,
        "mammo_tensors": mammo_tensors,
        "mammo_batch_indices": torch.tensor(mammo_batch_indices, dtype=torch.long),
        "us_tensors": us_tensors,
        "us_batch_indices": torch.tensor(us_batch_indices, dtype=torch.long),
        "batch_size": len(batch)
    }
