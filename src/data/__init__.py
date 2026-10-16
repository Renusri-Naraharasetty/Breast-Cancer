"""Data pipeline and dataset definitions for BCMID benchmarking."""

from .dataset import BCMIDModalityDataset, get_dataloader, detect_breast_side
from .multimodal_dataset import (
    BCMIDPatientMultimodalDataset,
    get_patient_multimodal_dataloader,
    collate_patient_multimodal,
)
from .split import create_patient_stratified_split

__all__ = [
    "BCMIDModalityDataset",
    "get_dataloader",
    "detect_breast_side",
    "BCMIDPatientMultimodalDataset",
    "get_patient_multimodal_dataloader",
    "collate_patient_multimodal",
    "create_patient_stratified_split",
]
