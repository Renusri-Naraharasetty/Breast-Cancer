import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import train_test_split
from .label_extractor import PatientLabel


def create_patient_stratified_split(
    patient_labels: Dict[str, PatientLabel],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    save_path: Optional[str] = None
) -> Dict[str, List[str]]:
    """Creates a patient-level stratified train / validation / test split.
    
    CRITICAL: All images belonging to a single patient stay strictly in one split.
    Patients with label == -1 (UNKNOWN) are isolated into a separate 'unknown' list.
    
    Returns:
        Dict containing lists of patient IDs:
        {
            'train': [...],
            'val': [...],
            'test': [...],
            'unknown': [...]
        }
    """
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-5, "Split ratios must sum to 1.0"
    
    random.seed(seed)
    
    labeled_patients = []
    labels = []
    unknown_patients = []
    
    for pid, plabel in sorted(patient_labels.items()):
        if plabel.label_id in [0, 1]:
            labeled_patients.append(pid)
            labels.append(plabel.label_id)
        else:
            unknown_patients.append(pid)
            
    # First split into train and temp (val + test)
    temp_ratio = val_ratio + test_ratio
    train_pids, temp_pids, train_y, temp_y = train_test_split(
        labeled_patients,
        labels,
        test_size=temp_ratio,
        stratify=labels,
        random_state=seed
    )
    
    # Next split temp into val and test
    relative_test_ratio = test_ratio / temp_ratio
    val_pids, test_pids, val_y, test_y = train_test_split(
        temp_pids,
        temp_y,
        test_size=relative_test_ratio,
        stratify=temp_y,
        random_state=seed
    )
    
    splits = {
        "train": sorted(train_pids),
        "val": sorted(val_pids),
        "test": sorted(test_pids),
        "unknown": sorted(unknown_patients)
    }
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(splits, f, indent=2)
            
    return splits
