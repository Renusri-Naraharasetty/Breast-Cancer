import os
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from sklearn.model_selection import StratifiedShuffleSplit


def create_patient_stratified_split(
    tsv_path: str = "BCMID_annotations.tsv",
    save_path: str = "data/splits/patient_stratified_split.json",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Dict[str, List[str]]:
    """Creates a patient-level stratified split ensuring zero cross-split leakage.

    Groups by root_patient_id so that all related patient examinations and bilateral
    assets remain in the exact same partition.
    """
    assert np.isclose(train_ratio + val_ratio + test_ratio, 1.0), "Splits must sum to 1.0"

    df = pd.read_csv(tsv_path, sep="\t")
    # Filter to eligible patients only
    eligible_df = df[df["label_id"] != -1].copy()

    # Aggregate by root_patient_id
    patient_df = (
        eligible_df.groupby("root_patient_id")
        .agg({
            "label_id": "first",
            "label_name": "first",
            "patient_id": list,
            "num_mammogram_images": "sum",
            "num_ultrasound_images": "sum",
        })
        .reset_index()
    )

    pids = patient_df["root_patient_id"].values
    labels = patient_df["label_id"].values

    # Step 1: Separate Train vs Temp (Val + Test)
    temp_ratio = val_ratio + test_ratio
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=temp_ratio, random_state=seed)
    train_idx, temp_idx = next(sss1.split(pids, labels))

    train_pids = pids[train_idx]
    temp_pids = pids[temp_idx]
    temp_labels = labels[temp_idx]

    # Step 2: Separate Val vs Test from Temp (50/50 of temp if val_ratio == test_ratio)
    val_share_of_temp = val_ratio / temp_ratio
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=(1.0 - val_share_of_temp), random_state=seed)
    val_idx_rel, test_idx_rel = next(sss2.split(temp_pids, temp_labels))

    val_pids = temp_pids[val_idx_rel]
    test_pids = temp_pids[test_idx_rel]

    # Verification: Zero leakage
    train_set = set(train_pids)
    val_set = set(val_pids)
    test_set = set(test_pids)

    assert len(train_set & val_set) == 0, "Leakage detected between Train and Val!"
    assert len(train_set & test_set) == 0, "Leakage detected between Train and Test!"
    assert len(val_set & test_set) == 0, "Leakage detected between Val and Test!"

    # Map back to all folder patient_ids (including aliases)
    root_to_folders = (
        eligible_df.groupby("root_patient_id")["patient_id"].apply(list).to_dict()
    )

    def expand_folders(root_list: np.ndarray) -> List[str]:
        folders = []
        for r in root_list:
            folders.extend(root_to_folders[r])
        return sorted(folders)

    split_dict = {
        "metadata": {
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "test_ratio": test_ratio,
            "seed": seed,
            "total_eligible_root_patients": len(pids),
            "total_eligible_patient_folders": len(eligible_df),
        },
        "root_patient_splits": {
            "train": sorted(list(train_pids)),
            "val": sorted(list(val_pids)),
            "test": sorted(list(test_pids)),
        },
        "folder_patient_splits": {
            "train": expand_folders(train_pids),
            "val": expand_folders(val_pids),
            "test": expand_folders(test_pids),
        },
    }

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(split_dict, f, indent=2)

    print(f"Stratified split saved to: {save_path}")
    print(f"Train root patients: {len(train_pids)} ({len(split_dict['folder_patient_splits']['train'])} folders)")
    print(f"Val root patients:   {len(val_pids)} ({len(split_dict['folder_patient_splits']['val'])} folders)")
    print(f"Test root patients:  {len(test_pids)} ({len(split_dict['folder_patient_splits']['test'])} folders)")

    return split_dict


if __name__ == "__main__":
    create_patient_stratified_split()
