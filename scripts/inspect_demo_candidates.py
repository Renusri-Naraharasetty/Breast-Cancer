import json
import os
import glob
import pandas as pd
from PIL import Image

with open("data/splits/patient_stratified_split.json", "r", encoding="utf-8") as f:
    splits = json.load(f)

test_patients = splits["folder_patient_splits"]["test"]
df = pd.read_csv("BCMID_annotations.tsv", sep="\t")
test_df = df[df["patient_id"].isin(test_patients)].copy()

print("=" * 80)
print("1. SUSPICIOUS_ACTIONABLE PATIENTS (TEST SPLIT)")
print("=" * 80)
susp_df = test_df[test_df["label_name"] == "SUSPICIOUS_ACTIONABLE"]
for _, row in susp_df.iterrows():
    pid = row["patient_id"]
    mams = sorted(glob.glob(f"BCMID/{pid}/Mammogram/*.png"))
    uss = sorted(glob.glob(f"BCMID/{pid}/Ultrasound/*.png"))
    print(f"Patient ID: {pid}")
    print(f"  BI-RADS: {row['highest_birads']} | Pathology Confirmed: {row['is_pathology_confirmed']}")
    print(f"  Mammograms ({len(mams)}): {[os.path.basename(m) for m in mams]}")
    print(f"  Ultrasounds ({len(uss)}): {[os.path.basename(u) for u in uss]}")
    print(f"  Evidence: {row['evidence_snippet']}")
    print("-" * 60)

print("\n" + "=" * 80)
print("2. SAMPLE LOW_RISK PATIENTS (TEST SPLIT)")
print("=" * 80)
low_df = test_df[test_df["label_name"] == "LOW_RISK"]
for _, row in low_df.head(10).iterrows():
    pid = row["patient_id"]
    mams = sorted(glob.glob(f"BCMID/{pid}/Mammogram/*.png"))
    uss = sorted(glob.glob(f"BCMID/{pid}/Ultrasound/*.png"))
    print(f"Patient ID: {pid}")
    print(f"  BI-RADS: {row['highest_birads']} | Pathology Confirmed: {row['is_pathology_confirmed']}")
    print(f"  Mammograms ({len(mams)}): {[os.path.basename(m) for m in mams]}")
    print(f"  Ultrasounds ({len(uss)}): {[os.path.basename(u) for u in uss]}")
    print(f"  Evidence: {row['evidence_snippet']}")
    print("-" * 60)
