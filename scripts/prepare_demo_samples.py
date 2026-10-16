"""Prepares the demo samples and manifest for testing trained models."""

import os
import shutil
import pandas as pd
from PIL import Image

demo_dir = "data/samples/demo"
low_dir = os.path.join(demo_dir, "low_risk")
susp_dir = os.path.join(demo_dir, "suspicious_actionable")

os.makedirs(low_dir, exist_ok=True)
os.makedirs(susp_dir, exist_ok=True)

sources = {
    "low_mam": {
        "pid": "0423y18",
        "modality": "mammogram",
        "src": "BCMID/0423y18/Mammogram/0423y18_L_MLO_1.png",
        "dst": os.path.join(low_dir, "mammogram.png"),
        "label_id": 0,
        "label_name": "LOW_RISK",
        "birads": "2",
        "evidence": "Screening study for 52ys old female with history of thyroid cancer , the current study shows :- Unremarkable for normal fibro glandular breasts, apart from left intrammmary lymph nodes ?.. For annual follow up. BIRAD 2",
        "split": "test",
    },
    "low_us": {
        "pid": "0423y18",
        "modality": "ultrasound",
        "src": "BCMID/0423y18/Ultrasound/0423y18_US_L_2.png",
        "dst": os.path.join(low_dir, "ultrasound.png"),
        "label_id": 0,
        "label_name": "LOW_RISK",
        "birads": "2",
        "evidence": "Screening study for 52ys old female with history of thyroid cancer , the current study shows :- Unremarkable for normal fibro glandular breasts, apart from left intrammmary lymph nodes ?.. For annual follow up. BIRAD 2",
        "split": "test",
    },
    "susp_mam": {
        "pid": "00506y21",
        "modality": "mammogram",
        "src": "BCMID/00506y21/Mammogram/00506y21_L_MLO_1.png",
        "dst": os.path.join(susp_dir, "mammogram.png"),
        "label_id": 1,
        "label_name": "SUSPICIOUS_ACTIONABLE",
        "birads": "4B",
        "evidence": "Diagnostic study revealed left breast suspicious lesion and indeterminate left axillary LN ??for histopathological correlation ?BIRADS 4b Dr. Weaam Mohamed",
        "split": "test",
    },
    "susp_us": {
        "pid": "00506y21",
        "modality": "ultrasound",
        "src": "BCMID/00506y21/Ultrasound/00506y21_US_L_5.png",
        "dst": os.path.join(susp_dir, "ultrasound.png"),
        "label_id": 1,
        "label_name": "SUSPICIOUS_ACTIONABLE",
        "birads": "4B",
        "evidence": "Diagnostic study revealed left breast suspicious lesion and indeterminate left axillary LN ??for histopathological correlation ?BIRADS 4b Dr. Weaam Mohamed",
        "split": "test",
    },
}

manifest_rows = []
for k, item in sources.items():
    src_path = item["src"]
    dst_path = item["dst"]
    shutil.copy2(src_path, dst_path)

    # Verify PIL readability
    with Image.open(dst_path) as im:
        w, h = im.size
        fmt = im.format
        mode = im.mode

    print(f"Copied {src_path} -> {dst_path} [{w}x{h}, {mode}, {fmt}, {os.path.getsize(dst_path):,} bytes]")

    manifest_rows.append({
        "patient_id": item["pid"],
        "modality": item["modality"],
        "demo_target_path": dst_path.replace("\\", "/"),
        "original_filename": os.path.basename(src_path),
        "original_source_path": src_path.replace("\\", "/"),
        "ground_truth_label": item["label_id"],
        "label_name": item["label_name"],
        "birads_category": item["birads"],
        "report_evidence": item["evidence"],
        "source_split": item["split"],
    })

manifest_df = pd.DataFrame(manifest_rows)
manifest_csv = os.path.join(demo_dir, "demo_manifest.csv")
manifest_df.to_csv(manifest_csv, index=False)
print(f"\nManifest successfully created at: {manifest_csv}")
