import os
import re
import pypdf
import pandas as pd
from collections import Counter

bcmid_dir = r"c:\Users\admin\Music\Breast Cancer1\BCMID"
output_tsv = r"c:\Users\admin\Music\Breast Cancer1\BCMID_annotations.tsv"

patients = sorted(os.listdir(bcmid_dir))

footer_pattern = re.compile(
    r"(?:Category\s+Breast\s+imaging|BI-?RAD\s*\(?0\)?\s*incomplete|Breast\s+imaging\s+reporting\s+and\s+data\s+system)",
    re.IGNORECASE,
)
birads_pattern = re.compile(
    r"(?:B\s*I\s*[-_]?\s*[AR]{1,2}\s*D\s*[AS]{0,2}|CATEGORY)\s*\(?[:\-\s]*([0-6][a-c]?)\)?",
    re.IGNORECASE,
)

records = []

# Verified alias merges confirmed by report metadata and internal MRNs
alias_map = {
    "01566y211": "01566y21",
    "03128y211": "03128y21",
    "04412y221": "04412y22",
    "1000y16": "05128y22",
    "7708y181": "7708y18",
}

score_rank = {
    "0": 0,
    "1": 1,
    "2": 2,
    "3": 3,
    "4A": 4,
    "4": 4,
    "4B": 5,
    "4C": 6,
    "5": 7,
    "6": 8,
}

for p in patients:
    p_path = os.path.join(bcmid_dir, p)
    root_id = alias_map.get(p, p)

    # Modalities
    mg_dir = os.path.join(p_path, "Mammogram")
    us_dir = os.path.join(p_path, "Ultrasound")
    mg_files = [f for f in os.listdir(mg_dir) if f.lower().endswith(".png")] if os.path.exists(mg_dir) else []
    us_files = [f for f in os.listdir(us_dir) if f.lower().endswith(".png")] if os.path.exists(us_dir) else []

    has_corrupt = False
    valid_us = 0
    for uf in us_files:
        fp = os.path.join(us_dir, uf)
        if os.path.getsize(fp) == 0:
            has_corrupt = True
        else:
            valid_us += 1

    # PDF
    pdfs = [f for f in os.listdir(p_path) if f.lower().endswith(".pdf")]
    if not pdfs:
        records.append({
            "patient_id": p,
            "root_patient_id": root_id,
            "highest_birads": "NONE",
            "label_name": "EXCLUDED",
            "label_id": -1,
            "is_pathology_confirmed": False,
            "triage_confidence": 0.0,
            "extraction_status": "EXCLUDED_MISSING_REPORT",
            "side_findings": "NONE",
            "evidence_snippet": "No diagnostic PDF report present in patient directory",
            "pdf_report": "NONE",
            "num_mammogram_images": len(mg_files),
            "num_ultrasound_images": valid_us,
            "has_corrupted_image": has_corrupt,
        })
        continue

    pdf_name = pdfs[0]
    pdf_path = os.path.join(p_path, pdf_name)
    try:
        reader = pypdf.PdfReader(pdf_path)
        text = reader.pages[0].extract_text()
    except Exception as e:
        records.append({
            "patient_id": p,
            "root_patient_id": root_id,
            "highest_birads": "ERROR",
            "label_name": "EXCLUDED",
            "label_id": -1,
            "is_pathology_confirmed": False,
            "triage_confidence": 0.0,
            "extraction_status": "EXCLUDED_PARSE_ERROR",
            "side_findings": "NONE",
            "evidence_snippet": str(e),
            "pdf_report": pdf_name,
            "num_mammogram_images": len(mg_files),
            "num_ultrasound_images": valid_us,
            "has_corrupted_image": has_corrupt,
        })
        continue

    # Strip generic dictionary footer
    fmatch = footer_pattern.search(text)
    main_text = text[: fmatch.start()] if fmatch else text
    op_match = re.search(r"(?:OPINION|CONCLUSION|IMPRESSION)\s*[:\-]?(.*)", main_text, re.DOTALL | re.IGNORECASE)
    op_text = op_match.group(1).strip() if op_match else main_text.strip()

    # Side-specific matches
    right_m = re.findall(
        r"right[^\n\r\.;]*(?:birads|bi-rads|b-irads|category)\s*\(?[:\-\s]*([0-6][a-c]?)", main_text, re.IGNORECASE
    )
    left_m = re.findall(
        r"left[^\n\r\.;]*(?:birads|bi-rads|b-irads|category)\s*\(?[:\-\s]*([0-6][a-c]?)", main_text, re.IGNORECASE
    )

    # Generic matches in opinion then main
    op_scores = birads_pattern.findall(op_text)
    main_scores = birads_pattern.findall(main_text)
    all_scores = set(s.upper() for s in (op_scores or main_scores))

    # Handle spelling quirks
    if not all_scores:
        if re.search(r"biarads\s*6", main_text, re.IGNORECASE):
            all_scores.add("6")
        elif re.search(r"b-irads\s*1", main_text, re.IGNORECASE):
            all_scores.add("1")
        elif re.search(r"birad-s\s*3", main_text, re.IGNORECASE):
            all_scores.add("3")

    side_str = f"Right: {right_m}, Left: {left_m}" if (right_m or left_m) else "Bilateral/General"
    clean_op = op_text.replace("\n", " ").encode("ascii", errors="replace").decode("ascii").strip()
    evidence_snippet = clean_op[:250]

    # Determine highest severity score
    if "0" in all_scores:
        highest_score = "0"
        label_name = "EXCLUDED"
        label_id = -1
        is_pathology = False
        triage_conf = 0.0
        status = "EXCLUDED_INCOMPLETE"
    elif all_scores:
        sorted_scores = sorted(all_scores, key=lambda s: score_rank.get(s, -1), reverse=True)
        highest_score = sorted_scores[0]
        if score_rank.get(highest_score, -1) >= 4:
            label_name = "SUSPICIOUS_ACTIONABLE"
            label_id = 1
            is_pathology = (highest_score == "6")
            triage_conf = 1.0
            status = "CONFIRMED_BIRADS"
        else:
            label_name = "LOW_RISK"
            label_id = 0
            is_pathology = False
            triage_conf = 1.0
            status = "CONFIRMED_BIRADS"
    else:
        # Clinical narrative fallback
        lower_main = main_text.lower()
        if "pathologically proved" in lower_main and any(k in lower_main for k in ["cancer", "malignan", "carcinoma"]):
            highest_score = "6_PATHOLOGY"
            label_name = "SUSPICIOUS_ACTIONABLE"
            label_id = 1
            is_pathology = True
            triage_conf = 0.95
            status = "INFERRED_CLINICAL"
        elif "clear" in lower_main and "mastectomy" in lower_main and "satisfactory" in lower_main:
            highest_score = "2_CLEAR_FOLLOWUP"
            label_name = "LOW_RISK"
            label_id = 0
            is_pathology = False
            triage_conf = 0.90
            status = "INFERRED_CLINICAL"
        elif "fibro adenoma" in lower_main or "probably benign" in lower_main:
            highest_score = "3_PROBABLY_BENIGN"
            label_name = "LOW_RISK"
            label_id = 0
            is_pathology = False
            triage_conf = 0.85
            status = "INFERRED_CLINICAL"
        else:
            highest_score = "INDETERMINATE"
            label_name = "EXCLUDED"
            label_id = -1
            is_pathology = False
            triage_conf = 0.0
            status = "EXCLUDED_INDETERMINATE"

    records.append({
        "patient_id": p,
        "root_patient_id": root_id,
        "highest_birads": highest_score,
        "label_name": label_name,
        "label_id": label_id,
        "is_pathology_confirmed": is_pathology,
        "triage_confidence": triage_conf,
        "extraction_status": status,
        "side_findings": side_str,
        "evidence_snippet": evidence_snippet,
        "pdf_report": pdf_name,
        "num_mammogram_images": len(mg_files),
        "num_ultrasound_images": valid_us,
        "has_corrupted_image": has_corrupt,
    })

df = pd.DataFrame(records)
df.to_csv(output_tsv, sep="\t", index=False)
print("Generated BCMID_annotations.tsv with updated columns and semantics!")
print(f"Total rows: {len(df)}")
print("\nLabel distribution across 332 folders:")
print(df["label_name"].value_counts())

root_summary = df.groupby("root_patient_id").agg({
    "label_name": "first",
    "label_id": "first",
    "num_mammogram_images": "sum",
    "num_ultrasound_images": "sum",
}).reset_index()

print(f"\nTotal unique root patients: {len(root_summary)}")
print("Root patient label distribution:")
print(root_summary["label_name"].value_counts())

usable = root_summary[root_summary["label_id"] != -1]
n_low = (usable["label_id"] == 0).sum()
n_susp = (usable["label_id"] == 1).sum()
print(f"\nTotal eligible unique patients: {len(usable)}")
print(f"  LOW_RISK (0): {n_low}")
print(f"  SUSPICIOUS_ACTIONABLE (1): {n_susp}")
print(f"Eligible Mammogram images: {usable['num_mammogram_images'].sum()}")
print(f"Eligible Ultrasound images: {usable['num_ultrasound_images'].sum()}")

excluded = root_summary[root_summary["label_id"] == -1]
print(f"\nTotal excluded unique patients: {len(excluded)}")
print(f"Excluded Mammogram images: {excluded['num_mammogram_images'].sum()}")
print(f"Excluded Ultrasound images: {excluded['num_ultrasound_images'].sum()}")
