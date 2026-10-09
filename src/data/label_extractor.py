import os
import re
import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
from .pdf_parser import parse_pdf_report


@dataclass
class PatientLabel:
    patient_id: str
    label_str: str          # "BENIGN", "MALIGNANT", or "UNKNOWN"
    label_id: int           # 0 for BENIGN, 1 for MALIGNANT, -1 for UNKNOWN
    confidence: float       # 0.0 to 1.0
    status: str             # "CONFIDENT", "INFERRED_CLINICAL", "UNKNOWN"
    reason: str             # Detailed explanation
    birads_score: Optional[str] = None
    pdf_filename: Optional[str] = None


def extract_label_from_text(text: str) -> Tuple[str, int, float, str, str, Optional[str]]:
    """Extracts diagnostic label from clinical report text.
    
    Returns:
        (label_str, label_id, confidence, status, reason, birads_score)
    """
    if not text or not text.strip():
        return "UNKNOWN", -1, 0.0, "UNKNOWN", "Empty or unreadable text", None

    # 1. Strip the standard template footer containing the generic BI-RADS dictionary
    footer_pattern = r'(?:Category\s+Breast\s+imaging|BI-?RAD\s*\(?0\)?\s*incomplete|Breast\s+imaging\s+reporting\s+and\s+data\s+system)'
    footer_match = re.search(footer_pattern, text, re.IGNORECASE)
    main_text = text[:footer_match.start()] if footer_match else text

    # 2. Extract OPINION / CONCLUSION / IMPRESSION section if present
    op_match = re.search(r'(?:OPINION|CONCLUSION|IMPRESSION)\s*[:\-]?(.*)', main_text, re.DOTALL | re.IGNORECASE)
    op_text = op_match.group(1) if op_match else main_text

    # 3. Regex for BI-RADS scores (handles variations: BIRADS, BI-RADS, BIRAD, BIARADS, BIRDAS, spaces)
    birads_pattern = r'(?:B\s*I\s*[-_]?\s*[AR]{1,2}\s*D\s*[AS]{0,2}|CATEGORY)\s*\(?[:\-\s]*([0-6][a-c]?)\)?'
    matches_op = re.findall(birads_pattern, op_text, re.IGNORECASE)
    matches_main = re.findall(birads_pattern, main_text, re.IGNORECASE)

    chosen_birads = matches_op[-1].upper() if matches_op else (matches_main[-1].upper() if matches_main else None)

    if chosen_birads:
        if chosen_birads in ['1', '2', '3']:
            return "BENIGN", 0, 0.95, "CONFIDENT", f"BI-RADS {chosen_birads} in report", chosen_birads
        elif chosen_birads in ['4', '4A', '4B', '4C', '5', '6']:
            return "MALIGNANT", 1, 0.95, "CONFIDENT", f"BI-RADS {chosen_birads} in report", chosen_birads
        elif chosen_birads == '0':
            # BI-RADS 0: Incomplete assessment. Check clinical context below.
            pass

    # 4. Clinical Contextual Patterns (interventions, pathology, active treatment)
    lower_main = main_text.lower()
    
    # Check for pathologically proved cancer/malignancy
    if re.search(r'pathologically\s+prov(?:ed|en)\s+(?:as\s+)?(?:bilateral\s+|left\s+|right\s+)?(?:malignan|breast\s+cancer|neoplastic)', lower_main):
        return "MALIGNANT", 1, 0.90, "INFERRED_CLINICAL", "Pathologically proved malignant neoplastic lesion", chosen_birads

    # Check for neoadjuvant chemotherapy markers or surgical clip insertion for known cancer
    if "neoadjuvant" in lower_main and any(kw in lower_main for kw in ["clip", "marker", "wire", "tumor", "lesion"]):
        return "MALIGNANT", 1, 0.90, "INFERRED_CLINICAL", "Neoadjuvant treatment localization for known breast carcinoma", chosen_birads

    # Check for procedure indicating active cancer management
    proc_match = re.search(r'procedure\s*[:\-]\s*([^\n\r]+)', lower_main)
    clin_match = re.search(r'clinical data\s*[:\-]\s*([^\n\r]+)', lower_main)
    proc = proc_match.group(1) if proc_match else ""
    clin = clin_match.group(1) if clin_match else ""
    
    if any(k in clin for k in ["cancer", "carcinoma", "malignan"]) and any(p in proc for p in ["clip", "marker", "wire"]):
        return "MALIGNANT", 1, 0.88, "INFERRED_CLINICAL", f"Active interventional procedure ({proc.strip()}) for {clin.strip()}", chosen_birads

    # Benign patterns (clear post-treatment follow-up, fibroadenoma, simple cysts)
    if re.search(r'clear\s+(?:left|right|both)\s+(?:breast|mastectomy\s+bed)', lower_main) or "satisfactory follow up" in lower_main:
        return "BENIGN", 0, 0.85, "INFERRED_CLINICAL", "Satisfactory clear follow-up with no residual/recurrent mass", chosen_birads

    if re.search(r'probably\s+benign|fibro\s*adenoma|simple\s+cyst', lower_main) and not re.search(r'malignan|carcinoma', lower_main):
        return "BENIGN", 0, 0.85, "INFERRED_CLINICAL", "Benign morphological findings (fibroadenoma/cyst)", chosen_birads

    # If inconclusive, do NOT guess
    return "UNKNOWN", -1, 0.0, "UNKNOWN", "No conclusive BI-RADS or definitive diagnostic indicator", chosen_birads


def extract_patient_labels(dataset_dir: str) -> Dict[str, PatientLabel]:
    """Iterates through all patient directories in dataset_dir and extracts labels.
    
    Args:
        dataset_dir: Path to BCMID directory.
        
    Returns:
        Dictionary mapping patient_id to PatientLabel dataclass.
    """
    patient_dirs = sorted([d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))])
    patient_labels = {}

    for p in patient_dirs:
        p_dir = os.path.join(dataset_dir, p)
        pdf_files = [f for f in os.listdir(p_dir) if f.lower().endswith('.pdf')]
        
        if not pdf_files:
            patient_labels[p] = PatientLabel(
                patient_id=p,
                label_str="UNKNOWN",
                label_id=-1,
                confidence=0.0,
                status="UNKNOWN",
                reason="Missing PDF report file",
                birads_score=None,
                pdf_filename=None
            )
            continue

        pdf_name = pdf_files[0]
        pdf_path = os.path.join(p_dir, pdf_name)
        text, err = parse_pdf_report(pdf_path)

        if err:
            patient_labels[p] = PatientLabel(
                patient_id=p,
                label_str="UNKNOWN",
                label_id=-1,
                confidence=0.0,
                status="UNKNOWN",
                reason=err,
                birads_score=None,
                pdf_filename=pdf_name
            )
            continue

        label_str, label_id, conf, status, reason, birads = extract_label_from_text(text)
        patient_labels[p] = PatientLabel(
            patient_id=p,
            label_str=label_str,
            label_id=label_id,
            confidence=conf,
            status=status,
            reason=reason,
            birads_score=birads,
            pdf_filename=pdf_name
        )

    return patient_labels
