import os, re, json, hashlib
import pandas as pd

d10_dir = "outputs/phase_d/phase_d10_final_integrity_audit"
manuscript_md = os.path.join(d10_dir, "final_research_manuscript.md")
exec_summary_md = os.path.join(d10_dir, "executive_summary.md")
refs_md = os.path.join(d10_dir, "references.md")
checklist_md = os.path.join(d10_dir, "submission_readiness_checklist.md")
audit_report_md = os.path.join(d10_dir, "final_integrity_audit_report.md")

results = {}

# 1. Patient Identifiers and Privacy Issues
pid_regex = re.compile(r'\b\d{3,5}y\d{2,3}\b')
privacy_violations = []

for dirpath, _, filenames in os.walk(d10_dir):
    for f in filenames:
        if f.endswith(('.md', '.csv', '.json', '.html', '.txt')) and f != 'dataset_count_discrepancies.csv' and f != 'metric_consistency_audit.csv' and f != 'phase_d10_manifest.json':
            # Skip audit reports that specifically discuss the historical IDs in the audit context if needed,
            # but let's check publication files
            if 'corrected_manuscript' in dirpath or f in ['final_research_manuscript.md', 'executive_summary.md', 'references.md', 'limitations_and_future_work.md', 'final_results_tables.csv']:
                fpath = os.path.join(dirpath, f)
                with open(fpath, 'r', encoding='utf-8', errors='ignore') as fp:
                    matches = pid_regex.findall(fp.read())
                    if matches:
                        privacy_violations.append((fpath, matches))

results['check_1_patient_privacy'] = {
    'passed': len(privacy_violations) == 0,
    'violations': privacy_violations
}

# 2. Unsupported cancer-diagnosis claims
with open(manuscript_md, 'r', encoding='utf-8') as f:
    ms_text = f.read()

bad_phrases = [
    'cancer caught', 'cancers caught', 'cancers detected', 'cancer detected',
    'malignancies caught', 'malignancy caught', 'pathology confirmed', 'definitive diagnosis'
]
pathology_claims = []
for p in bad_phrases:
    if p in ms_text.lower():
        # verify if it's in a disclaimer like 'not a definitive diagnosis'
        for line in ms_text.splitlines():
            if p in line.lower() and not ('not ' in line.lower() or 'boundary' in line.lower() or 'research' in line.lower() or 'prioritize' in line.lower()):
                pathology_claims.append(line.strip())

results['check_2_clinical_framing'] = {
    'passed': len(pathology_claims) == 0,
    'claims': pathology_claims
}

# 3. Incorrect dataset counts or performance metrics
counts_to_check = {
    'raw_folders': '332',
    'exclusions': '17',
    'eligible_folders': '315',
    'root_patients': '311',
    'train_folders': '220',
    'train_roots': '217',
    'train_low_risk': '184',
    'train_suspicious': '36',
    'val_folders': '48',
    'val_roots': '47',
    'val_low_risk': '40',
    'val_suspicious': '8',
    'test_folders': '47',
    'test_roots': '47',
    'test_low_risk': '40',
    'test_suspicious': '7',
    'c3_acc': '85.11%',
    'c3_sens': '85.71%',
    'c3_spec': '85.00%',
    'c3_f1': '0.6316',
    'c3_auc': '0.9036',
    'd5_acc': '80.85%',
    'd5_sens': '85.71%',
    'd5_spec': '80.00%',
    'd5_f1': '0.5714',
    'd5_auc': '0.8964',
    'mcnemar_p': '0.5000'
}

metric_issues = []
for k, v in counts_to_check.items():
    if v not in ms_text:
        metric_issues.append(f"{k} with value {v} not found in manuscript text")

results['check_3_metrics_and_counts'] = {
    'passed': len(metric_issues) == 0,
    'issues': metric_issues
}

# 4. Missing or inconsistent citations
ref_section = ms_text[ms_text.find('## 13. References'):ms_text.find('## 14. Reproducibility Appendix')]
refs_found = re.findall(r'^(\d+)\.\s+', ref_section, re.MULTILINE)
total_refs = len(refs_found)

citations_in_body = set()
for m in re.findall(r'\[([0-9,\s–-]+)\]', ms_text[:ms_text.find('## 13. References')]):
    cleaned = m.replace('–', ' ').replace('-', ' ').strip()
    tokens = re.split(r'[\s,]+', cleaned)
    for t in tokens:
        if t.isdigit() and int(t) <= total_refs:
            citations_in_body.add(int(t))

missing_citations = [i for i in range(1, total_refs + 1) if i not in citations_in_body]

results['check_4_citations'] = {
    'passed': len(missing_citations) == 0,
    'total_refs': total_refs,
    'cited_in_body': sorted(list(citations_in_body)),
    'missing_citations': missing_citations
}

# 5. Broken figure and table references
figs_referenced = re.findall(r'Figure\s+(\d+)', ms_text)
tables_referenced = re.findall(r'Table\s+(\d+)', ms_text)

# Check physical figure files
fig_files = [
    "dataset_split_flowchart.png",
    "architecture_pipeline_diagram.png",
    "threshold_sensitivity_specificity_tradeoff.png",
    "reliability_diagram_c3_vs_d5.png",
    "confusion_matrices_c3_vs_d5.png",
    "roc_curves_c3_vs_d5.png",
    "pr_curves_c3_vs_d5.png",
    "reliability_curves_c3_vs_d5.png",
    "paired_prediction_scatter.png"
]
missing_figs = []
for f in fig_files:
    if not os.path.exists(os.path.join(d10_dir, "figures", f)):
        missing_figs.append(f)

results['check_5_figures_tables'] = {
    'passed': len(missing_figs) == 0,
    'figs_referenced': sorted(list(set(figs_referenced))),
    'missing_figures': missing_figs
}

# 6. Outdated or missing manuscript exports
exports = [
    "final_research_manuscript.pdf",
    "final_research_manuscript.docx",
    "final_research_manuscript.html"
]
missing_exports = []
for e in exports:
    epath = os.path.join(d10_dir, e)
    if not os.path.exists(epath) or os.path.getsize(epath) == 0:
        missing_exports.append(e)

results['check_6_exports'] = {
    'passed': len(missing_exports) == 0,
    'missing_exports': missing_exports
}

# 7. Unsupported claims of clinical readiness
claims_regex = re.compile(r'(clinically validated|ready for clinical (?:use|deployment)|replacement for (?:radiologist|biopsy))', re.IGNORECASE)
found_readiness = []
for line in ms_text.splitlines():
    if claims_regex.search(line):
        if not ('not ' in line.lower() or 'no claims' in line.lower() or 'does not' in line.lower() or 'unsupported' in line.lower()):
            found_readiness.append(line.strip())

results['check_7_clinical_readiness_claims'] = {
    'passed': len(found_readiness) == 0,
    'unsupported_claims': found_readiness
}

# 8. Missing required submission files
required_files = [
    "final_integrity_audit_report.md",
    "submission_readiness_checklist.md",
    "dataset_count_discrepancies.csv",
    "metric_consistency_audit.csv",
    "privacy_and_citation_audit.md",
    "final_research_manuscript.md",
    "executive_summary.md",
    "final_results_tables.csv",
    "limitations_and_future_work.md",
    "references.md",
    "final_research_manuscript.pdf",
    "final_research_manuscript.docx",
    "final_research_manuscript.html"
]
missing_req = []
for r in required_files:
    rpath = os.path.join(d10_dir, r)
    if not os.path.exists(rpath) or os.path.getsize(rpath) == 0:
        missing_req.append(r)

results['check_8_required_files'] = {
    'passed': len(missing_req) == 0,
    'missing_required_files': missing_req
}

all_passed = all(v['passed'] for v in results.values())
results['OVERALL_STATUS'] = "READY FOR SUBMISSION REVIEW" if all_passed else "NOT READY — ISSUES REMAIN"

print("=================================================================")
print(f"AUTOMATED VERIFICATION AUDIT: {results['OVERALL_STATUS']}")
print("=================================================================")
for k, v in results.items():
    if k != 'OVERALL_STATUS':
        status_str = "PASS" if v['passed'] else "FAIL"
        print(f"[{status_str}] {k}: {v}")
