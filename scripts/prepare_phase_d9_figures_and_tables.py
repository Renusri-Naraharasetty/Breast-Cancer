"""Phase D9: Figure and Table Generation Script.

Creates publication-quality figures and consolidated comparison tables for the final research deliverables:
- Architecture pipeline diagram (high-resolution PNG)
- Cohort and split flowchart diagram (high-resolution PNG)
- Copies existing verified figures from Phase D7 and D8
- Exports all structured comparative tables (CSV and Markdown)
- Generates master final_results_tables.csv
"""

import os
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Set output directories
out_dir = "outputs/phase_d/phase_d9_research_consolidation"
fig_dir = os.path.join(out_dir, "figures")
tbl_dir = os.path.join(out_dir, "tables")

os.makedirs(fig_dir, exist_ok=True)
os.makedirs(tbl_dir, exist_ok=True)

# -------------------------------------------------------------------------
# 1. Generate Architecture Pipeline Diagram
# -------------------------------------------------------------------------
def generate_architecture_diagram(save_path: str):
    fig, ax = plt.subplots(figsize=(13, 6.5), dpi=300)
    ax.axis("off")
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 6.5)

    # Style definitions
    box_props_mg = dict(boxstyle="round,pad=0.5", facecolor="#EFF6FF", edgecolor="#2563EB", lw=2)
    box_props_us = dict(boxstyle="round,pad=0.5", facecolor="#F0FDF4", edgecolor="#16A34A", lw=2)
    box_props_gat = dict(boxstyle="round,pad=0.5", facecolor="#FAF5FF", edgecolor="#9333EA", lw=2)
    box_props_tf = dict(boxstyle="round,pad=0.5", facecolor="#FFFBEB", edgecolor="#D97706", lw=2)
    box_props_head = dict(boxstyle="round,pad=0.5", facecolor="#FEF2F2", edgecolor="#DC2626", lw=2)

    # 1. Inputs: Mammogram
    ax.text(1.5, 5.0, "Mammogram Views\n(CC / MLO)\nRaw PNGs (224x224)", ha="center", va="center", fontsize=10, weight="bold", bbox=box_props_mg)
    ax.annotate("", xy=(3.0, 5.0), xytext=(2.3, 5.0), arrowprops=dict(arrowstyle="->", lw=1.8, color="#2563EB"))
    ax.text(3.9, 5.0, "EfficientNet-B0\nEncoder (Frozen)\n1280-dim vector", ha="center", va="center", fontsize=9.5, bbox=box_props_mg)

    # 2. Inputs: Ultrasound
    ax.text(1.5, 2.0, "Ultrasound Views\n(Multi-focal)\nRaw PNGs (224x224)", ha="center", va="center", fontsize=10, weight="bold", bbox=box_props_us)
    ax.annotate("", xy=(3.0, 2.0), xytext=(2.3, 2.0), arrowprops=dict(arrowstyle="->", lw=1.8, color="#16A34A"))
    ax.text(3.9, 2.0, "ViT-B/16\nEncoder (Frozen)\n768-dim vector", ha="center", va="center", fontsize=9.5, bbox=box_props_us)

    # Anatomical Side Embedding
    ax.text(2.7, 3.5, "Anatomical Side Embedding\n(Left, Right, Bilateral, Unknown)", ha="center", va="center", fontsize=9, style="italic",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#F3F4F6", edgecolor="#6B7280", lw=1.2))

    # Connect to GAT
    ax.annotate("", xy=(5.8, 4.2), xytext=(4.8, 5.0), arrowprops=dict(arrowstyle="->", lw=1.8, color="#4B5563"))
    ax.annotate("", xy=(5.8, 3.5), xytext=(3.9, 3.5), arrowprops=dict(arrowstyle="->", lw=1.5, color="#6B7280", linestyle="--"))
    ax.annotate("", xy=(5.8, 2.8), xytext=(4.8, 2.0), arrowprops=dict(arrowstyle="->", lw=1.8, color="#4B5563"))

    # 3. GAT Fusion Layer
    gat_text = "Side-Aware GAT Module\n(2 Layers, 4 Heads)\n" \
               "• Hidden Dim = 256\n" \
               "• Graph adjacency across views\n" \
               "• Side-aware attention biases"
    ax.text(6.8, 3.5, gat_text, ha="center", va="center", fontsize=9.5, bbox=box_props_gat)

    # Connect GAT to Transformer
    ax.annotate("", xy=(7.9, 3.5), xytext=(7.7, 3.5), arrowprops=dict(arrowstyle="->", lw=2, color="#9333EA"))

    # 4. Cross-Attention Transformer
    tf_text = "Multimodal Transformer\n(2 Layers, 8 Heads)\n" \
              "• Cross-attention aggregation\n" \
              "• Feedforward Dim = 1024\n" \
              "• Dropout Rate = 0.30\n" \
              "• Patient-level pooled representation"
    ax.text(9.1, 3.5, tf_text, ha="center", va="center", fontsize=9.5, bbox=box_props_tf)

    # Connect Transformer to Classifier
    ax.annotate("", xy=(10.3, 3.5), xytext=(10.1, 3.5), arrowprops=dict(arrowstyle="->", lw=2, color="#D97706"))

    # 5. Classification Head & Output
    head_text = "Linear Classification Head\n(2 Classes, Softmax)\n" \
                "• Class-Balanced Focal Loss\n" \
                "  (gamma = 1.5, alpha = [1.0, 5.11])\n" \
                "• Output: P(Suspicious | Images)\n" \
                "• Threshold tau = 0.50"
    ax.text(11.6, 3.5, head_text, ha="center", va="center", fontsize=9.5, bbox=box_props_head)

    # Title & Subtitle
    plt.title("Multimodal Graph-Infused Transformer (GIT) Architecture for Radiologic Breast Triage", fontsize=13, weight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Generated {save_path}")

# -------------------------------------------------------------------------
# 2. Generate Dataset & Cohort Flowchart
# -------------------------------------------------------------------------
def generate_cohort_flowchart(save_path: str):
    fig, ax = plt.subplots(figsize=(12, 7.0), dpi=300)
    ax.axis("off")
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7.0)

    # Style definitions
    box_all = dict(boxstyle="round,pad=0.5", facecolor="#F8FAFC", edgecolor="#334155", lw=2)
    box_excl = dict(boxstyle="round,pad=0.5", facecolor="#FEF2F2", edgecolor="#EF4444", lw=1.8, linestyle="--")
    box_elig = dict(boxstyle="round,pad=0.5", facecolor="#EFF6FF", edgecolor="#2563EB", lw=2)
    box_train = dict(boxstyle="round,pad=0.5", facecolor="#F0FDF4", edgecolor="#16A34A", lw=2)
    box_val = dict(boxstyle="round,pad=0.5", facecolor="#FFFBEB", edgecolor="#D97706", lw=2)
    box_test = dict(boxstyle="round,pad=0.5", facecolor="#FAF5FF", edgecolor="#9333EA", lw=2)

    # Top: Raw Cohort
    ax.text(6.0, 6.3, "BCMID Raw Patient Folders\nTotal Folders = 332 | Annotation Metadata in BCMID_annotations.tsv",
            ha="center", va="center", fontsize=10.5, weight="bold", bbox=box_all)

    # Split to Excluded and Eligible
    ax.annotate("", xy=(2.5, 4.8), xytext=(5.0, 6.0), arrowprops=dict(arrowstyle="->", lw=1.8, color="#EF4444"))
    ax.annotate("", xy=(8.5, 4.8), xytext=(7.0, 6.0), arrowprops=dict(arrowstyle="->", lw=2.0, color="#2563EB"))

    # Excluded Box
    excl_text = "Excluded Non-Diagnostic Folders (N = 17)\n" \
                "• BI-RADS NONE: 11 folders\n" \
                "• INDETERMINATE: 5 folders\n" \
                "• Category 0: 1 folder\n" \
                "(Strictly isolated from all training & testing)"
    ax.text(2.5, 4.8, excl_text, ha="center", va="center", fontsize=9, bbox=box_excl)

    # Eligible Cohort Box
    elig_text = "Audited Eligible Cohort\n" \
                "• Patient Folders: N = 315\n" \
                "• Unique Biological Root Patients: N = 311\n" \
                "• Repeat-Examination Patients: N = 4 (8 folders)\n" \
                "• Class 0 (LOW_RISK, BI-RADS 1-3): 264 folders (83.8%)\n" \
                "• Class 1 (SUSPICIOUS, BI-RADS 4-6): 51 folders (16.2%)"
    ax.text(8.5, 4.8, elig_text, ha="center", va="center", fontsize=9.5, weight="bold", bbox=box_elig)

    # Connect Eligible to Splits
    ax.annotate("", xy=(2.2, 2.5), xytext=(7.2, 4.0), arrowprops=dict(arrowstyle="->", lw=1.8, color="#16A34A"))
    ax.annotate("", xy=(6.0, 2.5), xytext=(8.5, 4.0), arrowprops=dict(arrowstyle="->", lw=1.8, color="#D97706"))
    ax.annotate("", xy=(9.8, 2.5), xytext=(9.8, 4.0), arrowprops=dict(arrowstyle="->", lw=1.8, color="#9333EA"))

    # Training Split
    train_text = "Training Partition (70%)\n" \
                 "• Folders: 220\n" \
                 "• Root Patients: 217\n" \
                 "• Multi-visit Roots: 3 (6 folders)\n" \
                 "• Class 0 (LOW_RISK): 184 (83.6%)\n" \
                 "• Class 1 (SUSPICIOUS): 36 (16.4%)\n" \
                 "• Ratio (Neg/Pos): 5.1111 : 1\n" \
                 "• Images: 644 MG / 671 US"
    ax.text(2.2, 2.2, train_text, ha="center", va="center", fontsize=9, bbox=box_train)

    # Validation Split
    val_text = "Validation Partition (15%)\n" \
               "• Folders: 48\n" \
               "• Root Patients: 47\n" \
               "• Multi-visit Roots: 1 (2 folders)\n" \
               "• Class 0 (LOW_RISK): 40 (83.3%)\n" \
               "• Class 1 (SUSPICIOUS): 8 (16.7%)\n" \
               "• Ratio (Neg/Pos): 5.0000 : 1\n" \
               "• Images: 138 MG / 135 US"
    ax.text(6.0, 2.2, val_text, ha="center", va="center", fontsize=9, bbox=box_val)

    # Held-Out Test Split
    test_text = "Held-Out Test Partition (15%)\n" \
                "• Folders: 47\n" \
                "• Root Patients: 47 (100% unique)\n" \
                "• Multi-visit Roots: 0\n" \
                "• Class 0 (LOW_RISK): 40 (85.1%)\n" \
                "• Class 1 (SUSPICIOUS): 7 (14.9%)\n" \
                "• Ratio (Neg/Pos): 5.7143 : 1\n" \
                "• Images: 132 MG / 150 US\n" \
                "• STRICTLY FROZEN GATE"
    ax.text(9.8, 2.2, test_text, ha="center", va="center", fontsize=9, weight="bold", bbox=box_test)

    # Footnote
    ax.text(6.0, 0.4, "Patient-Level Stratification Guarantee: Zero root patient overlap across splits (Train ∩ Val = ∅, Train ∩ Test = ∅, Val ∩ Test = ∅).\n"
                       "Cohort reconciliation: 311 unique patients + 4 repeat visits = 315 patient folders. Legacy '314' was an arithmetic typo in an earlier table sum (220+48+47=315).",
            ha="center", va="center", fontsize=8.5, style="italic", color="#475569")

    plt.title("BCMID Patient-Level Stratified Cohort Flowchart and Split Architecture", fontsize=13, weight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Generated {save_path}")

# -------------------------------------------------------------------------
# 3. Copy Existing Figures from Phase D7 and D8
# -------------------------------------------------------------------------
def copy_verified_figures():
    src_d7_fig = "outputs/phase_d/figures"
    src_d8_fig = "outputs/phase_d/phase_d8_final_evaluation/figures"

    copied_files = []
    # D7 Figures
    for fname in ["threshold_sensitivity_specificity_tradeoff.png", "reliability_diagram_c3_vs_d5.png", "threshold_metrics_curve_c3_vs_d5.png"]:
        src_path = os.path.join(src_d7_fig, fname)
        dst_path = os.path.join(fig_dir, fname)
        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
            copied_files.append(dst_path)

    # D8 Figures
    for fname in ["roc_curves_c3_vs_d5.png", "pr_curves_c3_vs_d5.png", "confusion_matrices_c3_vs_d5.png", "paired_prediction_scatter.png", "reliability_curves_c3_vs_d5.png"]:
        src_path = os.path.join(src_d8_fig, fname)
        dst_path = os.path.join(fig_dir, fname)
        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
            copied_files.append(dst_path)

    print(f"Copied {len(copied_files)} verified figures to {fig_dir}")

# -------------------------------------------------------------------------
# 4. Generate All Standardized Tables (CSV and Markdown)
# -------------------------------------------------------------------------
def generate_tables():
    # Table 1: Individual Modality Baselines (Exp 1 - 4)
    t1_data = [
        {"Model": "Exp 1: EfficientNet-B0", "Modality": "Mammogram", "Level": "Patient", "Accuracy": "85.11%", "Sensitivity": "14.29% (1/7)", "Specificity": "97.50% (39/40)", "Precision": "50.00% (1/2)", "F1-Score": "0.2222", "Balanced Acc": "55.89%", "ROC-AUC": "0.6571", "PR-AUC": "0.3079", "CM [TN, FP, FN, TP]": "[39, 1, 6, 1]"},
        {"Model": "Exp 2: ViT-B/16", "Modality": "Ultrasound", "Level": "Patient", "Accuracy": "91.49%", "Sensitivity": "57.14% (4/7)", "Specificity": "97.50% (39/40)", "Precision": "80.00% (4/5)", "F1-Score": "0.6667", "Balanced Acc": "77.32%", "ROC-AUC": "0.8357", "PR-AUC": "0.7099", "CM [TN, FP, FN, TP]": "[39, 1, 3, 4]"},
        {"Model": "Exp 3: ViT-B/16", "Modality": "Mammogram", "Level": "Patient", "Accuracy": "87.23%", "Sensitivity": "14.29% (1/7)", "Specificity": "100.00% (40/40)", "Precision": "100.00% (1/1)", "F1-Score": "0.2500", "Balanced Acc": "57.14%", "ROC-AUC": "0.7893", "PR-AUC": "0.4130", "CM [TN, FP, FN, TP]": "[40, 0, 6, 1]"},
        {"Model": "Exp 4: EfficientNet-B0", "Modality": "Ultrasound", "Level": "Patient", "Accuracy": "48.94%", "Sensitivity": "100.00% (7/7)", "Specificity": "40.00% (16/40)", "Precision": "22.58% (7/31)", "F1-Score": "0.3684", "Balanced Acc": "70.00%", "ROC-AUC": "0.8357", "PR-AUC": "0.4999", "CM [TN, FP, FN, TP]": "[16, 24, 0, 7]"},
    ]
    df_t1 = pd.DataFrame(t1_data)
    df_t1.to_csv(os.path.join(tbl_dir, "table1_individual_baselines.csv"), index=False)
    with open(os.path.join(tbl_dir, "table1_individual_baselines.md"), "w") as f:
        f.write(df_t1.to_markdown(index=False))

    # Table 2: Multimodal GIT Baselines
    t2_data = [
        {"Model Designation": "GIT 1 Baseline", "Architecture": "EffNet MG + ViT US", "Selection Protocol": "Min Val Loss (Ep 3)", "Accuracy": "82.98% (39/47)", "Sensitivity": "71.43% (5/7)", "Specificity": "85.00% (34/40)", "Precision": "45.45% (5/11)", "F1-Score": "0.5556", "Balanced Acc": "78.21%", "ROC-AUC": "0.8857", "PR-AUC": "0.5729", "CM [TN, FP, FN, TP]": "[34, 6, 2, 5]"},
        {"Model Designation": "GIT 2 Baseline", "Architecture": "ViT MG + EffNet US", "Selection Protocol": "Min Val Loss (Ep 2)", "Accuracy": "63.83% (30/47)", "Sensitivity": "100.00% (7/7)", "Specificity": "57.50% (23/40)", "Precision": "29.17% (7/24)", "F1-Score": "0.4516", "Balanced Acc": "78.75%", "ROC-AUC": "0.8321", "PR-AUC": "0.4048", "CM [TN, FP, FN, TP]": "[23, 17, 0, 7]"},
    ]
    df_t2 = pd.DataFrame(t2_data)
    df_t2.to_csv(os.path.join(tbl_dir, "table2_git_baselines.csv"), index=False)
    with open(os.path.join(tbl_dir, "table2_git_baselines.md"), "w") as f:
        f.write(df_t2.to_markdown(index=False))

    # Table 3: Phase C Focal Experiments (Validation Cohort N=48)
    t3_data = [
        {"Experiment ID": "Original GIT 1", "Loss Formulation": "Weighted BCE", "Selected Epoch": 3, "Val Loss": "0.4069", "Composite Score (S)": "0.7922", "Val ROC-AUC": "0.9250", "Val PR-AUC": "0.7436", "Val Sensitivity": "50.00% (4/8)", "Val Specificity": "92.50% (37/40)", "Val F1": "0.5333", "Val CM": "[37, 3, 4, 4]"},
        {"Experiment ID": "C1 (BCE Repro)", "Loss Formulation": "Weighted BCE", "Selected Epoch": 4, "Val Loss": "0.4272", "Composite Score (S)": "0.8086", "Val ROC-AUC": "0.9187", "Val PR-AUC": "0.7432", "Val Sensitivity": "75.00% (6/8)", "Val Specificity": "87.50% (35/40)", "Val F1": "0.6316", "Val CM": "[35, 5, 2, 6]"},
        {"Experiment ID": "C2 (Focal gamma=2.0)", "Loss Formulation": "Focal Loss (gamma=2.0)", "Selected Epoch": 10, "Val Loss": "0.2537", "Composite Score (S)": "0.8276", "Val ROC-AUC": "0.9219", "Val PR-AUC": "0.7516", "Val Sensitivity": "75.00% (6/8)", "Val Specificity": "92.50% (37/40)", "Val F1": "0.7059", "Val CM": "[37, 3, 2, 6]"},
        {"Experiment ID": "C3 (Focal gamma=1.5, PRIMARY)", "Loss Formulation": "Focal Loss (gamma=1.5)", "Selected Epoch": 4, "Val Loss": "0.1679", "Composite Score (S)": "0.8351", "Val ROC-AUC": "0.9281", "Val PR-AUC": "0.7661", "Val Sensitivity": "75.00% (6/8)", "Val Specificity": "92.50% (37/40)", "Val F1": "0.7059", "Val CM": "[37, 3, 2, 6]"},
        {"Experiment ID": "C4 (W-BCE low-lr)", "Loss Formulation": "Weighted BCE (lr=5e-5)", "Selected Epoch": 7, "Val Loss": "0.4664", "Composite Score (S)": "0.8338", "Val ROC-AUC": "0.9219", "Val PR-AUC": "0.7667", "Val Sensitivity": "62.50% (5/8)", "Val Specificity": "97.50% (39/40)", "Val F1": "0.7143", "Val CM": "[39, 1, 3, 5]"},
        {"Experiment ID": "C5 (Focal low-lr)", "Loss Formulation": "Focal Loss (lr=5e-5, wd=0.02)", "Selected Epoch": 8, "Val Loss": "0.1227", "Composite Score (S)": "0.8122", "Val ROC-AUC": "0.9219", "Val PR-AUC": "0.7708", "Val Sensitivity": "75.00% (6/8)", "Val Specificity": "85.00% (34/40)", "Val F1": "0.6000", "Val CM": "[34, 6, 2, 6]"},
    ]
    df_t3 = pd.DataFrame(t3_data)
    df_t3.to_csv(os.path.join(tbl_dir, "table3_phase_c_experiments.csv"), index=False)
    with open(os.path.join(tbl_dir, "table3_phase_c_experiments.md"), "w") as f:
        f.write(df_t3.to_markdown(index=False))

    # Table 4: Phase D Ablation Experiments (D1 - D5)
    t4_data = [
        {"Experiment ID": "D1 (C3 Repro)", "Parameter Varied": "Exact Baseline Repro", "Selected Epoch": 4, "Composite Score (S)": "0.8351", "Delta vs C3": "0.0000", "Val ROC-AUC": "0.9281", "Val PR-AUC": "0.7661", "Val Sensitivity": "75.00% (6/8)", "Val Specificity": "92.50% (37/40)", "Val F1": "0.7059", "Val Loss": "0.1679", "Train Loss": "0.1590", "Loss Gap": "+0.0089", "Val CM": "[37, 3, 2, 6]"},
        {"Experiment ID": "D2 (Focal gamma=1.0)", "Parameter Varied": "gamma = 1.0", "Selected Epoch": 4, "Composite Score (S)": "0.8351", "Delta vs C3": "0.0000", "Val ROC-AUC": "0.9281", "Val PR-AUC": "0.7661", "Val Sensitivity": "75.00% (6/8)", "Val Specificity": "92.50% (37/40)", "Val F1": "0.7059", "Val Loss": "0.2250", "Train Loss": "0.2026", "Loss Gap": "+0.0224", "Val CM": "[37, 3, 2, 6]"},
        {"Experiment ID": "D3 (Focal gamma=2.0)", "Parameter Varied": "gamma = 2.0", "Selected Epoch": 10, "Composite Score (S)": "0.8276", "Delta vs C3": "-0.0075", "Val ROC-AUC": "0.9219", "Val PR-AUC": "0.7516", "Val Sensitivity": "75.00% (6/8)", "Val Specificity": "92.50% (37/40)", "Val F1": "0.7059", "Val Loss": "0.2537", "Train Loss": "0.0075", "Loss Gap": "+0.2462", "Val CM": "[37, 3, 2, 6]"},
        {"Experiment ID": "D4 (Dropout=0.40)", "Parameter Varied": "dropout = 0.40", "Selected Epoch": 18, "Composite Score (S)": "0.8334", "Delta vs C3": "-0.0017", "Val ROC-AUC": "0.9313", "Val PR-AUC": "0.7816", "Val Sensitivity": "50.00% (4/8)", "Val Specificity": "100.00% (40/40)", "Val F1": "0.6667", "Val Loss": "0.5854", "Train Loss": "0.0068", "Loss Gap": "+0.5786", "Val CM": "[40, 0, 4, 4]"},
        {"Experiment ID": "D5 (Conservative LR)", "Parameter Varied": "lr = 5e-5", "Selected Epoch": 4, "Composite Score (S)": "0.8468", "Delta vs C3": "+0.0117", "Val ROC-AUC": "0.9281", "Val PR-AUC": "0.7573", "Val Sensitivity": "87.50% (7/8)", "Val Specificity": "92.50% (37/40)", "Val F1": "0.7778", "Val Loss": "0.1390", "Train Loss": "0.1585", "Loss Gap": "-0.0195", "Val CM": "[37, 3, 1, 7]"},
    ]
    df_t4 = pd.DataFrame(t4_data)
    df_t4.to_csv(os.path.join(tbl_dir, "table4_phase_d_ablations.csv"), index=False)
    with open(os.path.join(tbl_dir, "table4_phase_d_ablations.md"), "w") as f:
        f.write(df_t4.to_markdown(index=False))

    # Table 5: Multi-Seed Stability Summary (Seeds 42, 123, 456)
    t5_data = [
        {"Model Configuration": "C3 Benchmark (lr=1e-4)", "Selected Epoch (Mean +/- SD)": "4.67 +/- 0.58", "Composite Score S": "0.8527 +/- 0.0224", "ROC-AUC": "0.9406 +/- 0.0125", "PR-AUC": "0.8036 +/- 0.0511", "Sensitivity (Recall)": "75.00% +/- 12.50%", "Specificity": "92.50% +/- 5.00%", "Precision (PPV)": "69.44% +/- 12.73%", "F1-Score": "0.7067 +/- 0.0072", "Balanced Acc": "83.75% +/- 3.75%", "Val Loss": "0.1813 +/- 0.0553", "Mean TP": "6.00 +/- 1.00", "Mean FP": "3.00 +/- 2.00"},
        {"Model Configuration": "Candidate D5 (lr=5e-5)", "Selected Epoch (Mean +/- SD)": "4.33 +/- 2.52", "Composite Score S": "0.8532 +/- 0.0200", "ROC-AUC": "0.9385 +/- 0.0130", "PR-AUC": "0.7864 +/- 0.0457", "Sensitivity (Recall)": "83.33% +/- 7.22%", "Specificity": "91.67% +/- 1.44%", "Precision (PPV)": "66.77% +/- 3.18%", "F1-Score": "0.7402 +/- 0.0361", "Balanced Acc": "87.50% +/- 3.31%", "Val Loss": "0.1530 +/- 0.0131", "Mean TP": "6.67 +/- 0.58", "Mean FP": "3.33 +/- 0.58"},
        {"Model Configuration": "Delta (D5 - C3)", "Selected Epoch (Mean +/- SD)": "-0.33", "Composite Score S": "+0.0005", "ROC-AUC": "-0.0021", "PR-AUC": "-0.0172", "Sensitivity (Recall)": "+8.33%", "Specificity": "-0.83%", "Precision (PPV)": "-2.67%", "F1-Score": "+0.0334", "Balanced Acc": "+3.75%", "Val Loss": "-0.0283", "Mean TP": "+0.67", "Mean FP": "+0.33"},
    ]
    df_t5 = pd.DataFrame(t5_data)
    df_t5.to_csv(os.path.join(tbl_dir, "table5_multi_seed_stability.csv"), index=False)
    with open(os.path.join(tbl_dir, "table5_multi_seed_stability.md"), "w") as f:
        f.write(df_t5.to_markdown(index=False))

    # Table 6: Locked Held-Out Test Evaluation Comparison
    t6_data = [
        {"Metric": "Confusion Matrix [TN, FP, FN, TP]", "C3 Benchmark (lr=1e-4)": "[34, 6, 1, 6]", "Candidate D5 (lr=5e-5)": "[32, 8, 1, 6]", "Delta (D5 - C3)": "TN: -2, FP: +2, FN: 0, TP: 0"},
        {"Metric": "Accuracy", "C3 Benchmark (lr=1e-4)": "85.11% (40/47)", "Candidate D5 (lr=5e-5)": "80.85% (38/47)", "Delta (D5 - C3)": "-4.26%"},
        {"Metric": "Sensitivity (Recall)", "C3 Benchmark (lr=1e-4)": "85.71% (6/7)", "Candidate D5 (lr=5e-5)": "85.71% (6/7)", "Delta (D5 - C3)": "0.00%"},
        {"Metric": "Specificity", "C3 Benchmark (lr=1e-4)": "85.00% (34/40)", "Candidate D5 (lr=5e-5)": "80.00% (32/40)", "Delta (D5 - C3)": "-5.00%"},
        {"Metric": "Precision (PPV)", "C3 Benchmark (lr=1e-4)": "50.00% (6/12)", "Candidate D5 (lr=5e-5)": "42.86% (6/14)", "Delta (D5 - C3)": "-7.14%"},
        {"Metric": "Negative Predictive Value (NPV)", "C3 Benchmark (lr=1e-4)": "97.14% (34/35)", "Candidate D5 (lr=5e-5)": "96.97% (32/33)", "Delta (D5 - C3)": "-0.17%"},
        {"Metric": "F1-Score", "C3 Benchmark (lr=1e-4)": "0.6316", "Candidate D5 (lr=5e-5)": "0.5714", "Delta (D5 - C3)": "-0.0602"},
        {"Metric": "Balanced Accuracy", "C3 Benchmark (lr=1e-4)": "85.36%", "Candidate D5 (lr=5e-5)": "82.86%", "Delta (D5 - C3)": "-2.50%"},
        {"Metric": "ROC-AUC", "C3 Benchmark (lr=1e-4)": "0.9036", "Candidate D5 (lr=5e-5)": "0.8964", "Delta (D5 - C3)": "-0.0072"},
        {"Metric": "PR-AUC (Prevalence: 14.89%)", "C3 Benchmark (lr=1e-4)": "0.6188", "Candidate D5 (lr=5e-5)": "0.6151", "Delta (D5 - C3)": "-0.0037"},
        {"Metric": "Brier Score", "C3 Benchmark (lr=1e-4)": "0.1133", "Candidate D5 (lr=5e-5)": "0.1281", "Delta (D5 - C3)": "+0.0148"},
        {"Metric": "Log Loss", "C3 Benchmark (lr=1e-4)": "0.3519", "Candidate D5 (lr=5e-5)": "0.4075", "Delta (D5 - C3)": "+0.0556"},
        {"Metric": "Sensitivity 95% CI (Wilson)", "C3 Benchmark (lr=1e-4)": "[0.4869, 0.9743]", "Candidate D5 (lr=5e-5)": "[0.4869, 0.9743]", "Delta (D5 - C3)": "Identical"},
        {"Metric": "Sensitivity 95% CI (Clopper-Pearson)", "C3 Benchmark (lr=1e-4)": "[0.4213, 0.9964]", "Candidate D5 (lr=5e-5)": "[0.4213, 0.9964]", "Delta (D5 - C3)": "Identical"},
        {"Metric": "Specificity 95% CI (Wilson)", "C3 Benchmark (lr=1e-4)": "[0.7093, 0.9294]", "Candidate D5 (lr=5e-5)": "[0.6524, 0.8950]", "Delta (D5 - C3)": "Shifted by 2 FP"},
        {"Metric": "ROC-AUC 95% CI (Bootstrap)", "C3 Benchmark (lr=1e-4)": "[0.7928, 0.9821]", "Candidate D5 (lr=5e-5)": "[0.7750, 0.9786]", "Delta (D5 - C3)": "Broad overlap"},
        {"Metric": "PR-AUC 95% CI (Bootstrap)", "C3 Benchmark (lr=1e-4)": "[0.3821, 0.9286]", "Candidate D5 (lr=5e-5)": "[0.3831, 0.9137]", "Delta (D5 - C3)": "Broad overlap"},
    ]
    df_t6 = pd.DataFrame(t6_data)
    df_t6.to_csv(os.path.join(tbl_dir, "table6_locked_test_comparison.csv"), index=False)
    with open(os.path.join(tbl_dir, "table6_locked_test_comparison.md"), "w") as f:
        f.write(df_t6.to_markdown(index=False))

    # Table 7: Paired Contingency Table
    t7_data = [
        {"D5 Classification": "D5 Correct", "C3 Correct": 38, "C3 Incorrect": 0, "Total D5 Correct": 38},
        {"D5 Classification": "D5 Incorrect", "C3 Correct": 2, "C3 Incorrect": 7, "Total D5 Incorrect": 9},
        {"D5 Classification": "Total C3", "C3 Correct": 40, "C3 Incorrect": 7, "Total D5 Correct": 47},
    ]
    df_t7 = pd.DataFrame(t7_data)
    df_t7.to_csv(os.path.join(tbl_dir, "table7_paired_contingency.csv"), index=False)
    with open(os.path.join(tbl_dir, "table7_paired_contingency.md"), "w") as f:
        f.write(df_t7.to_markdown(index=False))

    # Master Table: final_results_tables.csv
    master_rows = []
    # Add Phase 5 Baselines
    for r in t1_data:
        master_rows.append({"Phase": "Phase 1-5 (Single Modality)", "Model": r["Model"], "Modality": r["Modality"], "Cohort": "Held-Out Test (N=47)", "Accuracy": r["Accuracy"], "Sensitivity": r["Sensitivity"], "Specificity": r["Specificity"], "Precision": r["Precision"], "F1-Score": r["F1-Score"], "Balanced Acc": r["Balanced Acc"], "ROC-AUC": r["ROC-AUC"], "PR-AUC": r["PR-AUC"]})
    for r in t2_data:
        master_rows.append({"Phase": "Phase 5 (Multimodal GIT)", "Model": r["Model Designation"], "Modality": "Multimodal", "Cohort": "Held-Out Test (N=47)", "Accuracy": r["Accuracy"], "Sensitivity": r["Sensitivity"], "Specificity": r["Specificity"], "Precision": r["Precision"], "F1-Score": r["F1-Score"], "Balanced Acc": r["Balanced Acc"], "ROC-AUC": r["ROC-AUC"], "PR-AUC": r["PR-AUC"]})
    # Add Locked Final Test
    master_rows.append({"Phase": "Phase D8 (Locked Final Test)", "Model": "C3 Benchmark (lr=1e-4)", "Modality": "Multimodal", "Cohort": "Held-Out Test (N=47)", "Accuracy": "85.11% (40/47)", "Sensitivity": "85.71% (6/7)", "Specificity": "85.00% (34/40)", "Precision": "50.00% (6/12)", "F1-Score": "0.6316", "Balanced Acc": "85.36%", "ROC-AUC": "0.9036", "PR-AUC": "0.6188"})
    master_rows.append({"Phase": "Phase D8 (Locked Final Test)", "Model": "Candidate D5 (lr=5e-5)", "Modality": "Multimodal", "Cohort": "Held-Out Test (N=47)", "Accuracy": "80.85% (38/47)", "Sensitivity": "85.71% (6/7)", "Specificity": "80.00% (32/40)", "Precision": "42.86% (6/14)", "F1-Score": "0.5714", "Balanced Acc": "82.86%", "ROC-AUC": "0.8964", "PR-AUC": "0.6151"})

    df_master = pd.DataFrame(master_rows)
    master_csv_path = os.path.join(out_dir, "final_results_tables.csv")
    df_master.to_csv(master_csv_path, index=False)
    print(f"Generated master results table at {master_csv_path}")

def main():
    print("Preparing Phase D9 figures and tables...")
    generate_architecture_diagram(os.path.join(fig_dir, "architecture_pipeline_diagram.png"))
    generate_cohort_flowchart(os.path.join(fig_dir, "dataset_split_flowchart.png"))
    copy_verified_figures()
    generate_tables()
    print("All Phase D9 figures and tables prepared successfully.")

if __name__ == "__main__":
    main()
