"""Inference Consistency Audit: Evaluation Pipeline vs Application Inference Engine."""

import os
import sys
import json
import yaml
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.app.inference_engine import MultimodalInferenceEngine
from scripts.evaluate import build_model, MODEL_REGISTRY, run_inference_on_loader
from src.data.dataset import BCMIDDataset, collate_patient_batch
from src.data.label_extractor import PatientLabel
from torch.utils.data import DataLoader


def run_audit():
    print("=" * 70)
    print("STARTING INFERENCE CONSISTENCY AUDIT")
    print("=" * 70)

    # 1. Config and paths
    config_path = "configs/config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = torch.device("cpu") # Test on CPU for exact deterministic comparison

    # CHECK 1: Checkpoints
    print("\n--- CHECK 1: MODEL CHECKPOINTS ---")
    ckpt_gt = "outputs/experiments/graph_transformer/checkpoints/best_model.pth"
    ckpt_mammo = "outputs/experiments/densenet_mammogram/checkpoints/best_model.pth"
    ckpt_us = "outputs/experiments/resnet_ultrasound/checkpoints/best_model.pth"

    for name, p in [("Graph Transformer", ckpt_gt), ("DenseNet-121", ckpt_mammo), ("ResNet-50", ckpt_us)]:
        if not os.path.exists(p):
            print(f"ERROR: Checkpoint missing for {name} at {p}")
        else:
            data = torch.load(p, map_location="cpu")
            epoch = data.get("epoch")
            is_smoke = data.get("is_smoke_test", False)
            print(f"{name}: Exists at {p}, Epoch={epoch}, SmokeTest={is_smoke}")

    # Initialize Application Engine
    app_engine = MultimodalInferenceEngine(
        config_path=config_path,
        gt_checkpoint_path=ckpt_gt,
        densenet_checkpoint_path=ckpt_mammo,
        resnet_checkpoint_path=ckpt_us,
        device="cpu"
    )

    # CHECK 2: Class Mapping
    print("\n--- CHECK 2: CLASS MAPPING ---")
    print("Class 0: Negative / Benign")
    print("Class 1: Positive / Suspicious / Malignant")
    print("Softmax[:, 0] = prob_benign, Softmax[:, 1] = prob_malignant")

    # CHECK 3: Thresholds
    print("\n--- CHECK 3: THRESHOLDS ---")
    print(f"Graph Transformer Calibrated Threshold: {app_engine.THRESHOLDS['graph_transformer']} (Expected: 0.2565)")
    print(f"DenseNet-121 Calibrated Threshold:      {app_engine.THRESHOLDS['densenet']} (Expected: 0.2932)")
    print(f"ResNet-50 Calibrated Threshold:         {app_engine.THRESHOLDS['resnet']} (Expected: 0.3844)")

    # CHECK 6: Test Cohort Audit
    print("\n--- CHECK 6: TEST-SET PATIENTS AUDIT ---")
    splits = json.load(open(config["data"]["splits_path"]))
    manifest = json.load(open(config["data"]["manifest_path"]))
    test_pids = splits["test"]

    benign_pids = [p for p in test_pids if manifest[p]["label_id"] == 0]
    malignant_pids = [p for p in test_pids if manifest[p]["label_id"] == 1]

    # Select 3 benign and 3 malignant patients
    selected_benign = benign_pids[:3]
    selected_mal = malignant_pids[:3]
    selected_pids = selected_benign + selected_mal

    patient_labels = {pid: PatientLabel(**manifest[pid]) for pid in selected_pids}

    print(f"Selected Benign PIDs: {selected_benign}")
    print(f"Selected Malignant PIDs: {selected_mal}")

    # Evaluate using the exact evaluation dataset & loader
    eval_dataset = BCMIDDataset(
        dataset_dir=config["data"]["dataset_dir"],
        patient_ids=selected_pids,
        patient_labels=patient_labels,
        image_size=tuple(config["data"]["image_size"]),
        is_training=False,
        modality="multimodal"
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_patient_batch
    )

    # 1. Evaluate with Graph Transformer via Evaluation Pipeline
    model_gt_eval = build_model("graph_transformer", config)
    model_gt_eval.load_state_dict(torch.load(ckpt_gt, map_location="cpu")["model_state_dict"])
    model_gt_eval.eval()

    y_true, y_prob_eval, pids_eval = run_inference_on_loader(model_gt_eval, eval_loader, device)

    # 2. Evaluate each patient through the Application Inference Engine
    results_table = []

    dataset_dir = Path(config["data"]["dataset_dir"])

    for i, pid in enumerate(selected_pids):
        gt_label = "Benign (0)" if manifest[pid]["label_id"] == 0 else "Malignant (1)"
        eval_prob = y_prob_eval[i]
        eval_pred = "Malignant/Suspicious" if eval_prob >= 0.2565 else "Benign"

        # Find first mammogram and ultrasound image for this patient
        p_dir = dataset_dir / pid
        m_imgs = sorted(list((p_dir / "Mammogram").glob("*.png")) + list((p_dir / "Mammogram").glob("*.jpg")))
        u_imgs = sorted(list((p_dir / "Ultrasound").glob("*.png")) + list((p_dir / "Ultrasound").glob("*.jpg")))

        if not m_imgs or not u_imgs:
            print(f"Warning: Missing scan for {pid}")
            continue

        with open(m_imgs[0], "rb") as fm:
            m_bytes = fm.read()
        with open(u_imgs[0], "rb") as fu:
            u_bytes = fu.read()

        # Run application predict
        app_res = app_engine.predict(mammo_input=m_bytes, us_input=u_bytes)
        app_prob = app_res["prediction"]["prob_malignant"]
        app_pred = "Malignant/Suspicious" if app_res["prediction"]["is_positive_suspicious"] else "Benign"

        results_table.append({
            "pid": pid,
            "gt": gt_label,
            "eval_prob": eval_prob,
            "app_prob": app_prob,
            "thresh": 0.2565,
            "eval_pred": eval_pred,
            "app_pred": app_pred,
            "diff": abs(eval_prob - app_prob)
        })

    print("\n" + "=" * 95)
    print(f"{'Patient':<10} | {'Ground Truth':<14} | {'Eval Prob':<10} | {'App Prob':<10} | {'Diff':<8} | {'Eval Pred':<12} | {'App Pred':<12}")
    print("=" * 95)
    for r in results_table:
        print(f"{r['pid']:<10} | {r['gt']:<14} | {r['eval_prob']:<10.4f} | {r['app_prob']:<10.4f} | {r['diff']:<8.4f} | {r['eval_pred']:<12} | {r['app_pred']:<12}")
    print("=" * 95)

    # CHECK 7: Unimodal test on benign scan (Patient 0055y17 and 0072y16)
    print("\n--- CHECK 7: UNIMODAL AUDIT ON BENIGN TEST-SET SCANS ---")
    for b_pid in ["0055y17", "0072y16"]:
        p_dir = dataset_dir / b_pid
        m_imgs = sorted(list((p_dir / "Mammogram").glob("*.png")))
        u_imgs = sorted(list((p_dir / "Ultrasound").glob("*.png")))

        print(f"\nEvaluating Benign Patient: {b_pid} (Ground Truth: Benign)")

        # Mammogram only
        if m_imgs:
            with open(m_imgs[0], "rb") as f:
                res_m = app_engine.predict(mammo_input=f.read())
            print(f"  Mammogram Only (DenseNet-121): Prob={res_m['prediction']['prob_malignant']:.4f}, Thresh={res_m['prediction']['calibrated_threshold']}, Pred={res_m['prediction']['class_name']}, Suspicious={res_m['prediction']['is_positive_suspicious']}")

        # Ultrasound only
        if u_imgs:
            with open(u_imgs[0], "rb") as f:
                res_u = app_engine.predict(us_input=f.read())
            print(f"  Ultrasound Only (ResNet-50):   Prob={res_u['prediction']['prob_malignant']:.4f}, Thresh={res_u['prediction']['calibrated_threshold']}, Pred={res_u['prediction']['class_name']}, Suspicious={res_u['prediction']['is_positive_suspicious']}")

        # Both (Multimodal)
        if m_imgs and u_imgs:
            with open(m_imgs[0], "rb") as fm, open(u_imgs[0], "rb") as fu:
                res_both = app_engine.predict(mammo_input=fm.read(), us_input=fu.read())
            print(f"  Both Scans (Graph Transformer):Prob={res_both['prediction']['prob_malignant']:.4f}, Thresh={res_both['prediction']['calibrated_threshold']}, Pred={res_both['prediction']['class_name']}, Suspicious={res_both['prediction']['is_positive_suspicious']}")

    # CHECK 8: Arbitrary Non-Medical Image Test
    print("\n--- CHECK 8: ARBITRARY / BLANK IMAGE TEST ---")
    blank_img = Image.new("RGB", (224, 224), color=(255, 255, 255))
    buf = io.BytesIO()
    blank_img.save(buf, format="PNG")
    blank_bytes = buf.getvalue()

    res_blank_m = app_engine.predict(mammo_input=blank_bytes)
    res_blank_u = app_engine.predict(us_input=blank_bytes)
    res_blank_both = app_engine.predict(mammo_input=blank_bytes, us_input=blank_bytes)

    print(f"Blank Image -> Mammo Only: Prob={res_blank_m['prediction']['prob_malignant']:.4f}, Pred={res_blank_m['prediction']['class_name']}")
    print(f"Blank Image -> US Only:    Prob={res_blank_u['prediction']['prob_malignant']:.4f}, Pred={res_blank_u['prediction']['class_name']}")
    print(f"Blank Image -> Both:       Prob={res_blank_both['prediction']['prob_malignant']:.4f}, Pred={res_blank_both['prediction']['class_name']}")


if __name__ == "__main__":
    import io
    run_audit()
