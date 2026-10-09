"""Comprehensive inference consistency audit script."""

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
from scripts.evaluate import build_model
from src.data.preprocessing import get_mammogram_transforms, get_ultrasound_transforms


def run_comprehensive_audit():
    config = yaml.safe_load(open("configs/config.yaml"))
    splits = json.load(open(config["data"]["splits_path"]))
    manifest = json.load(open(config["data"]["manifest_path"]))
    dataset_dir = Path(config["data"]["dataset_dir"])

    app_engine = MultimodalInferenceEngine(config_path="configs/config.yaml", device="cpu")

    # Models from evaluation
    ckpt_gt = "outputs/experiments/graph_transformer/checkpoints/best_model.pth"
    ckpt_mammo = "outputs/experiments/densenet_mammogram/checkpoints/best_model.pth"
    ckpt_us = "outputs/experiments/resnet_ultrasound/checkpoints/best_model.pth"

    m_gt = build_model("graph_transformer", config)
    m_gt.load_state_dict(torch.load(ckpt_gt, map_location="cpu")["model_state_dict"])
    m_gt.eval()

    m_dense = build_model("densenet", config)
    m_dense.load_state_dict(torch.load(ckpt_mammo, map_location="cpu")["model_state_dict"])
    m_dense.eval()

    m_res = build_model("resnet", config)
    m_res.load_state_dict(torch.load(ckpt_us, map_location="cpu")["model_state_dict"])
    m_res.eval()

    # Select 4 benign patients and 4 malignant patients
    test_pids = splits["test"]
    benign_pids = [p for p in test_pids if manifest[p]["label_id"] == 0][:4]
    mal_pids = [p for p in test_pids if manifest[p]["label_id"] == 1][:4]

    patients = [
        # (pid, ground_truth)
        (benign_pids[0], "Benign (0)"),
        (benign_pids[1], "Benign (0)"),
        (benign_pids[2], "Benign (0)"),
        (benign_pids[3], "Benign (0)"),
        (mal_pids[0], "Malignant (1)"),
        (mal_pids[1], "Malignant (1)"),
        (mal_pids[2], "Malignant (1)"),
        (mal_pids[3], "Malignant (1)"),
    ]

    print("=" * 105)
    print(f"{'Patient':<10} | {'Ground Truth':<14} | {'Eval Prob':<11} | {'App Prob':<11} | {'Diff':<8} | {'Threshold':<9} | {'Eval Pred':<14} | {'App Pred':<14}")
    print("=" * 105)

    idx_zero = torch.zeros(1, dtype=torch.long)

    for pid, gt in patients:
        p_dir = dataset_dir / pid
        m_files = sorted(list((p_dir / "Mammogram").glob("*.png")) + list((p_dir / "Mammogram").glob("*.jpg")))
        u_files = sorted(list((p_dir / "Ultrasound").glob("*.png")) + list((p_dir / "Ultrasound").glob("*.jpg")))

        if not m_files or not u_files:
            continue

        with open(m_files[0], "rb") as fm, open(u_files[0], "rb") as fu:
            m_bytes = fm.read()
            u_bytes = fu.read()

        # Application engine
        app_res = app_engine.predict(mammo_input=m_bytes, us_input=u_bytes)
        app_prob = app_res["prediction"]["prob_malignant"]
        app_pred = "Malignant" if app_res["prediction"]["is_positive_suspicious"] else "Benign"

        # Evaluation pipeline on the same input
        t_m = app_engine.preprocess_image(m_bytes, "mammogram")
        t_u = app_engine.preprocess_image(u_bytes, "ultrasound")
        with torch.no_grad():
            logits, _ = m_gt(t_m, idx_zero, t_u, idx_zero, batch_size=1)
            eval_prob = F.softmax(logits, dim=1)[0, 1].item()
        eval_pred = "Malignant" if eval_prob >= 0.2565 else "Benign"
        diff = abs(eval_prob - app_prob)

        print(f"{pid:<10} | {gt:<14} | {eval_prob:<11.6f} | {app_prob:<11.6f} | {diff:<8.6f} | {'0.2565':<9} | {eval_pred:<14} | {app_pred:<14}")

    print("=" * 105)


if __name__ == "__main__":
    run_comprehensive_audit()
