import os
import sys
import json
import yaml
from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import BCMIDDataset, collate_patient_batch
from src.data.label_extractor import PatientLabel
from src.models.densenet_mammogram import MammogramDenseNet
from src.models.resnet_ultrasound import UltrasoundResNet
from src.explainability.gradcam import GradCAM, overlay_cam_on_image
from src.utils.logging import get_logger


def generate_gradcam_demo(config_path: str = "configs/config.yaml"):
    logger = get_logger("GradCAM-Demo")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Running Grad-CAM on device: {device}")

    dataset_dir = config["data"]["dataset_dir"]
    manifest_path = config["data"]["manifest_path"]
    splits_path = config["data"]["splits_path"]

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest_raw = json.load(f)
    patient_labels = {pid: PatientLabel(**data) for pid, data in manifest_raw.items()}

    with open(splits_path, "r", encoding="utf-8") as f:
        splits = json.load(f)

    test_pids = splits["test"]

    test_dataset = BCMIDDataset(
        dataset_dir=dataset_dir,
        patient_ids=test_pids[:3],
        patient_labels=patient_labels,
        image_size=tuple(config["data"]["image_size"]),
        is_training=False,
        modality="multimodal"
    )

    out_dir = os.path.join(config["output"]["base_dir"], "explainability")
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # 1. Mammogram DenseNet-121 Grad-CAM
    mammo_net = MammogramDenseNet(pretrained=True, feature_dim=1024, num_classes=2).to(device)
    ckpt_path = os.path.join(config["output"]["base_dir"], "densenet_mammogram", "checkpoints", "best_model.pth")
    if os.path.exists(ckpt_path):
        mammo_net.load_state_dict(torch.load(ckpt_path, map_location=device)["model_state_dict"])

    # Target last dense block
    target_layer_m = mammo_net.features.denseblock4
    cam_generator_m = GradCAM(model=mammo_net, target_layer=target_layer_m)

    sample = test_dataset[0]
    pid = sample["patient_id"]
    m_img = sample["mammo_tensors"][0:1].to(device) # [1, 3, H, W]

    cam_m = cam_generator_m.generate_cam(m_img, target_class=1)
    overlay_cam_on_image(
        image_tensor=m_img,
        cam_heatmap=cam_m,
        save_path=os.path.join(out_dir, f"gradcam_mammogram_{pid}.png"),
        title=f"Mammogram DenseNet-121 Grad-CAM (Patient: {pid})"
    )
    cam_generator_m.remove_hooks()
    logger.info(f"Saved Mammogram Grad-CAM to: {os.path.join(out_dir, f'gradcam_mammogram_{pid}.png')}")

    # 2. Ultrasound ResNet-50 Grad-CAM
    us_net = UltrasoundResNet(pretrained=True, feature_dim=2048, num_classes=2).to(device)
    ckpt_path_u = os.path.join(config["output"]["base_dir"], "resnet_ultrasound", "checkpoints", "best_model.pth")
    if os.path.exists(ckpt_path_u):
        us_net.load_state_dict(torch.load(ckpt_path_u, map_location=device)["model_state_dict"])

    target_layer_u = us_net.layer4
    cam_generator_u = GradCAM(model=us_net, target_layer=target_layer_u)

    u_img = sample["us_tensors"][0:1].to(device) # [1, 3, H, W]
    cam_u = cam_generator_u.generate_cam(u_img, target_class=1)
    overlay_cam_on_image(
        image_tensor=u_img,
        cam_heatmap=cam_u,
        save_path=os.path.join(out_dir, f"gradcam_ultrasound_{pid}.png"),
        title=f"Ultrasound ResNet-50 Grad-CAM (Patient: {pid})"
    )
    cam_generator_u.remove_hooks()
    logger.info(f"Saved Ultrasound Grad-CAM to: {os.path.join(out_dir, f'gradcam_ultrasound_{pid}.png')}")


if __name__ == "__main__":
    generate_gradcam_demo()
