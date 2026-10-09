import sys
sys.path.insert(0, ".")
import torch
import yaml
import json
from src.data.dataset import BCMIDDataset, collate_patient_batch
from src.data.label_extractor import PatientLabel
from src.models.multimodal_model import MultimodalGraphTransformer
from src.training.losses import get_loss_function

with open("configs/config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

with open(config["data"]["manifest_path"], "r", encoding="utf-8") as f:
    manifest_raw = json.load(f)
patient_labels = {pid: PatientLabel(**data) for pid, data in manifest_raw.items()}

with open(config["data"]["splits_path"], "r", encoding="utf-8") as f:
    splits = json.load(f)

ds = BCMIDDataset(
    dataset_dir=config["data"]["dataset_dir"],
    patient_ids=splits["train"][:4],
    patient_labels=patient_labels,
    image_size=(224, 224),
    is_training=True,
    modality="multimodal"
)

loader = torch.utils.data.DataLoader(ds, batch_size=4, collate_fn=collate_patient_batch)
batch = next(iter(loader))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MultimodalGraphTransformer(pretrained=True).to(device)
criterion = get_loss_function(device=device)

# Test without AMP (fp32)
m_imgs = batch["mammo_tensors"].to(device)
m_idx = batch["mammo_batch_indices"].to(device)
u_imgs = batch["us_tensors"].to(device)
u_idx = batch["us_batch_indices"].to(device)
labels = batch["labels"].to(device)
bs = batch["batch_size"]

model.train()
logits, _ = model(m_imgs, m_idx, u_imgs, u_idx, bs)
loss_fp32 = criterion(logits, labels)
print("FP32 Logits:", logits)
print("FP32 Loss:", loss_fp32.item(), "Is NaN:", torch.isnan(loss_fp32).item())

# Test with AMP (autocast)
with torch.amp.autocast('cuda'):
    logits_amp, _ = model(m_imgs, m_idx, u_imgs, u_idx, bs)
    loss_amp = criterion(logits_amp, labels)
print("AMP Logits:", logits_amp)
print("AMP Loss:", loss_amp.item(), "Is NaN:", torch.isnan(loss_amp).item())
