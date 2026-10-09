"""FastAPI server for Multimodal and Single-Modality Breast Cancer Research Demo.

Serves the research UI, provides inference endpoints for single-modality and multimodal analysis,
and serves test-set samples without exposing ground-truth labels.

DISCLAIMER: Research prototype — not for clinical diagnosis.
"""

import os
import sys
import io
import json
from pathlib import Path
from typing import Optional, List
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, Response, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.app.inference_engine import MultimodalInferenceEngine

app = FastAPI(
    title="Multimodal Breast Cancer Classification Research Demo",
    description="Research prototype for dual-modality (Mammography + Ultrasound) and single-modality classification.",
    version="1.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = Path(__file__).resolve().parent / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Singleton inference engine
_engine: Optional[MultimodalInferenceEngine] = None


def get_engine() -> MultimodalInferenceEngine:
    global _engine
    if _engine is None:
        config_path = PROJECT_ROOT / "configs" / "config.yaml"
        gt_ckpt = PROJECT_ROOT / "outputs" / "experiments" / "graph_transformer" / "checkpoints" / "best_model.pth"
        densenet_ckpt = PROJECT_ROOT / "outputs" / "experiments" / "densenet" / "checkpoints" / "best_model.pth"
        resnet_ckpt = PROJECT_ROOT / "outputs" / "experiments" / "resnet" / "checkpoints" / "best_model.pth"

        _engine = MultimodalInferenceEngine(
            config_path=str(config_path),
            gt_checkpoint_path=str(gt_ckpt),
            densenet_checkpoint_path=str(densenet_ckpt),
            resnet_checkpoint_path=str(resnet_ckpt)
        )
    return _engine


SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def validate_optional_image_file(file: Optional[UploadFile], modality_name: str) -> Optional[bytes]:
    """Validates an optional uploaded image file. Returns bytes if valid file present, else None."""
    if file is None:
        return None

    # Check if filename is empty (e.g. empty multipart field)
    filename = getattr(file, "filename", None)
    if not filename or filename.strip() == "":
        return None

    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}' for {modality_name}. Allowed formats: PNG, JPG, JPEG, TIFF, BMP."
        )

    try:
        content = file.file.read()
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to read uploaded {modality_name} file: {str(e)}"
        )

    if len(content) == 0:
        raise HTTPException(
            status_code=400,
            detail=f"The uploaded {modality_name} file is empty (0 bytes)."
        )

    try:
        with Image.open(io.BytesIO(content)) as img:
            img.verify()
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid or corrupted image file for {modality_name}. Please upload a valid image."
        )

    return content


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        return HTMLResponse("<h1>BCMID Multimodal Research Demo UI is loading...</h1>", status_code=200)
    return HTMLResponse(index_file.read_text(encoding="utf-8"))


@app.get("/api/health")
async def health_check():
    """System health check and checkpoint metadata for all supported models."""
    try:
        engine = get_engine()
        gt_meta = engine.models_meta.get("graph_transformer", {})
        densenet_meta = engine.models_meta.get("densenet", {})
        resnet_meta = engine.models_meta.get("resnet", {})

        return {
            "status": "online",
            "model_ready": True,
            "architecture": engine.checkpoint_meta["architecture"],
            "checkpoint_epoch": engine.checkpoint_meta["epoch"],
            "calibrated_threshold": engine.DEFAULT_THRESHOLD,
            "validation_roc_auc": engine.checkpoint_meta.get("val_roc_auc", 0.8406),
            "test_roc_auc": 0.8594,
            "test_sensitivity": 0.7500,
            "device": str(engine.device),
            "models": {
                "multimodal": {
                    "model_name": "Graph Transformer",
                    "mode": "Multimodal Analysis",
                    "calibrated_threshold": engine.THRESHOLDS["graph_transformer"],
                    "checkpoint_epoch": gt_meta.get("epoch", 11)
                },
                "mammogram_only": {
                    "model_name": "DenseNet-121",
                    "mode": "Single-Modality Analysis — Mammogram",
                    "calibrated_threshold": engine.THRESHOLDS["densenet"],
                    "checkpoint_epoch": densenet_meta.get("epoch")
                },
                "ultrasound_only": {
                    "model_name": "ResNet-50",
                    "mode": "Single-Modality Analysis — Ultrasound",
                    "calibrated_threshold": engine.THRESHOLDS["resnet"],
                    "checkpoint_epoch": resnet_meta.get("epoch")
                }
            },
            "disclaimer": "Research prototype — not for clinical diagnosis."
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": f"Inference engine failure: {str(e)}"}
        )


@app.post("/api/analyze")
async def analyze_images(
    mammogram: Optional[UploadFile] = File(None, description="Mammography image file (optional if ultrasound provided)"),
    ultrasound: Optional[UploadFile] = File(None, description="Breast Ultrasound image file (optional if mammogram provided)"),
    threshold: Optional[float] = Form(None)
):
    """Performs inference for single-modality (Mammogram or Ultrasound) or paired multimodal input."""
    engine = get_engine()

    # Validate uploaded files
    mammo_bytes = validate_optional_image_file(mammogram, "Mammogram")
    us_bytes = validate_optional_image_file(ultrasound, "Ultrasound")

    # Reject if neither modality is uploaded
    if mammo_bytes is None and us_bytes is None:
        raise HTTPException(
            status_code=400,
            detail="No image provided. Please upload at least one image (Mammogram or Ultrasound) to perform analysis."
        )

    try:
        results = engine.predict(
            mammo_input=mammo_bytes,
            us_input=us_bytes,
            threshold=threshold
        )
        return results
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Model inference failed: {str(e)}"
        )


# Sample test cases from BCMID test cohort (labels and patient IDs strictly omitted from frontend/API)
SAMPLE_CASES = {
    "case_1": {
        "id": "case_1",
        "title": "Low-Risk Sample",
        "patient_dir": "0072y16",
        "mammo_file": "0072y16_L_CC_1.png",
        "us_file": "0072y16_US_L_1.png"
    },
    "case_2": {
        "id": "case_2",
        "title": "High-Risk Sample",
        "patient_dir": "2829y16",
        "mammo_file": "2829y16_L_CC_1.png",
        "us_file": "2829y16_US_L_1.png"
    }
}


@app.get("/api/samples")
async def list_sample_cases():
    """Lists available test-set sample cases (ground-truth labels strictly omitted)."""
    return [
        {
            "id": k,
            "title": v["title"],
            "has_mammogram": True,
            "has_ultrasound": True
        }
        for k, v in SAMPLE_CASES.items()
    ]


@app.get("/api/samples/{case_id}/{modality}")
async def get_sample_image(case_id: str, modality: str):
    """Fetches raw sample image bytes for demonstration."""
    if case_id not in SAMPLE_CASES:
        raise HTTPException(status_code=404, detail="Sample case not found.")

    case = SAMPLE_CASES[case_id]
    dataset_dir = PROJECT_ROOT / "BCMID" / case["patient_dir"]

    if modality == "mammogram":
        img_path = dataset_dir / "Mammogram" / case["mammo_file"]
    elif modality == "ultrasound":
        img_path = dataset_dir / "Ultrasound" / case["us_file"]
    else:
        raise HTTPException(status_code=400, detail="Invalid modality. Choose 'mammogram' or 'ultrasound'.")

    if not img_path.exists():
        raise HTTPException(status_code=404, detail=f"Image file not found at {img_path}")

    return FileResponse(
        str(img_path),
        media_type="image/png",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0"
        }
    )
