"""Unit tests for predict_sample.py single-sample inference script."""

import pytest
import os
from predict_sample import predict_sample, MODEL_REGISTRY


def test_model_registry_integrity():
    assert set(MODEL_REGISTRY.keys()) == {"exp1", "exp2", "exp3", "exp4"}
    for key, meta in MODEL_REGISTRY.items():
        assert os.path.exists(meta["checkpoint_path"]), f"Checkpoint missing for {key}: {meta['checkpoint_path']}"


def test_predict_sample_exp1_mammogram():
    image_path = "data/samples/sample_mammogram.png"
    assert os.path.exists(image_path)
    res = predict_sample(image_path=image_path, model_key="exp1")

    assert res["predicted_class"] in ["LOW_RISK", "SUSPICIOUS_ACTIONABLE"]
    assert 0.0 <= res["probability_low_risk"] <= 1.0
    assert 0.0 <= res["probability_suspicious"] <= 1.0
    assert round(res["probability_low_risk"] + res["probability_suspicious"], 2) == 1.0
    assert res["model_choice"] == "exp1"
    assert "disclaimer" in res


def test_predict_sample_exp2_ultrasound():
    image_path = "data/samples/sample_ultrasound.png"
    assert os.path.exists(image_path)
    res = predict_sample(image_path=image_path, model_key="exp2")

    assert res["predicted_class"] in ["LOW_RISK", "SUSPICIOUS_ACTIONABLE"]
    assert 0.0 <= res["probability_low_risk"] <= 1.0
    assert 0.0 <= res["probability_suspicious"] <= 1.0
    assert res["model_choice"] == "exp2"


def test_predict_sample_invalid_image():
    with pytest.raises(FileNotFoundError):
        predict_sample(image_path="non_existent_file.png", model_key="exp1")


def test_predict_sample_invalid_model():
    image_path = "data/samples/sample_mammogram.png"
    with pytest.raises(ValueError, match="Invalid model choice"):
        predict_sample(image_path=image_path, model_key="exp99")
