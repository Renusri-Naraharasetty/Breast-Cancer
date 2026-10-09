"""Automated unit and integration test suite for BCMID Single-Modality and Multimodal Application.

Tests all required specifications:
1. Mammogram only → HTTP 200 + DenseNet-121 prediction
2. Ultrasound only → HTTP 200 + ResNet-50 prediction
3. Mammogram + ultrasound → HTTP 200 + Graph Transformer prediction
4. No files → HTTP 400
5. Invalid mammogram → HTTP 400
6. Invalid ultrasound → HTTP 400
7. Correct model is reported for each mode
8. Correct threshold is reported for each model (0.2932, 0.3844, 0.2565)
9. Explainability works for each supported mode (Grad-CAM for unimodal, attention weights & plot for multimodal)
10. Repeated inference remains stable
11. Existing low-risk/high-risk multimodal samples still work
12. No ground-truth labels or patient IDs are exposed
"""

import io
import sys
import unittest
from pathlib import Path
from PIL import Image
from fastapi.testclient import TestClient

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.app.server import app, get_engine


class TestMultimodalApplication(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Initialize test client and verify engine pre-warming."""
        cls.client = TestClient(app)
        # Preload engine to ensure checkpoint validity
        cls.engine = get_engine()

    def _create_synthetic_image_bytes(self, color=(128, 128, 128), size=(224, 224), fmt="PNG") -> bytes:
        """Helper to create dummy RGB image bytes."""
        img = Image.new("RGB", size, color=color)
        buf = io.BytesIO()
        img.save(buf, format=fmt)
        return buf.getvalue()

    def test_01_health_endpoint(self):
        """Verifies system health status, checkpoint epochs, and calibrated thresholds."""
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertEqual(data["status"], "online")
        self.assertTrue(data["model_ready"])
        self.assertEqual(data["architecture"], "Multimodal Graph-Infused Transformer")
        self.assertAlmostEqual(data["calibrated_threshold"], 0.2565, places=4)
        self.assertEqual(data["disclaimer"], "Research prototype — not for clinical diagnosis.")

        # Check multi-model reporting
        self.assertIn("models", data)
        models = data["models"]
        self.assertEqual(models["multimodal"]["model_name"], "Graph Transformer")
        self.assertAlmostEqual(models["multimodal"]["calibrated_threshold"], 0.2565, places=4)

        self.assertEqual(models["mammogram_only"]["model_name"], "DenseNet-121")
        self.assertAlmostEqual(models["mammogram_only"]["calibrated_threshold"], 0.2932, places=4)

        self.assertEqual(models["ultrasound_only"]["model_name"], "ResNet-50")
        self.assertAlmostEqual(models["ultrasound_only"]["calibrated_threshold"], 0.3844, places=4)

    def test_02_samples_listing_label_privacy(self):
        """Req 12: Verifies sample cases do not leak ground-truth labels, patient IDs, or diagnosis."""
        response = self.client.get("/api/samples")
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertIsInstance(data, list)
        self.assertGreaterEqual(len(data), 2)

        for sample in data:
            self.assertIn("id", sample)
            self.assertIn("title", sample)
            # Ensure no label/diagnosis leakage or patient ID exposure
            self.assertNotIn("label", sample)
            self.assertNotIn("ground_truth", sample)
            self.assertNotIn("class", sample)
            self.assertNotIn("malignant", sample)
            self.assertNotIn("benign", sample)
            self.assertNotIn("patient_id", sample)
            self.assertNotIn("patient_dir", sample)
            self.assertNotIn("birads", sample)
            self.assertNotIn("pathology", sample)
            self.assertIn("sample", sample["title"].lower())
            self.assertNotIn("0072y16", sample["title"])
            self.assertNotIn("2829y16", sample["title"])

    def test_03_samples_image_fetching(self):
        """Verifies sample image files are properly retrieved from test cohort."""
        # Case 1 mammogram
        resp_m = self.client.get("/api/samples/case_1/mammogram")
        self.assertEqual(resp_m.status_code, 200)
        self.assertTrue(resp_m.headers["content-type"].startswith("image/"))
        self.assertGreater(len(resp_m.content), 1000)

        # Case 1 ultrasound
        resp_u = self.client.get("/api/samples/case_1/ultrasound")
        self.assertEqual(resp_u.status_code, 200)
        self.assertTrue(resp_u.headers["content-type"].startswith("image/"))
        self.assertGreater(len(resp_u.content), 1000)

        # Invalid case
        resp_invalid = self.client.get("/api/samples/nonexistent_case/mammogram")
        self.assertEqual(resp_invalid.status_code, 404)

        # Invalid modality
        resp_inv_mod = self.client.get("/api/samples/case_1/mri")
        self.assertEqual(resp_inv_mod.status_code, 400)

    def test_04_no_files_validation(self):
        """Req 4: Verifies rejection with HTTP 400 when no files are uploaded."""
        # Completely empty POST
        resp = self.client.post("/api/analyze")
        self.assertEqual(resp.status_code, 400)
        detail = resp.json().get("detail", "")
        self.assertTrue(
            "at least one" in detail.lower() or "no image" in detail.lower(),
            f"Expected clear message asking to upload at least one image, got: {detail}"
        )

        # POST with empty multipart files
        resp2 = self.client.post("/api/analyze", files={})
        self.assertEqual(resp2.status_code, 400)

    def test_05_invalid_mammogram_validation(self):
        """Req 5: Verifies rejection with HTTP 400 on corrupted or non-image mammogram."""
        valid_img = self._create_synthetic_image_bytes()
        corrupted_bytes = b"CORRUPTED_NOT_A_VALID_IMAGE"

        # Corrupted mammogram alone
        resp1 = self.client.post(
            "/api/analyze",
            files={"mammogram": ("mammo.png", corrupted_bytes, "image/png")}
        )
        self.assertEqual(resp1.status_code, 400)
        self.assertIn("Invalid or corrupted", resp1.json()["detail"])

        # Corrupted mammogram with valid ultrasound
        resp2 = self.client.post(
            "/api/analyze",
            files={
                "mammogram": ("mammo.png", corrupted_bytes, "image/png"),
                "ultrasound": ("us.png", valid_img, "image/png")
            }
        )
        self.assertEqual(resp2.status_code, 400)
        self.assertIn("Invalid or corrupted", resp2.json()["detail"])

        # Unsupported format (PDF) for mammogram
        resp3 = self.client.post(
            "/api/analyze",
            files={"mammogram": ("scan.pdf", valid_img, "application/pdf")}
        )
        self.assertEqual(resp3.status_code, 400)
        self.assertIn("Unsupported file type", resp3.json()["detail"])

    def test_06_invalid_ultrasound_validation(self):
        """Req 6: Verifies rejection with HTTP 400 on corrupted or non-image ultrasound."""
        valid_img = self._create_synthetic_image_bytes()
        corrupted_bytes = b"CORRUPTED_ULTRASOUND_DATA"

        # Corrupted ultrasound alone
        resp1 = self.client.post(
            "/api/analyze",
            files={"ultrasound": ("us.png", corrupted_bytes, "image/png")}
        )
        self.assertEqual(resp1.status_code, 400)
        self.assertIn("Invalid or corrupted", resp1.json()["detail"])

        # Corrupted ultrasound with valid mammogram
        resp2 = self.client.post(
            "/api/analyze",
            files={
                "mammogram": ("mammo.png", valid_img, "image/png"),
                "ultrasound": ("us.png", corrupted_bytes, "image/png")
            }
        )
        self.assertEqual(resp2.status_code, 400)
        self.assertIn("Invalid or corrupted", resp2.json()["detail"])

        # Unsupported format (PDF) for ultrasound
        resp3 = self.client.post(
            "/api/analyze",
            files={"ultrasound": ("us.pdf", valid_img, "application/pdf")}
        )
        self.assertEqual(resp3.status_code, 400)
        self.assertIn("Unsupported file type", resp3.json()["detail"])

    def test_07_mammogram_only_inference(self):
        """Req 1, 7, 8, 9: Verifies Mammogram only -> HTTP 200 + DenseNet-121 + threshold 0.2932 + Grad-CAM."""
        resp_m = self.client.get("/api/samples/case_1/mammogram")
        self.assertEqual(resp_m.status_code, 200)

        response = self.client.post(
            "/api/analyze",
            files={"mammogram": ("mammo.png", resp_m.content, "image/png")}
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()

        # Req 7: Model name and analysis mode
        self.assertEqual(data["model_name"], "DenseNet-121")
        self.assertEqual(data["analysis_mode"], "Single-Modality Analysis — Mammogram")
        self.assertEqual(data["model_metadata"]["architecture"], "DenseNet-121")

        # Req 8: Calibrated threshold
        pred = data["prediction"]
        self.assertAlmostEqual(pred["calibrated_threshold"], 0.2932, places=4)
        self.assertIn(pred["class_name"], [
            "BI-RADS-derived positive/suspicious class",
            "BI-RADS-derived negative/benign class"
        ])
        self.assertAlmostEqual(pred["prob_malignant"] + pred["prob_benign"], 1.0, places=4)

        # Threshold decision logic
        if pred["prob_malignant"] >= 0.2932:
            self.assertEqual(pred["class_id"], 1)
            self.assertTrue(pred["is_positive_suspicious"])
        else:
            self.assertEqual(pred["class_id"], 0)
            self.assertFalse(pred["is_positive_suspicious"])

        # Req 9: Grad-CAM explainability
        exp = data["explainability"]
        self.assertEqual(exp["type"], "gradcam")
        self.assertIn("plot_base64", exp)
        self.assertTrue(exp["plot_base64"].startswith("data:image/png;base64,"))
        self.assertGreater(len(exp["plot_base64"]), 1000)

        # Disclaimer
        self.assertEqual(data["disclaimer"], "Research prototype — not for clinical diagnosis.")

    def test_08_ultrasound_only_inference(self):
        """Req 2, 7, 8, 9: Verifies Ultrasound only -> HTTP 200 + ResNet-50 + threshold 0.3844 + Grad-CAM."""
        resp_u = self.client.get("/api/samples/case_1/ultrasound")
        self.assertEqual(resp_u.status_code, 200)

        response = self.client.post(
            "/api/analyze",
            files={"ultrasound": ("us.png", resp_u.content, "image/png")}
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()

        # Req 7: Model name and analysis mode
        self.assertEqual(data["model_name"], "ResNet-50")
        self.assertEqual(data["analysis_mode"], "Single-Modality Analysis — Ultrasound")
        self.assertEqual(data["model_metadata"]["architecture"], "ResNet-50")

        # Req 8: Calibrated threshold
        pred = data["prediction"]
        self.assertAlmostEqual(pred["calibrated_threshold"], 0.3844, places=4)
        self.assertIn(pred["class_name"], [
            "BI-RADS-derived positive/suspicious class",
            "BI-RADS-derived negative/benign class"
        ])
        self.assertAlmostEqual(pred["prob_malignant"] + pred["prob_benign"], 1.0, places=4)

        # Threshold decision logic
        if pred["prob_malignant"] >= 0.3844:
            self.assertEqual(pred["class_id"], 1)
            self.assertTrue(pred["is_positive_suspicious"])
        else:
            self.assertEqual(pred["class_id"], 0)
            self.assertFalse(pred["is_positive_suspicious"])

        # Req 9: Grad-CAM explainability
        exp = data["explainability"]
        self.assertEqual(exp["type"], "gradcam")
        self.assertIn("plot_base64", exp)
        self.assertTrue(exp["plot_base64"].startswith("data:image/png;base64,"))
        self.assertGreater(len(exp["plot_base64"]), 1000)

        # Disclaimer
        self.assertEqual(data["disclaimer"], "Research prototype — not for clinical diagnosis.")

    def test_09_multimodal_inference(self):
        """Req 3, 7, 8, 9: Verifies Mammogram + Ultrasound -> HTTP 200 + Graph Transformer + threshold 0.2565 + Attention."""
        resp_m = self.client.get("/api/samples/case_1/mammogram")
        resp_u = self.client.get("/api/samples/case_1/ultrasound")
        self.assertEqual(resp_m.status_code, 200)
        self.assertEqual(resp_u.status_code, 200)

        response = self.client.post(
            "/api/analyze",
            files={
                "mammogram": ("mammo.png", resp_m.content, "image/png"),
                "ultrasound": ("us.png", resp_u.content, "image/png")
            }
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()

        # Req 7: Model name and analysis mode
        self.assertEqual(data["model_name"], "Graph Transformer")
        self.assertEqual(data["analysis_mode"], "Multimodal Analysis")
        self.assertIn("Graph-Infused Transformer", data["model_metadata"]["architecture"])

        # Req 8: Calibrated threshold
        pred = data["prediction"]
        self.assertAlmostEqual(pred["calibrated_threshold"], 0.2565, places=4)
        self.assertIn(pred["class_name"], [
            "BI-RADS-derived positive/suspicious class",
            "BI-RADS-derived negative/benign class"
        ])
        self.assertAlmostEqual(pred["prob_malignant"] + pred["prob_benign"], 1.0, places=4)

        # Req 9: Multimodal attention explainability
        exp = data["explainability"]
        self.assertEqual(exp["type"], "multimodal_attention")
        self.assertIn("plot_base64", exp)
        self.assertTrue(exp["plot_base64"].startswith("data:image/png;base64,"))
        self.assertIn("attention_weights", exp)
        self.assertIn("transformer_mammo_to_us", exp["attention_weights"])
        self.assertIn("transformer_us_to_mammo", exp["attention_weights"])

        # Disclaimer
        self.assertEqual(data["disclaimer"], "Research prototype — not for clinical diagnosis.")

    def test_10_multimodal_sample_cases(self):
        """Req 11: Verifies existing low-risk (Case 1) and high-risk (Case 2) multimodal test samples still work."""
        for case_id in ["case_1", "case_2"]:
            resp_m = self.client.get(f"/api/samples/{case_id}/mammogram")
            resp_u = self.client.get(f"/api/samples/{case_id}/ultrasound")
            self.assertEqual(resp_m.status_code, 200)
            self.assertEqual(resp_u.status_code, 200)

            response = self.client.post(
                "/api/analyze",
                files={
                    "mammogram": ("mammo.png", resp_m.content, "image/png"),
                    "ultrasound": ("us.png", resp_u.content, "image/png")
                }
            )
            self.assertEqual(response.status_code, 200)
            data = response.json()

            self.assertEqual(data["model_name"], "Graph Transformer")
            self.assertEqual(data["analysis_mode"], "Multimodal Analysis")
            self.assertAlmostEqual(data["prediction"]["calibrated_threshold"], 0.2565, places=4)
            self.assertIn("BI-RADS-derived", data["prediction"]["class_name"])
            self.assertTrue(data["explainability"]["plot_base64"].startswith("data:image/png;base64,"))

    def test_11_repeated_inference_stability(self):
        """Req 10: Verifies sequential inference calls across all modes produce deterministic, stable outputs."""
        resp_m = self.client.get("/api/samples/case_2/mammogram")
        resp_u = self.client.get("/api/samples/case_2/ultrasound")

        # 1. Multimodal stability
        res_multi = []
        for _ in range(2):
            resp = self.client.post(
                "/api/analyze",
                files={
                    "mammogram": ("mammo.png", resp_m.content, "image/png"),
                    "ultrasound": ("us.png", resp_u.content, "image/png")
                }
            )
            self.assertEqual(resp.status_code, 200)
            res_multi.append(resp.json()["prediction"]["prob_malignant"])
        self.assertAlmostEqual(res_multi[0], res_multi[1], places=5)

        # 2. Mammogram-only stability
        res_mammo = []
        for _ in range(2):
            resp = self.client.post(
                "/api/analyze",
                files={"mammogram": ("mammo.png", resp_m.content, "image/png")}
            )
            self.assertEqual(resp.status_code, 200)
            res_mammo.append(resp.json()["prediction"]["prob_malignant"])
        self.assertAlmostEqual(res_mammo[0], res_mammo[1], places=5)

        # 3. Ultrasound-only stability
        res_us = []
        for _ in range(2):
            resp = self.client.post(
                "/api/analyze",
                files={"ultrasound": ("us.png", resp_u.content, "image/png")}
            )
            self.assertEqual(resp.status_code, 200)
            res_us.append(resp.json()["prediction"]["prob_malignant"])
        self.assertAlmostEqual(res_us[0], res_us[1], places=5)

    def test_12_custom_threshold_override(self):
        """Verifies custom threshold override applies correctly across all modes."""
        resp_m = self.client.get("/api/samples/case_1/mammogram")
        resp_u = self.client.get("/api/samples/case_1/ultrasound")

        # Multimodal with extreme threshold 0.9999
        resp = self.client.post(
            "/api/analyze",
            files={
                "mammogram": ("mammo.png", resp_m.content, "image/png"),
                "ultrasound": ("us.png", resp_u.content, "image/png")
            },
            data={"threshold": 0.9999}
        )
        self.assertEqual(resp.status_code, 200)
        pred = resp.json()["prediction"]
        self.assertAlmostEqual(pred["calibrated_threshold"], 0.9999, places=4)
        if pred["prob_malignant"] < 0.9999:
            self.assertEqual(pred["class_id"], 0)

        # Mammogram-only with custom threshold 0.0001
        resp_mo = self.client.post(
            "/api/analyze",
            files={"mammogram": ("mammo.png", resp_m.content, "image/png")},
            data={"threshold": 0.0001}
        )
        self.assertEqual(resp_mo.status_code, 200)
        pred_mo = resp_mo.json()["prediction"]
        self.assertAlmostEqual(pred_mo["calibrated_threshold"], 0.0001, places=4)
        if pred_mo["prob_malignant"] > 0.0001:
            self.assertEqual(pred_mo["class_id"], 1)

    def test_13_no_ground_truth_or_patient_ids_in_inference_responses(self):
        """Req 12: Verifies inference payloads never expose ground-truth labels or patient IDs."""
        resp_m = self.client.get("/api/samples/case_1/mammogram")
        resp_u = self.client.get("/api/samples/case_1/ultrasound")

        for files_dict in [
            {"mammogram": ("mammo.png", resp_m.content, "image/png")},
            {"ultrasound": ("us.png", resp_u.content, "image/png")},
            {"mammogram": ("mammo.png", resp_m.content, "image/png"), "ultrasound": ("us.png", resp_u.content, "image/png")}
        ]:
            resp = self.client.post("/api/analyze", files=files_dict)
            self.assertEqual(resp.status_code, 200)
            data_str = resp.text.lower()
            self.assertNotIn("0072y16", data_str)
            self.assertNotIn("2829y16", data_str)
            self.assertNotIn("ground_truth", data_str)
            self.assertNotIn("patient_id", data_str)
            self.assertNotIn("pathology", data_str)


if __name__ == "__main__":
    unittest.main()
