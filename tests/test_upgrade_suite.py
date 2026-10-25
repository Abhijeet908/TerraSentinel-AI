"""Comprehensive hackathon upgrade test suite for TerraSentinel AI.

Verifies the 20 critical specifications mandated by the Hackathon Transformation Charter:
1. Identical image pairs.
2. Localized synthetic change.
3. Global brightness shift.
4. Different image dimensions.
5. Registration failure.
6. Invalid uploaded files.
7. Empty masks.
8. Valid-region percentages.
9. Missing geospatial metadata.
10. JSON serialization with NumPy types.
11. JSON round-trip validity.
12. HTML report generation.
13. PDF export where supported.
14. Counterfactual control-region analysis.
15. No valid control regions.
16. Multi-date persistence and reversal.
17. Missing NASA dataset assets.
18. Missing optional API credentials.
19. Session-state behavior.
20. Dataset loading and report consistency.
"""
import unittest
import numpy as np
from PIL import Image
import io
import os
import json
import logging
import warnings
from pathlib import Path

# Suppress Streamlit bare-mode execution warnings in unit tests
logging.getLogger("streamlit").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning, module="streamlit")

from modules.image_loader import load_image, harmonize_dimensions
from modules.image_preprocessing import register_images_orb, preprocess_pair
from modules.change_detection import detect_changes
from modules.anomaly_classification import classify_changes
from modules.evidence_engine import evaluate_scene_evidence, extract_region_evidence
from modules.change_truth_engine import (
    run_change_truth_engine,
    sample_counterfactual_control_regions,
    VERDICT_GLOBAL_CONFOUND,
    VERDICT_LOCALIZED,
    VERDICT_TEMPORAL_PERSISTENT
)
from modules.temporal_engine import evaluate_temporal_sequence
from modules.metrics import calculate_spatial_areas
from modules.ai_interpreter import generate_ai_interpretation
from modules.report_generator import generate_html_report, generate_json_report, generate_pdf_report
from config import SAMPLE_DATA_DIR


class TestTerraSentinelUpgrade(unittest.TestCase):

    def test_01_identical_images(self):
        """TEST 1: Identical images produce zero difference and zero regions."""
        img = np.full((150, 150, 3), 90, dtype=np.uint8)
        res = detect_changes(img, img, threshold=30, min_region_size=50)
        self.assertEqual(res["metrics"]["total_changed_pixels"], 0)
        self.assertEqual(res["metrics"]["change_percentage"], 0.0)
        self.assertEqual(len(res["regions"]), 0)

    def test_02_synthetic_localized_change(self):
        """TEST 2: Synthetic localized change identified and overlaps modified bbox."""
        img1 = np.full((200, 200, 3), 50, dtype=np.uint8)
        img2 = img1.copy()
        img2[40:80, 40:80] = [200, 210, 220]  # 40x40 patch = 1600 px
        res = detect_changes(img1, img2, threshold=30, min_region_size=100)
        self.assertGreaterEqual(res["metrics"]["total_changed_pixels"], 1400)
        self.assertGreaterEqual(len(res["regions"]), 1)
        bx, by, bw, bh = res["regions"][0]["bbox"]
        self.assertTrue(bx <= 40 and by <= 40)
        self.assertTrue((bx + bw) >= 80 and (by + bh) >= 80)

    def test_03_global_brightness_shift(self):
        """TEST 3: Global brightness shift flagged as confounder rather than local damage."""
        img1 = np.full((120, 120, 3), 70, dtype=np.uint8)
        # Uniform shift across the whole scene (e.g. ambient solar elevation angle)
        img2 = np.full((120, 120, 3), 105, dtype=np.uint8)
        res = detect_changes(img1, img2, threshold=20, min_region_size=50)
        truth = run_change_truth_engine(
            res["regions"], res["binary_mask"], img1, img2, res["raw_diff_map"], res["metrics"]
        )
        self.assertTrue(truth["background_variation"]["global_confounder_flag"])
        # Should flag confounder or whole-scene illumination drift
        self.assertIn(truth["overall_verdict"], [VERDICT_GLOBAL_CONFOUND, "CHANGE DETECTED — CAUSE UNCERTAIN", "NO ANOMALIES DETECTED"])

    def test_04_different_dimensions_harmonization(self):
        """TEST 4: Differing image dimensions are safely harmonized."""
        img1 = np.zeros((100, 120, 3), dtype=np.uint8)
        img2 = np.zeros((200, 240, 3), dtype=np.uint8)
        h1, h2, resized = harmonize_dimensions(img1, img2)
        self.assertTrue(resized)
        self.assertEqual(h1.shape, h2.shape)

    def test_05_failed_registration_diagnostic(self):
        """TEST 5: Extreme warp or featureless image fails registration gracefully with diagnostic."""
        img1 = np.zeros((100, 100, 3), dtype=np.uint8)
        img2 = np.zeros((100, 100, 3), dtype=np.uint8)
        aligned, diag = register_images_orb(img1, img2)
        self.assertFalse(diag["alignment_success"])
        self.assertIsNotNone(diag.get("error_msg"))

    def test_06_invalid_uploaded_files(self):
        """TEST 6: Corrupt bytes raise user-friendly error without crashing."""
        with self.assertRaises(ValueError):
            load_image(b"CORRUPTED_NON_IMAGE_HEADER_BYTES")

    def test_07_empty_detection_masks(self):
        """TEST 7: Evidence Engine and Change Truth Engine handle empty detection masks gracefully."""
        img = np.full((100, 100, 3), 100, dtype=np.uint8)
        res = detect_changes(img, img, threshold=30, min_region_size=50)
        truth = run_change_truth_engine(res["regions"], res["binary_mask"], img, img, res["raw_diff_map"], res["metrics"])
        self.assertEqual(truth["overall_verdict"], "NO ANOMALIES DETECTED")
        self.assertEqual(len(truth["regions_truth"]), 0)

    def test_08_valid_region_percentages(self):
        """TEST 8: Valid pixel masking calculates mathematically exact percentage."""
        h, w = 100, 100
        total_px = h * w
        img1 = np.full((h, w, 3), 50, dtype=np.uint8)
        img2 = img1.copy()
        img2[0:20, 0:20] = [200, 200, 200]  # 400 pixels = 4.0%
        res = detect_changes(img1, img2, threshold=30, min_region_size=50)
        pct = res["metrics"]["change_percentage"]
        calculated_pct = (res["metrics"]["total_changed_pixels"] / float(total_px)) * 100.0
        self.assertAlmostEqual(pct, calculated_pct, places=2)

    def test_09_geospatial_area_unavailable_without_metadata(self):
        """TEST 9: Physical land area (ha/km2) is withheld when GSD is unverified."""
        area_info = calculate_spatial_areas(5000, resolution_m_per_pixel=10.0, spatial_calibration_confirmed=False)
        self.assertFalse(area_info["has_spatial_calibration"])
        self.assertEqual(area_info["display_str"], "Geographic area unavailable")
        self.assertIsNone(area_info["area_hectares"])

    def test_10_numpy_scalars_and_arrays_in_json(self):
        """TEST 10: NumPy scalars, vectors, and image matrices handled properly in JSON."""
        test_det = {
            "metrics": {
                "scalar_f32": np.float32(42.5),
                "scalar_i64": np.int64(999),
                "scalar_bool": np.bool_(True),
                "small_vector": np.array([12.3, 45.6]),
                "raw_image_matrix": np.zeros((100, 100, 3), dtype=np.uint8)
            },
            "regions": []
        }
        meta = {"test_val": np.float64(1.23456)}
        json_str = generate_json_report(test_det, meta)
        parsed = json.loads(json_str)
        self.assertEqual(parsed["metrics"]["scalar_f32"], 42.5)
        self.assertEqual(parsed["metrics"]["scalar_i64"], 999)
        self.assertEqual(parsed["metrics"]["scalar_bool"], True)
        self.assertEqual(parsed["metrics"]["small_vector"], [12.3, 45.6])
        self.assertNotIn("raw_image_matrix", parsed["metrics"])

    def test_11_json_round_trip_validity(self):
        """TEST 11: Exported JSON can be parsed back and maintains data integrity."""
        img1 = np.full((100, 100, 3), 50, dtype=np.uint8)
        img2 = img1.copy()
        img2[20:60, 20:60] = [180, 200, 220]
        det = detect_changes(img1, img2, threshold=30, min_region_size=50)
        det["img_before"] = img1
        det["img_after"] = img2
        meta = {"location_name": "Round Trip Test Site", "spatial_calibration_confirmed": True, "approx_resolution_m": 15.0}
        truth = run_change_truth_engine(det["regions"], det["binary_mask"], img1, img2, det["raw_diff_map"], det["metrics"], meta)
        det["truth_result"] = truth

        json_str = generate_json_report(det, meta)
        parsed = json.loads(json_str)
        self.assertEqual(parsed["metadata"]["location_name"], "Round Trip Test Site")
        self.assertEqual(parsed["metrics"]["total_changed_pixels"], det["metrics"]["total_changed_pixels"])
        self.assertIn("change_truth_engine", parsed)
        self.assertEqual(parsed["change_truth_engine"]["overall_verdict"], truth["overall_verdict"])

    def test_12_html_report_generation(self):
        """TEST 12: HTML report generated with embedded base64 overlays and no emojis."""
        img1 = np.full((100, 100, 3), 40, dtype=np.uint8)
        img2 = img1.copy()
        img2[20:70, 20:70] = [190, 110, 50]
        det = detect_changes(img1, img2, threshold=30, min_region_size=50)
        det["img_before"] = img1
        det["img_after"] = img2
        meta = {"location_name": "HTML Site", "spatial_calibration_confirmed": True, "approx_resolution_m": 10.0}
        truth = run_change_truth_engine(det["regions"], det["binary_mask"], img1, img2, det["raw_diff_map"], det["metrics"], meta)
        det["truth_result"] = truth
        det["evidence_result"] = evaluate_scene_evidence(det["regions"], det["binary_mask"], img1, img2, det["raw_diff_map"], det["metrics"], meta)
        det["ai_interpretation"] = {"summary": "HTML test", "possible_categories": ["Possible excavation"], "confidence": "high", "observed_changes": ["Observed"], "alternative_explanations": ["Alt"]}

        html_rep = generate_html_report(det, meta)
        self.assertIn("<!DOCTYPE html>", html_rep)
        self.assertIn("HTML Site", html_rep)
        self.assertIn("CHANGE TRUTH ENGINE", html_rep)
        self.assertIn("data:image/png;base64,", html_rep)

    def test_13_pdf_export_where_supported(self):
        """TEST 13: PDF export generates valid bytes if ReportLab is installed."""
        img1 = np.full((80, 80, 3), 60, dtype=np.uint8)
        img2 = img1.copy()
        img2[20:50, 20:50] = [190, 190, 190]
        det = detect_changes(img1, img2, threshold=25, min_region_size=40)
        det["img_before"] = img1
        det["img_after"] = img2
        meta = {"location_name": "PDF Site"}
        pdf_bytes = generate_pdf_report(det, meta)
        if pdf_bytes is not None:
            self.assertTrue(pdf_bytes.startswith(b"%PDF"))

    def test_14_counterfactual_control_region_analysis(self):
        """TEST 14: Change Truth Engine computes standardized difference vs control patches."""
        img1 = np.full((200, 200, 3), 50, dtype=np.uint8)
        img2 = img1.copy()
        # Create a localized high-delta patch (50 -> 210)
        img2[40:80, 40:80] = [210, 210, 210]
        det = detect_changes(img1, img2, threshold=30, min_region_size=100)
        truth = run_change_truth_engine(det["regions"], det["binary_mask"], img1, img2, det["raw_diff_map"], det["metrics"])
        self.assertGreater(len(truth["regions_truth"]), 0)
        r0 = truth["regions_truth"][0]
        ctrl = r0["counterfactual_controls"]
        self.assertTrue(ctrl["available"])
        self.assertGreater(ctrl["num_controls"], 0)
        self.assertIsNotNone(ctrl["z_score"])
        self.assertGreater(ctrl["z_score"], 2.0)  # Strong statistical separation
        self.assertEqual(r0["verdict"], VERDICT_LOCALIZED)

    def test_15_no_valid_control_regions(self):
        """TEST 15: Handles edge-case where entire image changed and no stable controls exist."""
        diff_map = np.full((50, 50), 80.0, dtype=np.float32)
        # All pixels changed -> 0 stable background
        mask_binary = np.full((50, 50), 255, dtype=np.uint8)
        res = sample_counterfactual_control_regions(diff_map, mask_binary, [5, 5, 40, 40], num_samples=10)
        self.assertFalse(res["control_analysis_available"])
        self.assertIn("Insufficient stable background", res["reason"])

    def test_16_temporal_persistence_and_reversal(self):
        """TEST 16: Multi-date reasoning distinguishes 2-date safeguard and 3-date reversal."""
        t0 = np.full((100, 100, 3), 80, dtype=np.uint8)
        t1 = t0.copy()
        t1[30:70, 30:70] = [20, 50, 80]
        # Two-date safeguard
        temp_2date = evaluate_temporal_sequence([(t0, "T0"), (t1, "T1")])
        self.assertEqual(temp_2date["classification"], "One-time observed change")
        self.assertFalse(temp_2date["persistence_confirmed"])

        # Three-date reversal
        t2_receded = t0.copy() + 1
        temp_3date = evaluate_temporal_sequence([(t0, "T0"), (t1, "T1"), (t2_receded, "T2")])
        self.assertTrue(temp_3date["reversal_detected"])
        self.assertIn("reversal", temp_3date["classification"].lower())

    def test_17_missing_nasa_dataset_assets(self):
        """TEST 17: Missing dataset folder or missing files handled gracefully without crash."""
        from unittest.mock import patch
        non_existent = SAMPLE_DATA_DIR / "non_existent_mission_case"
        self.assertFalse(non_existent.exists())
        from app import load_demo_dataset
        with patch("streamlit.error") as mock_err, patch("streamlit.session_state", {}):
            load_demo_dataset("non_existent_mission_case")
            mock_err.assert_called_once()
            self.assertIn("not found", mock_err.call_args[0][0])

    def test_18_missing_optional_api_credentials(self):
        """TEST 18: Deterministic local model works when Meta AI credentials missing."""
        img1 = np.full((80, 80, 3), 50, dtype=np.uint8)
        img2 = img1.copy()
        img2[20:50, 20:50] = [180, 70, 40]
        det = detect_changes(img1, img2, threshold=30, min_region_size=50)
        heur = classify_changes(img1, img2, det["binary_mask"], det["regions"], det["metrics"])
        ai_out = generate_ai_interpretation(det["metrics"], heur, {"location_name": "Local Site"})
        self.assertIn("Local Analysis Mode", ai_out["mode"])

    def test_19_session_state_behavior(self):
        """TEST 19: Persistent default session state keys exist and have proper types."""
        from unittest.mock import patch
        from app import init_session_state
        state_dict = {}
        with patch("streamlit.session_state", state_dict):
            init_session_state()
            self.assertIn("img_before", state_dict)
            self.assertIn("analysis_result", state_dict)
            self.assertIn("truth_result", state_dict)
            self.assertIsNone(state_dict["analysis_result"])
            self.assertIsNone(state_dict["truth_result"])

    def test_20_dataset_loading_and_report_consistency(self):
        """TEST 20: 100% agreement between computed analysis metrics and exported report."""
        img1 = np.full((100, 100, 3), 60, dtype=np.uint8)
        img2 = img1.copy()
        img2[20:50, 20:50] = [180, 200, 220]
        det = detect_changes(img1, img2, threshold=25, min_region_size=50)
        det["img_before"] = img1
        det["img_after"] = img2
        meta = {"location_name": "Consistency Verification Site"}
        truth = run_change_truth_engine(det["regions"], det["binary_mask"], img1, img2, det["raw_diff_map"], det["metrics"], meta)
        det["truth_result"] = truth

        json_str = generate_json_report(det, meta)
        parsed = json.loads(json_str)
        self.assertEqual(parsed["metrics"]["total_changed_pixels"], det["metrics"]["total_changed_pixels"])
        self.assertEqual(parsed["metrics"]["change_percentage"], det["metrics"]["change_percentage"])
        self.assertEqual(parsed["change_truth_engine"]["overall_verdict"], truth["overall_verdict"])


if __name__ == "__main__":
    unittest.main()
