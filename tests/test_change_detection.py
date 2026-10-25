"""Automated test suite for TerraSentinel AI core algorithms."""
import numpy as np
from modules.change_detection import detect_changes
from modules.anomaly_classification import classify_changes
from modules.ai_interpreter import generate_ai_interpretation

def test_1_identical_images_zero_difference():
    """TEST 1: Identical Images -> Approximately zero difference and no regions."""
    # Synthetic 200x200 image
    img = np.full((200, 200, 3), 100, dtype=np.uint8)
    res = detect_changes(img, img, threshold=30, min_region_size=50)

    assert res["metrics"]["total_changed_pixels"] == 0
    assert res["metrics"]["change_percentage"] == 0.0
    assert len(res["regions"]) == 0
    assert res["metrics"]["largest_region_pixels"] == 0


def test_2_localized_change_detection():
    """TEST 2: Localized Change -> Known rectangular region modified in image 2."""
    h, w = 300, 300
    img1 = np.full((h, w, 3), 60, dtype=np.uint8)
    img2 = img1.copy()

    # Modify rectangular patch of 40x40 = 1600 pixels
    rx, ry, rw, rh = 50, 50, 40, 40
    img2[ry:ry+rh, rx:rx+rw] = [200, 220, 240]

    res = detect_changes(img1, img2, threshold=30, min_region_size=100)

    assert res["metrics"]["total_changed_pixels"] >= 1400
    assert len(res["regions"]) >= 1

    # Check top detected region bbox overlaps with modified coordinates
    top_region = res["regions"][0]
    bx, by, bw, bh = top_region["bbox"]
    # Check bounding box encompasses or overlaps the modified area
    assert bx <= rx and by <= ry
    assert (bx + bw) >= (rx + rw)
    assert (by + bh) >= (ry + rh)
    assert abs(bw - rw) <= 15  # Accounts for mild Gaussian filter spread


def test_5_blank_uniform_images():
    """TEST 5: Blank Images -> Two uniform images produce no false anomaly."""
    img1 = np.zeros((150, 150, 3), dtype=np.uint8)
    img2 = np.zeros((150, 150, 3), dtype=np.uint8)

    res = detect_changes(img1, img2, threshold=25, min_region_size=50)
    assert res["metrics"]["total_changed_pixels"] == 0
    assert len(res["regions"]) == 0


def test_7_missing_ai_credentials_local_fallback():
    """TEST 7: Missing AI Credentials -> Local analysis still works reliably."""
    img1 = np.full((100, 100, 3), 50, dtype=np.uint8)
    img2 = img1.copy()
    img2[20:60, 20:60] = [180, 70, 40]

    det = detect_changes(img1, img2, threshold=30, min_region_size=50)
    heur = classify_changes(img1, img2, det["binary_mask"], det["regions"], det["metrics"])

    meta = {"location_name": "Test Site", "sensor_name": "Optical Test"}
    ai_out = generate_ai_interpretation(det["metrics"], heur, meta)

    assert "Local Analysis Mode" in ai_out["mode"]
    assert "summary" in ai_out
    assert isinstance(ai_out["observed_changes"], list)
    assert len(ai_out["observed_changes"]) > 0
    assert "confidence" in ai_out


def test_8_no_change_report_classification():
    """TEST 8: No-Change Report -> Identical images classified as No significant change."""
    img = np.full((100, 100, 3), 120, dtype=np.uint8)
    det = detect_changes(img, img, threshold=30, min_region_size=50)
    heur = classify_changes(img, img, det["binary_mask"], det["regions"], det["metrics"])

    assert heur["primary_category"] == "No significant change detected"
    assert heur["confidence"] == "High"
    assert heur["confidence_score"] >= 0.90
