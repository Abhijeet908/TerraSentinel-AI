"""Tests for HTML, JSON, and PDF report generation."""
import json
import numpy as np
from modules.change_detection import detect_changes
from modules.anomaly_classification import classify_changes
from modules.ai_interpreter import generate_ai_interpretation
from modules.report_generator import generate_html_report, generate_json_report, generate_pdf_report

def test_6_report_generation_html_and_json():
    """TEST 6: Valid computed analysis result produces HTML, JSON, and PDF reports."""
    # Create realistic scenario
    h, w = 150, 150
    img1 = np.full((h, w, 3), 40, dtype=np.uint8)
    img2 = img1.copy()
    img2[30:80, 30:80] = [180, 120, 70]  # Soil exposure patch

    det = detect_changes(img1, img2, threshold=25, min_region_size=50)
    det["img_before"] = img1
    det["img_after"] = img2

    heur = classify_changes(img1, img2, det["binary_mask"], det["regions"], det["metrics"])
    meta = {
        "location_name": "Sector Alpha Testing",
        "date_before": "2024-01-01",
        "date_after": "2024-02-01",
        "sensor_name": "Sensor Unit 9",
        "approx_resolution_m": 10.0
    }
    ai_interp = generate_ai_interpretation(det["metrics"], heur, meta)
    det["ai_interpretation"] = ai_interp

    # 1. Test JSON Export
    json_str = generate_json_report(det, meta, report_id="TSA-TEST-001")
    parsed_json = json.loads(json_str)

    assert parsed_json["report_id"] == "TSA-TEST-001"
    assert parsed_json["metrics"]["total_changed_pixels"] == det["metrics"]["total_changed_pixels"]
    assert parsed_json["metadata"]["location_name"] == "Sector Alpha Testing"
    assert "disclaimer" in parsed_json

    # 2. Test HTML Report
    html_str = generate_html_report(det, meta, report_id="TSA-TEST-001")
    assert "<!DOCTYPE html>" in html_str
    assert "TSA-TEST-001" in html_str
    assert "Sector Alpha Testing" in html_str
    assert "data:image/png;base64," in html_str  # Visuals embedded
    assert "SCIENTIFIC &amp; LEGAL DISCLAIMER" in html_str or "SCIENTIFIC & LEGAL DISCLAIMER" in html_str

    # 3. Test PDF Report
    pdf_bytes = generate_pdf_report(det, meta, report_id="TSA-TEST-001")
    if pdf_bytes is not None:
        assert len(pdf_bytes) > 1000
        assert pdf_bytes.startswith(b"%PDF")
