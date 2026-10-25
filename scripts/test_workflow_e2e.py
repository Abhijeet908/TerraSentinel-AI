"""End-to-End Workflow Verification Script for TerraSentinel AI."""
import sys
import os
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.image_loader import load_image, harmonize_dimensions
from modules.image_preprocessing import preprocess_pair
from modules.change_detection import detect_changes
from modules.evidence_engine import evaluate_scene_evidence
from modules.temporal_engine import evaluate_temporal_sequence
from modules.report_generator import generate_html_report, generate_json_report, generate_pdf_report
from modules.metrics import calculate_spatial_areas

def test_full_pipeline():
    print("STEP 1: Ingesting Real NASA Amazon Deforestation Satellite Data...")
    ds_dir = os.path.join(os.path.dirname(__file__), "..", "sample_data", "real_amazon_deforestation")
    b_path = os.path.join(ds_dir, "before.png")
    a_path = os.path.join(ds_dir, "after.png")
    t1_path = os.path.join(ds_dir, "t1_2006.png")

    img_b, meta_b = load_image(b_path)
    img_a, meta_a = load_image(a_path)
    img_t1, _ = load_image(t1_path)

    with open(os.path.join(ds_dir, "metadata.json")) as f:
        meta = json.load(f)

    print(f"  Loaded Before: {img_b.shape}, After: {img_a.shape}, Sensor: {meta.get('sensor_name')}")

    print("STEP 2: Harmonizing & Preprocessing...")
    norm_b, norm_a, was_resized = harmonize_dimensions(img_b, img_a)
    proc_b, proc_a, diag = preprocess_pair(norm_b, norm_a, enable_alignment=True, blur_radius=3)
    print(f"  Registration Diagnostic: {diag.get('registration_status')}")

    print("STEP 3: Executing Computer Vision Change Differencing...")
    detection = detect_changes(proc_b, proc_a, threshold=32, min_region_size=150)
    detection["img_before"] = norm_b
    detection["img_after"] = norm_a
    m = detection["metrics"]
    print(f"  Changed Pixels: {m['total_changed_pixels']:,} / Valid: {m['valid_comparison_pixels']:,} ({m['change_percentage']:.2f}%)")
    print(f"  Regions Extracted: {m['regions_count']} clusters")

    print("STEP 4: Running Scientific Evidence Engine...")
    ev_res = evaluate_scene_evidence(detection["regions"], detection["binary_mask"], proc_b, proc_a, detection["raw_diff_map"], m, meta)
    detection["evidence_result"] = ev_res
    print(f"  Primary Category: '{ev_res['primary_assessment']}' ({ev_res['assessment_plausibility']} Plausibility)")
    print(f"  Regions Evaluated: {len(ev_res['regions_evidence'])}")
    if ev_res["regions_evidence"]:
        r1 = ev_res["regions_evidence"][0]
        print(f"  Top Region #{r1['region_id']}: Compactness={r1['geometry']['compactness']}, Complexity={r1['geometry']['boundary_complexity']}, Delta_VARI={r1['spectral_evidence']['delta_vari']}")

    print("STEP 5: Running Multi-Date Temporal Sequence Reasoning...")
    seq = [(norm_b, "2000-07-30 (T0)"), (img_t1, "2006-07-17 (T1)"), (norm_a, "2012-07-18 (T2)")]
    temp_res = evaluate_temporal_sequence(seq, threshold=32, min_region_size=150)
    detection["temporal_result"] = temp_res
    print(f"  Temporal Dynamics: {temp_res['classification']}")
    print(f"  Persistence State: {temp_res['persistence_state']}")

    print("STEP 6: Evaluating Defensible Physical Area Scaling...")
    area_info = calculate_spatial_areas(m["total_changed_pixels"], meta.get("approx_resolution_m"), meta.get("spatial_calibration_confirmed"))
    print(f"  Physical Area Display: {area_info['display_str']}")
    print(f"  Methodology: {area_info['methodology_note']}")

    print("STEP 7: Generating JSON Dossier Export...")
    json_str = generate_json_report(detection, meta)
    parsed_json = json.loads(json_str)
    assert parsed_json["metrics"]["total_changed_pixels"] == m["total_changed_pixels"]
    assert parsed_json["metrics"]["change_percentage"] == m["change_percentage"]
    assert "evidence_engine" in parsed_json
    assert "temporal_analysis" in parsed_json
    print(f"  JSON Export: VALID ({len(json_str):,} characters, 0 crashes)")

    print("STEP 8: Generating Self-Contained HTML Dossier...")
    html_str = generate_html_report(detection, meta)
    assert "<!DOCTYPE html>" in html_str
    assert "data:image/png;base64," in html_str
    print(f"  HTML Export: VALID ({len(html_str):,} characters, base64 overlays embedded)")

    print("STEP 9: Generating Executive PDF Briefing via ReportLab...")
    pdf_bytes = generate_pdf_report(detection, meta)
    if pdf_bytes:
        assert pdf_bytes.startswith(b"%PDF")
        print(f"  PDF Export: VALID ({len(pdf_bytes):,} bytes)")
    else:
        print("  PDF Export: Skipped (ReportLab optional)")

    print("\nALL WORKFLOW PIPELINE VERIFICATION STEPS PASSED SUCCESSFULLY!")

def test_kerala_flood_pipeline():
    print("\nSTEP 10: Ingesting Real NASA Kerala Floods 2018 Satellite Data (NASA EO 92669)...")
    ds_dir = os.path.join(os.path.dirname(__file__), "..", "sample_data", "real_kerala_flood_2018")
    b_path = os.path.join(ds_dir, "before.png")
    a_path = os.path.join(ds_dir, "after.png")

    img_b, meta_b = load_image(b_path)
    img_a, meta_a = load_image(a_path)

    with open(os.path.join(ds_dir, "metadata.json")) as f:
        meta = json.load(f)

    print(f"  Loaded Kerala Before: {img_b.shape}, After: {img_a.shape}, Location: {meta.get('location_name')}")

    norm_b, norm_a, was_resized = harmonize_dimensions(img_b, img_a)
    proc_b, proc_a, diag = preprocess_pair(norm_b, norm_a, enable_alignment=False, blur_radius=3)

    detection = detect_changes(proc_b, proc_a, threshold=30, min_region_size=150)
    detection["img_before"] = norm_b
    detection["img_after"] = norm_a
    m = detection["metrics"]
    print(f"  Kerala Changed Pixels: {m['total_changed_pixels']:,} ({m['change_percentage']:.2f}%)")
    print(f"  Kerala Regions: {m['regions_count']} clusters")

    ev_res = evaluate_scene_evidence(detection["regions"], detection["binary_mask"], proc_b, proc_a, detection["raw_diff_map"], m, meta)
    detection["evidence_result"] = ev_res
    print(f"  Kerala Primary Assessment: '{ev_res['primary_assessment']}' ({ev_res['assessment_plausibility']} Plausibility)")

    json_str = generate_json_report(detection, meta)
    parsed = json.loads(json_str)
    assert parsed["metrics"]["total_changed_pixels"] == m["total_changed_pixels"]
    print("  Kerala JSON Export: VALID")

    html_str = generate_html_report(detection, meta)
    assert "<!DOCTYPE html>" in html_str
    print("  Kerala HTML Export: VALID")

from modules.change_truth_engine import run_change_truth_engine

def test_yellowstone_wildfire_pipeline():
    print("\nSTEP 11: Ingesting Real NASA Yellowstone Wildfire & Recovery Data (Landsat 5 & 7)...")
    ds_dir = os.path.join(os.path.dirname(__file__), "..", "sample_data", "real_yellowstone_wildfire")
    b_path = os.path.join(ds_dir, "before.png")
    a_path = os.path.join(ds_dir, "after.png")
    t2_path = os.path.join(ds_dir, "t2_recovery.png")

    img_b, _ = load_image(b_path)
    img_a, _ = load_image(a_path)
    img_t2, _ = load_image(t2_path)

    with open(os.path.join(ds_dir, "metadata.json")) as f:
        meta = json.load(f)

    print(f"  Loaded Yellowstone Before (1987): {img_b.shape}, After (1989): {img_a.shape}, Recovery (2010): {img_t2.shape}")
    norm_b, norm_a, _ = harmonize_dimensions(img_b, img_a)
    detection = detect_changes(norm_b, norm_a, threshold=30, min_region_size=100)
    detection["img_before"] = norm_b
    detection["img_after"] = norm_a
    m = detection["metrics"]
    print(f"  Yellowstone Changed Pixels (1987->1989): {m['total_changed_pixels']:,} ({m['change_percentage']:.2f}%)")

    # Change Truth Engine evaluation
    truth = run_change_truth_engine(detection["regions"], detection["binary_mask"], norm_b, norm_a, detection["raw_diff_map"], m, meta)
    detection["truth_result"] = truth
    print(f"  Yellowstone Change Truth Verdict: {truth['overall_verdict']}")
    if truth["regions_truth"]:
        ctrl_info = truth["regions_truth"][0].get("control_analysis", {})
        print(f"  Controls Sampled (Region #1): {ctrl_info.get('num_controls', 0)}, Z-Score: {ctrl_info.get('z_score')}")

    # 3-date temporal analysis
    seq = [(norm_b, "1987-08-05 (T0 Baseline)"), (norm_a, "1989-08-02 (T1 Burn Scar)"), (img_t2, "2010-08-25 (T2 20-Yr Recovery)")]
    temp_res = evaluate_temporal_sequence(seq, threshold=30, min_region_size=100)
    detection["temporal_result"] = temp_res
    print(f"  Yellowstone Multi-Date Trajectory: {temp_res['classification']}")

    json_str = generate_json_report(detection, meta)
    parsed = json.loads(json_str)
    assert "change_truth_engine" in parsed
    print("  Yellowstone JSON Export: VALID")

    html_str = generate_html_report(detection, meta)
    assert "<!DOCTYPE html>" in html_str
    print("  Yellowstone HTML Export: VALID")

if __name__ == "__main__":
    test_full_pipeline()
    test_kerala_flood_pipeline()
    test_yellowstone_wildfire_pipeline()
