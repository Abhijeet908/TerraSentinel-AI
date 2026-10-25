"""Automated Report Generation module for TerraSentinel AI.

Generates:
1. Self-contained HTML dossier with embedded base64 overlays, region evidence cards,
   competing hypotheses, and scientific disclaimers.
2. Structured machine-readable JSON analysis export.
3. High-resolution PDF executive briefing using ReportLab.
"""
import io
import os
import base64
import json
import uuid
from datetime import datetime, date
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
from PIL import Image

# PDF generation imports
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib import colors
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, KeepTogether
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False


def _array_to_base64_png(arr: np.ndarray, max_width: int = 600) -> str:
    """Converts a numpy RGB array to a base64 encoded PNG data URI."""
    pil_img = Image.fromarray(arr)
    if pil_img.width > max_width:
        h = int(pil_img.height * (max_width / float(pil_img.width)))
        pil_img = pil_img.resize((max_width, h), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG", optimize=True)
    b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64_str}"


def _array_to_temp_png(arr: np.ndarray, max_width: int = 500) -> io.BytesIO:
    """Converts a numpy RGB array to BytesIO stream for ReportLab."""
    pil_img = Image.fromarray(arr)
    if pil_img.width > max_width:
        h = int(pil_img.height * (max_width / float(pil_img.width)))
        pil_img = pil_img.resize((max_width, h), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)
    return buf


# Keys that contain large raw image matrices or masks that must never be serialized into JSON
EXCLUDED_IMAGE_KEYS = {
    "img_before", "img_after", "raw_diff_map", "heatmap", "binary_mask",
    "overlay", "annotated_boxes", "res_01", "res_02", "res_12"
}

_EXCLUDED = object()

def sanitize_for_json(obj: Any, max_array_len: int = 100, depth: int = 0) -> Any:
    """Recursively converts NumPy scalars, non-finite floats, arrays, and complex types

    into standard JSON-compliant Python primitives. Excludes raw image arrays.
    """
    if depth > 25:
        return str(obj)

    # None, bool, str
    if obj is None:
        return None
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, str):
        return obj

    # NumPy and Python integers
    if isinstance(obj, (int, np.integer)):
        return int(obj)

    # NumPy and Python floats (map NaN and Inf to None -> JSON null)
    if isinstance(obj, (float, np.floating)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return round(float(obj), 6)

    # NumPy arrays
    if isinstance(obj, np.ndarray):
        if obj.size > max_array_len or obj.ndim > 1:
            # Exclude large multi-dimensional matrices (images, heatmaps, masks)
            return _EXCLUDED
        # Small 1D arrays (e.g. coordinates, RGB shifts)
        return [sanitize_for_json(x, max_array_len, depth + 1) for x in obj.tolist()]

    # Dates and timestamps
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()

    # Path objects
    if isinstance(obj, (Path, os.PathLike)):
        return str(obj)

    # Dictionaries
    if isinstance(obj, dict):
        sanitized_dict = {}
        for k, v in obj.items():
            str_key = str(k)
            # Skip heavy image / mask arrays
            if str_key in EXCLUDED_IMAGE_KEYS:
                continue
            val = sanitize_for_json(v, max_array_len, depth + 1)
            if val is not _EXCLUDED:
                sanitized_dict[str_key] = val
        return sanitized_dict

    # Lists, Tuples, Sets
    if isinstance(obj, (list, tuple, set)):
        sanitized_list = []
        for item in obj:
            sanitized_item = sanitize_for_json(item, max_array_len, depth + 1)
            if sanitized_item is not _EXCLUDED:
                sanitized_list.append(sanitized_item)
        return sanitized_list

    # Fallback to string representation for unsupported custom types
    return str(obj)


def generate_json_report(
    analysis_data: Dict[str, Any],
    metadata: Dict[str, Any],
    report_id: Optional[str] = None
) -> str:
    """Generates a comprehensive JSON export of the full analysis.

    Converts all NumPy scalars/arrays to valid JSON types and safely excludes
    large image arrays.
    """
    rep_id = report_id or f"TSA-{datetime.utcnow().strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}"

    export_obj = {
        "report_id": rep_id,
        "platform": "TerraSentinel AI — Earth Intelligence Platform",
        "generated_at_utc": datetime.utcnow().isoformat() + "Z",
        "metadata": sanitize_for_json(metadata),
        "metrics": sanitize_for_json(analysis_data.get("metrics", {})),
        "ai_interpretation": sanitize_for_json(analysis_data.get("ai_interpretation", {})),
        "evidence_engine": sanitize_for_json(analysis_data.get("evidence_result", {})),
        "change_truth_engine": sanitize_for_json(analysis_data.get("truth_result", analysis_data.get("evidence_result", {}))),
        "temporal_analysis": sanitize_for_json(analysis_data.get("temporal_result", {})),
        "detected_regions": sanitize_for_json(analysis_data.get("regions", [])),
        "disclaimer": (
            "Preliminary automated assessment based on uncalibrated optical imagery. "
            "Detected pixel differences do not establish ground-truth damage, deforestation, "
            "or human intent. Ground verification or radar cross-validation is required."
        )
    }

    clean_payload = sanitize_for_json(export_obj)
    return json.dumps(clean_payload, indent=2, allow_nan=False)


def generate_html_report(
    analysis_data: Dict[str, Any],
    metadata: Dict[str, Any],
    report_id: Optional[str] = None
) -> str:
    """Generates an executive, self-contained HTML dossier with embedded images."""
    rep_id = report_id or f"TSA-{datetime.utcnow().strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}"
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

    # Extract base64 images
    img_before_b64 = _array_to_base64_png(analysis_data["img_before"])
    img_after_b64 = _array_to_base64_png(analysis_data["img_after"])
    heatmap_b64 = _array_to_base64_png(analysis_data["heatmap"])
    overlay_b64 = _array_to_base64_png(analysis_data["overlay"])

    metrics = analysis_data.get("metrics", {})
    interp = analysis_data.get("ai_interpretation", {})
    ev_res = analysis_data.get("evidence_result", {})
    truth_res = analysis_data.get("truth_result", {})
    temp_res = analysis_data.get("temporal_result", {})

    obs_list = "".join(f"<li>{item}</li>" for item in interp.get("observed_changes", []))
    alt_list = "".join(f"<li>{item}</li>" for item in interp.get("alternative_explanations", []))
    followup_list = "".join(f"<li>{item}</li>" for item in interp.get("recommended_follow_up", []))
    limit_list = "".join(f"<li>{item}</li>" for item in interp.get("limitations", []))

    confidence = interp.get("confidence", "moderate").upper()
    conf_color = "#00E676" if confidence == "HIGH" else ("#FFD600" if confidence == "MODERATE" else "#FF5252")

    # Change Truth Engine block
    change_truth_html = ""
    if truth_res and truth_res.get("regions_truth"):
        top_truth = truth_res["regions_truth"][0]
        v = top_truth.get("verdict", "EVALUATING")
        v_color = "var(--accent)" if "LOCALIZED" in v or "PERSISTENCE" in v else ("#ffd600" if "CONFOUND" in v or "UNCERTAIN" in v else "var(--red)")
        ctrl = top_truth.get("counterfactual_controls", {})
        z_str = f"+{ctrl['z_score']}σ" if ctrl.get("z_score") is not None else "N/A"
        pct_str = f">{ctrl['empirical_percentile']}%" if ctrl.get("empirical_percentile") is not None else "N/A"

        strength_list = "".join(f"<li>{s}</li>" for s in top_truth.get("what_would_change_this_verdict", {}).get("strengthening_evidence", []))
        weaken_list = "".join(f"<li>{w}</li>" for w in top_truth.get("what_would_change_this_verdict", {}).get("weakening_evidence", []))

        change_truth_html = f"""
        <div class="card" style="border-left: 3px solid {v_color};">
            <h2>CHANGE TRUTH ENGINE // COUNTERFACTUAL EVIDENCE CHALLENGE</h2>
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; flex-wrap: wrap; gap: 8px;">
                <div>
                    <span style="font-size: 10px; font-family: 'IBM Plex Mono', monospace; color: var(--text-muted); text-transform: uppercase;">PRIMARY VERDICT:</span>
                    <div style="font-size: 17px; font-weight: 700; color: {v_color}; margin-top: 2px;">{v}</div>
                </div>
                <div style="font-size: 10px; font-family: 'IBM Plex Mono', monospace; background: rgba(56, 255, 142, 0.08); border: 1px solid var(--border); padding: 4px 10px;">
                    EFFECT SIZE: <b style="color: {v_color};">{z_str}</b> | CONTROL RANK: <b>{pct_str}</b>
                </div>
            </div>
            <p style="font-size: 12.5px; color: var(--text-body);">{top_truth.get('verdict_rationale', '')}</p>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-top: 14px; font-size: 11px;">
                <div style="background: #020303; border: 1px solid var(--border); padding: 10px;">
                    <b style="color: var(--accent); font-family: 'IBM Plex Mono', monospace; text-transform: uppercase;">WHAT WOULD STRENGTHEN THIS VERDICT?</b>
                    <ul style="margin: 6px 0 0 0; padding-left: 18px; color: var(--text-body);">{strength_list}</ul>
                </div>
                <div style="background: #020303; border: 1px solid var(--border); padding: 10px;">
                    <b style="color: var(--amber); font-family: 'IBM Plex Mono', monospace; text-transform: uppercase;">WHAT WOULD REFUTE OR WEAKEN THIS VERDICT?</b>
                    <ul style="margin: 6px 0 0 0; padding-left: 18px; color: var(--text-body);">{weaken_list}</ul>
                </div>
            </div>
        </div>
        """

    # Confounding warnings HTML block
    confounding_warnings = ev_res.get("confounding_warnings", [])
    confounding_html = ""
    if confounding_warnings:
        items = "".join(f"<li>[FACTOR] {w}</li>" for w in confounding_warnings)
        confounding_html = f"""
        <div style="background: rgba(255, 214, 0, 0.08); border-left: 3px solid #ffd600; padding: 12px 16px; margin-bottom: 22px;">
            <strong style="font-family: 'IBM Plex Mono', monospace; color: #ffd600; font-size: 11px; text-transform: uppercase;">Confounding Optical & Registration Factors Detected:</strong>
            <ul style="margin: 6px 0 0 0; padding-left: 18px; font-size: 12px; color: #ffd600;">{items}</ul>
        </div>
        """

    # Temporal dynamics block
    temporal_html = ""
    if temp_res:
        temporal_html = f"""
        <div class="card">
            <h2>Temporal Dynamics & Persistence Reasoning</h2>
            <p><strong>Temporal Tier:</strong> {temp_res.get('temporal_tier', 'Two-Acquisition Baseline')}</p>
            <p><strong>Persistence State:</strong> <span style="color: var(--cyan); font-weight: 600;">{temp_res.get('persistence_state', 'Undetermined')}</span></p>
            <p><strong>Classification:</strong> {temp_res.get('classification', '')}</p>
            <p style="font-size: 13px; color: var(--text-muted);">{temp_res.get('scientific_rationale', '')}</p>
        </div>
        """

    # Area methodology note
    area_note = metadata.get("area_methodology_note", "Geographic area unavailable (uncalibrated spatial resolution).")

    # Render table of top 10 regions
    regions = analysis_data.get("regions", [])[:10]
    region_rows = ""
    for r in regions:
        region_rows += f"""
        <tr>
            <td>#{r['id']}</td>
            <td>{r['area_pixels']:,} px</td>
            <td>{r['fraction_percent']:.3f}%</td>
            <td>{r['mean_intensity']:.1f} DN</td>
            <td>({r['centroid'][0]}, {r['centroid'][1]})</td>
            <td>{r.get('aspect_ratio', '-')}</td>
        </tr>
        """
    if not region_rows:
        region_rows = "<tr><td colspan='6' style='text-align:center;'>No anomaly clusters above minimum area threshold.</td></tr>"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>TerraSentinel AI Analysis Dossier - {rep_id}</title>
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600;700&family=IBM+Plex+Sans:wght@400;500;600;700;800&display=swap');

    :root {{
        --bg-main: #020303;
        --bg-card: #090e0b;
        --border: #304c3b;
        --accent: #38ff8e;
        --cyan: #65f8ff;
        --green: #38ff8e;
        --amber: #ffd600;
        --red: #ff65d8;
        --text: #f8fffa;
        --text-body: #d3dbd5;
        --text-muted: #c2cbc5;
    }}
    * {{
        box-sizing: border-box;
        border-radius: 0 !important;
    }}
    body {{
        font-family: 'IBM Plex Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        background-color: var(--bg-main);
        color: var(--text-body);
        margin: 0;
        padding: 40px 20px;
        line-height: 1.6;
    }}
    .container {{
        max-width: 1100px;
        margin: 0 auto;
        border-left: 2px solid var(--accent);
        padding-left: 25px;
    }}
    .header {{
        border-bottom: 1px solid var(--border);
        padding-bottom: 16px;
        margin-bottom: 25px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
    }}
    .header h1 {{
        margin: 0;
        color: var(--text);
        font-size: 24px;
        letter-spacing: 1.5px;
        font-weight: 800;
        text-transform: uppercase;
    }}
    .header .subtitle {{
        color: var(--accent);
        font-size: 12px;
        font-family: 'IBM Plex Mono', monospace;
        letter-spacing: 0.8px;
        text-transform: uppercase;
        margin-top: 4px;
    }}
    .badge {{
        display: inline-block;
        padding: 3px 10px;
        font-size: 11px;
        font-family: 'IBM Plex Mono', monospace;
        font-weight: 700;
        background: rgba(56, 255, 142, 0.08);
        color: var(--accent);
        border: 1px solid var(--accent);
        letter-spacing: 0.8px;
    }}
    .meta-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 12px;
        margin-bottom: 25px;
    }}
    .meta-box {{
        background: var(--bg-card);
        padding: 12px 14px;
        border: 1px solid var(--border);
    }}
    .meta-box .label {{
        font-size: 10px;
        font-family: 'IBM Plex Mono', monospace;
        text-transform: uppercase;
        color: var(--text-muted);
        letter-spacing: 0.8px;
    }}
    .meta-box .val {{
        font-size: 14px;
        font-weight: 600;
        color: var(--text);
        margin-top: 4px;
    }}
    .metrics-bar {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 12px;
        margin-bottom: 25px;
    }}
    .metric-card {{
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-top: 2px solid var(--accent);
        padding: 14px;
        text-align: center;
    }}
    .metric-card .num {{
        font-size: 22px;
        font-family: 'IBM Plex Mono', monospace;
        font-weight: 700;
        color: var(--accent);
    }}
    .metric-card .desc {{
        font-size: 10px;
        font-family: 'IBM Plex Mono', monospace;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: var(--text-muted);
        margin-top: 4px;
    }}
    .gallery {{
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 16px;
        margin-bottom: 30px;
    }}
    .img-box {{
        background: var(--bg-card);
        border: 1px solid var(--border);
        overflow: hidden;
    }}
    .img-box img {{
        width: 100%;
        display: block;
        height: auto;
    }}
    .img-title {{
        padding: 8px 12px;
        font-size: 11px;
        font-family: 'IBM Plex Mono', monospace;
        font-weight: 600;
        background: #020303;
        border-top: 1px solid var(--border);
        color: var(--cyan);
        text-transform: uppercase;
    }}
    .card {{
        background: var(--bg-card);
        border: 1px solid var(--border);
        padding: 20px;
        margin-bottom: 20px;
    }}
    .card h2 {{
        margin-top: 0;
        color: var(--text);
        font-size: 15px;
        font-family: 'IBM Plex Mono', monospace;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        border-bottom: 1px solid var(--border);
        padding-bottom: 8px;
    }}
    table {{
        width: 100%;
        border-collapse: collapse;
        margin-top: 12px;
        font-size: 12px;
        font-family: 'IBM Plex Mono', monospace;
    }}
    th, td {{
        padding: 8px 12px;
        text-align: left;
        border-bottom: 1px solid var(--border);
    }}
    th {{
        color: var(--text-muted);
        text-transform: uppercase;
        font-size: 10px;
        background: #020303;
    }}
    .disclaimer {{
        background: rgba(255, 214, 0, 0.05);
        border-left: 3px solid var(--amber);
        padding: 14px;
        font-size: 11px;
        font-family: 'IBM Plex Mono', monospace;
        margin-top: 25px;
        color: #ffd600;
        line-height: 1.5;
    }}
    @media print {{
        body {{ background: #fff; color: #111; }}
        .meta-box, .card, .metric-card {{ background: #fafafa; border: 1px solid #ccc; }}
        .header h1, .card h2, .img-title {{ color: #000; }}
        .badge {{ background: #eee; border: 1px solid #999; color: #111; }}
        .disclaimer {{ background: #fffbe6; color: #664d03; border-left: 3px solid #ffc107; }}
    }}
</style>
</head>
<body>
<div class="container">
    <div class="header">
        <div>
            <h1>TERRASENTINEL AI</h1>
            <div class="subtitle">Earth Observation Intelligence & Evidence Dossier</div>
        </div>
        <div>
            <span class="badge">{rep_id}</span>
        </div>
    </div>

    {confounding_html}

    <div class="meta-grid">
        <div class="meta-box">
            <div class="label">Target Location</div>
            <div class="val">{metadata.get('location_name') or 'Unspecified'}</div>
        </div>
        <div class="meta-box">
            <div class="label">Baseline Date (Before)</div>
            <div class="val">{metadata.get('date_before') or 'Unspecified'}</div>
        </div>
        <div class="meta-box">
            <div class="label">Analysis Date (After)</div>
            <div class="val">{metadata.get('date_after') or 'Unspecified'}</div>
        </div>
        <div class="meta-box">
            <div class="label">Data Source & Attribution</div>
            <div class="val" style="font-size:12px;">{metadata.get('attribution') or metadata.get('sensor_name') or 'Optical Satellite Sensor'}</div>
        </div>
    </div>

    <div class="metrics-bar">
        <div class="metric-card">
            <div class="num">{metrics.get('change_percentage', 0.0):.2f}%</div>
            <div class="desc">Surface Change Coverage</div>
        </div>
        <div class="metric-card">
            <div class="num">{metrics.get('total_changed_pixels', 0):,}</div>
            <div class="desc">Changed Pixels</div>
        </div>
        <div class="metric-card">
            <div class="num">{metrics.get('regions_count', 0)}</div>
            <div class="desc">Distinct Anomaly Clusters</div>
        </div>
        <div class="metric-card">
            <div class="num">{metrics.get('processing_time_sec', 0.0)}s</div>
            <div class="desc">Computation Latency</div>
        </div>
    </div>

    <div class="card">
        <h2>Evidence-Based Anomaly Assessment</h2>
        <p><strong>Primary Categorization:</strong> <span style="color:var(--cyan); font-weight:600;">{interp.get('possible_categories', ['Unclassified'])[0]}</span></p>
        <p><strong>Plausibility:</strong> <span style="color:{conf_color}; font-weight:600;">{confidence}</span> — {interp.get('confidence_basis', '')}</p>
        <p><strong>Analysis Mode:</strong> {interp.get('mode', 'Local Deterministic Model')}</p>
        <p>{interp.get('summary', '')}</p>

        <h3 style="font-size:14px; margin-top:18px; color:var(--text-muted); text-transform:uppercase;">Observed Visual Indicators</h3>
        <ul>{obs_list}</ul>

        <h3 style="font-size:14px; margin-top:18px; color:var(--text-muted); text-transform:uppercase;">Alternative Environmental Explanations</h3>
        <ul>{alt_list}</ul>
    </div>

    {change_truth_html}

    {temporal_html}

    <div class="gallery">
        <div class="img-box">
            <img src="{img_before_b64}" alt="Baseline Optical Image">
            <div class="img-title">Baseline Optical Imagery (Before)</div>
        </div>
        <div class="img-box">
            <img src="{img_after_b64}" alt="Target Optical Image">
            <div class="img-title">Target Optical Imagery (After)</div>
        </div>
        <div class="img-box">
            <img src="{heatmap_b64}" alt="Difference Heatmap">
            <div class="img-title">Difference Heatmap (Turbo Spectral Scale)</div>
        </div>
        <div class="img-box">
            <img src="{overlay_b64}" alt="Change Detection Overlay">
            <div class="img-title">Synthesized Multi-Tier Overlay (Red=High, Amber=Moderate)</div>
        </div>
    </div>

    <div class="card">
        <h2>Top Detected Anomaly Regions</h2>
        <table>
            <thead>
                <tr>
                    <th>Cluster</th>
                    <th>Area (Pixels)</th>
                    <th>Image Fraction</th>
                    <th>Mean Intensity</th>
                    <th>Centroid (X, Y)</th>
                    <th>Aspect Ratio</th>
                </tr>
            </thead>
            <tbody>
                {region_rows}
            </tbody>
        </table>
    </div>

    <div class="card">
        <h2>Spatial Scale & Area Methodology</h2>
        <p style="font-size: 13px; color: var(--text);">{area_note}</p>
    </div>

    <div class="card">
        <h2>Recommended Follow-Up & Scientific Constraints</h2>
        <div style="display:grid; grid-template-columns: 1fr 1fr; gap:20px;">
            <div>
                <h3 style="font-size:13px; color:var(--cyan); text-transform:uppercase;">Recommended Next Steps</h3>
                <ul>{followup_list}</ul>
            </div>
            <div>
                <h3 style="font-size:13px; color:var(--amber); text-transform:uppercase;">Scientific Constraints</h3>
                <ul>{limit_list}</ul>
            </div>
        </div>
    </div>

    <div class="disclaimer">
        <strong>SCIENTIFIC & LEGAL DISCLAIMER:</strong> This analysis is derived from computer-vision differencing on uncalibrated optical imagery. Detected pixel variations do not prove legal liability, malicious intent, or permanent ground destruction. Physical ground surveys or high-resolution radar (SAR) validation are required prior to operational or regulatory action.
    </div>
</div>
</body>
</html>
"""
    return html


def generate_pdf_report(
    analysis_data: Dict[str, Any],
    metadata: Dict[str, Any],
    report_id: Optional[str] = None
) -> Optional[bytes]:
    """Generates an executive PDF report using ReportLab if available."""
    if not HAS_REPORTLAB:
        return None

    rep_id = report_id or f"TSA-{datetime.utcnow().strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}"
    buffer = io.BytesIO()

    try:
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            rightMargin=36,
            leftMargin=36,
            topMargin=36,
            bottomMargin=36
        )

        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            "DocTitle",
            parent=styles["Heading1"],
            fontSize=17,
            leading=21,
            textColor=colors.HexColor("#020303")
        )
        subtitle_style = ParagraphStyle(
            "DocSubtitle",
            parent=styles["Normal"],
            fontSize=8.5,
            textColor=colors.HexColor("#304c3b"),
            spaceAfter=12
        )
        h2_style = ParagraphStyle(
            "DocH2",
            parent=styles["Heading2"],
            fontSize=11,
            leading=14,
            textColor=colors.HexColor("#020303"),
            spaceBefore=8,
            spaceAfter=5
        )
        body_style = ParagraphStyle(
            "DocBody",
            parent=styles["Normal"],
            fontSize=8.5,
            leading=11,
            textColor=colors.HexColor("#111111")
        )
        disclaimer_style = ParagraphStyle(
            "Disclaimer",
            parent=styles["Normal"],
            fontSize=7.5,
            leading=9.5,
            textColor=colors.HexColor("#554400"),
            backColor=colors.HexColor("#fffdf0"),
            borderColor=colors.HexColor("#ffd600"),
            borderWidth=1,
            borderPadding=5,
            spaceBefore=10
        )

        story = []

        # Header
        story.append(Paragraph("TERRASENTINEL AI — EARTH INTELLIGENCE DOSSIER", title_style))
        story.append(Paragraph(f"Dossier ID: {rep_id} | Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}", subtitle_style))
        story.append(Spacer(1, 6))

        # Metadata table
        meta_table_data = [
            [
                Paragraph(f"<b>Location:</b> {metadata.get('location_name') or 'Unspecified'}", body_style),
                Paragraph(f"<b>Baseline Date:</b> {metadata.get('date_before') or 'Unspecified'}", body_style),
            ],
            [
                Paragraph(f"<b>Sensor / Attribution:</b> {metadata.get('sensor_name') or 'Optical Sensor'}", body_style),
                Paragraph(f"<b>Analysis Date:</b> {metadata.get('date_after') or 'Unspecified'}", body_style),
            ]
        ]
        meta_table = Table(meta_table_data, colWidths=[270, 270])
        meta_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#f0f4f8")),
            ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor("#d0d7de")),
            ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#e1e4e8")),
            ('TOPPADDING', (0,0), (-1,-1), 4),
            ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ]))
        story.append(meta_table)
        story.append(Spacer(1, 8))

        # Metrics Table
        metrics = analysis_data.get("metrics", {})
        metrics_data = [
            ["Metric", "Measured Value", "Scientific Description"],
            ["Surface Change Coverage", f"{metrics.get('change_percentage', 0.0):.2f}%", "Proportion of optical frame exceeding anomaly threshold"],
            ["Changed Pixels", f"{metrics.get('total_changed_pixels', 0):,}", "Cumulative pixel count in validated clusters"],
            ["Anomaly Clusters", f"{metrics.get('regions_count', 0)}", "Discrete contiguous change components extracted"],
            ["Largest Region", f"{metrics.get('largest_region_pixels', 0):,} px", f"{metrics.get('largest_region_percentage', 0.0):.2f}% of total scene area"],
            ["Processing Latency", f"{metrics.get('processing_time_sec', 0.0)}s", "End-to-end computer-vision pipeline runtime"]
        ]
        m_table = Table(metrics_data, colWidths=[130, 90, 320])
        m_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#020303")),
            ('TEXTCOLOR', (0,0), (-1,0), colors.HexColor("#38ff8e")),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (-1,-1), 7.5),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#d0d7de")),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#f8fafc")]),
            ('TOPPADDING', (0,0), (-1,-1), 3),
            ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ]))
        story.append(m_table)
        story.append(Spacer(1, 8))

        # Evidence & Assessment
        interp = analysis_data.get("ai_interpretation", {})
        truth_res = analysis_data.get("truth_result", {})
        story.append(Paragraph("Evidence-Based Anomaly Assessment & Change Truth", h2_style))
        story.append(Paragraph(f"<b>Primary Category:</b> {interp.get('possible_categories', ['Unclassified'])[0]} ({interp.get('confidence', 'moderate').upper()} Plausibility)", body_style))
        if truth_res and truth_res.get("overall_verdict"):
            story.append(Paragraph(f"<b>Change Truth Verdict:</b> {truth_res['overall_verdict']}", body_style))
            story.append(Paragraph(f"<b>Truth Challenge Rationale:</b> {truth_res.get('overall_rationale', '')}", body_style))
        story.append(Paragraph(f"<b>Assessment Summary:</b> {interp.get('summary', '')}", body_style))
        story.append(Spacer(1, 6))

        # Embedded comparison images (2x2 grid)
        buf_b = _array_to_temp_png(analysis_data["img_before"], 240)
        buf_a = _array_to_temp_png(analysis_data["img_after"], 240)
        buf_h = _array_to_temp_png(analysis_data["heatmap"], 240)
        buf_o = _array_to_temp_png(analysis_data["overlay"], 240)

        img_table_data = [
            [Paragraph("<b>Baseline Optical (Before)</b>", body_style), Paragraph("<b>Target Optical (After)</b>", body_style)],
            [RLImage(buf_b, width=255, height=155), RLImage(buf_a, width=255, height=155)],
            [Paragraph("<b>Difference Heatmap</b>", body_style), Paragraph("<b>Synthesized Anomaly Overlay</b>", body_style)],
            [RLImage(buf_h, width=255, height=155), RLImage(buf_o, width=255, height=155)],
        ]
        img_table = Table(img_table_data, colWidths=[270, 270])
        img_table.setStyle(TableStyle([
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('TOPPADDING', (0,0), (-1,-1), 2),
            ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ]))
        story.append(img_table)
        story.append(Spacer(1, 8))

        # Disclaimer
        story.append(Paragraph(
            "<b>SCIENTIFIC NOTICE:</b> This dossier is an automated preliminary evaluation. "
            "Optical pixel differences do not constitute legal proof of intentional damage, "
            "unauthorized deforestation, or land violation. Ground truthing and radar validation required.",
            disclaimer_style
        ))

        doc.build(story)
        buffer.seek(0)
        return buffer.getvalue()

    except Exception as e:
        return None
