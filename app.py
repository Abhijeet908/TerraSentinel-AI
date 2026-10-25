"""TerraSentinel AI — Satellite Anomaly Detection & Earth Intelligence Platform.

Main Application Entrypoint (Cyberpunk Design System & Change Truth Engine Edition).
"""
import os
import sys
import tempfile
import json
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List
import numpy as np
from PIL import Image
import streamlit as st

# Suppress Matplotlib cache permission warnings on Windows
os.environ["MPLCONFIGDIR"] = tempfile.gettempdir()

# Ensure project root is in sys.path for clean container and cloud execution
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Local Modules
from config import (
    APP_NAME, APP_SUBTITLE, APP_TAGLINE, APP_VERSION,
    THEME_COLORS, DESIGN_TOKENS, SAMPLE_DATA_DIR, ASSETS_DIR, META_AI_CONFIG, DEFAULT_SETTINGS
)
from modules.image_loader import load_image, harmonize_dimensions
from modules.image_preprocessing import preprocess_pair
from modules.change_detection import detect_changes
from modules.anomaly_classification import classify_changes
from modules.evidence_engine import evaluate_scene_evidence
from modules.change_truth_engine import run_change_truth_engine
from modules.temporal_engine import evaluate_temporal_sequence
from modules.ai_interpreter import generate_ai_interpretation
from modules.report_generator import generate_html_report, generate_json_report, generate_pdf_report
from components.sidebar import render_sidebar, NAVIGATION_OPTIONS
from components.comparison_view import render_comparison_workspace
from components.metrics_panel import render_metrics_cards, render_anomaly_banner
from components.visualizations import (
    render_region_table,
    render_region_evidence_inspector,
    render_change_truth_inspector,
    render_spectral_and_spatial_charts
)

# Page Configuration with clean icon asset (zero emojis)
favicon_path = ASSETS_DIR / "favicon.png"
icon_arg = str(favicon_path) if favicon_path.exists() else None

st.set_page_config(
    page_title=f"{APP_NAME} // SATELLITE INTELLIGENCE",
    page_icon=icon_arg,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Authoritative Cyberpunk Design System CSS Injection
CUSTOM_CSS = f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600;700&family=IBM+Plex+Sans:wght@400;500;600;700;800&display=swap');

    :root {{
        --dp-color-ground: {DESIGN_TOKENS['dp_color_ground']};
        --dp-color-accent: {DESIGN_TOKENS['dp_color_accent']};
        --dp-color-interference: {DESIGN_TOKENS['dp_color_interference']};
        --dp-color-registration: {DESIGN_TOKENS['dp_color_registration']};
        --dp-color-ink: {DESIGN_TOKENS['dp_color_ink']};
        --dp-color-ink-body: {DESIGN_TOKENS['dp_color_ink_body']};
        --dp-color-ink-muted: {DESIGN_TOKENS['dp_color_ink_muted']};
        --dp-color-plate: {DESIGN_TOKENS['dp_color_plate']};
        --dp-color-line: {DESIGN_TOKENS['dp_color_line']};
        --dp-color-control-line: {DESIGN_TOKENS['dp_color_control-line'] if 'dp_color_control-line' in DESIGN_TOKENS else DESIGN_TOKENS['dp_color_control_line']};
        --dp-typography-display: "Cyberpunk Plex Sans", "IBM Plex Sans", sans-serif;
        --dp-typography-mono: "Cyberpunk Plex Mono", "IBM Plex Mono", monospace;
        --dp-typography-small: 14px;
        --dp-typography-label: 12px;
        --dp-typography-metadata: 11px;
        --dp-spacing-gutter: clamp(22px, 4.5cqi, 72px);
        --dp-spacing-column-gap: 50px;
        --dp-radius-control: 0;
        --dp-border-hairline: 1px;
        --dp-motion-interaction: 0s;
    }}

    /* Universal Reset & Geometric Hardening */
    * {{
        border-radius: var(--dp-radius-control) !important;
        box-sizing: border-box;
    }}

    /* Dark Theme Core */
    .stApp {{
        background-color: var(--dp-color-ground);
        color: var(--dp-color-ink-body);
        font-family: 'IBM Plex Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }}

    /* Signal Rail: Continuous Electric Green Line down content left edge */
    [data-testid="block-container"] {{
        border-left: 2px solid var(--dp-color-accent);
        padding-left: 2.2rem !important;
        padding-right: 2rem !important;
        padding-top: 1.5rem !important;
        padding-bottom: 3rem !important;
        max-width: 1560px !important;
    }}

    /* Sidebar background */
    [data-testid="stSidebar"] {{
        background-color: var(--dp-color-ground);
        border-right: 1px solid var(--dp-color-line);
    }}

    /* Headings */
    h1, h2, h3, h4, h5, h6 {{
        font-family: 'IBM Plex Sans', sans-serif !important;
        color: var(--dp-color-ink) !important;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }}

    /* Monospace elements */
    code, pre, .telemetry-label, .telemetry-sub, .view-label, [data-testid="stMetricLabel"] {{
        font-family: 'IBM Plex Mono', monospace !important;
    }}

    /* Telemetry HUD Cards */
    .telemetry-card {{
        background: var(--dp-color-plate) !important;
        border: 1px solid var(--dp-color-line) !important;
        border-top: 2px solid var(--dp-color-accent) !important;
        padding: 12px 14px !important;
        text-align: left !important;
    }}
    .telemetry-label {{
        font-size: 10px !important;
        text-transform: uppercase !important;
        color: var(--dp-color-ink-muted) !important;
        letter-spacing: 1px !important;
        font-weight: 600 !important;
    }}
    .telemetry-value {{
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 22px !important;
        font-weight: 700 !important;
        margin-top: 4px !important;
        color: var(--dp-color-ink) !important;
    }}
    .telemetry-sub {{
        font-size: 9.5px !important;
        color: var(--dp-color-ink-muted) !important;
        margin-top: 3px !important;
    }}
    .highlight-accent {{ color: var(--dp-color-accent) !important; }}
    .highlight-registration {{ color: var(--dp-color-registration) !important; }}
    .highlight-interference {{ color: var(--dp-color-interference) !important; }}

    .view-label {{
        font-size: 11px !important;
        font-weight: 700 !important;
        letter-spacing: 1.2px !important;
        color: var(--dp-color-accent) !important;
        margin-bottom: 6px !important;
        text-transform: uppercase !important;
    }}

    /* Action Buttons */
    .stButton>button {{
        background: var(--dp-color-plate) !important;
        color: var(--dp-color-accent) !important;
        border: 1px solid var(--dp-color-control-line) !important;
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 11px !important;
        font-weight: 700 !important;
        letter-spacing: 1px !important;
        text-transform: uppercase !important;
        padding: 0.5rem 1rem !important;
        transition: all var(--dp-motion-interaction) ease-in-out !important;
    }}
    .stButton>button:hover {{
        background: var(--dp-color-accent) !important;
        color: var(--dp-color-ground) !important;
        border-color: var(--dp-color-accent) !important;
        box-shadow: 0 0 12px rgba(56, 255, 142, 0.4) !important;
    }}
    .stButton>button[kind="primary"] {{
        background: var(--dp-color-accent) !important;
        color: var(--dp-color-ground) !important;
        border-color: var(--dp-color-accent) !important;
    }}
    .stButton>button[kind="primary"]:hover {{
        background: #50ffa4 !important;
        color: var(--dp-color-ground) !important;
        box-shadow: 0 0 16px rgba(56, 255, 142, 0.6) !important;
    }}

    /* Download Buttons */
    .stDownloadButton>button {{
        background: var(--dp-color-plate) !important;
        color: var(--dp-color-registration) !important;
        border: 1px solid var(--dp-color-control-line) !important;
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 11px !important;
        font-weight: 700 !important;
        letter-spacing: 0.8px !important;
        text-transform: uppercase !important;
        width: 100% !important;
    }}
    .stDownloadButton>button:hover {{
        background: var(--dp-color-registration) !important;
        color: var(--dp-color-ground) !important;
        border-color: var(--dp-color-registration) !important;
        box-shadow: 0 0 12px rgba(101, 248, 255, 0.4) !important;
    }}

    /* Radio buttons */
    .stRadio [role="radiogroup"] {{
        gap: 8px;
    }}
    .stRadio label {{
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 11px !important;
        letter-spacing: 0.5px;
    }}

    /* Form controls & inputs */
    input, select, textarea {{
        background-color: var(--dp-color-plate) !important;
        color: var(--dp-color-ink) !important;
        border: 1px solid var(--dp-color-line) !important;
        font-family: 'IBM Plex Mono', monospace !important;
    }}

    /* Sliders */
    [data-testid="stSlider"] {{
        font-family: 'IBM Plex Mono', monospace !important;
    }}

    /* Streamlit Metric Overrides */
    [data-testid="stMetricValue"] {{
        font-family: 'IBM Plex Mono', monospace !important;
        color: var(--dp-color-accent) !important;
    }}
    [data-testid="stMetricLabel"] {{
        font-size: 10px !important;
        text-transform: uppercase !important;
        color: var(--dp-color-ink-muted) !important;
    }}

    /* Expanders */
    [data-testid="stExpander"] {{
        background-color: var(--dp-color-plate) !important;
        border: 1px solid var(--dp-color-line) !important;
    }}

    /* Tables and Dataframes */
    [data-testid="stDataFrame"] {{
        border: 1px solid var(--dp-color-line) !important;
        font-family: 'IBM Plex Mono', monospace !important;
    }}

    /* Horizontal Rules */
    hr {{
        border: none !important;
        border-bottom: 1px solid var(--dp-color-line) !important;
        margin: 18px 0 !important;
    }}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def init_session_state():
    """Initializes persistent application session state."""
    defaults = {
        "active_nav": NAVIGATION_OPTIONS[0],
        "img_before": None,
        "img_after": None,
        "img_t2": None,  # Optional third temporal acquisition
        "meta_before": {},
        "meta_after": {},
        "user_meta": {
            "location_name": "",
            "date_before": "",
            "date_after": "",
            "date_t2": "",
            "sensor_name": "Optical Multispectral / RGB",
            "approx_resolution_m": None,
            "spatial_calibration_confirmed": False,
            "data_tier": "User Uploaded Imagery",
            "attribution": "",
            "notes": ""
        },
        "analysis_result": None,
        "evidence_result": None,
        "truth_result": None,
        "temporal_result": None,
        "heuristic_result": None,
        "ai_result": None,
        "was_resized": False,
        "preprocess_diag": {}
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session_state()

# Render Mission Control Sidebar
sidebar_cfg = render_sidebar()


def run_pipeline():
    """Executes the change detection, Evidence Engine, Change Truth challenge, and temporal reasoning."""
    if st.session_state["img_before"] is None or st.session_state["img_after"] is None:
        st.error("[ERROR] Both BASELINE and TARGET satellite imagery must be supplied before initiating analysis.")
        return

    with st.spinner("Executing Change Truth Engine and Computer Vision Differencing Pipeline..."):
        # Harmonize dimensions if different
        raw_b = st.session_state["img_before"]
        raw_a = st.session_state["img_after"]
        norm_b, norm_a, was_resized = harmonize_dimensions(raw_b, raw_a)
        st.session_state["was_resized"] = was_resized

        # Preprocessing & Optional Registration
        proc_b, proc_a, diag = preprocess_pair(
            norm_b,
            norm_a,
            enable_alignment=sidebar_cfg["enable_alignment"],
            blur_radius=sidebar_cfg["blur_radius"]
        )
        st.session_state["preprocess_diag"] = diag

        # Change Detection
        detection = detect_changes(
            proc_b,
            proc_a,
            threshold=sidebar_cfg["threshold"],
            min_region_size=sidebar_cfg["min_region_size"],
            structural_weight=sidebar_cfg["structural_weight"],
            overlay_opacity=sidebar_cfg["overlay_opacity"]
        )
        # Store original normalized images for display
        detection["img_before"] = norm_b
        detection["img_after"] = norm_a

        # 1. Evidence Engine Analysis (Per-region geometric & spectral evidence)
        evidence_res = evaluate_scene_evidence(
            detection["regions"],
            detection["binary_mask"],
            proc_b,
            proc_a,
            detection["raw_diff_map"],
            detection["metrics"],
            metadata={
                **st.session_state["user_meta"],
                "was_resized": was_resized,
                "registration_status": diag.get("registration_status", "Default grid")
            }
        )
        st.session_state["evidence_result"] = evidence_res
        detection["evidence_result"] = evidence_res

        # 2. Heuristic Classification (Legacy baseline compatibility)
        heuristic = classify_changes(
            proc_b,
            proc_a,
            detection["binary_mask"],
            detection["regions"],
            detection["metrics"]
        )
        st.session_state["heuristic_result"] = heuristic

        # 3. Temporal Sequence Reasoning
        seq = [
            (norm_b, st.session_state["user_meta"].get("date_before") or "T0 Baseline"),
            (norm_a, st.session_state["user_meta"].get("date_after") or "T1 Disturbance")
        ]
        if st.session_state["img_t2"] is not None:
            _, norm_t2, _ = harmonize_dimensions(norm_b, st.session_state["img_t2"])
            seq.append((norm_t2, st.session_state["user_meta"].get("date_t2") or "T2 Follow-up"))

        temporal_res = evaluate_temporal_sequence(
            seq,
            threshold=sidebar_cfg["threshold"],
            min_region_size=sidebar_cfg["min_region_size"]
        )
        st.session_state["temporal_result"] = temporal_res
        detection["temporal_result"] = temporal_res

        # 4. SIGNATURE FEATURE: Change Truth Engine (Counterfactual Challenge & Effect Size)
        truth_res = run_change_truth_engine(
            detection["regions"],
            detection["binary_mask"],
            proc_b,
            proc_a,
            detection["raw_diff_map"],
            detection["metrics"],
            metadata={
                **st.session_state["user_meta"],
                "was_resized": was_resized,
                "registration_status": diag.get("registration_status", "Default grid")
            },
            temporal_result=temporal_res,
            seed=42
        )
        st.session_state["truth_result"] = truth_res
        detection["truth_result"] = truth_res

        # 5. AI Interpretation Adapter
        combined_meta = {
            **st.session_state["user_meta"],
            "registration_status": diag.get("registration_status", "Default grid alignment")
        }
        ai_interp = generate_ai_interpretation(
            detection["metrics"],
            heuristic,
            combined_meta
        )
        detection["ai_interpretation"] = ai_interp
        st.session_state["ai_result"] = ai_interp
        st.session_state["analysis_result"] = detection

    st.success("[STATUS: VALIDATED] Anomaly detection and Change Truth evaluation computed successfully.")


def load_demo_dataset(dataset_key: str):
    """Loads a pre-bundled dataset (Real NASA imagery or synthetic benchmark) into session state."""
    ds_dir = SAMPLE_DATA_DIR / dataset_key
    if not ds_dir.exists():
        st.error(f"[ERROR] Dataset directory {dataset_key} not found.")
        return

    before_path = ds_dir / "before.png"
    after_path = ds_dir / "after.png"
    meta_path = ds_dir / "metadata.json"
    t2_path = ds_dir / "t2_recovery.png"
    if not t2_path.exists():
        t2_path = ds_dir / "t2_2012.png"

    if before_path.exists() and after_path.exists():
        arr_b, m_b = load_image(before_path)
        arr_a, m_a = load_image(after_path)
        st.session_state["img_before"] = arr_b
        st.session_state["img_after"] = arr_a
        st.session_state["meta_before"] = m_b
        st.session_state["meta_after"] = m_a

        if t2_path.exists():
            arr_t2, _ = load_image(t2_path)
            st.session_state["img_t2"] = arr_t2
        else:
            st.session_state["img_t2"] = None

        if meta_path.exists():
            with open(meta_path, "r") as f:
                d_meta = json.load(f)
                dates = d_meta.get("dates", {})
                st.session_state["user_meta"] = {
                    "location_name": d_meta.get("location_name", ""),
                    "date_before": d_meta.get("date_before", dates.get("t0_baseline", "")),
                    "date_after": d_meta.get("date_after", dates.get("t1_event", dates.get("t2_target", ""))),
                    "date_t2": dates.get("t2_recovery", dates.get("t2_target", "")),
                    "sensor_name": d_meta.get("sensor_name", ""),
                    "approx_resolution_m": d_meta.get("approx_resolution_m", None),
                    "spatial_calibration_confirmed": d_meta.get("spatial_calibration_confirmed", False),
                    "data_tier": d_meta.get("data_tier", "Documented Dataset"),
                    "attribution": d_meta.get("attribution", ""),
                    "notes": d_meta.get("description", "")
                }

        run_pipeline()
        st.session_state["active_nav"] = NAVIGATION_OPTIONS[0]
        st.rerun()


# ==============================================================================
# MAIN HEADER & EDITORIAL COMPOSITION (Part 2 Specification)
# ==============================================================================
h_col1, h_col2 = st.columns([2.8, 2.2])
with h_col1:
    st.markdown(f"""
    <div style="margin-bottom: 4px;">
        <div style="font-family: 'IBM Plex Sans', sans-serif; font-size: 32px; font-weight: 800; color: var(--dp-color-ink); letter-spacing: 1.2px; text-transform: uppercase; line-height: 1.1;">
            {APP_NAME}
        </div>
        <div style="font-family: 'IBM Plex Mono', monospace; font-size: 11px; color: var(--dp-color-accent); letter-spacing: 1.8px; text-transform: uppercase; font-weight: 700; margin-top: 4px;">
            {APP_SUBTITLE}
        </div>
        <div style="font-family: 'IBM Plex Mono', monospace; font-size: 10px; color: var(--dp-color-ink-muted); margin-top: 4px; letter-spacing: 0.5px; text-transform: uppercase;">
            {APP_TAGLINE}
        </div>
    </div>
    """, unsafe_allow_html=True)

with h_col2:
    st.markdown("<div style='text-align: right; padding-top: 4px;'>", unsafe_allow_html=True)
    # The two clear primary actions mandated by specification
    btn_a1, btn_a2 = st.columns(2)
    with btn_a1:
        if st.button("LOAD VERIFIED CASE", use_container_width=True, help="Navigate to verified NASA Earth Observation case library"):
            st.session_state["active_nav"] = "06 / DATA & BENCHMARKS"
            st.rerun()
    with btn_a2:
        if st.button("ANALYZE IMAGES", type="primary", use_container_width=True, help="Execute Change Detection and Change Truth Engine"):
            run_pipeline()

    # Direct 1-Click NASA Case Pills for the 90-Second Live Demonstration
    p1, p2, p3, p4 = st.columns(4)
    with p1:
        if st.button("KERALA '18", help="NASA EO 92669 Landsat-8/Sentinel-2 Kerala monsoon inundation"):
            load_demo_dataset("real_kerala_flood_2018")
    with p2:
        if st.button("AMAZON", help="NASA Terra MODIS Rondônia deforestation 3-date sequence"):
            load_demo_dataset("real_amazon_deforestation")
    with p3:
        if st.button("LAKE MEAD", help="NASA/USGS Landsat shoreline recession sequence"):
            load_demo_dataset("real_lakemead_shoreline")
    with p4:
        if st.button("WILDFIRE", help="NASA/USGS Landsat Yellowstone historic wildfire & recovery"):
            load_demo_dataset("real_yellowstone_wildfire")
    st.markdown("</div>", unsafe_allow_html=True)

# Ruled Technical Facts Row (Strictly Factual Metadata)
loc_str = st.session_state["user_meta"].get("location_name") or "UNSPECIFIED COORDINATES"
sens_str = st.session_state["user_meta"].get("sensor_name") or "OPTICAL MULTISPECTRAL"
st.markdown(f"""
<div style="display: flex; justify-content: space-between; align-items: center; border-top: 1px solid var(--dp-color-line); border-bottom: 1px solid var(--dp-color-line); padding: 6px 4px; margin: 12px 0 20px 0; font-family: 'IBM Plex Mono', monospace; font-size: 10px; color: var(--dp-color-ink-muted); flex-wrap: wrap; gap: 8px;">
    <span>LOCATION // <b style="color: var(--dp-color-ink);">{loc_str.upper()}</b></span>
    <span>SENSOR // <b style="color: var(--dp-color-registration);">{sens_str.upper()}</b></span>
    <span>DETECTION // <b style="color: var(--dp-color-accent);">LAB-ΔE + SSIM (DETERMINISTIC)</b></span>
    <span>EVALUATION // <b style="color: var(--dp-color-accent);">CHANGE TRUTH ENGINE ACTIVE</b></span>
</div>
""", unsafe_allow_html=True)


# ==============================================================================
# TAB 1: 01 / CHANGE DETECTION
# ==============================================================================
if "CHANGE DETECTION" in sidebar_cfg["nav"]:
    st.markdown("""
    <div style="font-family: 'IBM Plex Mono', monospace; font-size: 11px; color: var(--dp-color-accent); letter-spacing: 1.5px; text-transform: uppercase; margin-bottom: 2px; font-weight: 700;">
        // MODULE 01
    </div>
    <div style="font-family: 'IBM Plex Sans', sans-serif; font-size: 18px; font-weight: 700; color: var(--dp-color-ink); text-transform: uppercase; margin-bottom: 12px;">
        SATELLITE IMAGERY INGESTION & DIFFERENCING WORKSPACE
    </div>
    """, unsafe_allow_html=True)

    # Data Tier Indicator Badge
    tier_label = st.session_state["user_meta"].get("data_tier", "User Uploaded Imagery")
    is_real = "Real" in tier_label
    badge_bg = "rgba(56, 255, 142, 0.1)" if is_real else "rgba(255, 214, 0, 0.1)"
    badge_color = "var(--dp-color-accent)" if is_real else "#ffd600"

    st.markdown(f"""
    <div style="margin-bottom: 15px; font-family: 'IBM Plex Mono', monospace; font-size: 11px;">
        <span style="color: var(--dp-color-ink-muted); font-weight: 600;">ACTIVE DATA PROVENANCE:</span>
        <span style="background: {badge_bg}; color: {badge_color}; border: 1px solid {badge_color}; padding: 2px 8px; font-weight: 700; margin-left: 6px; letter-spacing: 0.5px;">
            {tier_label.upper()}
        </span>
    </div>
    """, unsafe_allow_html=True)

    # Ingestion Upload Section
    up_col1, up_col2 = st.columns(2)

    with up_col1:
        date_b_label = st.session_state["user_meta"].get("date_before") or "Baseline"
        st.markdown(f"""
        <div style="font-family: 'IBM Plex Mono', monospace; font-size: 11px; font-weight: 700; color: var(--dp-color-ink); text-transform: uppercase; margin-bottom: 6px;">
            01 // BASELINE IMAGERY (BEFORE // {date_b_label})
        </div>
        """, unsafe_allow_html=True)
        file_before = st.file_uploader(
            "Upload Before Image (PNG, JPG, TIFF)",
            type=["png", "jpg", "jpeg", "tif", "tiff"],
            key="upload_before",
            label_visibility="collapsed",
            help="Pre-event or historical reference optical scene."
        )
        if file_before:
            try:
                arr, meta = load_image(file_before)
                st.session_state["img_before"] = arr
                st.session_state["meta_before"] = meta
                st.session_state["user_meta"]["data_tier"] = "User Uploaded Imagery (Uncalibrated GSD)"
                st.session_state["user_meta"]["spatial_calibration_confirmed"] = False
            except Exception as e:
                st.error(f"[ERROR] Decoding BASELINE image: {e}")

    with up_col2:
        date_a_label = st.session_state["user_meta"].get("date_after") or "Target"
        st.markdown(f"""
        <div style="font-family: 'IBM Plex Mono', monospace; font-size: 11px; font-weight: 700; color: var(--dp-color-ink); text-transform: uppercase; margin-bottom: 6px;">
            02 // TARGET IMAGERY (AFTER // {date_a_label})
        </div>
        """, unsafe_allow_html=True)
        file_after = st.file_uploader(
            "Upload After Image (PNG, JPG, TIFF)",
            type=["png", "jpg", "jpeg", "tif", "tiff"],
            key="upload_after",
            label_visibility="collapsed",
            help="Post-event or target surveillance optical scene."
        )
        if file_after:
            try:
                arr, meta = load_image(file_after)
                st.session_state["img_after"] = arr
                st.session_state["meta_after"] = meta
            except Exception as e:
                st.error(f"[ERROR] Decoding TARGET image: {e}")

    # Metadata & Spatial Calibration Expander
    with st.expander("SCENE METADATA & SPATIAL CALIBRATION CONFIGURATION", expanded=False):
        mc1, mc2, mc3 = st.columns(3)
        with mc1:
            st.session_state["user_meta"]["location_name"] = st.text_input(
                "Geographic Location Name",
                value=st.session_state["user_meta"]["location_name"],
                placeholder="e.g. Kerala Basin, India"
            )
            st.session_state["user_meta"]["sensor_name"] = st.text_input(
                "Satellite / Sensor Name",
                value=st.session_state["user_meta"]["sensor_name"],
                placeholder="e.g. Landsat-8 OLI or Sentinel-2"
            )
        with mc2:
            st.session_state["user_meta"]["date_before"] = st.text_input(
                "Baseline Acquisition Date",
                value=st.session_state["user_meta"]["date_before"],
                placeholder="YYYY-MM-DD"
            )
            st.session_state["user_meta"]["date_after"] = st.text_input(
                "Target Acquisition Date",
                value=st.session_state["user_meta"]["date_after"],
                placeholder="YYYY-MM-DD"
            )
        with mc3:
            res_val = st.number_input(
                "Ground Sample Distance GSD (m/px)",
                min_value=0.1,
                max_value=1000.0,
                value=float(st.session_state["user_meta"]["approx_resolution_m"]) if st.session_state["user_meta"]["approx_resolution_m"] else 10.0,
                step=1.0,
                help="GSD in meters per pixel."
            )
            st.session_state["user_meta"]["approx_resolution_m"] = res_val
            calib_check = st.checkbox(
                "Confirm verified spatial calibration metadata",
                value=bool(st.session_state["user_meta"].get("spatial_calibration_confirmed", False)),
                help="Only check if GSD is verified from sensor metadata. Prevents inventing fake physical areas."
            )
            st.session_state["user_meta"]["spatial_calibration_confirmed"] = calib_check

    # Action Bar
    btn_col1, btn_col2 = st.columns([1.4, 3.6])
    with btn_col1:
        if st.button("RUN CHANGE DETECTION", type="primary", use_container_width=True):
            run_pipeline()

    # Confounding factors alerts
    ev_res = st.session_state.get("evidence_result")
    if ev_res and ev_res.get("confounding_warnings"):
        for cw in ev_res["confounding_warnings"]:
            st.warning(f"[OPTICAL/REGISTRATION NOTICE] {cw}")

    # Render Telemetry and Anomaly Banners
    if st.session_state.get("analysis_result") is not None:
        st.markdown("<hr>", unsafe_allow_html=True)
        render_metrics_cards(st.session_state["analysis_result"]["metrics"], st.session_state["user_meta"])
        render_anomaly_banner(st.session_state.get("heuristic_result"), st.session_state.get("ai_result"), ev_res)

    # Comparison Workspace
    if st.session_state["img_before"] is not None and st.session_state["img_after"] is not None:
        st.markdown("<hr>", unsafe_allow_html=True)
        render_comparison_workspace(
            st.session_state["img_before"],
            st.session_state["img_after"],
            analysis_result=st.session_state.get("analysis_result"),
            metadata=st.session_state["user_meta"]
        )
    else:
        st.info("[READY] Select an authentic NASA benchmark above or upload image files to initiate analysis.")


# ==============================================================================
# TAB 2: 02 / CHANGE TRUTH ENGINE (SIGNATURE FEATURE)
# ==============================================================================
elif "TRUTH" in sidebar_cfg["nav"] or "EVIDENCE" in sidebar_cfg["nav"]:
    st.markdown("""
    <div style="font-family: 'IBM Plex Mono', monospace; font-size: 11px; color: var(--dp-color-accent); letter-spacing: 1.5px; text-transform: uppercase; margin-bottom: 2px; font-weight: 700;">
        // MODULE 02 // SIGNATURE DIFFERENTIATOR
    </div>
    <div style="font-family: 'IBM Plex Sans', sans-serif; font-size: 22px; font-weight: 800; color: var(--dp-color-ink); text-transform: uppercase; margin-bottom: 4px;">
        CHANGE TRUTH ENGINE
    </div>
    <div style="font-family: 'IBM Plex Mono', monospace; font-size: 11.5px; color: var(--dp-color-accent); font-weight: 500; margin-bottom: 12px; letter-spacing: 0.5px;">
        An evidence challenge for every detected anomaly.
    </div>
    <div style="font-size: 12.5px; color: var(--dp-color-ink-body); margin-bottom: 16px; line-height: 1.6;">
        Traditional satellite differencing naively classifies every changed pixel, producing false alarms from illumination shifts, atmospheric haze, and misregistration. The <b>Change Truth Engine</b> actively challenges its own detections through 8 stages: measuring candidate metrics, establishing background variance, conducting counterfactual control-region sampling, challenging competing explanations, verifying multi-date temporal persistence, and issuing descriptive evidence verdicts.
    </div>
    """, unsafe_allow_html=True)

    analysis_res = st.session_state.get("analysis_result")
    truth_res = st.session_state.get("truth_result")

    if not analysis_res or not truth_res:
        st.info("[AWAITING DATA] No active analysis in memory. Please execute change detection on an image pair first.")
    else:
        # Dominant Scene Verdict Banner
        ov = truth_res.get("overall_verdict", "CHANGE DETECTED")
        ov_color = "var(--dp-color-accent)" if ("LOCALIZED" in ov or "PERSISTENCE" in ov) else ("#ffd600" if "CONFOUND" in ov or "UNCERTAIN" in ov else "var(--dp-color-interference)")
        bg_data = truth_res.get("background_variation", {})

        st.markdown(f"""
        <div style="background: var(--dp-color-plate); border: 1px solid var(--dp-color-line); border-left: 4px solid {ov_color}; padding: 16px 20px; margin-bottom: 20px;">
            <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 8px;">
                <div>
                    <span style="font-family: 'IBM Plex Mono', monospace; font-size: 10px; color: var(--dp-color-ink-muted); text-transform: uppercase; letter-spacing: 1.5px; font-weight: 700;">
                        // DOMINANT SCENE-LEVEL TRUTH VERDICT
                    </span>
                    <div style="font-family: 'IBM Plex Sans', sans-serif; font-size: 20px; font-weight: 800; color: {ov_color}; margin-top: 4px;">
                        {ov}
                    </div>
                </div>
                <div style="font-family: 'IBM Plex Mono', monospace; font-size: 10px; background: rgba(56, 255, 142, 0.08); border: 1px solid var(--dp-color-line); padding: 4px 10px;">
                    BACKGROUND SHIFT: <b>{bg_data.get('bg_mean_delta', 0.0):.1f} DN</b> (σ = {bg_data.get('bg_std_delta', 0.0):.1f})
                </div>
            </div>
            <div style="font-family: 'IBM Plex Sans', sans-serif; font-size: 13px; color: var(--dp-color-ink-body); margin-top: 6px;">
                {truth_res.get('overall_rationale', '')}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Stage G: Interactive Demonstration Inspector
        render_change_truth_inspector(
            truth_res,
            analysis_res["img_before"],
            analysis_res["img_after"],
            analysis_res["overlay"]
        )

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("""
        <div style="font-family: 'IBM Plex Mono', monospace; font-size: 11px; color: var(--dp-color-accent); font-weight: 700; text-transform: uppercase; margin-bottom: 4px;">
            // SCENE CLUSTERS
        </div>
        <div style="font-family: 'IBM Plex Sans', sans-serif; font-size: 15px; font-weight: 700; color: var(--dp-color-ink); text-transform: uppercase; margin-bottom: 8px;">
            TOPOLOGICAL CLUSTER DECOMPOSITION & EVIDENCE TABLE
        </div>
        """, unsafe_allow_html=True)
        render_region_table(analysis_res["regions"], st.session_state.get("evidence_result"))


# ==============================================================================
# TAB 3: 03 / TEMPORAL DYNAMICS
# ==============================================================================
elif "TEMPORAL" in sidebar_cfg["nav"]:
    st.markdown("""
    <div style="font-family: 'IBM Plex Mono', monospace; font-size: 11px; color: var(--dp-color-accent); letter-spacing: 1.5px; text-transform: uppercase; margin-bottom: 2px; font-weight: 700;">
        // MODULE 03
    </div>
    <div style="font-family: 'IBM Plex Sans', sans-serif; font-size: 18px; font-weight: 700; color: var(--dp-color-ink); text-transform: uppercase; margin-bottom: 8px;">
        MULTI-DATE TEMPORAL REASONING & PERSISTENCE
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="background: var(--dp-color-plate); border: 1px solid var(--dp-color-line); border-left: 3px solid var(--dp-color-accent); padding: 14px 18px; margin-bottom: 18px; font-size: 12px; line-height: 1.6;">
        <b style="font-family: 'IBM Plex Mono', monospace; color: var(--dp-color-accent); text-transform: uppercase;">Scientific Principle: Never infer persistence from two dates.</b><br>
        A pair comparison proves only that a difference exists at T1, but cannot establish whether that anomaly is permanent land destruction or transient inundation.
        Ingesting a third chronological timestamp (T2) allows TerraSentinel AI to distinguish persistent land disturbance from transient events that subsequently reverse.
    </div>
    """, unsafe_allow_html=True)

    temp_res = st.session_state.get("temporal_result")
    analysis_res = st.session_state.get("analysis_result")

    # Quick loaders for temporal datasets
    t_btn1, t_btn2, t_btn3 = st.columns(3)
    with t_btn1:
        if st.button("AMAZON LOGGING 3-DATE", use_container_width=True):
            load_demo_dataset("real_amazon_deforestation")
    with t_btn2:
        if st.button("YELLOWSTONE 20-YR RECOVERY", use_container_width=True):
            load_demo_dataset("real_yellowstone_wildfire")
    with t_btn3:
        if st.button("SYNTHETIC FLOOD & RECESSION", use_container_width=True):
            load_demo_dataset("synthetic_flood_temporal")

    if not temp_res:
        st.info("[AWAITING SEQUENCE] No multi-date sequence loaded. Select one of the 3-date benchmark datasets above.")
    else:
        st.markdown(f"""
        <div style="background: var(--dp-color-plate); border: 1px solid var(--dp-color-line); border-left: 4px solid var(--dp-color-accent); padding: 16px 20px; margin: 15px 0;">
            <div style="font-family: 'IBM Plex Mono', monospace; font-size: 10px; color: var(--dp-color-accent); text-transform: uppercase; font-weight: 700; letter-spacing: 1px;">
                // TEMPORAL DYNAMICS CLASSIFICATION
            </div>
            <div style="font-family: 'IBM Plex Sans', sans-serif; font-size: 18px; font-weight: 700; color: var(--dp-color-ink); margin-top: 4px;">
                {temp_res['classification']}
            </div>
            <div style="font-family: 'IBM Plex Sans', sans-serif; font-size: 13px; color: var(--dp-color-ink-body); margin-top: 8px;">
                {temp_res['scientific_rationale']}
            </div>
            <div style="font-family: 'IBM Plex Mono', monospace; font-size: 11px; color: var(--dp-color-ink-muted); margin-top: 8px; border-top: 1px solid var(--dp-color-line); padding-top: 6px;">
                PERSISTENCE STATE // <span style="color: {'var(--dp-color-interference)' if temp_res['persistence_confirmed'] else ('var(--dp-color-accent)' if temp_res['reversal_detected'] else '#ffd600')}; font-weight: 700;">{temp_res['persistence_state'].upper()}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if "metrics" in temp_res and "persistent_px" in temp_res["metrics"]:
            tm = temp_res["metrics"]
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("T0 -> T1 DISTURBED", f"{tm['t0_t1_changed_px']:,} px")
            with m2:
                st.metric("T0 -> T2 NET DELTA", f"{tm['t0_t2_changed_px']:,} px")
            with m3:
                st.metric("PERSISTENT AREA", f"{tm['persistent_px']:,} px", f"{tm['persistence_ratio']*100:.1f}%")
            with m4:
                st.metric("REVERTED AREA", f"{tm['reverted_px']:,} px", f"{tm['reversal_ratio']*100:.1f}%")

        # View multi-date images
        if st.session_state.get("img_t2") is not None:
            st.markdown("""
            <div style="font-family: 'IBM Plex Mono', monospace; font-size: 11px; color: var(--dp-color-accent); font-weight: 700; text-transform: uppercase; margin: 16px 0 6px 0;">
                // SEQUENCE VIEWPORT
            </div>
            <div style="font-family: 'IBM Plex Sans', sans-serif; font-size: 15px; font-weight: 700; color: var(--dp-color-ink); text-transform: uppercase; margin-bottom: 10px;">
                CHRONOLOGICAL THREE-STAGE VIEWPORT
            </div>
            """, unsafe_allow_html=True)
            iv1, iv2, iv3 = st.columns(3)
            with iv1:
                st.markdown(f"<div class='view-label'>T0 BASELINE // {st.session_state['user_meta'].get('date_before', 'T0')}</div>", unsafe_allow_html=True)
                st.image(st.session_state["img_before"], use_container_width=True)
            with iv2:
                st.markdown(f"<div class='view-label'>T1 EVENT // {st.session_state['user_meta'].get('date_after', 'T1')}</div>", unsafe_allow_html=True)
                st.image(st.session_state["img_after"], use_container_width=True)
            with iv3:
                st.markdown(f"<div class='view-label'>T2 FOLLOW-UP // {st.session_state['user_meta'].get('date_t2', 'T2')}</div>", unsafe_allow_html=True)
                st.image(st.session_state["img_t2"], use_container_width=True)


# ==============================================================================
# TAB 4: 04 / QUANTITATIVE ANALYSIS
# ==============================================================================
elif "QUANTITATIVE" in sidebar_cfg["nav"]:
    st.markdown("""
    <div style="font-family: 'IBM Plex Mono', monospace; font-size: 11px; color: var(--dp-color-accent); letter-spacing: 1.5px; text-transform: uppercase; margin-bottom: 2px; font-weight: 700;">
        // MODULE 04
    </div>
    <div style="font-family: 'IBM Plex Sans', sans-serif; font-size: 18px; font-weight: 700; color: var(--dp-color-ink); text-transform: uppercase; margin-bottom: 12px;">
        QUANTITATIVE TELEMETRY & SPATIAL SCALE
    </div>
    """, unsafe_allow_html=True)

    analysis_res = st.session_state.get("analysis_result")
    if not analysis_res:
        st.info("[AWAITING DATA] No quantitative telemetry available. Run an analysis from the Change Detection Studio.")
    else:
        render_metrics_cards(analysis_res["metrics"], st.session_state["user_meta"])

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("""
        <div style="font-family: 'IBM Plex Mono', monospace; font-size: 11px; color: var(--dp-color-accent); font-weight: 700; text-transform: uppercase; margin-bottom: 4px;">
            // SPATIAL CALIBRATION
        </div>
        <div style="font-family: 'IBM Plex Sans', sans-serif; font-size: 15px; font-weight: 700; color: var(--dp-color-ink); text-transform: uppercase; margin-bottom: 8px;">
            GROUND SAMPLE DISTANCE (GSD) & PHYSICAL AREA POLICY
        </div>
        """, unsafe_allow_html=True)

        meta = st.session_state["user_meta"]
        is_calib = meta.get("spatial_calibration_confirmed", False)
        res_m = meta.get("approx_resolution_m")

        if is_calib and res_m:
            st.success(
                f"[SPATIAL CALIBRATION CONFIRMED] Sensor GSD is calibrated at {res_m} m/pixel. "
                f"Physical ground area computed via planar pixel-cell scaling: Area = Pixels * GSD². "
                f"Total changed area: {((analysis_res['metrics']['total_changed_pixels'] * (res_m**2)) / 10000.0):,.2f} hectares."
            )
        else:
            st.warning(
                "[SPATIAL CALIBRATION WITHHELD] Physical ground area (hectares / km²) is withheld because verified "
                "sensor metadata is unconfirmed. TerraSentinel AI refuses to fabricate real-world land areas without confirmed calibration."
            )

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("""
        <div style="font-family: 'IBM Plex Mono', monospace; font-size: 11px; color: var(--dp-color-accent); font-weight: 700; text-transform: uppercase; margin-bottom: 4px;">
            // ANOMALY CLUSTERS
        </div>
        <div style="font-family: 'IBM Plex Sans', sans-serif; font-size: 15px; font-weight: 700; color: var(--dp-color-ink); text-transform: uppercase; margin-bottom: 8px;">
            DETECTED ANOMALY CLUSTERS
        </div>
        """, unsafe_allow_html=True)
        render_region_table(analysis_res["regions"], st.session_state.get("evidence_result"))

        st.markdown("<hr>", unsafe_allow_html=True)
        render_spectral_and_spatial_charts(
            analysis_res["raw_diff_map"],
            analysis_res["regions"],
            st.session_state["heuristic_result"]
        )


# ==============================================================================
# TAB 5: 05 / REPORTS
# ==============================================================================
elif "REPORT" in sidebar_cfg["nav"]:
    st.markdown("""
    <div style="font-family: 'IBM Plex Mono', monospace; font-size: 11px; color: var(--dp-color-accent); letter-spacing: 1.5px; text-transform: uppercase; margin-bottom: 2px; font-weight: 700;">
        // MODULE 05
    </div>
    <div style="font-family: 'IBM Plex Sans', sans-serif; font-size: 18px; font-weight: 700; color: var(--dp-color-ink); text-transform: uppercase; margin-bottom: 12px;">
        AUTOMATED INTELLIGENCE DOSSIER & REPORT EXPORTS
    </div>
    """, unsafe_allow_html=True)

    analysis_res = st.session_state.get("analysis_result")
    if not analysis_res:
        st.info("[AWAITING DATA] Please execute an anomaly analysis before generating intelligence reports.")
    else:
        st.markdown(
            "TerraSentinel AI generates executive-ready, evidence-based dossiers packaged with "
            "embedded visual overlays, quantitative telemetry, Change Truth counterfactual controls, and scientific disclaimers."
        )

        meta = {
            **st.session_state["user_meta"],
            "registration_status": st.session_state.get("preprocess_diag", {}).get("registration_status", "Default grid")
        }

        try:
            html_content = generate_html_report(analysis_res, meta)
            json_content = generate_json_report(analysis_res, meta)
            pdf_bytes = generate_pdf_report(analysis_res, meta)
        except Exception as e:
            st.error(f"[ERROR] Compiling intelligence report export: {e}")
            json_content = json.dumps({"error": str(e), "status": "failed"}, indent=2)
            html_content = f"<html><body><h3>Report Compilation Error</h3><p>{e}</p></body></html>"
            pdf_bytes = None

        rc1, rc2, rc3 = st.columns(3)

        with rc1:
            st.download_button(
                label="DOWNLOAD HTML DOSSIER",
                data=html_content,
                file_name="TerraSentinel_AI_Dossier.html",
                mime="text/html",
                use_container_width=True
            )
            st.caption("Self-contained interactive HTML with embedded base64 overlays.")

        with rc2:
            st.download_button(
                label="DOWNLOAD JSON DATA EXPORT",
                data=json_content,
                file_name="TerraSentinel_AI_Metrics.json",
                mime="application/json",
                use_container_width=True
            )
            st.caption("Standardized machine-readable geospatial telemetry export.")

        with rc3:
            if pdf_bytes is not None:
                st.download_button(
                    label="DOWNLOAD PDF BRIEFING",
                    data=pdf_bytes,
                    file_name="TerraSentinel_AI_Report.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
                st.caption("High-resolution publication PDF generated via ReportLab.")
            else:
                st.caption("PDF generation offline; HTML & JSON available.")

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("""
        <div style="font-family: 'IBM Plex Mono', monospace; font-size: 11px; color: var(--dp-color-accent); font-weight: 700; text-transform: uppercase; margin-bottom: 4px;">
            // DOSSIER PREVIEW
        </div>
        """, unsafe_allow_html=True)
        st.components.v1.html(html_content, height=650, scrolling=True)


# ==============================================================================
# TAB 6: 06 / DATA & BENCHMARKS
# ==============================================================================
elif "DATA" in sidebar_cfg["nav"] or "BENCHMARK" in sidebar_cfg["nav"]:
    st.markdown("""
    <div style="font-family: 'IBM Plex Mono', monospace; font-size: 11px; color: var(--dp-color-accent); letter-spacing: 1.5px; text-transform: uppercase; margin-bottom: 2px; font-weight: 700;">
        // MODULE 06
    </div>
    <div style="font-family: 'IBM Plex Sans', sans-serif; font-size: 18px; font-weight: 700; color: var(--dp-color-ink); text-transform: uppercase; margin-bottom: 12px;">
        SATELLITE DATA & NASA BENCHMARK LIBRARY
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="font-family: 'IBM Plex Mono', monospace; font-size: 12px; color: var(--dp-color-accent); font-weight: 700; text-transform: uppercase; margin-bottom: 4px;">
        TIER 1 // VERIFIED NASA & USGS EARTH OBSERVATION CASES
    </div>
    <div style="font-size: 12px; color: var(--dp-color-ink-muted); margin-bottom: 14px;">
        Authentic public-domain Earth observation satellite missions from NASA Earth Observatory, USGS EROS, and Copernicus ESA.
    </div>
    """, unsafe_allow_html=True)

    real_datasets = [
        {
            "id": "real_kerala_flood_2018",
            "title": "CASE A // Kerala Monsoon Floods & Basin Inundation",
            "badge": "NASA EO 92669 // LANDSAT-8 & SENTINEL-2",
            "badge_color": "var(--dp-color-accent)",
            "dates": "2018-02-06 to 2018-08-22",
            "res": "30 m / px",
            "desc": "Authentic multi-sensor satellite surveillance capturing catastrophic monsoon inundation across the Kuttanad agricultural basin and Vembanad Lake corridor in August 2018 compared to dry-season baseline in February 2018.",
            "limitations": "Combines Landsat 8 and Sentinel-2 editorial false-color composites. Raw pixel differencing illustrates flood boundary but is uncalibrated for absolute hydrological volume."
        },
        {
            "id": "real_amazon_deforestation",
            "title": "CASE B // Rondônia Amazon Rainforest Canopy Disturbance",
            "badge": "NASA TERRA MODIS // 3-DATE SEQUENCE",
            "badge_color": "var(--dp-color-accent)",
            "dates": "2000-07-30 to 2012-07-18",
            "res": "250 m / px",
            "desc": "Authentic NASA satellite observation documenting fishbone logging corridors across the Brazilian Amazon. Includes 3-date chronological sequence (2000, 2006, 2012).",
            "limitations": "250m coarse spatial resolution is optimized for macro-deforestation corridors; sub-canopy selective logging below pixel scale is unmeasured."
        },
        {
            "id": "real_lakemead_shoreline",
            "title": "CASE C // Lake Mead Shoreline Recession & Drought",
            "badge": "NASA / USGS LANDSAT // 20-YEAR CHRONOLOGY",
            "badge_color": "var(--dp-color-registration)",
            "dates": "2000-07-06 to 2022-07-03",
            "res": "30 m / px",
            "desc": "Authentic Landsat 7 & Landsat 8/9 imagery documenting historic reservoir shrinkage and the exposed mineralized shoreline across the Hoover Dam reservoir.",
            "limitations": "Planar pixel calculation on steep canyon topography underestimates 3D slope surface area without an ingested digital elevation model (DEM)."
        },
        {
            "id": "real_yellowstone_wildfire",
            "title": "CASE D // Yellowstone Historic Wildfire & 20-Year Natural Recovery",
            "badge": "NASA / USGS LANDSAT 5 & 7 // BURN SCAR CHRONOLOGY",
            "badge_color": "var(--dp-color-interference)",
            "dates": "1987-08-05 to 2010-08-04",
            "res": "30 m / px",
            "desc": "Authentic multi-decadal satellite surveillance of the historic 1988 Yellowstone fires which burned nearly 800,000 acres, followed by post-fire burn scar mapping and natural forest regeneration.",
            "limitations": "3-band visible spectrum proxy. Integration with NASA FIRMS thermal anomaly points or Normalized Burn Ratio (NBR) recommended for burn severity zonation."
        }
    ]

    for d in real_datasets:
        with st.container():
            c_info, c_action = st.columns([4, 1.2])
            with c_info:
                st.markdown(f"""
                <div style="background: var(--dp-color-plate); border: 1px solid var(--dp-color-line); border-left: 3px solid {d['badge_color']}; padding: 14px 18px; margin-bottom: 12px;">
                    <div style="display: flex; align-items: center; gap: 10px; flex-wrap: wrap;">
                        <span style="font-family: 'IBM Plex Sans', sans-serif; font-size: 15px; font-weight: 700; color: var(--dp-color-ink);">{d['title']}</span>
                        <span style="font-family: 'IBM Plex Mono', monospace; background: rgba(56, 255, 142, 0.08); color: {d['badge_color']}; border: 1px solid {d['badge_color']}; font-size: 9.5px; padding: 2px 7px; font-weight: 700;">
                            {d['badge']}
                        </span>
                    </div>
                    <div style="font-family: 'IBM Plex Sans', sans-serif; font-size: 12px; color: var(--dp-color-ink-body); margin-top: 6px;">{d['desc']}</div>
                    <div style="font-family: 'IBM Plex Mono', monospace; font-size: 10px; color: var(--dp-color-ink-muted); margin-top: 6px;">
                        ACQUISITION DATES: {d['dates']} // CALIBRATED GSD: {d['res']} // VERIFIED CALIBRATION
                    </div>
                    <div style="font-family: 'IBM Plex Mono', monospace; font-size: 9.5px; color: #ffd600; margin-top: 4px;">
                        KNOWN LIMITATIONS: {d['limitations']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            with c_action:
                st.markdown("<div style='padding-top: 10px;'>", unsafe_allow_html=True)
                if st.button(f"LOAD CASE", key=f"btn_real_{d['id']}", use_container_width=True):
                    load_demo_dataset(d["id"])
                st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("""
    <div style="font-family: 'IBM Plex Mono', monospace; font-size: 12px; color: #ffd600; font-weight: 700; text-transform: uppercase; margin-bottom: 4px;">
        TIER 2 // SYNTHETIC CALIBRATION BENCHMARKS
    </div>
    <div style="font-size: 12px; color: var(--dp-color-ink-muted); margin-bottom: 14px;">
        Explicitly labeled synthetic datasets for algorithm verification, edge-case testing, and zero-anomaly control.
    </div>
    """, unsafe_allow_html=True)

    synthetic_datasets = [
        {
            "id": "synthetic_flood_temporal",
            "title": "Brahmaputra Basin Flood Inundation & Recession",
            "badge": "SYNTHETIC BENCHMARK",
            "badge_color": "#ffd600",
            "desc": "Models 3-stage temporal dynamics: dry floodplain -> monsoon inundation -> post-monsoon water recession."
        },
        {
            "id": "synthetic_mining",
            "title": "Open-Cast Pit Excavation & Exposed Soil",
            "badge": "SYNTHETIC BENCHMARK",
            "badge_color": "#ffd600",
            "desc": "Models open-cast terraced excavation exposing subsoil minerals."
        },
        {
            "id": "synthetic_unchanged_control",
            "title": "Baseline Stability Zero-Anomaly Control",
            "badge": "SYNTHETIC CONTROL",
            "badge_color": "#ffd600",
            "desc": "Control pair with identical scene content and sensor noise to verify false-alarm rejection."
        }
    ]

    for d in synthetic_datasets:
        with st.container():
            c_info, c_action = st.columns([4, 1.2])
            with c_info:
                st.markdown(f"""
                <div style="background: var(--dp-color-plate); border: 1px solid var(--dp-color-line); border-left: 3px solid #ffd600; padding: 14px 18px; margin-bottom: 12px;">
                    <div style="display: flex; align-items: center; gap: 10px; flex-wrap: wrap;">
                        <span style="font-family: 'IBM Plex Sans', sans-serif; font-size: 15px; font-weight: 700; color: var(--dp-color-ink);">{d['title']}</span>
                        <span style="font-family: 'IBM Plex Mono', monospace; background: rgba(255, 214, 0, 0.08); color: {d['badge_color']}; border: 1px solid {d['badge_color']}; font-size: 9.5px; padding: 2px 7px; font-weight: 700;">
                            {d['badge']}
                        </span>
                    </div>
                    <div style="font-family: 'IBM Plex Sans', sans-serif; font-size: 12px; color: var(--dp-color-ink-body); margin-top: 6px;">{d['desc']}</div>
                    <div style="font-family: 'IBM Plex Mono', monospace; font-size: 10px; color: var(--dp-color-ink-muted); margin-top: 6px;">
                        DATA PROVENANCE: SYNTHETIC CALIBRATION BENCHMARK (ALGORITHMIC VERIFICATION)
                    </div>
                </div>
                """, unsafe_allow_html=True)
            with c_action:
                st.markdown("<div style='padding-top: 10px;'>", unsafe_allow_html=True)
                if st.button(f"LOAD BENCHMARK", key=f"btn_syn_{d['id']}", use_container_width=True):
                    load_demo_dataset(d["id"])
                st.markdown("</div>", unsafe_allow_html=True)


# ==============================================================================
# TAB 7: 07 / METHODOLOGY
# ==============================================================================
elif "METHODOLOGY" in sidebar_cfg["nav"]:
    st.markdown("""
    <div style="font-family: 'IBM Plex Mono', monospace; font-size: 11px; color: var(--dp-color-accent); letter-spacing: 1.5px; text-transform: uppercase; margin-bottom: 2px; font-weight: 700;">
        // MODULE 07
    </div>
    <div style="font-family: 'IBM Plex Sans', sans-serif; font-size: 18px; font-weight: 700; color: var(--dp-color-ink); text-transform: uppercase; margin-bottom: 12px;">
        SCIENTIFIC METHODOLOGY & RELIABILITY SAFEGUARDS
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    #### 1. Core Scientific Principle: Observation vs. Inferred Cause
    In Earth observation, **a changed pixel is merely a difference in radiometric reflectance**.
    An optical satellite sensor measures digital numbers (DN) representing reflected solar radiation.
    Pixel variations can stem from:
    - Genuine surface land-cover transformation (e.g. logging, road construction, flooding).
    - Ambient solar azimuth and elevation differences between acquisition seasons.
    - Transient cloud shadows, smoke plumes, or atmospheric water vapor.
    - Phenological vegetation cycles (e.g. seasonal deciduous leaf-fall or agricultural harvest).

    **TerraSentinel AI strictly isolates algorithmic measurements from causal hypotheses.**

    ---

    #### 2. Computer Vision Change Detection Architecture
    Our image processing pipeline combines complementary algorithms:
    - **Perceptual Color Metric (CIE-Lab $\\Delta E$):** Computes Euclidean distance in perceptually uniform Lab color space, decoupling surface illumination changes ($L^*$) from chromatic shifts ($a^*, b^*$).
    - **Structural Similarity Index (SSIM):** Compares local luminance, contrast, and structural texture correlation across Gaussian sliding windows:
      $$\\text{SSIM}(x, y) = \\frac{(2\\mu_x\\mu_y + C_1)(2\\sigma_{xy} + C_2)}{(\\mu_x^2 + \\mu_y^2 + C_1)(\\sigma_x^2 + \\sigma_y^2 + C_2)}$$
    - **Morphological Spatial Filtering:** Applies morphological opening (erosion followed by dilation) to eradicate isolated single-pixel sensor noise, followed by closing to bridge interior voids in contiguous disturbance patches.
    - **Connected Component Labelling (8-Way Connectivity):** Extracts discrete topological regions, quantifying bounding boxes, pixel areas, centroids, and aspect ratios.

    ---

    #### 3. Change Truth Engine & Counterfactual Baseline Controls
    To challenge potential false positives, TerraSentinel AI conducts counterfactual spatial sampling:
    - Samples non-anomalous background patches of comparable scale.
    - Evaluates standardized effect size ($Z$-score) of candidate anomaly against background variation.
    - Actively challenges alternative explanations (misregistration, illumination drift, seasonal dieback).
    - Exposes concrete conditions that would strengthen or overturn the verdict.

    ---

    #### 4. Temporal Persistence vs. Transient Reversal
    Persistence **CANNOT** be inferred from two dates alone.
    A third timestamp ($T_2$) is required to evaluate whether an anomaly persists or represents transient seasonal dynamics (e.g. floodwater recession).
    """)
