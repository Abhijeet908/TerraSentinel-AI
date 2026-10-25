"""Metrics display panel for TerraSentinel AI (Cyberpunk Design System)."""
import streamlit as st
from typing import Dict, Any, Optional
from modules.metrics import calculate_spatial_areas

def render_metrics_cards(
    metrics: Optional[Dict[str, Any]],
    metadata: Optional[Dict[str, Any]] = None
):
    """Renders cyber-spatial telemetry metrics cards."""
    if not metrics:
        st.info("No active telemetry. Upload imagery or load a benchmark dataset to calculate change statistics.")
        return

    # Calculate calibrated area only if spatial calibration is confirmed
    res_m = metadata.get("approx_resolution_m") if metadata else None
    calib_confirmed = bool(metadata.get("spatial_calibration_confirmed", False)) if metadata else False
    spatial_info = calculate_spatial_areas(
        metrics.get("total_changed_pixels", 0),
        res_m,
        spatial_calibration_confirmed=calib_confirmed
    )

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        st.markdown(f"""
        <div class="telemetry-card">
            <div class="telemetry-label">CHANGE COVERAGE</div>
            <div class="telemetry-value highlight-accent">{metrics.get('change_percentage', 0.0):.2f}%</div>
            <div class="telemetry-sub">OF SCENE PIXELS</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="telemetry-card">
            <div class="telemetry-label">CHANGED PIXELS</div>
            <div class="telemetry-value">{metrics.get('total_changed_pixels', 0):,}</div>
            <div class="telemetry-sub">TOTAL: {metrics.get('total_image_pixels', 0):,} PX</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        area_display = spatial_info["display_str"]
        is_calib = spatial_info["has_spatial_calibration"]
        sub_text = f"CALIBRATED GSD: {res_m} M/PX" if is_calib else "UNCALIBRATED (WITHHELD)"
        st.markdown(f"""
        <div class="telemetry-card">
            <div class="telemetry-label">PHYSICAL GROUND AREA</div>
            <div class="telemetry-value {'highlight-registration' if is_calib else ''}" style="font-size: {'14px' if is_calib else '11px'}; margin-top: {'6px' if is_calib else '8px'}; color: {'var(--dp-color-registration)' if is_calib else 'var(--dp-color-ink-muted)'};">
                {area_display.upper()}
            </div>
            <div class="telemetry-sub">{sub_text}</div>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown(f"""
        <div class="telemetry-card">
            <div class="telemetry-label">ANOMALY CLUSTERS</div>
            <div class="telemetry-value highlight-interference">{metrics.get('regions_count', 0)}</div>
            <div class="telemetry-sub">MIN CLUSTER: {metrics.get('min_region_size_used', 150)} PX</div>
        </div>
        """, unsafe_allow_html=True)

    with c5:
        st.markdown(f"""
        <div class="telemetry-card">
            <div class="telemetry-label">PROCESSING LATENCY</div>
            <div class="telemetry-value">{metrics.get('processing_time_sec', 0.0)}s</div>
            <div class="telemetry-sub">DETERMINISTIC CV</div>
        </div>
        """, unsafe_allow_html=True)


def render_anomaly_banner(
    heuristic: Optional[Dict[str, Any]],
    ai_interp: Optional[Dict[str, Any]] = None,
    evidence_res: Optional[Dict[str, Any]] = None
):
    """Renders prominent top-level assessment banner strictly separating

    observed change magnitude from heuristic hypothesis plausibility.
    No emojis.
    """
    if not heuristic and not evidence_res:
        return

    cat = evidence_res.get("primary_assessment") if evidence_res else heuristic.get("primary_category", "Unclassified visual change")
    plausibility = (evidence_res.get("assessment_plausibility") if evidence_res else heuristic.get("confidence", "Moderate")).upper()

    conf_bg = "rgba(56, 255, 142, 0.1)" if plausibility == "HIGH" else ("rgba(255, 214, 0, 0.1)" if plausibility == "MODERATE" else "rgba(255, 101, 216, 0.1)")
    conf_border = "var(--dp-color-accent)" if plausibility == "HIGH" else ("#ffd600" if plausibility == "MODERATE" else "var(--dp-color-interference)")
    conf_color = conf_border

    obs_text = evidence_res.get("observation_summary") if evidence_res else heuristic.get('observation', '')

    st.markdown(f"""
    <div style="background: var(--dp-color-plate); border: 1px solid var(--dp-color-line); border-left: 4px solid var(--dp-color-accent); padding: 16px 20px; margin-bottom: 20px;">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 10px;">
            <div>
                <div style="font-family: 'IBM Plex Mono', monospace; font-size: 10px; color: var(--dp-color-accent); text-transform: uppercase; letter-spacing: 1.5px; font-weight: 700;">
                    // PRIMARY HYPOTHESIS FORMULATION
                </div>
                <div style="font-family: 'IBM Plex Sans', sans-serif; font-size: 19px; font-weight: 700; color: var(--dp-color-ink); margin-top: 4px; letter-spacing: 0.5px;">
                    {cat}
                </div>
            </div>
            <div>
                <span style="font-family: 'IBM Plex Mono', monospace; display: inline-block; padding: 3px 10px; font-size: 10px; font-weight: 700; background: {conf_bg}; color: {conf_color}; border: 1px solid {conf_border}; letter-spacing: 0.8px;">
                    {plausibility} PLAUSIBILITY
                </span>
            </div>
        </div>
        <div style="font-family: 'IBM Plex Sans', sans-serif; font-size: 13px; color: var(--dp-color-ink-body); margin-top: 8px; line-height: 1.5;">
            {obs_text}
        </div>
        <div style="font-family: 'IBM Plex Mono', monospace; font-size: 10px; color: var(--dp-color-ink-muted); margin-top: 8px; border-top: 1px solid var(--dp-color-line); padding-top: 6px;">
            EVALUATION BASIS // Multi-spectral transitions and cluster topologies. Empirical indicators, not confirmed ground-truth legal violation.
        </div>
    </div>
    """, unsafe_allow_html=True)
