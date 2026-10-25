"""Visual analytics, Evidence Engine inspector, and Plotly charts for TerraSentinel AI (Cyberpunk Design System)."""
import streamlit as st
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
from modules.metrics import compute_difference_histogram

def render_region_table(regions: List[Dict[str, Any]], evidence_result: Optional[Dict[str, Any]] = None):
    """Renders a structured, sortable telemetry table of detected anomaly clusters."""
    if not regions:
        st.info("No anomaly clusters detected matching the active area threshold.")
        return

    # Map region evidence if available
    reg_ev_map = {}
    if evidence_result and "regions_evidence" in evidence_result:
        for ev in evidence_result["regions_evidence"]:
            reg_ev_map[ev["region_id"]] = ev

    table_data = []
    for r in regions:
        rid = r["id"]
        ev = reg_ev_map.get(rid, {})
        geom = ev.get("geometry", {})

        table_data.append({
            "CLUSTER ID": f"#{rid:02d}",
            "PIXEL AREA": f"{r['area_pixels']:,} px",
            "SCENE COVERAGE": f"{r['fraction_percent']:.3f}%",
            "MEAN DELTA": f"{r['mean_intensity']:.1f} DN",
            "COMPACTNESS": f"{geom.get('compactness', 0.0):.3f}" if 'compactness' in geom else "-",
            "COMPLEXITY": f"{geom.get('boundary_complexity', 0.0):.2f}" if 'boundary_complexity' in geom else "-",
            "PRIMARY HYPOTHESIS": ev.get("leading_hypothesis", "Evaluating..."),
            "PLAUSIBILITY": ev.get("plausibility", "-"),
            "CENTROID": f"({r['centroid'][0]}, {r['centroid'][1]})"
        })

    df = pd.DataFrame(table_data)
    st.dataframe(df, use_container_width=True, hide_index=True)


def render_change_truth_inspector(
    truth_result: Optional[Dict[str, Any]],
    img_before: np.ndarray,
    img_after: np.ndarray,
    overlay_img: np.ndarray
):
    """Stage G: Interactive demonstration inspector for the Change Truth Engine."""
    if not truth_result or not truth_result.get("regions_truth"):
        st.caption("No Change Truth evidence available for inspection. Run change detection first.")
        return

    regions_truth = truth_result["regions_truth"]
    options = [
        f"CLUSTER #{r['region_id']:02d} ({r['observed_measurements']['area_pixels']:,} px) // {r['verdict']}"
        for r in regions_truth
    ]

    selected_label = st.selectbox("SELECT ANOMALY CANDIDATE TO CHALLENGE EVIDENCE:", options, index=0)
    selected_idx = options.index(selected_label)
    target_truth = regions_truth[selected_idx]

    x, y, w, h = target_truth["bbox"]
    pad = 15
    y1 = max(0, y - pad)
    y2 = min(img_after.shape[0], y + h + pad)
    x1 = max(0, x - pad)
    x2 = min(img_after.shape[1], x + w + pad)

    crop_before = img_before[y1:y2, x1:x2]
    crop_after = img_after[y1:y2, x1:x2]
    crop_overlay = overlay_img[y1:y2, x1:x2]

    # Control patch crop if available
    ctrl_bbox = target_truth.get("control_bbox")
    crop_control = None
    if ctrl_bbox:
        cx, cy, cw, ch = ctrl_bbox
        cy1, cy2 = max(0, cy), min(img_after.shape[0], cy + ch)
        cx1, cx2 = max(0, cx), min(img_after.shape[1], cx + cw)
        crop_control = img_after[cy1:cy2, cx1:cx2]

    v = target_truth["verdict"]
    v_color = "var(--dp-color-accent)" if ("LOCALIZED" in v or "PERSISTENCE" in v) else ("#ffd600" if "CONFOUND" in v or "UNCERTAIN" in v else "var(--dp-color-interference)")
    ctrl_data = target_truth.get("counterfactual_controls", {})
    z_str = f"+{ctrl_data['z_score']}σ" if ctrl_data.get("z_score") is not None else "N/A"
    pct_str = f">{ctrl_data['empirical_percentile']}%" if ctrl_data.get("empirical_percentile") is not None else "N/A"

    # Verdict Banner
    st.markdown(f"""
    <div style="background: var(--dp-color-plate); border: 1px solid var(--dp-color-line); border-left: 4px solid {v_color}; padding: 14px 18px; margin: 12px 0 16px 0;">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;">
            <div>
                <span style="font-family: 'IBM Plex Mono', monospace; font-size: 10px; color: var(--dp-color-ink-muted); text-transform: uppercase; letter-spacing: 1.5px; font-weight: 700;">
                    // CHANGE TRUTH VERDICT // CLUSTER #{target_truth['region_id']:02d}
                </span>
                <div style="font-family: 'IBM Plex Sans', sans-serif; font-size: 18px; font-weight: 700; color: {v_color}; margin-top: 2px;">
                    {v}
                </div>
            </div>
            <div style="font-family: 'IBM Plex Mono', monospace; font-size: 10.5px; background: rgba(56, 255, 142, 0.08); border: 1px solid var(--dp-color-line); padding: 4px 10px;">
                EFFECT SIZE: <b style="color: {v_color};">{z_str}</b> | CONTROL RANK: <b>{pct_str}</b>
            </div>
        </div>
        <div style="font-family: 'IBM Plex Sans', sans-serif; font-size: 12.5px; color: var(--dp-color-ink-body); margin-top: 6px;">
            {target_truth['verdict_rationale']}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 4-Stage Visual Crop Viewport
    col_crops = st.columns(4 if crop_control is not None else 3)
    with col_crops[0]:
        st.markdown("<div class='view-label'>01 // BASELINE (BEFORE)</div>", unsafe_allow_html=True)
        st.image(crop_before, use_container_width=True)
    with col_crops[1]:
        st.markdown("<div class='view-label'>02 // TARGET (AFTER)</div>", unsafe_allow_html=True)
        st.image(crop_after, use_container_width=True)
    with col_crops[2]:
        st.markdown("<div class='view-label'>03 // ANOMALY MASK</div>", unsafe_allow_html=True)
        st.image(crop_overlay, use_container_width=True)
    if crop_control is not None and len(col_crops) > 3:
        with col_crops[3]:
            st.markdown("<div class='view-label'>04 // CONTROL WINDOW</div>", unsafe_allow_html=True)
            st.image(crop_control, use_container_width=True)

    # 3-Column Evidence Analysis
    m_col1, m_col2, m_col3 = st.columns([1, 1, 1.4])

    obs_m = target_truth["observed_measurements"]
    with m_col1:
        st.markdown("""
        <div style="font-family: 'IBM Plex Mono', monospace; font-size: 10px; font-weight: 700; color: var(--dp-color-accent); text-transform: uppercase; margin-bottom: 6px;">
            CANDIDATE MEASUREMENTS
        </div>
        """, unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-family: 'IBM Plex Mono', monospace; font-size: 11px; color: var(--dp-color-ink-body); line-height: 1.7; background: var(--dp-color-plate); border: 1px solid var(--dp-color-line); padding: 10px;">
            <div>AREA: <b>{obs_m['area_pixels']:,} px</b></div>
            <div>MEAN DELTA: <b>{obs_m['mean_delta_dn']:.1f} DN</b></div>
            <div>MAX DELTA: <b>{obs_m['max_delta_dn']:.1f} DN</b></div>
            <div>LIGHTNESS SHIFT: <b>{obs_m['delta_lightness']:+.1f} DN</b></div>
            <div>VARI SHIFT: <b>{obs_m['delta_vari']:+.3f}</b></div>
            <div>COMPACTNESS: <b>{obs_m['compactness']:.3f}</b></div>
            <div>COMPLEXITY: <b>{obs_m['boundary_complexity']:.2f}</b></div>
        </div>
        """, unsafe_allow_html=True)

    with m_col2:
        st.markdown("""
        <div style="font-family: 'IBM Plex Mono', monospace; font-size: 10px; font-weight: 700; color: var(--dp-color-registration); text-transform: uppercase; margin-bottom: 6px;">
            COUNTERFACTUAL CONTROLS
        </div>
        """, unsafe_allow_html=True)
        c_avail = ctrl_data.get("available", False)
        c_mean_val = f"{ctrl_data.get('control_mean_delta', 0.0):.1f} DN" if c_avail else "N/A"
        c_std_val = f"{ctrl_data.get('control_std_delta', 0.0):.1f} DN" if c_avail else "N/A"
        st.markdown(f"""
        <div style="font-family: 'IBM Plex Mono', monospace; font-size: 11px; color: var(--dp-color-ink-body); line-height: 1.7; background: var(--dp-color-plate); border: 1px solid var(--dp-color-line); padding: 10px;">
            <div>CONTROLS SAMPLED: <b>{ctrl_data.get('num_controls', 0)}</b></div>
            <div>CONTROL MEAN DELTA: <b>{c_mean_val}</b></div>
            <div>CONTROL STD (σ): <b>{c_std_val}</b></div>
            <div>Z-SCORE (EFFECT SIZE): <b style="color: {v_color};">{z_str}</b></div>
            <div>EMPIRICAL PERCENTILE: <b>{pct_str}</b></div>
            <div style="font-size: 9.5px; color: var(--dp-color-ink-muted); margin-top: 4px;">METHOD: Non-anomalous background patches</div>
        </div>
        """, unsafe_allow_html=True)

    with m_col3:
        st.markdown("""
        <div style="font-family: 'IBM Plex Mono', monospace; font-size: 10px; font-weight: 700; color: var(--dp-color-ink); text-transform: uppercase; margin-bottom: 6px;">
            COMPETING HYPOTHESES EVALUATION
        </div>
        """, unsafe_allow_html=True)
        for h_item in target_truth["competing_explanations"][:4]:
            supp = h_item["supported"]
            h_badge_color = "var(--dp-color-accent)" if supp else "var(--dp-color-ink-muted)"
            h_status = "SUPPORTED" if supp else "DISFAVORED"
            st.markdown(f"""
            <div style="background: var(--dp-color-plate); border: 1px solid var(--dp-color-line); padding: 6px 10px; margin-bottom: 6px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-family: 'IBM Plex Sans', sans-serif; font-size: 11px; font-weight: 700; color: var(--dp-color-ink);">{h_item['hypothesis']}</span>
                    <span style="font-family: 'IBM Plex Mono', monospace; font-size: 9px; font-weight: 700; color: {h_badge_color}; border: 1px solid {h_badge_color}; padding: 0px 4px;">
                        {h_status}
                    </span>
                </div>
                <div style="font-family: 'IBM Plex Mono', monospace; font-size: 9.5px; color: var(--dp-color-ink-muted); margin-top: 2px;">
                    {h_item['evidence']}
                </div>
            </div>
            """, unsafe_allow_html=True)

    # WHAT WOULD CHANGE THIS VERDICT?
    st.markdown("""
    <div style="margin: 16px 0 6px 0; border-top: 1px solid var(--dp-color-line); padding-top: 12px;">
        <span style="font-family: 'IBM Plex Mono', monospace; font-size: 10px; color: var(--dp-color-accent); letter-spacing: 1.5px; text-transform: uppercase; font-weight: 700;">
            // ADVERSARIAL CHALLENGE
        </span>
        <div style="font-family: 'IBM Plex Sans', sans-serif; font-size: 14px; font-weight: 700; color: var(--dp-color-ink); text-transform: uppercase;">
            WHAT WOULD CHANGE THIS VERDICT?
        </div>
    </div>
    """, unsafe_allow_html=True)

    w_col1, w_col2 = st.columns(2)
    wwc = target_truth.get("what_would_change_this_verdict", {})
    with w_col1:
        st.markdown(f"""
        <div style="background: var(--dp-color-plate); border: 1px solid var(--dp-color-line); border-top: 2px solid var(--dp-color-accent); padding: 12px;">
            <b style="font-family: 'IBM Plex Mono', monospace; font-size: 10px; color: var(--dp-color-accent); text-transform: uppercase;">
                EVIDENCE THAT WOULD STRENGTHEN / PROVE VERDICT:
            </b>
            <ul style="margin: 6px 0 0 0; padding-left: 16px; font-size: 11px; font-family: 'IBM Plex Sans', sans-serif; color: var(--dp-color-ink-body);">
                {"".join(f"<li>{s}</li>" for s in wwc.get('strengthening_evidence', []))}
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with w_col2:
        st.markdown(f"""
        <div style="background: var(--dp-color-plate); border: 1px solid var(--dp-color-line); border-top: 2px solid #ffd600; padding: 12px;">
            <b style="font-family: 'IBM Plex Mono', monospace; font-size: 10px; color: #ffd600; text-transform: uppercase;">
                EVIDENCE THAT WOULD REFUTE / WEAKEN VERDICT:
            </b>
            <ul style="margin: 6px 0 0 0; padding-left: 16px; font-size: 11px; font-family: 'IBM Plex Sans', sans-serif; color: var(--dp-color-ink-body);">
                {"".join(f"<li>{w}</li>" for w in wwc.get('weakening_evidence', []))}
            </ul>
        </div>
        """, unsafe_allow_html=True)


def render_region_evidence_inspector(
    evidence_result: Optional[Dict[str, Any]],
    img_before: np.ndarray,
    img_after: np.ndarray,
    overlay_img: np.ndarray
):
    """Interactive deep-dive inspector for individual detected anomaly clusters."""
    if not evidence_result or not evidence_result.get("regions_evidence"):
        st.caption("No region evidence available for inspection.")
        return

    regions_ev = evidence_result["regions_evidence"]
    options = [f"CLUSTER #{r['region_id']:02d} ({r['area_pixels']:,} px) // {r['leading_hypothesis']}" for r in regions_ev]

    selected_label = st.selectbox("SELECT ANOMALY CLUSTER FOR DEEP EVIDENCE INSPECTION:", options, index=0)
    selected_idx = options.index(selected_label)
    target_ev = regions_ev[selected_idx]

    x, y, w, h = target_ev["bbox"]

    # Crop thumbnails with padding
    pad = 15
    y1 = max(0, y - pad)
    y2 = min(img_after.shape[0], y + h + pad)
    x1 = max(0, x - pad)
    x2 = min(img_after.shape[1], x + w + pad)

    crop_before = img_before[y1:y2, x1:x2]
    crop_after = img_after[y1:y2, x1:x2]
    crop_overlay = overlay_img[y1:y2, x1:x2]

    st.markdown(f"""
    <div style="margin: 16px 0 10px 0; border-top: 1px solid var(--dp-color-line); padding-top: 14px;">
        <span style="font-family: 'IBM Plex Mono', monospace; font-size: 10px; color: var(--dp-color-accent); letter-spacing: 1.5px; text-transform: uppercase; font-weight: 700;">
            // TOPOLOGICAL DECOMPOSITION
        </span>
        <div style="font-family: 'IBM Plex Sans', sans-serif; font-size: 16px; font-weight: 700; color: var(--dp-color-ink); text-transform: uppercase;">
            REGION EVIDENCE DOSSIER // CLUSTER #{target_ev['region_id']:02d}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 3 Viewport crops
    v1, v2, v3 = st.columns(3)
    with v1:
        st.markdown("<div class='view-label'>BASELINE OPTICAL (BEFORE)</div>", unsafe_allow_html=True)
        st.image(crop_before, use_container_width=True)
    with v2:
        st.markdown("<div class='view-label'>TARGET OPTICAL (AFTER)</div>", unsafe_allow_html=True)
        st.image(crop_after, use_container_width=True)
    with v3:
        st.markdown("<div class='view-label'>ANOMALY MASK OVERLAY</div>", unsafe_allow_html=True)
        st.image(crop_overlay, use_container_width=True)

    # Telemetry and Evidence Grid
    e_col1, e_col2, e_col3 = st.columns([1, 1, 2])

    with e_col1:
        st.markdown("""
        <div style="font-family: 'IBM Plex Mono', monospace; font-size: 11px; font-weight: 700; color: var(--dp-color-registration); text-transform: uppercase; margin-bottom: 8px;">
            GEOMETRIC TOPOLOGY
        </div>
        """, unsafe_allow_html=True)
        geom = target_ev["geometry"]
        st.markdown(f"""
        <div style="font-family: 'IBM Plex Mono', monospace; font-size: 11px; color: var(--dp-color-ink-body); line-height: 1.7; background: var(--dp-color-plate); border: 1px solid var(--dp-color-line); padding: 10px;">
            <div>PIXEL AREA: <b>{target_ev['area_pixels']:,} px</b></div>
            <div>SCENE FRACTION: <b>{target_ev['fraction_percent']:.3f}%</b></div>
            <div>COMPACTNESS: <b>{geom['compactness']:.3f}</b></div>
            <div>COMPLEXITY: <b>{geom['boundary_complexity']:.2f}</b></div>
            <div>ASPECT RATIO: <b>{geom['aspect_ratio']}</b></div>
            <div>EXTENT: <b>{geom['extent']:.2f}</b></div>
        </div>
        """, unsafe_allow_html=True)

    with e_col2:
        st.markdown("""
        <div style="font-family: 'IBM Plex Mono', monospace; font-size: 11px; font-weight: 700; color: var(--dp-color-accent); text-transform: uppercase; margin-bottom: 8px;">
            SPECTRAL DELTAS
        </div>
        """, unsafe_allow_html=True)
        spec = target_ev["spectral_evidence"]
        rgb = spec["delta_rgb"]
        st.markdown(f"""
        <div style="font-family: 'IBM Plex Mono', monospace; font-size: 11px; color: var(--dp-color-ink-body); line-height: 1.7; background: var(--dp-color-plate); border: 1px solid var(--dp-color-line); padding: 10px;">
            <div>MEAN DELTA: <b>{spec['mean_difference_dn']} DN</b></div>
            <div>MAX DELTA: <b>{spec['max_difference_dn']} DN</b></div>
            <div>DELTA LIGHTNESS: <b>{spec['delta_lightness_dn']:+.1f} DN</b></div>
            <div>VARI SHIFT: <b>{spec['delta_vari']:+.3f}</b></div>
            <div>DELTA RGB: <b>{rgb[0]:+.0f} / {rgb[1]:+.0f} / {rgb[2]:+.0f}</b></div>
            <div>SOIL DETECT: <b>{'YES' if spec['soil_tendency'] else 'NO'}</b></div>
            <div>WATER ABSORP: <b>{'YES' if spec['water_absorption'] else 'NO'}</b></div>
        </div>
        """, unsafe_allow_html=True)

    with e_col3:
        st.markdown("""
        <div style="font-family: 'IBM Plex Mono', monospace; font-size: 11px; font-weight: 700; color: var(--dp-color-ink); text-transform: uppercase; margin-bottom: 8px;">
            COMPETING HYPOTHESES & PLAUSIBILITY RANK
        </div>
        """, unsafe_allow_html=True)
        for h_item in target_ev["competing_hypotheses"]:
            plaus = h_item['plausibility']
            p_color = "var(--dp-color-accent)" if plaus == "High" else "#ffd600"
            sig_list = "".join(f"<li>{s}</li>" for s in h_item['supporting_signals'])
            alt_str = "; ".join(h_item['competing_explanations'])
            st.markdown(f"""
            <div style="background: var(--dp-color-plate); border: 1px solid var(--dp-color-line); border-left: 3px solid {p_color}; padding: 10px 12px; margin-bottom: 8px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-family: 'IBM Plex Sans', sans-serif; font-weight: 700; color: var(--dp-color-ink); font-size: 12px;">{h_item['category']}</span>
                    <span style="font-family: 'IBM Plex Mono', monospace; font-size: 9px; font-weight: 700; color: {p_color}; border: 1px solid {p_color}; padding: 1px 6px;">
                        {plaus.upper()} PLAUSIBILITY
                    </span>
                </div>
                <div style="font-family: 'IBM Plex Sans', sans-serif; font-size: 11px; color: var(--dp-color-ink-body); margin-top: 5px;">
                    <strong style="color: var(--dp-color-ink-muted);">SUPPORTING SIGNALS:</strong>
                    <ul style="margin: 2px 0 4px 0; padding-left: 16px;">
                        {sig_list}
                    </ul>
                </div>
                <div style="font-family: 'IBM Plex Mono', monospace; font-size: 10px; color: var(--dp-color-ink-muted);">
                    ALTERNATIVES: {alt_str}
                </div>
            </div>
            """, unsafe_allow_html=True)


def render_spectral_and_spatial_charts(
    raw_diff_map,
    regions: List[Dict[str, Any]],
    heuristic: Dict[str, Any]
):
    """Renders interactive Plotly visual charts formatted with Cyberpunk palette."""
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("""
        <div style="font-family: 'IBM Plex Mono', monospace; font-size: 11px; color: var(--dp-color-accent); font-weight: 700; text-transform: uppercase; margin-bottom: 4px;">
            // SPECTRAL TELEMETRY
        </div>
        <div style="font-family: 'IBM Plex Sans', sans-serif; font-size: 14px; font-weight: 700; color: var(--dp-color-ink); text-transform: uppercase; margin-bottom: 8px;">
            DIFFERENCE INTENSITY HISTOGRAM
        </div>
        """, unsafe_allow_html=True)
        hist_data = compute_difference_histogram(raw_diff_map)

        fig_hist = go.Figure()
        fig_hist.add_trace(go.Bar(
            x=hist_data["bin_centers"],
            y=hist_data["counts"],
            marker_color="#65f8ff",
            opacity=0.9,
            name="Pixel Difference"
        ))
        fig_hist.update_layout(
            paper_bgcolor="#090e0b",
            plot_bgcolor="#020303",
            font=dict(family="IBM Plex Mono, monospace", color="#c2cbc5", size=10),
            margin=dict(l=20, r=20, t=25, b=20),
            height=280,
            xaxis=dict(title="Radiometric Difference (DN)", gridcolor="#304c3b", zerolinecolor="#304c3b"),
            yaxis=dict(title="Log Frequency", gridcolor="#304c3b", type="log", zerolinecolor="#304c3b"),
            showlegend=False
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with c2:
        st.markdown("""
        <div style="font-family: 'IBM Plex Mono', monospace; font-size: 11px; color: var(--dp-color-accent); font-weight: 700; text-transform: uppercase; margin-bottom: 4px;">
            // SPATIAL DISTRIBUTION
        </div>
        <div style="font-family: 'IBM Plex Sans', sans-serif; font-size: 14px; font-weight: 700; color: var(--dp-color-ink); text-transform: uppercase; margin-bottom: 8px;">
            ANOMALY CLUSTER AREA HIERARCHY
        </div>
        """, unsafe_allow_html=True)
        if regions:
            top_regions = regions[:10]
            ids = [f"#{r['id']:02d}" for r in top_regions]
            areas = [r["area_pixels"] for r in top_regions]

            fig_bar = go.Figure(go.Bar(
                x=ids,
                y=areas,
                marker=dict(
                    color=areas,
                    colorscale=[[0.0, "#304c3b"], [0.5, "#65f8ff"], [1.0, "#38ff8e"]],
                    showscale=False
                )
            ))
            fig_bar.update_layout(
                paper_bgcolor="#090e0b",
                plot_bgcolor="#020303",
                font=dict(family="IBM Plex Mono, monospace", color="#c2cbc5", size=10),
                margin=dict(l=20, r=20, t=25, b=20),
                height=280,
                xaxis=dict(title="Cluster Identifier", gridcolor="#304c3b", zerolinecolor="#304c3b"),
                yaxis=dict(title="Pixel Area (px)", gridcolor="#304c3b", zerolinecolor="#304c3b"),
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.caption("No clusters available for hierarchy charting.")
