"""Visual comparison workspace component for TerraSentinel AI (Cyberpunk Design System)."""
import streamlit as st
from typing import Dict, Any, Optional
import numpy as np

def render_comparison_workspace(
    img_before: np.ndarray,
    img_after: np.ndarray,
    analysis_result: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None
):
    """Renders the comprehensive multi-mode image comparison workspace without emojis."""
    st.markdown("""
    <div style="font-family: 'IBM Plex Mono', monospace; font-size: 11px; color: var(--dp-color-accent); letter-spacing: 1.5px; text-transform: uppercase; margin-bottom: 4px; font-weight: 700;">
        // VIEWPORT ANALYTICS
    </div>
    <div style="font-family: 'IBM Plex Sans', sans-serif; font-size: 18px; font-weight: 700; color: var(--dp-color-ink); text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 12px;">
        EARTH OBSERVATION COMPARISON WORKSPACE
    </div>
    """, unsafe_allow_html=True)

    if analysis_result is None:
        # Initial preview before running analysis
        col_b, col_a = st.columns(2)
        with col_b:
            date_b = metadata.get("date_before", "Unspecified Date") if metadata else "Baseline"
            st.markdown(f"**BASELINE IMAGERY (BEFORE)** — *{date_b}*")
            st.image(img_before, use_container_width=True)
            st.caption(f"Shape: {img_before.shape[1]}x{img_before.shape[0]} px | RGB 8-bit")

        with col_a:
            date_a = metadata.get("date_after", "Unspecified Date") if metadata else "Target"
            st.markdown(f"**TARGET IMAGERY (AFTER)** — *{date_a}*")
            st.image(img_after, use_container_width=True)
            st.caption(f"Shape: {img_after.shape[1]}x{img_after.shape[0]} px | RGB 8-bit")

        return

    # Visual Inspection Mode Tabs (No emojis)
    modes = [
        "SIDE-BY-SIDE SPLIT",
        "MULTI-TIER OVERLAY",
        "DIFFERENCE HEATMAP",
        "ANNOTATED CLUSTERS",
        "BINARY MASK",
        "BASELINE (BEFORE)",
        "TARGET (AFTER)",
        "INTERACTIVE FLICKER"
    ]

    selected_mode = st.radio(
        "VISUALIZATION MODE",
        modes,
        horizontal=True,
        label_visibility="collapsed"
    )

    st.markdown("<div style='margin-bottom: 14px;'></div>", unsafe_allow_html=True)

    date_b = metadata.get("date_before", "Unspecified Date") if metadata else "Baseline"
    date_a = metadata.get("date_after", "Unspecified Date") if metadata else "Target"

    if selected_mode == "SIDE-BY-SIDE SPLIT":
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"""
            <div class='view-label'>BASELINE OPTICAL (BEFORE // {date_b})</div>
            """, unsafe_allow_html=True)
            st.image(img_before, use_container_width=True)
            st.caption(f"Acquisition baseline reference scene ({date_b}).")
        with c2:
            st.markdown(f"""
            <div class='view-label'>TARGET OPTICAL (AFTER // {date_a})</div>
            """, unsafe_allow_html=True)
            st.image(img_after, use_container_width=True)
            st.caption(f"Target surveillance acquisition scene ({date_a}).")

    elif selected_mode == "MULTI-TIER OVERLAY":
        st.markdown("<div class='view-label'>MULTI-TIER SYNTHESIZED ANOMALY OVERLAY</div>", unsafe_allow_html=True)
        st.image(analysis_result["overlay"], use_container_width=True)

        # Legend
        st.markdown("""
        <div style="display: flex; gap: 20px; align-items: center; background: var(--dp-color-plate); border: 1px solid var(--dp-color-line); padding: 10px 16px; margin-top: 8px; font-size: 11px; font-family: 'IBM Plex Mono', monospace;">
            <span style="font-weight: 700; color: var(--dp-color-ink-muted);">OVERLAY LEGEND:</span>
            <span style="display: flex; align-items: center; gap: 6px;">
                <span style="width: 10px; height: 10px; background: var(--dp-color-interference); display: inline-block;"></span>
                <span style="color: var(--dp-color-interference); font-weight: 700;">HIGH-MAGNITUDE DELTA</span> (Above 1.5x threshold)
            </span>
            <span style="display: flex; align-items: center; gap: 6px;">
                <span style="width: 10px; height: 10px; background: #ffd600; display: inline-block;"></span>
                <span style="color: #ffd600; font-weight: 700;">MODERATE TRANSITION</span> (Candidate anomaly)
            </span>
            <span style="display: flex; align-items: center; gap: 6px;">
                <span style="width: 10px; height: 10px; background: rgba(255,255,255,0.1); border: 1px solid var(--dp-color-line); display: inline-block;"></span>
                <span style="color: var(--dp-color-ink-muted);">STABLE SURFACE</span>
            </span>
        </div>
        """, unsafe_allow_html=True)

    elif selected_mode == "DIFFERENCE HEATMAP":
        st.markdown("<div class='view-label'>DIFFERENCE HEATMAP // TURBO SPECTRAL COLORMAP</div>", unsafe_allow_html=True)
        st.image(analysis_result["heatmap"], use_container_width=True)
        st.caption("Continuous perceptual difference gradient: Deep Blue = Stable -> Cyan/Green = Moderate -> Yellow/Red = Severe Anomaly.")

    elif selected_mode == "ANNOTATED CLUSTERS":
        st.markdown("<div class='view-label'>BOUNDING BOXES & CENTROID ANNOTATIONS</div>", unsafe_allow_html=True)
        st.image(analysis_result["annotated_boxes"], use_container_width=True)
        st.caption(f"Connected components with area >= {analysis_result['metrics']['min_region_size_used']} px. Centroids marked in gold.")

    elif selected_mode == "BINARY MASK":
        st.markdown("<div class='view-label'>FILTERED BINARY CHANGE MASK</div>", unsafe_allow_html=True)
        st.image(analysis_result["binary_mask"], use_container_width=True)
        st.caption("Post-morphological filtered binary mask (255=Changed cluster, 0=Background).")

    elif selected_mode == "BASELINE (BEFORE)":
        st.markdown(f"<div class='view-label'>BASELINE OPTICAL (BEFORE // {date_b}) FULL VIEW</div>", unsafe_allow_html=True)
        st.image(img_before, use_container_width=True)

    elif selected_mode == "TARGET (AFTER)":
        st.markdown(f"<div class='view-label'>TARGET OPTICAL (AFTER // {date_a}) FULL VIEW</div>", unsafe_allow_html=True)
        st.image(img_after, use_container_width=True)

    elif selected_mode == "INTERACTIVE FLICKER":
        st.caption("Use the selector below to instantly compare Before, After, and Overlay in the exact same viewport.")
        toggle_phase = st.select_slider(
            "Scene Viewport Selector",
            options=["BASELINE (BEFORE)", "TARGET (AFTER)", "OVERLAY (ANOMALY)"],
            value="TARGET (AFTER)"
        )
        if toggle_phase == "BASELINE (BEFORE)":
            st.image(img_before, use_container_width=True)
        elif toggle_phase == "TARGET (AFTER)":
            st.image(img_after, use_container_width=True)
        else:
            st.image(analysis_result["overlay"], use_container_width=True)
