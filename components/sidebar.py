"""Sidebar component for TerraSentinel AI (Cyberpunk Design System)."""
import streamlit as st
from config import APP_NAME, APP_SUBTITLE, APP_TAGLINE, APP_VERSION, DEFAULT_SETTINGS, META_AI_CONFIG, THEME_COLORS

NAVIGATION_OPTIONS = [
    "01 / CHANGE DETECTION",
    "02 / CHANGE TRUTH ENGINE",
    "03 / TEMPORAL DYNAMICS",
    "04 / QUANTITATIVE ANALYSIS",
    "05 / REPORTS",
    "06 / DATA & BENCHMARKS",
    "07 / METHODOLOGY"
]

def render_sidebar():
    """Renders the persistent TerraSentinel AI mission-control sidebar without emojis.

    Returns:
        dict: User-selected navigation and detection settings.
    """
    with st.sidebar:
        # Technical Brand Header
        st.markdown(f"""
        <div style="padding: 12px 4px 18px 4px; border-bottom: 1px solid var(--dp-color-line); margin-bottom: 16px;">
            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 4px;">
                <span style="font-family: 'IBM Plex Mono', monospace; font-size: 10px; color: var(--dp-color-accent); letter-spacing: 1.5px; text-transform: uppercase; font-weight: 700;">
                    SYSTEM NODE // 01
                </span>
                <span style="font-family: 'IBM Plex Mono', monospace; font-size: 9px; color: var(--dp-color-ink-muted); border: 1px solid var(--dp-color-line); padding: 1px 4px;">
                    v{APP_VERSION}
                </span>
            </div>
            <div style="font-family: 'IBM Plex Sans', sans-serif; font-size: 20px; font-weight: 800; letter-spacing: 1.2px; color: var(--dp-color-ink); text-transform: uppercase;">
                {APP_NAME}
            </div>
            <div style="font-family: 'IBM Plex Mono', monospace; font-size: 9.5px; font-weight: 500; color: var(--dp-color-accent); letter-spacing: 0.8px; margin-top: 2px; text-transform: uppercase;">
                {APP_SUBTITLE}
            </div>
            <div style="font-family: 'IBM Plex Mono', monospace; font-size: 9px; color: var(--dp-color-ink-muted); margin-top: 6px; letter-spacing: 0.3px;">
                {APP_TAGLINE}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Allow URL or session state navigation
        current_nav = st.session_state.get("active_nav", NAVIGATION_OPTIONS[0])
        default_index = 0
        for idx, item in enumerate(NAVIGATION_OPTIONS):
            core_keyword = item.split("/")[1].strip().split()[0]
            if item == current_nav or core_keyword in current_nav:
                default_index = idx
                break

        st.markdown("""
        <div style="font-family: 'IBM Plex Mono', monospace; font-size: 10px; color: var(--dp-color-ink-muted); letter-spacing: 1.2px; margin-bottom: 8px; text-transform: uppercase; font-weight: 600;">
            NAVIGATION // OPERATIONAL MODULES
        </div>
        """, unsafe_allow_html=True)

        selected_nav = st.radio(
            "MISSION CONTROL MODULES",
            NAVIGATION_OPTIONS,
            index=default_index,
            label_visibility="collapsed"
        )
        st.session_state["active_nav"] = selected_nav

        st.markdown("<div style='border-bottom: 1px solid var(--dp-color-line); margin: 16px 0;'></div>", unsafe_allow_html=True)

        # Operational Status Card
        has_ai = bool(META_AI_CONFIG.get("api_key") and "your_meta_ai" not in META_AI_CONFIG.get("api_key"))
        status_color = "var(--dp-color-registration)" if has_ai else "var(--dp-color-accent)"
        status_text = "META LLM ONLINE" if has_ai else "DETERMINISTIC CV"
        mode_desc = "Meta-Llama Engine active" if has_ai else "Offline Local Pipeline (100% Deterministic)"
        has_analysis = "analysis_result" in st.session_state and st.session_state["analysis_result"] is not None

        st.markdown(f"""
        <div style="background: var(--dp-color-plate); border: 1px solid var(--dp-color-line); border-left: 3px solid {status_color}; padding: 10px 12px; margin-bottom: 16px;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="font-family: 'IBM Plex Mono', monospace; font-size: 9.5px; color: var(--dp-color-ink-muted); text-transform: uppercase; letter-spacing: 0.8px;">ENGINE STATUS</span>
                <span style="font-family: 'IBM Plex Mono', monospace; font-size: 9px; color: {status_color}; padding: 1px 5px; font-weight: 700; border: 1px solid {status_color};">
                    {status_text}
                </span>
            </div>
            <div style="font-family: 'IBM Plex Sans', sans-serif; font-size: 11px; color: var(--dp-color-ink-body); margin-top: 5px; font-weight: 500;">
                {mode_desc}
            </div>
            <div style="font-family: 'IBM Plex Mono', monospace; margin-top: 6px; font-size: 9.5px; color: {'var(--dp-color-accent)' if has_analysis else 'var(--dp-color-ink-muted)'}; display: flex; align-items: center; gap: 5px;">
                <span style="display: inline-block; width: 6px; height: 6px; background: {'var(--dp-color-accent)' if has_analysis else '#556980'};"></span>
                <span>{'ANALYSIS COMPUTED & VALIDATED' if has_analysis else 'AWAITING IMAGERY INGESTION'}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Advanced Detection Settings Expander
        with st.expander("DETECTION PARAMETERS & CV TUNING", expanded=False):
            st.caption("Computer vision segmentation and spatial filtering controls.")

            thresh = st.slider(
                "Difference Threshold (DN)",
                min_value=10,
                max_value=120,
                value=int(st.session_state.get("cfg_thresh", DEFAULT_SETTINGS["difference_threshold"])),
                step=2,
                help="Higher threshold suppresses seasonal reflectance variance and atmospheric haze."
            )
            st.session_state["cfg_thresh"] = thresh

            min_size = st.slider(
                "Minimum Cluster Area (px)",
                min_value=20,
                max_value=800,
                value=int(st.session_state.get("cfg_min_size", DEFAULT_SETTINGS["min_region_size"])),
                step=10,
                help="Filters isolated pixel noise and sensor artifacts."
            )
            st.session_state["cfg_min_size"] = min_size

            blur = st.slider(
                "Pre-Filter Smoothing Radius (px)",
                min_value=1,
                max_value=9,
                value=int(st.session_state.get("cfg_blur", DEFAULT_SETTINGS["blur_radius"])),
                step=2,
                help="Pre-smoothing Gaussian kernel size (odd integers)."
            )
            st.session_state["cfg_blur"] = blur

            struct_w = st.slider(
                "SSIM Structural Weight",
                min_value=0.0,
                max_value=1.0,
                value=float(st.session_state.get("cfg_struct_w", DEFAULT_SETTINGS["structural_weight"])),
                step=0.05,
                help="Balance between pixel color difference and structural SSIM gradient."
            )
            st.session_state["cfg_struct_w"] = struct_w

            opacity = st.slider(
                "Overlay Alpha Opacity",
                min_value=0.1,
                max_value=1.0,
                value=float(st.session_state.get("cfg_opacity", DEFAULT_SETTINGS["overlay_opacity"])),
                step=0.05,
                help="Alpha blending transparency for anomaly masks."
            )
            st.session_state["cfg_opacity"] = opacity

            enable_align = st.checkbox(
                "Conservative ORB Homography",
                value=bool(st.session_state.get("cfg_align", DEFAULT_SETTINGS["enable_alignment"])),
                help="Aligns target image using feature keypoints. Rejects unreliable warps."
            )
            st.session_state["cfg_align"] = enable_align

        st.markdown(f"""
        <div style="font-family: 'IBM Plex Mono', monospace; font-size: 9px; color: var(--dp-color-ink-muted); text-align: center; margin-top: 25px; padding-top: 15px; border-top: 1px solid var(--dp-color-line);">
            TERRASENTINEL AI // BUILD {APP_VERSION}<br>
            SPACETECH EARTH INTELLIGENCE
        </div>
        """, unsafe_allow_html=True)

    return {
        "nav": selected_nav,
        "threshold": st.session_state.get("cfg_thresh", DEFAULT_SETTINGS["difference_threshold"]),
        "min_region_size": st.session_state.get("cfg_min_size", DEFAULT_SETTINGS["min_region_size"]),
        "blur_radius": st.session_state.get("cfg_blur", DEFAULT_SETTINGS["blur_radius"]),
        "structural_weight": st.session_state.get("cfg_struct_w", DEFAULT_SETTINGS["structural_weight"]),
        "overlay_opacity": st.session_state.get("cfg_opacity", DEFAULT_SETTINGS["overlay_opacity"]),
        "enable_alignment": st.session_state.get("cfg_align", DEFAULT_SETTINGS["enable_alignment"]),
    }
