"""TerraSentinel AI Configuration Module.

Defines global constants, UI themes, analysis thresholds, and operational parameters.
"""
from dataclasses import dataclass
from typing import List, Dict, Any
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables if present
load_dotenv()

# Base directories
BASE_DIR = Path(__file__).resolve().parent
SAMPLE_DATA_DIR = BASE_DIR / "sample_data"
ASSETS_DIR = BASE_DIR / "assets"
REPORTS_DIR = BASE_DIR / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

# Application Identity
APP_NAME = "TerraSentinel AI"
APP_SUBTITLE = "Satellite Change & Damage Intelligence"
APP_TAGLINE = "See the change. Understand the risk. Protect the Earth."
APP_VERSION = "2.5.0-hackathon"

# Visual Theme (Cyberpunk High-Contrast Satellite Intelligence Design System)
DESIGN_TOKENS = {
    "dp_color_ground": "#020303",
    "dp_color_accent": "#38ff8e",
    "dp_color_interference": "#ff65d8",
    "dp_color_registration": "#65f8ff",
    "dp_color_ink": "#f8fffa",
    "dp_color_ink_body": "#d3dbd5",
    "dp_color_ink_muted": "#c2cbc5",
    "dp_color_plate": "#090e0b",
    "dp_color_line": "#304c3b",
    "dp_color_control_line": "#659579",
    "dp_typography_display": "'Cyberpunk Plex Sans', 'IBM Plex Sans', sans-serif",
    "dp_typography_mono": "'Cyberpunk Plex Mono', 'IBM Plex Mono', monospace",
    "dp_typography_small": "14px",
    "dp_typography_label": "12px",
    "dp_typography_metadata": "11px",
    "dp_spacing_gutter": "clamp(22px, 4.5cqi, 72px)",
    "dp_spacing_column_gap": "50px",
    "dp_radius_control": "0",
    "dp_border_hairline": "1px",
    "dp_motion_interaction": "0s",
}

THEME_COLORS = {
    "ground": "#020303",
    "accent": "#38ff8e",
    "interference": "#ff65d8",
    "registration": "#65f8ff",
    "ink": "#f8fffa",
    "ink_body": "#d3dbd5",
    "ink_muted": "#c2cbc5",
    "plate": "#090e0b",
    "line": "#304c3b",
    "control_line": "#659579",
    # Legacy fallbacks for compatibility
    "bg_dark": "#020303",
    "bg_card": "#090e0b",
    "bg_sidebar": "#020303",
    "accent_cyan": "#65f8ff",
    "accent_green": "#38ff8e",
    "accent_amber": "#ffd600",
    "accent_red": "#ff65d8",
    "text_main": "#f8fffa",
    "text_muted": "#c2cbc5",
    "border": "#304c3b",
}

# Supported Image Extensions
SUPPORTED_EXTENSIONS = [".png", ".jpg", ".jpeg", ".tif", ".tiff"]
MAX_IMAGE_DIM = int(os.getenv("TERRASENTINEL_MAX_IMAGE_DIM", "1600"))

# Default Detection Settings
DEFAULT_SETTINGS = {
    "difference_threshold": 32,      # 0-255 scale in Lab delta E or RGB norm
    "min_region_size": 150,          # Minimum pixel count for a change cluster
    "blur_radius": 3,                # Pre-smoothing Gaussian kernel size
    "overlay_opacity": 0.55,         # Opacity of visual change overlay
    "enable_alignment": False,       # Automatic ORB feature homography
    "morph_kernel_size": 3,          # Morphological filter size
    "structural_weight": 0.4,        # Weight given to SSIM structural difference
}

# Anomaly Classification Categories
ANOMALY_CATEGORIES = [
    "Vegetation or land-cover disturbance",
    "Possible excavation or exposed soil",
    "Possible water expansion or flooding",
    "Possible shoreline or water-body change",
    "Possible construction or infrastructure alteration",
    "Unclassified visual change",
    "No significant change detected",
]

def get_config_val(key: str, default: str = "") -> str:
    """Retrieves configuration from environment variables or Streamlit secrets."""
    val = os.getenv(key, "")
    if val:
        return val
    try:
        import streamlit as st
        if hasattr(st, "secrets") and key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass
    return default

# Meta AI / LLM Configuration
META_AI_CONFIG = {
    "api_endpoint": get_config_val("META_AI_API_ENDPOINT", ""),
    "api_key": get_config_val("META_AI_API_KEY", ""),
    "model": get_config_val("META_AI_MODEL", "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"),
    "timeout": 15,
}
