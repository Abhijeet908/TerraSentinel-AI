"""Transparent heuristic land-cover change interpretation engine.

Distinguishes rigorously between:
1. Objective visual observation (color shifts, edge density, spatial clustering)
2. Inferred possible classifications
3. Scientific confidence & limitations
4. Alternative environmental explanations
5. Recommended follow-up satellite/field evidence
"""
from typing import Dict, Any, List, Optional
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

def compute_vari(rgb_arr: np.ndarray) -> np.ndarray:
    """Computes Visible Atmospherically Resistant Index (VARI):

    VARI = (Green - Red) / (Green + Red - Blue + 1e-6)
    Useful for relative vegetation vigor tracking using standard RGB sensors.
    """
    r = rgb_arr[:, :, 0].astype(np.float32)
    g = rgb_arr[:, :, 1].astype(np.float32)
    b = rgb_arr[:, :, 2].astype(np.float32)
    denom = g + r - b
    denom[np.abs(denom) < 1.0] = 1.0
    vari = (g - r) / denom
    return np.clip(vari, -1.0, 1.0)


def classify_changes(
    img_before: np.ndarray,
    img_after: np.ndarray,
    binary_mask: np.ndarray,
    regions: List[Dict[str, Any]],
    metrics: Dict[str, Any]
) -> Dict[str, Any]:
    """Analyzes spectral and spatial transitions within changed masks to provide

    a scientifically grounded heuristic interpretation.
    """
    total_changed = metrics.get("total_changed_pixels", 0)
    change_pct = metrics.get("change_percentage", 0.0)

    # 1. No significant change scenario
    if total_changed == 0 or change_pct < 0.05 or len(regions) == 0:
        return {
            "primary_category": "No significant change detected",
            "possible_categories": ["No significant change detected"],
            "observation": (
                f"Computed difference across {metrics.get('total_image_pixels', 0):,} pixels identified only "
                f"{total_changed:,} changed pixels ({change_pct:.3f}% coverage), which falls below the noise "
                f"threshold ({metrics.get('min_region_size_used', 150)} pixels minimum cluster)."
            ),
            "confidence": "High",
            "confidence_score": 0.95,
            "confidence_basis": "Pixel difference distributions remain within baseline sensor noise and ambient lighting bounds.",
            "alternative_explanations": [
                "Identical or near-identical image acquisition state.",
                "Subtle atmospheric haze or sensor calibration offsets below detection threshold."
            ],
            "heuristic_indicators": {
                "vegetation_shift_score": 0.0,
                "soil_exposure_score": 0.0,
                "water_inundation_score": 0.0,
                "infrastructure_edge_score": 0.0
            },
            "limitations": [
                "Micro-scale changes smaller than pixel resolution cannot be resolved.",
                "Multispectral or radar (SAR) changes without visible manifestation are undetected."
            ],
            "recommended_follow_up": [
                "Acquire higher resolution optical imagery (sub-meter) if small target tracking is desired.",
                "Inspect Sentinel-1 synthetic aperture radar (SAR) for cloud-penetrating structural coherence."
            ]
        }

    # Extract pixel populations within the changed mask
    mask_bool = (binary_mask > 0)
    before_changed = img_before[mask_bool].astype(np.float32)
    after_changed = img_after[mask_bool].astype(np.float32)

    # Spectral analysis
    mean_r_b, mean_g_b, mean_b_b = np.mean(before_changed, axis=0)
    mean_r_a, mean_g_a, mean_b_a = np.mean(after_changed, axis=0)

    delta_r = mean_r_a - mean_r_b
    delta_g = mean_g_a - mean_g_b
    delta_b = mean_b_a - mean_b_b

    # Visible vegetation index (VARI)
    vari_before = compute_vari(img_before)
    vari_after = compute_vari(img_after)
    mean_vari_b = float(np.mean(vari_before[mask_bool]))
    mean_vari_a = float(np.mean(vari_after[mask_bool]))
    delta_vari = mean_vari_a - mean_vari_b

    # Water / Inundation index (Drop in lightness + increased relative blue/green)
    lightness_before = np.mean(before_changed)
    lightness_after = np.mean(after_changed)
    delta_lightness = lightness_after - lightness_before

    # Soil index (Red > Green > Blue and positive delta_r/delta_g with brown/tan tone)
    soil_tendency = (mean_r_a > mean_g_a) and (mean_g_a > mean_b_a) and (delta_r > 5)

    # Edge density / rectilinearity within changed regions (construction indicator)
    edge_density_score = 0.0
    if cv2 is not None:
        gray_after = cv2.cvtColor(img_after, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray_after, 50, 150)
        edges_in_mask = np.sum((edges > 0) & mask_bool)
        edge_density_score = float(edges_in_mask) / float(total_changed)

    # Scoring Heuristics
    veg_loss_score = 0.0
    water_expansion_score = 0.0
    excavation_score = 0.0
    construction_score = 0.0
    unclassified_score = 0.2

    # Evidence rules:
    # 1. Vegetation loss: Was green, now significantly less green, VARI dropped
    if delta_vari < -0.10:
        veg_loss_score += min(1.0, (-delta_vari) * 2.5 + (0.3 if delta_g < -15 else 0.0))
    elif delta_g < -25 and mean_g_b > mean_r_b and delta_lightness > -25:
        veg_loss_score += 0.5

    # 2. Water / Flooding: Drastic darkening (water absorbs light), reflectance drops, or blue shift
    if delta_lightness < -25 and lightness_after < 90:
        # Strong water signature: optical absorption
        water_expansion_score += min(1.0, 0.45 + (abs(delta_lightness) / 50.0))
        if mean_b_a >= mean_r_a:
            water_expansion_score += 0.25
        # If ground was flooded, reduce pure terrestrial deforestation score
        veg_loss_score *= 0.5

    # 3. Excavation / Soil: Significant increase in soil color, brightness increase in reddish/tan hues
    if soil_tendency:
        excavation_score += min(1.0, 0.6 + (delta_r / 50.0))
    elif (delta_r > 15 and delta_b < 10 and delta_vari < -0.05):
        excavation_score += min(1.0, 0.4 + (delta_r / 60.0))

    # 4. Construction / Infrastructure: High rectilinear edges + moderate lightness or angular boundaries
    rectilinear_shapes = sum(1 for r in regions if 0.6 <= r.get("aspect_ratio", 1.0) <= 1.6 and r["area_pixels"] > 300)
    rectilinear_ratio = rectilinear_shapes / max(1, len(regions))

    if edge_density_score > 0.08 or (rectilinear_ratio >= 0.3 and abs(delta_lightness) > 5):
        construction_score += min(1.0, 0.35 + (edge_density_score * 3.0) + (rectilinear_ratio * 0.4))

    # Candidate ranking
    # When high canopy greenness (VARI >= 0.45) drops, primary is vegetation/canopy disturbance
    if mean_vari_b >= 0.45 and delta_vari < -0.25:
        category_scores = [
            ("Vegetation or land-cover disturbance", veg_loss_score + 0.1),
            ("Possible excavation or exposed soil", excavation_score),
            ("Possible water expansion or flooding", water_expansion_score),
            ("Possible construction or infrastructure alteration", construction_score),
        ]
    elif excavation_score >= 0.7 and soil_tendency:
        category_scores = [
            ("Possible excavation or exposed soil", excavation_score + 0.05),
            ("Vegetation or land-cover disturbance", veg_loss_score),
            ("Possible water expansion or flooding", water_expansion_score),
            ("Possible construction or infrastructure alteration", construction_score),
        ]
    else:
        category_scores = [
            ("Vegetation or land-cover disturbance", veg_loss_score),
            ("Possible excavation or exposed soil", excavation_score),
            ("Possible water expansion or flooding", water_expansion_score),
            ("Possible construction or infrastructure alteration", construction_score),
        ]

    category_scores.sort(key=lambda x: x[1], reverse=True)
    top_cat, top_score = category_scores[0]

    if top_score < 0.35:
        primary_category = "Unclassified visual change"
        confidence_level = "Low"
        conf_numeric = 0.40
    elif top_score < 0.65:
        primary_category = top_cat
        confidence_level = "Moderate"
        conf_numeric = round(top_score, 2)
    else:
        primary_category = top_cat
        confidence_level = "High"
        conf_numeric = min(0.92, round(top_score, 2))

    possible_categories = [cat for cat, s in category_scores if s > 0.25]
    if not possible_categories:
        possible_categories = ["Unclassified visual change"]

    # Construct evidence-based descriptions
    observation = (
        f"Detected {len(regions)} changed cluster(s) spanning {total_changed:,} pixels "
        f"({change_pct:.2f}% of monitored frame). Largest contiguous anomaly spans "
        f"{metrics.get('largest_region_pixels', 0):,} pixels ({metrics.get('largest_region_percentage', 0):.2f}%). "
        f"Spectral delta: ΔRed={delta_r:+.1f}, ΔGreen={delta_g:+.1f}, ΔBlue={delta_b:+.1f}; "
        f"VARI vegetation metric shift: {delta_vari:+.3f}; Mean lightness transition: {delta_lightness:+.1f} DN."
    )

    alt_explanations = []
    if primary_category == "Vegetation or land-cover disturbance":
        alt_explanations = [
            "Normal seasonal senescence or crop harvest cycle rather than permanent deforestation.",
            "Soil moisture variation affecting canopy reflectance.",
            "Transient cloud shadow or smoke attenuation at acquisition time."
        ]
    elif primary_category == "Possible water expansion or flooding":
        alt_explanations = [
            "Topographic terrain shadowing caused by lower solar elevation angle.",
            "Controlled agricultural irrigation flooding (e.g. paddy fields).",
            "Seasonal reservoir volume fluctuation."
        ]
    elif primary_category == "Possible excavation or exposed soil":
        alt_explanations = [
            "Seasonal field plowing / agricultural preparation.",
            "Wildfire burn scar leaving exposed soil and ash.",
            "Road grading or temporary dredging spoil."
        ]
    elif primary_category == "Possible construction or infrastructure alteration":
        alt_explanations = [
            "Temporary heavy equipment or shipping container staging.",
            "Agricultural plastic mulch or greenhouse installation.",
            "Shadow shifts between seasonal sun azimuth angles."
        ]
    else:
        alt_explanations = [
            "Sensor radiometric calibration divergence between acquisition dates.",
            "Atmospheric aerosol or thin cirrus cloud interference.",
            "Minor geometric misalignment between optical baselines."
        ]

    recommended_follow_up = [
        "Incorporate calibrated Sentinel-2 NIR band (Band 8) to calculate genuine red-edge NDVI.",
        "Query Sentinel-1 dual-polarization SAR (VV/VH) to confirm surface roughness changes irrespective of cloud cover.",
        "Check multi-temporal meteorological records around acquisition timestamps for precipitation and sun angle.",
        "Cross-reference cadastral zoning or official land-use permit registries before assigning human causality."
    ]

    limitations = [
        "Analysis based exclusively on uncalibrated 3-band visible spectrum RGB imagery.",
        "Visual RGB differences do not constitute legal or scientific proof of human intent or unauthorized deforestation.",
        "Cloud shadows, seasonal phenology, and viewing geometry can induce false positive anomalies."
    ]

    return {
        "primary_category": primary_category,
        "possible_categories": possible_categories,
        "observation": observation,
        "confidence": confidence_level,
        "confidence_score": conf_numeric,
        "confidence_basis": (
            f"Derived from spectral shifts (VARI Δ={delta_vari:+.2f}, Lightness Δ={delta_lightness:+.1f}) "
            f"and spatial geometry ({len(regions)} clusters, edge density {edge_density_score:.2f}). "
            f"Reflects statistical heuristic correlation, not ground-truth inspection."
        ),
        "alternative_explanations": alt_explanations,
        "heuristic_indicators": {
            "vegetation_loss_score": round(veg_loss_score, 2),
            "soil_exposure_score": round(excavation_score, 2),
            "water_inundation_score": round(water_expansion_score, 2),
            "infrastructure_edge_score": round(construction_score, 2),
            "rectilinear_ratio": round(rectilinear_ratio, 2),
            "mean_vari_delta": round(delta_vari, 3)
        },
        "limitations": limitations,
        "recommended_follow_up": recommended_follow_up
    }
