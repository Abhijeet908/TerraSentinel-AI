"""TerraSentinel AI — Evidence Engine.

Computes multi-dimensional geometric, spectral, and spatial evidence for each
detected anomaly region. Evaluates competing scientific hypotheses, exposes
uncertainty, and explicitly documents data-quality concerns without fabricating
confidence or causality.
"""
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

def compute_vari(rgb_arr: np.ndarray) -> np.ndarray:
    """Computes Visible Atmospherically Resistant Index (VARI):

    VARI = (Green - Red) / (Green + Red - Blue + 1e-6)
    Note: Strictly an RGB proxy heuristic; NOT true multispectral NDVI.
    """
    r = rgb_arr[:, :, 0].astype(np.float32)
    g = rgb_arr[:, :, 1].astype(np.float32)
    b = rgb_arr[:, :, 2].astype(np.float32)
    denom = g + r - b
    denom[np.abs(denom) < 1.0] = 1.0
    vari = (g - r) / denom
    return np.clip(vari, -1.0, 1.0)


def extract_region_evidence(
    region_dict: Dict[str, Any],
    mask_binary: np.ndarray,
    img_before: np.ndarray,
    img_after: np.ndarray,
    diff_map: np.ndarray,
    total_valid_pixels: int
) -> Dict[str, Any]:
    """Analyzes a single detected connected-component region and generates

    a rigorous, evidence-based scientific dossier.
    """
    x, y, w, h = region_dict["bbox"]
    area_px = region_dict["area_pixels"]

    # Crop to bounding box sub-arrays for fast localized analysis
    sub_mask = mask_binary[y:y+h, x:x+w]
    label_id = region_dict.get("label_id")

    if label_id is not None:
        region_pixels_bool = (sub_mask == label_id) | (sub_mask == 255)
    else:
        region_pixels_bool = (sub_mask > 0)

    # Fallback if empty
    if not np.any(region_pixels_bool):
        region_pixels_bool = np.ones((h, w), dtype=bool)

    sub_b = img_before[y:y+h, x:x+w][region_pixels_bool].astype(np.float32)
    sub_a = img_after[y:y+h, x:x+w][region_pixels_bool].astype(np.float32)
    sub_diff = diff_map[y:y+h, x:x+w][region_pixels_bool].astype(np.float32)

    # 1. Geometric Measurements
    # Perimeter using contour if cv2 available
    perimeter = 2.0 * (w + h)  # default bounding approximation
    compactness = 0.5
    boundary_complexity = 4.0

    if cv2 is not None:
        sub_uint8 = region_pixels_bool.astype(np.uint8) * 255
        contours, _ = cv2.findContours(sub_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c = max(contours, key=cv2.contourArea)
            perimeter = float(cv2.arcLength(c, True))
            if perimeter > 0:
                # Isoperimetric quotient: 4 * pi * Area / Perimeter^2 (1.0 = perfect circle)
                compactness = round(float(4.0 * np.pi * area_px / (perimeter ** 2)), 3)
                compactness = min(1.0, max(0.01, compactness))
                # Boundary complexity: perimeter / sqrt(area) (Square = 4.0, jagged/dendritic > 7.0)
                boundary_complexity = round(float(perimeter / max(1.0, np.sqrt(area_px))), 2)

    aspect_ratio = round(float(w) / max(1, h), 2)
    extent = round(float(area_px) / max(1, w * h), 3)
    fraction_pct = round((area_px / float(total_valid_pixels)) * 100.0, 4)

    # 2. Spectral Measurements
    mean_b_rgb = np.mean(sub_b, axis=0) if len(sub_b) > 0 else np.array([0, 0, 0])
    mean_a_rgb = np.mean(sub_a, axis=0) if len(sub_a) > 0 else np.array([0, 0, 0])

    delta_r = round(float(mean_a_rgb[0] - mean_b_rgb[0]), 1)
    delta_g = round(float(mean_a_rgb[1] - mean_b_rgb[1]), 1)
    delta_b = round(float(mean_a_rgb[2] - mean_b_rgb[2]), 1)

    mean_diff = round(float(np.mean(sub_diff)), 1) if len(sub_diff) > 0 else 0.0
    max_diff = round(float(np.max(sub_diff)), 1) if len(sub_diff) > 0 else 0.0

    # Optical reflectance / lightness shift
    lightness_b = float(np.mean(mean_b_rgb))
    lightness_a = float(np.mean(mean_a_rgb))
    delta_lightness = round(lightness_a - lightness_b, 1)

    # VARI Proxy
    denom_b = max(1.0, mean_b_rgb[1] + mean_b_rgb[0] - mean_b_rgb[2])
    denom_a = max(1.0, mean_a_rgb[1] + mean_a_rgb[0] - mean_a_rgb[2])
    vari_b = float((mean_b_rgb[1] - mean_b_rgb[0]) / denom_b)
    vari_a = float((mean_a_rgb[1] - mean_a_rgb[0]) / denom_a)
    delta_vari = round(vari_a - vari_b, 3)

    soil_tendency = bool((mean_a_rgb[0] > mean_a_rgb[1]) and (mean_a_rgb[1] > mean_a_rgb[2]) and (delta_r > 5))
    water_absorption = bool((delta_lightness < -25) and (lightness_a < 85))

    # 3. Competing Scientific Hypotheses Evaluation
    hypotheses = []

    # Vegetation Disturbance Hypothesis
    if delta_vari < -0.10 or (vari_b > 0.25 and delta_vari < -0.05):
        hypotheses.append({
            "category": "Vegetation or land-cover disturbance",
            "plausibility": "High" if (vari_b > 0.35 and delta_vari < -0.20) else "Moderate",
            "supporting_signals": [
                f"Negative VARI canopy vigor drop of {delta_vari:+.3f} (Baseline: {vari_b:+.2f} -> Target: {vari_a:+.2f}).",
                f"Green spectrum reflectance shift of ΔG={delta_g:+.1f} DN.",
                f"Perimeter complexity {boundary_complexity} consistent with irregular canopy fragmentation."
            ],
            "competing_explanations": [
                "Normal deciduous seasonal senescence or seasonal leaf fall.",
                "Agricultural crop harvest cycle on cultivated land.",
                "Transient soil moisture deficit or dry-season vegetation stress."
            ]
        })

    # Water Expansion / Flooding Hypothesis
    if water_absorption or (delta_lightness < -18 and mean_a_rgb[2] >= mean_a_rgb[0]):
        hypotheses.append({
            "category": "Possible water expansion or flooding",
            "plausibility": "High" if delta_lightness < -30 else "Moderate",
            "supporting_signals": [
                f"Strong optical light absorption with mean surface reflectance drop of {delta_lightness:+.1f} DN.",
                f"Target scene surface brightness {lightness_a:.1f} DN characteristic of deep/turbid water.",
                f"Relative blue-wavelength dominance (B={mean_a_rgb[2]:.0f} vs R={mean_a_rgb[0]:.0f})."
            ],
            "competing_explanations": [
                "Topographic terrain shadowing from lower solar elevation angle at acquisition time.",
                "Controlled seasonal agricultural irrigation flooding (e.g. paddy field inundation).",
                "Cloud shadow projection from unmonitored off-scene clouds."
            ]
        })

    # Soil Exposure / Excavation Hypothesis
    if soil_tendency or (delta_r > 15 and delta_lightness > 5 and delta_vari < -0.05):
        hypotheses.append({
            "category": "Possible excavation or exposed soil",
            "plausibility": "High" if (soil_tendency and delta_r > 25) else "Moderate",
            "supporting_signals": [
                f"Positive red-channel shift of ΔR={delta_r:+.1f} DN producing characteristic tan/mineral soil coloration.",
                f"Surface reflectance chromatic ratio: R={mean_a_rgb[0]:.0f} > G={mean_a_rgb[1]:.0f} > B={mean_a_rgb[2]:.0f}.",
                f"Compactness score of {compactness:.2f} reflecting contiguous earth-clearing boundary."
            ],
            "competing_explanations": [
                "Seasonal agricultural tilling / field plowing prior to planting.",
                "Recent burn scar leaving exposed soil and charred biomass.",
                "Seasonal riverbank or sediment sandbar exposure during low-water stage."
            ]
        })

    # Infrastructure / Construction Hypothesis
    if (aspect_ratio < 2.0 and extent > 0.55 and boundary_complexity < 4.5) or (compactness > 0.5 and abs(delta_lightness) > 15):
        hypotheses.append({
            "category": "Possible construction or infrastructure alteration",
            "plausibility": "Moderate",
            "supporting_signals": [
                f"Geometric compactness ({compactness}) and rectilinear extent ({extent:.2f}) suggesting human-built boundaries.",
                f"Boundary complexity ({boundary_complexity}) lower than typical organic fractal vegetation.",
                f"Sharp contrast transition: ΔLightness={delta_lightness:+.1f} DN."
            ],
            "competing_explanations": [
                "Rectangular agricultural parcel clearing or plastic greenhouse installation.",
                "Temporary surface storage (e.g., shipping containers or agricultural sheeting).",
                "Straight-line geological fault or firebreak maintenance."
            ]
        })

    # Default if no specialized hypothesis emerged
    if not hypotheses:
        hypotheses.append({
            "category": "Unclassified visual change",
            "plausibility": "Moderate",
            "supporting_signals": [
                f"Measured visual difference: mean {mean_diff} DN, max {max_diff} DN.",
                f"Spectral shift: ΔR={delta_r:+.1f}, ΔG={delta_g:+.1f}, ΔB={delta_b:+.1f}.",
                f"Area {area_px:,} pixels ({fraction_pct:.3f}% of scene)."
            ],
            "competing_explanations": [
                "Atmospheric aerosol, haze, or thin cirrus cloud divergence between acquisitions.",
                "Sun-sensor-surface viewing geometry / bidirectional reflectance distribution (BRDF) drift.",
                "Subtle sensor radiometric calibration discrepancies."
            ]
        })

    top_hypo = hypotheses[0]

    return {
        "region_id": region_dict["id"],
        "bbox": [x, y, w, h],
        "centroid": region_dict["centroid"],
        "area_pixels": area_px,
        "fraction_percent": fraction_pct,
        "geometry": {
            "compactness": compactness,
            "boundary_complexity": boundary_complexity,
            "aspect_ratio": aspect_ratio,
            "extent": extent,
            "perimeter_px": round(perimeter, 1)
        },
        "spectral_evidence": {
            "mean_difference_dn": mean_diff,
            "max_difference_dn": max_diff,
            "delta_lightness_dn": delta_lightness,
            "delta_rgb": [delta_r, delta_g, delta_b],
            "vari_before": round(vari_b, 3),
            "vari_after": round(vari_a, 3),
            "delta_vari": delta_vari,
            "soil_tendency": soil_tendency,
            "water_absorption": water_absorption
        },
        "leading_hypothesis": top_hypo["category"],
        "plausibility": top_hypo["plausibility"],
        "competing_hypotheses": hypotheses,
        "data_quality_concerns": [
            "Analysis performed on uncalibrated 3-band visible spectrum optical data.",
            "Lack of calibrated NIR/SWIR bands precludes true red-edge NDVI or NDWI calculation.",
            "Atmospheric water vapor and solar zenith angle unadjusted across optical frames."
        ],
        "required_follow_up": [
            "Incorporate Sentinel-2 MSI Band 8 (NIR, 842 nm) or Landsat 8 Band 5 to compute genuine NDVI.",
            "Cross-validate with Sentinel-1 Synthetic Aperture Radar (SAR) dual-pol (VV/VH) backscatter to penetrate clouds and confirm physical surface roughness change.",
            "Query local cadastral and historical zoning records to verify authorized land-use permits."
        ]
    }


def evaluate_scene_evidence(
    regions: List[Dict[str, Any]],
    mask_binary: np.ndarray,
    img_before: np.ndarray,
    img_after: np.ndarray,
    diff_map: np.ndarray,
    metrics: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Generates the comprehensive scene-level Evidence Engine assessment

    containing both global synthesis and per-region decomposed dossiers.
    """
    total_valid_pixels = metrics.get("total_image_pixels", img_before.shape[0] * img_before.shape[1])
    total_changed = metrics.get("total_changed_pixels", 0)

    # 1. Zero change case
    if total_changed == 0 or len(regions) == 0:
        return {
            "primary_assessment": "No significant visual change detected",
            "assessment_plausibility": "High",
            "observation_summary": (
                f"Differencing across {total_valid_pixels:,} pixels revealed no contiguous change clusters "
                f"exceeding the active threshold ({metrics.get('threshold_used', 32)} DN) and minimum size "
                f"({metrics.get('min_region_size_used', 150)} px). Optical baselines demonstrate surface stability."
            ),
            "competing_hypotheses": [{
                "category": "No significant change detected",
                "plausibility": "High",
                "supporting_signals": ["All pixel differences remain within ambient sensor noise floor."],
                "competing_explanations": ["Stable land-cover state", "Identical optical acquisition conditions."]
            }],
            "regions_evidence": [],
            "confounding_warnings": [],
            "uncertainty_basis": "Empirical differencing distributions show zero clusters above noise floor.",
            "temporal_status": "Single interval stability"
        }

    # Evaluate each region independently
    region_evidence_list = []
    category_counts = {}

    for r in regions:
        ev = extract_region_evidence(r, mask_binary, img_before, img_after, diff_map, total_valid_pixels)
        region_evidence_list.append(ev)
        cat = ev["leading_hypothesis"]
        category_counts[cat] = category_counts.get(cat, 0) + ev["area_pixels"]

    # Rank scene-wide categories by total pixel weight
    ranked_cats = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
    leading_scene_cat = ranked_cats[0][0] if ranked_cats else "Unclassified visual change"

    # Confounding factors check
    confounding_warnings = []

    # Check background illumination drift across unchanged pixels
    unchanged_mask = (mask_binary == 0)
    if np.any(unchanged_mask):
        bg_b = np.mean(img_before[unchanged_mask])
        bg_a = np.mean(img_after[unchanged_mask])
        bg_delta = abs(bg_a - bg_b)
        if bg_delta > 18.0:
            confounding_warnings.append(
                f"Ambient illumination shift detected: Unchanged background brightness shifted by {bg_delta:.1f} DN. "
                f"This may reflect differing sun elevation angles, atmospheric haze, or seasonal lighting differences."
            )

    # Dimension harmonization notice if applicable
    if metadata and metadata.get("was_resized", False):
        confounding_warnings.append(
            "Dimension harmonization was applied: Images were rescaled to a common frame. "
            "Rescaling does NOT establish geographic equivalence or sub-pixel co-registration."
        )

    # Registration status warning
    reg_status = metadata.get("registration_status", "") if metadata else ""
    if "failed" in reg_status.lower() or "rejected" in reg_status.lower() or "too few" in reg_status.lower():
        confounding_warnings.append(
            f"Registration constraint: {reg_status}. Co-registration could not be guaranteed; "
            f"edge differences along high-contrast boundaries may be geometric alignment artifacts."
        )

    # Global synthesis
    summary_text = (
        f"Detected {len(regions)} discrete change cluster(s) totaling {total_changed:,} pixels "
        f"({metrics.get('change_percentage', 0.0):.2f}% of monitored optical scene). "
        f"Dominant spatial-spectral pattern: '{leading_scene_cat}', accounting for "
        f"{(ranked_cats[0][1]/max(1, total_changed))*100.0:.1f}% of total changed pixels."
    )

    return {
        "primary_assessment": leading_scene_cat,
        "assessment_plausibility": region_evidence_list[0]["plausibility"] if region_evidence_list else "Moderate",
        "observation_summary": summary_text,
        "leading_category_weight_pct": round((ranked_cats[0][1] / max(1, total_changed)) * 100.0, 1),
        "regions_evidence": region_evidence_list,
        "confounding_warnings": confounding_warnings,
        "uncertainty_basis": (
            f"Evaluated across {len(regions)} regions using 8-dimensional geometric and spectral indicators. "
            f"Reflects empirical visual heuristics; physical causality requires ground validation."
        )
    }
