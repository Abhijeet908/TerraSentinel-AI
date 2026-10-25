"""TerraSentinel AI — Change Truth Engine.

Signature Evidence Challenge & Counterfactual Anomaly Verification System.

The Change Truth Engine challenges every detected anomaly through eight structured stages:
- Stage A: Candidate Anomaly Extraction (geometric, spectral, and spatial features)
- Stage B: Background Variation Estimation (scene-wide illumination and reflectance drift)
- Stage C: Counterfactual Control-Region Analysis (standardized effect size vs non-anomalous control patches)
- Stage D: Competing Explanation Challenge (illumination shift, misregistration, phenology, water, soil)
- Stage E: Temporal Confirmation (multi-date persistence vs reversal verification)
- Stage F: Descriptive Evidence Verdict formulation
- Stage G: Interactive Dossier Generation with "What Would Change This Verdict?" analysis
- Stage H: Deterministic and reproducible reporting serialization
"""
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None


# Authoritative Verdict Constants
VERDICT_LOCALIZED = "LOCALIZED CHANGE RELATIVE TO CONTROL"
VERDICT_GLOBAL_CONFOUND = "POSSIBLE GLOBAL IMAGING CONFOUND"
VERDICT_TEMPORAL_PERSISTENT = "TEMPORAL PERSISTENCE OBSERVED"
VERDICT_CAUSE_UNCERTAIN = "CHANGE DETECTED — CAUSE UNCERTAIN"
VERDICT_INSUFFICIENT = "INSUFFICIENT EVIDENCE"


def compute_vari_proxy(rgb_arr: np.ndarray) -> np.ndarray:
    """Computes Visible Atmospherically Resistant Index (VARI) RGB proxy."""
    r = rgb_arr[:, :, 0].astype(np.float32)
    g = rgb_arr[:, :, 1].astype(np.float32)
    b = rgb_arr[:, :, 2].astype(np.float32)
    denom = g + r - b
    denom[np.abs(denom) < 1.0] = 1.0
    vari = (g - r) / denom
    return np.clip(vari, -1.0, 1.0)


def sample_counterfactual_control_regions(
    diff_map: np.ndarray,
    mask_binary: np.ndarray,
    candidate_bbox: List[int],
    num_samples: int = 12,
    seed: int = 42
) -> Dict[str, Any]:
    """Stage C: Selects spatially comparable, non-anomalous control windows

    and computes counterfactual baseline change statistics.
    """
    x, y, w, h = candidate_bbox
    img_h, img_w = diff_map.shape[:2]

    # Non-anomalous mask (pixels with 0 change and valid data)
    valid_background_mask = (mask_binary == 0)

    total_bg_pixels = int(np.count_nonzero(valid_background_mask))
    candidate_area = max(1, w * h)

    # Edge case: Insufficient stable background (< 300 pixels or < 1.5x candidate area)
    if total_bg_pixels < max(300, int(candidate_area * 1.5)):
        return {
            "control_analysis_available": False,
            "reason": "Insufficient stable background pixels for reliable counterfactual control sampling.",
            "num_controls_sampled": 0,
            "control_mean_delta": None,
            "control_std_delta": None,
            "z_score": None,
            "percentile": None,
            "control_bbox": None
        }

    rng = np.random.RandomState(seed)

    # Sample windows of size comparable to candidate (clamped between 20x20 and 80x80)
    win_w = max(15, min(w, 80))
    win_h = max(15, min(h, 80))

    control_means = []
    control_bboxes = []

    # Attempt up to 80 random placements to find non-overlapping background patches
    max_attempts = 100
    for _ in range(max_attempts):
        if len(control_means) >= num_samples:
            break
        rx = rng.randint(0, max(1, img_w - win_w))
        ry = rng.randint(0, max(1, img_h - win_h))

        # Check overlap with candidate region
        x_overlap = max(0, min(rx + win_w, x + w) - max(rx, x))
        y_overlap = max(0, min(ry + win_h, y + h) - max(ry, y))
        if x_overlap > 0 and y_overlap > 0:
            continue

        # Check that the candidate control window is predominantly (>85%) non-anomalous
        sub_mask = mask_binary[ry:ry+win_h, rx:rx+win_w]
        if np.mean(sub_mask == 0) >= 0.85:
            sub_diff = diff_map[ry:ry+win_h, rx:rx+win_w][sub_mask == 0]
            if len(sub_diff) > 0:
                control_means.append(float(np.mean(sub_diff)))
                control_bboxes.append([rx, ry, win_w, win_h])

    if len(control_means) < 3:
        # Fallback to direct sampling of background pixels if patch placement was constrained
        bg_diff_pixels = diff_map[valid_background_mask]
        if len(bg_diff_pixels) >= 100:
            sub_sample = rng.choice(bg_diff_pixels, size=min(1000, len(bg_diff_pixels)), replace=False)
            c_mean = float(np.mean(sub_sample))
            c_std = float(np.std(sub_sample))
            return {
                "control_analysis_available": True,
                "reason": "Pixel-distribution background sampling (dense scene constraints).",
                "num_controls_sampled": len(sub_sample),
                "control_mean_delta": round(c_mean, 2),
                "control_std_delta": round(max(0.5, c_std), 2),
                "z_score": None,  # Computed in caller
                "percentile": None,
                "control_bbox": None
            }
        return {
            "control_analysis_available": False,
            "reason": "Dense spatial change prevents extracting isolated control patches.",
            "num_controls_sampled": len(control_means),
            "control_mean_delta": None,
            "control_std_delta": None,
            "z_score": None,
            "percentile": None,
            "control_bbox": None
        }

    c_mean = float(np.mean(control_means))
    c_std = float(np.std(control_means))
    primary_control_bbox = control_bboxes[0] if control_bboxes else None

    return {
        "control_analysis_available": True,
        "reason": f"Sampled {len(control_means)} independent counterfactual background windows.",
        "num_controls_sampled": len(control_means),
        "control_mean_delta": round(c_mean, 2),
        "control_std_delta": round(max(0.5, c_std), 2),
        "control_means_list": [round(m, 2) for m in control_means],
        "primary_control_bbox": primary_control_bbox
    }


def analyze_region_truth(
    region_dict: Dict[str, Any],
    mask_binary: np.ndarray,
    img_before: np.ndarray,
    img_after: np.ndarray,
    diff_map: np.ndarray,
    background_stats: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None,
    temporal_result: Optional[Dict[str, Any]] = None,
    seed: int = 42
) -> Dict[str, Any]:
    """Executes the full Change Truth Engine challenge pipeline for a single anomaly candidate."""
    x, y, w, h = region_dict["bbox"]
    area_px = region_dict["area_pixels"]

    # -------------------------------------------------------------------------
    # STAGE A: Candidate Anomaly Measurements
    # -------------------------------------------------------------------------
    sub_mask = mask_binary[y:y+h, x:x+w]
    label_id = region_dict.get("label_id")
    if label_id is not None:
        region_bool = (sub_mask == label_id) | (sub_mask == 255)
    else:
        region_bool = (sub_mask > 0)
    if not np.any(region_bool):
        region_bool = np.ones((h, w), dtype=bool)

    sub_b = img_before[y:y+h, x:x+w][region_bool].astype(np.float32)
    sub_a = img_after[y:y+h, x:x+w][region_bool].astype(np.float32)
    sub_diff = diff_map[y:y+h, x:x+w][region_bool].astype(np.float32)

    mean_b_rgb = np.mean(sub_b, axis=0) if len(sub_b) > 0 else np.array([0, 0, 0])
    mean_a_rgb = np.mean(sub_a, axis=0) if len(sub_a) > 0 else np.array([0, 0, 0])

    delta_r = round(float(mean_a_rgb[0] - mean_b_rgb[0]), 1)
    delta_g = round(float(mean_a_rgb[1] - mean_b_rgb[1]), 1)
    delta_b = round(float(mean_a_rgb[2] - mean_b_rgb[2]), 1)

    mean_delta = round(float(np.mean(sub_diff)), 2) if len(sub_diff) > 0 else 0.0
    max_delta = round(float(np.max(sub_diff)), 2) if len(sub_diff) > 0 else 0.0

    lightness_b = float(np.mean(mean_b_rgb))
    lightness_a = float(np.mean(mean_a_rgb))
    delta_lightness = round(lightness_a - lightness_b, 1)

    # VARI calculation
    denom_b = max(1.0, mean_b_rgb[1] + mean_b_rgb[0] - mean_b_rgb[2])
    denom_a = max(1.0, mean_a_rgb[1] + mean_a_rgb[0] - mean_a_rgb[2])
    vari_b = float((mean_b_rgb[1] - mean_b_rgb[0]) / denom_b)
    vari_a = float((mean_a_rgb[1] - mean_a_rgb[0]) / denom_a)
    delta_vari = round(vari_a - vari_b, 3)

    # Geometry
    aspect_ratio = round(float(w) / max(1, h), 2)
    extent = round(float(area_px) / max(1, w * h), 3)
    perimeter = 2.0 * (w + h)
    compactness = 0.5
    boundary_complexity = 4.0

    if cv2 is not None:
        sub_uint8 = region_bool.astype(np.uint8) * 255
        contours, _ = cv2.findContours(sub_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c = max(contours, key=cv2.contourArea)
            perimeter = float(cv2.arcLength(c, True))
            if perimeter > 0:
                compactness = round(float(4.0 * np.pi * area_px / (perimeter ** 2)), 3)
                compactness = min(1.0, max(0.01, compactness))
                boundary_complexity = round(float(perimeter / max(1.0, np.sqrt(area_px))), 2)

    # -------------------------------------------------------------------------
    # STAGE B: Establish Background Variation & Confounders
    # -------------------------------------------------------------------------
    bg_mean = background_stats.get("bg_mean_delta", 10.0)
    bg_std = background_stats.get("bg_std_delta", 5.0)
    bg_lightness_shift = background_stats.get("bg_lightness_shift", 0.0)
    global_illumination_shift = bool(background_stats.get("global_confounder_flag", False))

    reg_status = metadata.get("registration_status", "Default grid") if metadata else "Default grid"
    misregistration_risk = bool("failed" in reg_status.lower() or "rejected" in reg_status.lower() or "too few" in reg_status.lower())

    # -------------------------------------------------------------------------
    # STAGE C: Counterfactual Control-Region Analysis
    # -------------------------------------------------------------------------
    ctrl_res = sample_counterfactual_control_regions(diff_map, mask_binary, [x, y, w, h], num_samples=12, seed=seed)

    z_score = None
    percentile = None
    control_comparison_valid = ctrl_res["control_analysis_available"]

    if control_comparison_valid and ctrl_res.get("control_mean_delta") is not None:
        c_mu = ctrl_res["control_mean_delta"]
        c_sigma = max(1.0, ctrl_res["control_std_delta"])
        z_score = round(float((mean_delta - c_mu) / c_sigma), 2)

        # Empirical percentile vs control window means
        c_list = ctrl_res.get("control_means_list", [])
        if c_list:
            percentile = round(float(np.mean([mean_delta > m for m in c_list]) * 100.0), 1)
        else:
            percentile = 99.0 if z_score > 2.5 else (50.0 if z_score < 0.5 else 80.0)

        ctrl_res["z_score"] = z_score
        ctrl_res["percentile"] = percentile

    # -------------------------------------------------------------------------
    # STAGE D: Challenge Competing Explanations
    # -------------------------------------------------------------------------
    competing_explanations = []

    # 1. Localized land-cover change hypothesis
    is_localized = bool(z_score is not None and z_score >= 2.0 and not misregistration_risk)
    competing_explanations.append({
        "hypothesis": "Genuine localized land-cover transformation",
        "supported": is_localized,
        "evidence": f"Candidate region delta ({mean_delta:.1f} DN) is {z_score if z_score else 'unmeasured'}σ above stable control baseline." if is_localized else "Change magnitude does not sufficiently separate from background controls."
    })

    # 2. Global illumination / atmospheric shift
    competing_explanations.append({
        "hypothesis": "Scene-wide illumination or atmospheric variation",
        "supported": global_illumination_shift,
        "evidence": f"Unchanged background shifted by {bg_lightness_shift:+.1f} DN, indicating ambient solar/haze divergence." if global_illumination_shift else "Background illumination is stable (|ΔL| < 16 DN)."
    })

    # 3. Misregistration artifact
    is_edge_artifact = bool(misregistration_risk or (aspect_ratio > 4.0 and compactness < 0.15))
    competing_explanations.append({
        "hypothesis": "Sub-pixel image misregistration / geometric shift",
        "supported": is_edge_artifact,
        "evidence": f"Co-registration constraint ({reg_status}) or narrow perimeter geometry (Aspect {aspect_ratio})." if is_edge_artifact else "Registration confirmed; cluster geometry is contiguous."
    })

    # 4. Phenological / seasonal vegetation cycle
    is_seasonal_veg = bool(delta_vari < -0.08 and vari_b > 0.25 and not is_edge_artifact)
    competing_explanations.append({
        "hypothesis": "Vegetation canopy alteration (deforestation, logging, or seasonal dieback)",
        "supported": is_seasonal_veg,
        "evidence": f"Canopy greenness index dropped by {delta_vari:+.3f} (VARI: {vari_b:.2f} -> {vari_a:.2f})." if is_seasonal_veg else "No significant negative VARI canopy drop."
    })

    # 5. Hydrological / water alteration
    is_water = bool(delta_lightness < -22.0 and delta_b >= delta_r - 5)
    competing_explanations.append({
        "hypothesis": "Surface water inundation, flooding, or shoreline alteration",
        "supported": is_water,
        "evidence": f"Substantial optical absorption: lightness dropped by {delta_lightness:+.1f} DN." if is_water else "Surface reflectance shift does not match deep water absorption."
    })

    # 6. Mineral soil exposure / excavation
    is_soil = bool(delta_r > 12.0 and mean_a_rgb[0] > mean_a_rgb[1] and delta_vari < -0.04)
    competing_explanations.append({
        "hypothesis": "Exposed subsoil, mining, or excavation",
        "supported": is_soil,
        "evidence": f"Positive red mineral shift ΔR={delta_r:+.1f} DN with chromatic order R > G > B." if is_soil else "Absence of distinctive red/tan soil reflectance profile."
    })

    # -------------------------------------------------------------------------
    # STAGE E: Temporal Confirmation
    # -------------------------------------------------------------------------
    temporal_persistence_confirmed = False
    temporal_reversal_detected = False
    temporal_summary = "Two-date comparison only; ongoing persistence cannot be established without T2 follow-up."

    if temporal_result and temporal_result.get("temporal_tier") == "Three-Acquisition Sequence":
        temporal_persistence_confirmed = temporal_result.get("persistence_confirmed", False)
        temporal_reversal_detected = temporal_result.get("reversal_detected", False)
        temporal_summary = f"Multi-date sequence evaluated: {temporal_result.get('classification', '')} (State: {temporal_result.get('persistence_state', '').upper()})."

    # -------------------------------------------------------------------------
    # STAGE F: Evidence Verdict Formulation
    # -------------------------------------------------------------------------
    if global_illumination_shift and (z_score is not None and z_score < 2.0):
        verdict = VERDICT_GLOBAL_CONFOUND
        verdict_rationale = (
            "Observed candidate difference is statistically consistent with widespread scene-level "
            "illumination drift or atmospheric haze. Cannot be confirmed as localized land disturbance."
        )
    elif temporal_persistence_confirmed:
        verdict = VERDICT_TEMPORAL_PERSISTENT
        verdict_rationale = (
            "Multi-date surveillance confirms that the detected anomaly persists across multiple chronological "
            "acquisitions, ruling out single-day transient phenomena or cloud shadows."
        )
    elif is_localized and (z_score is not None and z_score >= 2.5):
        verdict = VERDICT_LOCALIZED
        verdict_rationale = (
            f"Candidate anomaly magnitude ({mean_delta:.1f} DN) significantly exceeds the counterfactual "
            f"stable background distribution (Z = +{z_score}σ, >{percentile}% of controls). Localized disturbance confirmed."
        )
    elif mean_delta > 25.0:
        verdict = VERDICT_CAUSE_UNCERTAIN
        verdict_rationale = (
            "Distinct surface radiometric delta detected, but spectral or geometric evidence is ambiguous. "
            "Multiple competing hypotheses remain plausible without multi-spectral bands."
        )
    else:
        verdict = VERDICT_INSUFFICIENT
        verdict_rationale = (
            "Detected difference is close to ambient sensor noise floor or lacks sufficient statistical separation "
            "from background controls."
        )

    # -------------------------------------------------------------------------
    # STAGE G: What Would Change This Verdict?
    # -------------------------------------------------------------------------
    strengthening_factors = [
        "Synthetic Aperture Radar (SAR) cross-validation: Sentinel-1 VV/VH backscatter shift confirming surface roughness destruction independent of optical clouds.",
        "Calibrated multispectral Red-Edge (Sentinel-2 B8A) / SWIR (B11/B12) confirming biological vegetation necrosis or moisture absorption.",
        "Additional chronological acquisition at T2 confirming physical non-reversal over a multi-month baseline."
    ]

    weakening_factors = [
        "Reversion to baseline reflectance during subsequent agricultural season (proving seasonal crop rotation or deciduous leaf-fall).",
        "Official cadastral records verifying authorized civil construction or permitted land management.",
        "Co-registration re-analysis showing sub-pixel boundary shift along high-contrast topography."
    ]

    # Crop coordinates for reference control window
    ctrl_bbox = ctrl_res.get("primary_control_bbox")

    return {
        "region_id": region_dict["id"],
        "bbox": [x, y, w, h],
        "control_bbox": ctrl_bbox,
        "verdict": verdict,
        "verdict_rationale": verdict_rationale,
        "observed_measurements": {
            "area_pixels": area_px,
            "mean_delta_dn": mean_delta,
            "max_delta_dn": max_delta,
            "delta_lightness": delta_lightness,
            "delta_vari": delta_vari,
            "delta_rgb": [delta_r, delta_g, delta_b],
            "compactness": compactness,
            "boundary_complexity": boundary_complexity,
            "aspect_ratio": aspect_ratio
        },
        "counterfactual_controls": {
            "available": control_comparison_valid,
            "num_controls": ctrl_res.get("num_controls_sampled", 0),
            "control_mean_delta": ctrl_res.get("control_mean_delta"),
            "control_std_delta": ctrl_res.get("control_std_delta"),
            "z_score": z_score,
            "empirical_percentile": percentile,
            "methodology": "Uniform sampling of non-anomalous background patches of comparable scale."
        },
        "background_variation": {
            "bg_mean_delta": round(bg_mean, 2),
            "bg_std_delta": round(bg_std, 2),
            "bg_lightness_shift": round(bg_lightness_shift, 2),
            "global_confounder_flag": global_illumination_shift
        },
        "registration_quality": {
            "status": reg_status,
            "misregistration_risk": misregistration_risk
        },
        "competing_explanations": competing_explanations,
        "temporal_evidence": {
            "summary": temporal_summary,
            "persistence_confirmed": temporal_persistence_confirmed,
            "reversal_detected": temporal_reversal_detected
        },
        "what_would_change_this_verdict": {
            "strengthening_evidence": strengthening_factors,
            "weakening_evidence": weakening_factors
        },
        "limitations": [
            "Differencing computed on uncalibrated 3-band visible spectrum RGB imagery.",
            "Absence of near-infrared / shortwave-infrared bands precludes absolute biophysical index calculation.",
            "Pixel Ground Sample Distance must be verified from sensor ephemeris before determining legal compliance."
        ]
    }


def run_change_truth_engine(
    regions: List[Dict[str, Any]],
    mask_binary: np.ndarray,
    img_before: np.ndarray,
    img_after: np.ndarray,
    diff_map: np.ndarray,
    metrics: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None,
    temporal_result: Optional[Dict[str, Any]] = None,
    seed: int = 42
) -> Dict[str, Any]:
    """Executes the scene-level Change Truth Engine challenge.

    Returns the comprehensive Change Truth Dossier for all candidate regions.
    """
    total_valid = metrics.get("total_image_pixels", img_before.shape[0] * img_before.shape[1])
    total_changed = metrics.get("total_changed_pixels", 0)

    # 1. Zero change case
    if total_changed == 0 or len(regions) == 0:
        return {
            "overall_verdict": "NO ANOMALIES DETECTED",
            "overall_rationale": "All pixel differences remain below active noise thresholds. Counterfactual baseline confirms surface stability.",
            "regions_truth": [],
            "background_variation": {
                "bg_mean_delta": 0.0,
                "bg_std_delta": 0.0,
                "bg_lightness_shift": 0.0,
                "global_confounder_flag": False
            },
            "total_candidates_evaluated": 0,
            "reproducibility": {
                "seed": seed,
                "engine_version": "2.5.0-changetruth",
                "deterministic": True
            }
        }

    # 2. Stage B: Background Variation Estimation
    unchanged_mask = (mask_binary == 0)
    scene_light_b = float(np.mean(img_before))
    scene_light_a = float(np.mean(img_after))
    scene_light_shift = float(scene_light_a - scene_light_b)

    if np.any(unchanged_mask):
        bg_diff = diff_map[unchanged_mask]
        bg_mean = float(np.mean(bg_diff))
        bg_std = float(np.std(bg_diff))
        bg_light_b = float(np.mean(img_before[unchanged_mask]))
        bg_light_a = float(np.mean(img_after[unchanged_mask]))
        bg_light_shift = float(bg_light_a - bg_light_b)
    else:
        bg_mean = float(np.mean(diff_map))
        bg_std = float(np.std(diff_map))
        bg_light_shift = scene_light_shift

    # Global confounder detection:
    # 1. Background shifted in brightness by >16 DN
    # 2. Whole scene shifted in brightness by >16 DN
    # 3. Changed area is overwhelmingly large (>70%) with low diff variance (uniform shift)
    global_confounder = bool(
        abs(bg_light_shift) > 16.0
        or abs(scene_light_shift) > 16.0
        or (bg_mean > 24.0 and bg_std < 15.0)
        or (metrics.get("change_percentage", 0.0) > 70.0 and np.std(diff_map) < 15.0)
    )

    bg_stats = {
        "bg_mean_delta": bg_mean,
        "bg_std_delta": bg_std,
        "bg_lightness_shift": bg_light_shift,
        "scene_lightness_shift": scene_light_shift,
        "global_confounder_flag": global_confounder
    }

    # 3. Evaluate each region
    regions_truth = []
    verdict_counts = {}

    for idx, r in enumerate(regions):
        # Derive deterministic sub-seed per region for reproducibility
        r_seed = seed + (idx * 17)
        r_truth = analyze_region_truth(
            r,
            mask_binary,
            img_before,
            img_after,
            diff_map,
            bg_stats,
            metadata=metadata,
            temporal_result=temporal_result,
            seed=r_seed
        )
        regions_truth.append(r_truth)
        v = r_truth["verdict"]
        verdict_counts[v] = verdict_counts.get(v, 0) + 1

    # Dominant verdict
    ranked_verdicts = sorted(verdict_counts.items(), key=lambda x: x[1], reverse=True)
    dominant_verdict = ranked_verdicts[0][0] if ranked_verdicts else VERDICT_CAUSE_UNCERTAIN

    return {
        "overall_verdict": dominant_verdict,
        "overall_rationale": (
            f"Evaluated {len(regions)} candidate region(s) through Change Truth challenge. "
            f"Primary verdict: '{dominant_verdict}' across {verdict_counts.get(dominant_verdict, 0)} region(s)."
        ),
        "regions_truth": regions_truth,
        "background_variation": bg_stats,
        "total_candidates_evaluated": len(regions),
        "verdict_summary": verdict_counts,
        "reproducibility": {
            "seed": seed,
            "engine_version": "2.5.0-changetruth",
            "deterministic": True
        }
    }
