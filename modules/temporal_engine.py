"""TerraSentinel AI — Temporal Reasoning Engine.

Analyzes multi-temporal satellite sequences (T0 -> T1 -> T2) to rigorously distinguish:
1. One-time observed change (strictly flagged when only 2 dates are available)
2. Persistent land-cover disturbance (remains altered across sequential acquisitions)
3. Transient / Reversing events (e.g., temporary floodwaters that recede)
4. Progressive / Expanding disturbances (e.g., cumulative deforestation spurs)

CRITICAL SCIENTIFIC PRINCIPLE:
Persistence CANNOT be inferred from two images alone. A minimum of three acquisitions
is required to evaluate persistence versus transient reversal.
"""
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from modules.change_detection import detect_changes

def evaluate_temporal_sequence(
    images_seq: List[Tuple[np.ndarray, str]],
    threshold: int = 32,
    min_region_size: int = 150
) -> Dict[str, Any]:
    """Evaluates temporal dynamics across 2 or 3 chronological acquisitions.

    Args:
        images_seq: List of tuples (image_rgb_uint8, date_or_label_str)
    """
    n = len(images_seq)

    # 1. Two-date case (Standard before/after)
    if n < 3:
        return {
            "temporal_tier": "Two-Acquisition Baseline (Static)",
            "classification": "One-time observed change",
            "persistence_state": "Undetermined (Minimum 3 acquisitions required)",
            "reversal_detected": False,
            "persistence_confirmed": False,
            "scientific_rationale": (
                "SCIENTIFIC TEMPORAL SAFEGUARD: An optical comparison between only two dates can identify "
                "that a surface difference exists at T1 relative to T0, but CANNOT determine whether that "
                "change is permanent, seasonal, or transient. Follow-up imagery (T2) is mandatory to verify persistence."
            ),
            "recommendation": "Ingest a third temporal acquisition (T2) to assess persistence versus recovery."
        }

    # 2. Multi-date sequence (T0 -> T1 -> T2)
    img0, label0 = images_seq[0]
    img1, label1 = images_seq[1]
    img2, label2 = images_seq[2]

    # Change T0 -> T1 (Initial disturbance event)
    res_01 = detect_changes(img0, img1, threshold=threshold, min_region_size=min_region_size)
    mask_01 = (res_01["binary_mask"] > 0)
    changed_01_px = int(np.sum(mask_01))

    # Change T1 -> T2 (Subsequent transition)
    res_12 = detect_changes(img1, img2, threshold=threshold, min_region_size=min_region_size)
    mask_12 = (res_12["binary_mask"] > 0)

    # Net change T0 -> T2 (Long-term net delta relative to baseline)
    res_02 = detect_changes(img0, img2, threshold=threshold, min_region_size=min_region_size)
    mask_02 = (res_02["binary_mask"] > 0)
    changed_02_px = int(np.sum(mask_02))

    # Evaluate overlap and dynamics
    if changed_01_px == 0:
        return {
            "temporal_tier": "Three-Acquisition Sequence (Multi-Date)",
            "classification": "No initial disturbance detected at T1",
            "persistence_state": "Baseline stable",
            "reversal_detected": False,
            "persistence_confirmed": False,
            "scientific_rationale": "No disturbance exceeded threshold between T0 and T1.",
            "metrics": {
                "t0_label": label0,
                "t1_label": label1,
                "t2_label": label2,
                "t0_t1_changed_px": 0,
                "t0_t2_changed_px": changed_02_px
            }
        }

    # Intersection of T0->T1 disturbance that remains disturbed at T0->T2
    persistent_pixels = int(np.sum(mask_01 & mask_02))
    persistence_ratio = persistent_pixels / float(changed_01_px)

    # Pixels that were disturbed at T1 but reverted back towards baseline at T2
    reverted_pixels = int(np.sum(mask_01 & (~mask_02)))
    reversal_ratio = reverted_pixels / float(changed_01_px)

    # Dynamic Classification Logic
    if reversal_ratio > 0.45:
        temporal_class = "Transient change with documented reversal (Recovery / Receded)"
        persistence_state = "Transient / Reversing"
        reversal_detected = True
        persistence_confirmed = False
        rationale = (
            f"Of the {changed_01_px:,} pixels altered during interval T0 ({label0}) -> T1 ({label1}), "
            f"{reverted_pixels:,} pixels ({reversal_ratio*100.0:.1f}%) returned to baseline spectral state "
            f"by T2 ({label2}). This pattern is characteristic of transient events such as temporary floodwater "
            f"inundation that subsequently receded, or ephemeral vegetation greening."
        )
    elif persistence_ratio > 0.50:
        if changed_02_px > 1.25 * changed_01_px:
            temporal_class = "Persistent progressive disturbance (Cumulative expansion)"
            persistence_state = "Persistent & Expanding"
        else:
            temporal_class = "Persistent land-cover disturbance (Confirmed ongoing)"
            persistence_state = "Confirmed Persistent"
        reversal_detected = False
        persistence_confirmed = True
        rationale = (
            f"Of the {changed_01_px:,} pixels altered during interval T0 ({label0}) -> T1 ({label1}), "
            f"{persistent_pixels:,} pixels ({persistence_ratio*100.0:.1f}%) remained modified relative to baseline "
            f"at T2 ({label2}). Net cumulative disturbed area at T2 is {changed_02_px:,} pixels. "
            f"Confirmed persistence refutes short-term transient phenomena and indicates lasting land-use alteration."
        )
    else:
        temporal_class = "Inconclusive temporal progression (Evolving surface dynamics)"
        persistence_state = "Inconclusive"
        reversal_detected = False
        persistence_confirmed = False
        rationale = (
            f"Partial persistence ({persistence_ratio*100.0:.1f}%) and partial reversal ({reversal_ratio*100.0:.1f}%) "
            f"observed across sequence. The surface exhibits complex intermediate transitions."
        )

    return {
        "temporal_tier": "Three-Acquisition Sequence (Multi-Date Temporal Tracking)",
        "classification": temporal_class,
        "persistence_state": persistence_state,
        "reversal_detected": reversal_detected,
        "persistence_confirmed": persistence_confirmed,
        "scientific_rationale": rationale,
        "metrics": {
            "t0_label": label0,
            "t1_label": label1,
            "t2_label": label2,
            "t0_t1_changed_px": changed_01_px,
            "t1_t2_changed_px": int(np.sum(mask_12)),
            "t0_t2_changed_px": changed_02_px,
            "persistent_px": persistent_pixels,
            "persistence_ratio": round(persistence_ratio, 3),
            "reverted_px": reverted_pixels,
            "reversal_ratio": round(reversal_ratio, 3)
        },
        "res_01": res_01,
        "res_02": res_02,
        "res_12": res_12
    }
