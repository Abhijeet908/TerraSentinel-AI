"""Metrics calculation and spatial estimation module for TerraSentinel AI."""
from typing import Dict, Any, Optional, Tuple, List
import numpy as np

def calculate_spatial_areas(
    pixel_count: int,
    resolution_m_per_pixel: Optional[float] = None,
    spatial_calibration_confirmed: bool = False
) -> Dict[str, Any]:
    """Calculates ground area ONLY when verified spatial calibration is confirmed.

    Avoids fabricating physical land areas when spatial resolution is unverified or absent.
    """
    if not spatial_calibration_confirmed or resolution_m_per_pixel is None or resolution_m_per_pixel <= 0:
        return {
            "has_spatial_calibration": False,
            "pixel_count": pixel_count,
            "resolution_m_per_pixel": resolution_m_per_pixel,
            "area_sq_meters": None,
            "area_hectares": None,
            "area_sq_km": None,
            "display_str": "Geographic area unavailable",
            "methodology_note": (
                "Geographic physical area (hectares / km²) is withheld because spatial calibration (GSD) "
                "is unverified for this image upload. Estimating physical land area requires verified sensor "
                "metadata and map projection parameters."
            )
        }

    # Verified spatial calibration calculation
    area_m2 = float(pixel_count) * (float(resolution_m_per_pixel) ** 2)
    area_ha = area_m2 / 10_000.0
    area_km2 = area_m2 / 1_000_000.0

    disp = f"{area_ha:,.2f} ha ({area_m2:,.0f} m2)" if area_ha < 100 else f"{area_km2:,.2f} km2 ({area_ha:,.1f} ha)"

    return {
        "has_spatial_calibration": True,
        "pixel_count": pixel_count,
        "resolution_m_per_pixel": resolution_m_per_pixel,
        "area_sq_meters": round(area_m2, 2),
        "area_hectares": round(area_ha, 3),
        "area_sq_km": round(area_km2, 4),
        "display_str": disp,
        "methodology_note": (
            f"Physical area estimated using planar projection model: Area = N_pixels * GSD^2 "
            f"({pixel_count:,} px * {resolution_m_per_pixel}^2 m2/px = {area_m2:,.0f} m2). "
            f"Assumes near-nadir observation geometry and planar local topography."
        )
    }


def compute_difference_histogram(raw_diff_map: np.ndarray, num_bins: int = 30) -> Dict[str, Any]:
    """Computes histogram bins of difference values for scientific visualization."""
    flat_diff = raw_diff_map.flatten()
    counts, bin_edges = np.histogram(flat_diff, bins=num_bins, range=(0, 255))
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    return {
        "counts": counts.tolist(),
        "bin_centers": [round(float(c), 1) for c in bin_centers],
        "mean_diff": round(float(np.mean(flat_diff)), 2),
        "median_diff": round(float(np.median(flat_diff)), 2),
        "p95_diff": round(float(np.percentile(flat_diff, 95)), 2),
        "max_diff": round(float(np.max(flat_diff)), 2)
    }
