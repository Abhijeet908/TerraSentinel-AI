"""Core change detection engine for TerraSentinel AI.

Implements genuine computer-vision pipelines:
- Perceptual CIE-Lab delta E / Multi-spectral Euclidean difference
- Fast, robust SSIM (Structural Similarity) structural analysis
- Morphological noise suppression
- Connected-component region extraction with bounding boxes & centroids
- Two-tier confidence overlay synthesis (High=Red, Moderate=Amber)
"""
import time
from typing import Dict, Any, List, Tuple, Optional
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

def compute_ssim_map(gray1: np.ndarray, gray2: np.ndarray, ksize: int = 11, sigma: float = 1.5) -> np.ndarray:
    """Computes a localized SSIM map between two grayscale float arrays (0.0 - 1.0).

    Returns a map of SSIM index (1.0 = identical, 0.0 = completely different).
    """
    C1 = (0.01) ** 2
    C2 = (0.03) ** 2

    if cv2 is not None:
        mu1 = cv2.GaussianBlur(gray1, (ksize, ksize), sigma)
        mu2 = cv2.GaussianBlur(gray2, (ksize, ksize), sigma)
    else:
        # Fallback uniform average if cv2 is absent during unit test mock
        from scipy.ndimage import uniform_filter
        mu1 = uniform_filter(gray1, size=ksize)
        mu2 = uniform_filter(gray2, size=ksize)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    if cv2 is not None:
        sigma1_sq = cv2.GaussianBlur(gray1 * gray1, (ksize, ksize), sigma) - mu1_sq
        sigma2_sq = cv2.GaussianBlur(gray2 * gray2, (ksize, ksize), sigma) - mu2_sq
        sigma12 = cv2.GaussianBlur(gray1 * gray2, (ksize, ksize), sigma) - mu1_mu2
    else:
        from scipy.ndimage import uniform_filter
        sigma1_sq = uniform_filter(gray1 * gray1, size=ksize) - mu1_sq
        sigma2_sq = uniform_filter(gray2 * gray2, size=ksize) - mu2_sq
        sigma12 = uniform_filter(gray1 * gray2, size=ksize) - mu1_mu2

    # SSIM formula
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return np.clip(ssim_map, 0.0, 1.0)


def detect_changes(
    img_before: np.ndarray,
    img_after: np.ndarray,
    threshold: int = 32,
    min_region_size: int = 150,
    morph_kernel_size: int = 3,
    structural_weight: float = 0.35,
    overlay_opacity: float = 0.55
) -> Dict[str, Any]:
    """Executes the full scientific change detection pipeline.

    Returns:
        dict containing:
        - raw_diff_map: np.ndarray float 0-255
        - normalized_heatmap: np.ndarray uint8 RGB (Colormap)
        - binary_mask: np.ndarray uint8 0 or 255
        - overlay_img: np.ndarray uint8 RGB
        - annotated_boxes_img: np.ndarray uint8 RGB
        - regions: list of region dicts
        - metrics: quantitative metrics dict
        - processing_time_sec: float
    """
    start_time = time.perf_counter()

    h, w, c = img_before.shape
    total_pixels = h * w

    # Valid-pixel mask: exclude shared pure-black / nodata border margins
    is_zero_b = (img_before[:, :, 0] == 0) & (img_before[:, :, 1] == 0) & (img_before[:, :, 2] == 0)
    is_zero_a = (img_after[:, :, 0] == 0) & (img_after[:, :, 1] == 0) & (img_after[:, :, 2] == 0)
    invalid_nodata = is_zero_b & is_zero_a
    valid_mask = ~invalid_nodata
    valid_pixels = int(np.sum(valid_mask))
    if valid_pixels == 0:
        valid_pixels = total_pixels
        valid_mask = np.ones((h, w), dtype=bool)

    # 1. Pixel-level color difference
    # Convert to Lab color space for perceptual uniformity if cv2 is available
    if cv2 is not None:
        lab_before = cv2.cvtColor(img_before, cv2.COLOR_RGB2Lab).astype(np.float32)
        lab_after = cv2.cvtColor(img_after, cv2.COLOR_RGB2Lab).astype(np.float32)
        # Delta E approximation
        delta_l = lab_after[:, :, 0] - lab_before[:, :, 0]
        delta_a = lab_after[:, :, 1] - lab_before[:, :, 1]
        delta_b = lab_after[:, :, 2] - lab_before[:, :, 2]
        pixel_diff = np.sqrt(delta_l**2 + delta_a**2 + delta_b**2)
        # Normalize roughly to 0-255 range (max possible delta E is ~255)
        pixel_diff = np.clip(pixel_diff * 1.5, 0.0, 255.0)
    else:
        # Euclidean RGB distance fallback
        diff_rgb = np.linalg.norm(img_after.astype(np.float32) - img_before.astype(np.float32), axis=2)
        pixel_diff = np.clip(diff_rgb * (255.0 / np.sqrt(3 * 255**2)), 0.0, 255.0)

    # 2. Structural difference (SSIM)
    if cv2 is not None:
        gray_b = cv2.cvtColor(img_before, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        gray_a = cv2.cvtColor(img_after, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    else:
        gray_b = np.mean(img_before, axis=2).astype(np.float32) / 255.0
        gray_a = np.mean(img_after, axis=2).astype(np.float32) / 255.0

    ssim_map = compute_ssim_map(gray_b, gray_a)
    struct_diff = (1.0 - ssim_map) * 255.0  # 0 to 255

    # 3. Combined Difference Map
    w_s = float(np.clip(structural_weight, 0.0, 1.0))
    w_p = 1.0 - w_s
    combined_diff = (w_p * pixel_diff) + (w_s * struct_diff)
    combined_diff_uint8 = np.clip(combined_diff, 0, 255).astype(np.uint8)

    # 4. Thresholding to create candidate binary change mask
    thresh_val = max(5, int(threshold))
    binary_raw = (combined_diff_uint8 >= thresh_val).astype(np.uint8) * 255

    # 5. Morphological filtering (Opening then Closing)
    if cv2 is not None and morph_kernel_size >= 3:
        k = morph_kernel_size if morph_kernel_size % 2 != 0 else morph_kernel_size + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        # Open to eliminate single-pixel noise
        opened = cv2.morphologyEx(binary_raw, cv2.MORPH_OPEN, kernel)
        # Close to bridge interior voids
        cleaned_mask = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    else:
        cleaned_mask = binary_raw

    # Suppress changes on invalid/nodata border margins
    cleaned_mask[~valid_mask] = 0

    # 6. Connected Component Analysis & Region Extraction
    regions: List[Dict[str, Any]] = []
    filtered_mask = np.zeros_like(cleaned_mask)

    if cv2 is not None:
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cleaned_mask, connectivity=8)

        # stats columns: [x, y, width, height, area]
        # label 0 is background
        valid_regions_data = []
        for label_id in range(1, num_labels):
            area = int(stats[label_id, cv2.CC_STAT_AREA])
            if area >= min_region_size:
                valid_regions_data.append((label_id, area))

        # Sort descending by area
        valid_regions_data.sort(key=lambda item: item[1], reverse=True)

        total_changed_pixels = 0
        for rank, (label_id, area) in enumerate(valid_regions_data, start=1):
            x = int(stats[label_id, cv2.CC_STAT_LEFT])
            y = int(stats[label_id, cv2.CC_STAT_TOP])
            rw = int(stats[label_id, cv2.CC_STAT_WIDTH])
            rh = int(stats[label_id, cv2.CC_STAT_HEIGHT])
            cx, cy = centroids[label_id]

            # Mark in filtered mask
            region_pixel_mask = (labels == label_id)
            filtered_mask[region_pixel_mask] = 255
            total_changed_pixels += area

            # Calculate mean intensity in this region
            mean_intensity = float(np.mean(combined_diff[region_pixel_mask]))

            fraction_of_image = (area / float(valid_pixels)) * 100.0

            regions.append({
                "id": rank,
                "label_id": label_id,
                "bbox": [x, y, rw, rh],
                "area_pixels": area,
                "fraction_percent": round(fraction_of_image, 4),
                "mean_intensity": round(mean_intensity, 2),
                "centroid": [round(float(cx), 1), round(float(cy), 1)],
                "aspect_ratio": round(float(rw) / max(1, rh), 2),
            })
    else:
        # Fallback if cv2 is missing
        total_changed_pixels = int(np.sum(cleaned_mask > 0))
        filtered_mask = cleaned_mask
        if total_changed_pixels >= min_region_size:
            regions.append({
                "id": 1,
                "label_id": 1,
                "bbox": [0, 0, w, h],
                "area_pixels": total_changed_pixels,
                "fraction_percent": round((total_changed_pixels / float(total_pixels)) * 100.0, 4),
                "mean_intensity": round(float(np.mean(combined_diff[cleaned_mask > 0])), 2),
                "centroid": [w / 2.0, h / 2.0],
                "aspect_ratio": round(w / float(h), 2),
            })

    # Compute relative size compared to largest detected region
    largest_area = regions[0]["area_pixels"] if regions else 0
    for r in regions:
        r["relative_size_ratio"] = round(r["area_pixels"] / float(largest_area), 3) if largest_area > 0 else 0.0

    # 7. Multi-Tier Visual Overlay Creation
    # High change: Red (255, 23, 68), Moderate: Amber (255, 214, 0)
    high_thresh = thresh_val * 1.5
    overlay_rgb = img_after.copy()
    alpha = float(np.clip(overlay_opacity, 0.1, 1.0))

    if cv2 is not None:
        high_mask = (filtered_mask > 0) & (combined_diff >= high_thresh)
        med_mask = (filtered_mask > 0) & (combined_diff < high_thresh)

        # Blend High Changes (Red)
        color_red = np.array([255, 23, 68], dtype=np.uint8)
        overlay_rgb[high_mask] = (
            (1.0 - alpha) * overlay_rgb[high_mask].astype(np.float32) + alpha * color_red
        ).astype(np.uint8)

        # Blend Moderate Changes (Amber)
        color_amber = np.array([255, 214, 0], dtype=np.uint8)
        overlay_rgb[med_mask] = (
            (1.0 - alpha) * overlay_rgb[med_mask].astype(np.float32) + alpha * color_amber
        ).astype(np.uint8)

        # 8. Difference Heatmap (Turbo or Inferno colormap)
        # Normalize combined_diff to 0-255
        norm_diff = cv2.normalize(combined_diff_uint8, None, 0, 255, cv2.NORM_MINMAX)
        heatmap_bgr = cv2.applyColorMap(norm_diff, cv2.COLORMAP_TURBO)
        heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

        # 9. Annotated Bounding Boxes
        annotated_boxes = img_after.copy()
        for r in regions:
            x, y, rw, rh = r["bbox"]
            cv2.rectangle(annotated_boxes, (x, y), (x + rw, y + rh), (0, 229, 255), 2)
            cv2.circle(annotated_boxes, (int(r["centroid"][0]), int(r["centroid"][1])), 4, (255, 214, 0), -1)
            # Label
            label_text = f"#{r['id']}: {r['area_pixels']}px"
            cv2.putText(
                annotated_boxes, label_text, (x, max(14, y - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 229, 255), 1, cv2.LINE_AA
            )
    else:
        heatmap_rgb = np.stack([combined_diff_uint8]*3, axis=2)
        annotated_boxes = img_after.copy()

    # Metrics summary
    processing_time = round(time.perf_counter() - start_time, 3)
    change_pct = round((total_changed_pixels / float(valid_pixels)) * 100.0, 3)
    avg_intensity = float(np.mean(combined_diff[filtered_mask > 0])) if total_changed_pixels > 0 else 0.0

    metrics = {
        "total_changed_pixels": total_changed_pixels,
        "valid_comparison_pixels": valid_pixels,
        "total_image_pixels": total_pixels,
        "nodata_pixels_excluded": total_pixels - valid_pixels,
        "change_percentage": change_pct,
        "regions_count": len(regions),
        "largest_region_pixels": largest_area,
        "largest_region_percentage": round((largest_area / float(valid_pixels)) * 100.0, 3) if valid_pixels > 0 else 0.0,
        "average_change_intensity": round(avg_intensity, 2),
        "processing_time_sec": processing_time,
        "threshold_used": thresh_val,
        "min_region_size_used": min_region_size,
    }

    return {
        "raw_diff_map": combined_diff,
        "heatmap": heatmap_rgb,
        "binary_mask": filtered_mask,
        "overlay": overlay_rgb,
        "annotated_boxes": annotated_boxes,
        "regions": regions,
        "metrics": metrics,
        "processing_time_sec": processing_time
    }
