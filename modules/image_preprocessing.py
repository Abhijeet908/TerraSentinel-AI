"""Preprocessing pipeline for TerraSentinel AI satellite imagery.

Includes color space conversions, gaussian smoothing, conservative feature-based
registration (ORB + RANSAC), and registration diagnostic reporting.
"""
from typing import Tuple, Dict, Any, Optional
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

def preprocess_pair(
    img_before: np.ndarray,
    img_after: np.ndarray,
    enable_alignment: bool = False,
    blur_radius: int = 3
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Applies preprocessing to an image pair:

    1. Optional conservative feature-based registration (aligns after to before)
    2. Mild Gaussian smoothing for sensor noise reduction
    3. Diagnostic metadata extraction
    """
    diagnostics: Dict[str, Any] = {
        "alignment_attempted": enable_alignment,
        "alignment_success": False,
        "registration_status": "Not requested",
        "inliers_count": 0,
        "alignment_warning": None,
        "blur_radius_applied": blur_radius,
    }

    processed_before = img_before.copy()
    processed_after = img_after.copy()

    # Registration step
    if enable_alignment and cv2 is not None:
        aligned_after, reg_info = register_images_orb(processed_before, processed_after)
        diagnostics.update(reg_info)
        if reg_info.get("alignment_success", False):
            processed_after = aligned_after
        else:
            diagnostics["alignment_warning"] = reg_info.get("error_msg", "Registration failed; proceeding unaligned.")

    # Mild noise reduction if requested (blur_radius must be odd positive integer)
    if blur_radius > 1 and cv2 is not None:
        k = blur_radius if blur_radius % 2 != 0 else blur_radius + 1
        processed_before = cv2.GaussianBlur(processed_before, (k, k), 0)
        processed_after = cv2.GaussianBlur(processed_after, (k, k), 0)

    return processed_before, processed_after, diagnostics


def register_images_orb(
    ref_img: np.ndarray,
    target_img: np.ndarray,
    max_features: int = 2000,
    min_good_matches: int = 15
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Conservative ORB feature matching with RANSAC homography estimation.

    Aligns target_img to ref_img coordinate frame.
    Rejects degenerate, flipped, or extreme perspective warps.
    """
    if cv2 is None:
        return target_img, {
            "alignment_success": False,
            "registration_status": "OpenCV not available",
            "inliers_count": 0,
            "error_msg": "OpenCV library not found."
        }

    h, w = ref_img.shape[:2]
    gray_ref = cv2.cvtColor(ref_img, cv2.COLOR_RGB2GRAY)
    gray_target = cv2.cvtColor(target_img, cv2.COLOR_RGB2GRAY)

    orb = cv2.ORB_create(nfeatures=max_features)
    kp1, des1 = orb.detectAndCompute(gray_ref, None)
    kp2, des2 = orb.detectAndCompute(gray_target, None)

    if des1 is None or des2 is None or len(kp1) < min_good_matches or len(kp2) < min_good_matches:
        return target_img, {
            "alignment_success": False,
            "registration_status": "Insufficient keypoints detected",
            "inliers_count": 0,
            "error_msg": f"Detected keypoints too sparse ({len(kp1) if kp1 else 0} vs {len(kp2) if kp2 else 0})."
        }

    # Matcher with Hamming distance
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    raw_matches = bf.knnMatch(des2, des1, k=2)

    # Lowe's ratio test
    good_matches = []
    for m_pair in raw_matches:
        if len(m_pair) == 2:
            m, n = m_pair
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

    if len(good_matches) < min_good_matches:
        return target_img, {
            "alignment_success": False,
            "registration_status": f"Too few reliable matches ({len(good_matches)} < {min_good_matches})",
            "inliers_count": len(good_matches),
            "error_msg": f"Only {len(good_matches)} feature pairs matched reliably."
        }

    # Extract coordinates
    src_pts = np.float32([kp2[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp1[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # RANSAC homography
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    if H is None:
        return target_img, {
            "alignment_success": False,
            "registration_status": "Homography calculation failed",
            "inliers_count": 0,
            "error_msg": "RANSAC failed to find consistent geometric transformation."
        }

    inliers = int(np.sum(mask)) if mask is not None else 0
    if inliers < min_good_matches:
        return target_img, {
            "alignment_success": False,
            "registration_status": f"Low inlier ratio ({inliers}/{len(good_matches)})",
            "inliers_count": inliers,
            "error_msg": "Geometric consistency was below confidence threshold."
        }

    # Sanity check: Check determinant and condition of H to prevent inverted or wild warps
    det = np.linalg.det(H[:2, :2])
    if det <= 0.2 or det >= 5.0:
        return target_img, {
            "alignment_success": False,
            "registration_status": "Transformation rejected (Extreme scale/shear)",
            "inliers_count": inliers,
            "reprojection_rmse": None,
            "error_msg": f"Scale determinant {det:.2f} indicates unrealistic deformation."
        }

    # Calculate reprojection RMSE on inliers
    inlier_indices = np.where(mask.ravel() == 1)[0]
    if len(inlier_indices) > 0:
        pts_transformed = cv2.perspectiveTransform(src_pts[inlier_indices], H)
        errors = np.linalg.norm(pts_transformed - dst_pts[inlier_indices], axis=2)
        rmse = round(float(np.sqrt(np.mean(errors ** 2))), 2)
    else:
        rmse = 0.0

    # Apply warp
    aligned = cv2.warpPerspective(target_img, H, (w, h), borderMode=cv2.BORDER_REFLECT)

    return aligned, {
        "alignment_success": True,
        "registration_status": f"Registered (Inliers: {inliers}, RMSE: {rmse}px)",
        "inliers_count": inliers,
        "reprojection_rmse": rmse,
        "homography_matrix": H.tolist(),
        "error_msg": None
    }
