"""Global motion compensation via sparse optical flow + affine estimation.

Pure traditional CV: cv2.goodFeaturesToTrack + calcOpticalFlowPyrLK + estimateAffinePartial2D.
No AI/ML/DL — just corner tracking and RANSAC affine fitting.

Used by scene2 to compensate for drone camera motion (rising + rotating) during
occlusion, so the target's predicted position is relative to the moving background.
"""

import cv2
import numpy as np


def estimate_global_affine(prev_gray, curr_gray, exclude_bbox=None, cfg=None):
    """Estimate 2x3 affine transform from prev_gray to curr_gray.

    Steps:
    1. Detect corners on prev_gray (excluding target bbox area)
    2. Track corners to curr_gray via optical flow
    3. Estimate affine (translation + rotation + scale) via RANSAC

    Args:
        prev_gray: float32 [0,1] or uint8 previous frame.
        curr_gray: same format, current frame.
        exclude_bbox: [x, y, w, h] in original coords to exclude from tracking.
        cfg: scene config dict for parameter overrides.

    Returns:
        A: 2x3 float32 affine matrix, or None if estimation failed.
        num_inliers: int, number of RANSAC inlier matches.
    """
    # Default parameters
    max_corners = 300
    quality = 0.01
    min_dist = 8
    min_valid = 30
    exclude_margin = 80
    ransac_thresh = 3.0

    if cfg is not None:
        max_corners = cfg.get("scene2_gmc_max_corners", max_corners)
        quality = cfg.get("scene2_gmc_quality_level", quality)
        min_dist = cfg.get("scene2_gmc_min_distance", min_dist)
        min_valid = cfg.get("scene2_gmc_min_valid_tracks", min_valid)
        exclude_margin = cfg.get("scene2_gmc_exclude_margin", exclude_margin)
        ransac_thresh = cfg.get("scene2_gmc_ransac_thresh", ransac_thresh)

    # Convert to uint8 if float32
    if prev_gray.dtype == np.float32 and prev_gray.max() <= 1.5:
        prev_u8 = (prev_gray * 255).astype(np.uint8)
        curr_u8 = (curr_gray * 255).astype(np.uint8)
    else:
        prev_u8 = prev_gray.astype(np.uint8)
        curr_u8 = curr_gray.astype(np.uint8)

    h, w = prev_u8.shape

    # Build exclusion mask for target bbox
    mask = None
    if exclude_bbox is not None and len(exclude_bbox) == 4:
        bx, by, bw, bh = exclude_bbox
        mask = np.ones((h, w), dtype=np.uint8) * 255
        x1 = max(0, bx - exclude_margin)
        y1 = max(0, by - exclude_margin)
        x2 = min(w, bx + bw + exclude_margin)
        y2 = min(h, by + bh + exclude_margin)
        mask[y1:y2, x1:x2] = 0

    # Detect corners
    corners = cv2.goodFeaturesToTrack(
        prev_u8, maxCorners=max_corners, qualityLevel=quality,
        minDistance=min_dist, mask=mask
    )
    if corners is None or len(corners) < min_valid:
        return None, 0

    # Optical flow tracking
    new_corners, status, _ = cv2.calcOpticalFlowPyrLK(
        prev_u8, curr_u8, corners, None,
        winSize=(21, 21), maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
    )
    if new_corners is None or status is None:
        return None, 0

    # Filter valid tracks
    good_prev = corners[status.ravel() == 1]
    good_curr = new_corners[status.ravel() == 1]
    if len(good_prev) < min_valid:
        return None, len(good_prev)

    # Estimate affine via RANSAC
    try:
        A, inliers = cv2.estimateAffinePartial2D(
            good_prev, good_curr, method=cv2.RANSAC,
            ransacReprojThreshold=ransac_thresh, maxIters=2000, confidence=0.99
        )
    except Exception:
        return None, 0

    if A is None:
        return None, 0

    num_inliers = int(inliers.sum()) if inliers is not None else 0
    if num_inliers < min_valid:
        return None, num_inliers

    return A.astype(np.float32), num_inliers


def apply_affine(A, x, y):
    """Apply 2x3 affine matrix to a point (x, y). Returns (px, py)."""
    px = float(A[0, 0] * x + A[0, 1] * y + A[0, 2])
    py = float(A[1, 0] * x + A[1, 1] * y + A[1, 2])
    return px, py
