"""Scene3 motion-based small target detection via compensated frame differencing.

Pure traditional CV pipeline:
1. Global affine motion estimation (goodFeaturesToTrack + LK + RANSAC)
2. Compensated frame difference (warp prev → diff → threshold)
3. Connected component analysis → candidate bboxes
4. Candidate scoring (motion + shape + NCC + direction)

No AI/ML/DL. Requires self-implemented NCC from src/ncc.py.
"""

import cv2
import numpy as np

# Try to reuse global_motion if available
try:
    from .global_motion import estimate_global_affine as _gmc_affine
    from .global_motion import apply_affine as _apply_affine
except ImportError:
    _gmc_affine = None
    _apply_affine = None


def _estimate_affine(prev_gray, curr_gray, cfg, exclude_bbox=None):
    """Estimate global affine from prev to curr. Reuses global_motion if available."""
    if _gmc_affine is not None:
        return _gmc_affine(prev_gray, curr_gray, exclude_bbox=exclude_bbox, cfg=cfg)

    # Fallback implementation
    max_c = cfg.get("scene3_gmc_max_corners", 500)
    qual = cfg.get("scene3_gmc_quality_level", 0.01)
    min_d = cfg.get("scene3_gmc_min_distance", 8)
    min_v = cfg.get("scene3_gmc_min_valid_tracks", 40)
    ransac = cfg.get("scene3_gmc_ransac_thresh", 3.0)
    margin = cfg.get("scene3_gmc_exclude_margin", 60)

    if prev_gray.dtype == np.float32 and prev_gray.max() <= 1.5:
        pu = (prev_gray * 255).astype(np.uint8)
        cu = (curr_gray * 255).astype(np.uint8)
    else:
        pu = prev_gray.astype(np.uint8)
        cu = curr_gray.astype(np.uint8)

    h, w = pu.shape
    mask = None
    if exclude_bbox is not None and len(exclude_bbox) == 4:
        bx, by, bw, bh = exclude_bbox
        mask = np.ones((h, w), dtype=np.uint8) * 255
        x1 = max(0, bx - margin)
        y1 = max(0, by - margin)
        x2 = min(w, bx + bw + margin)
        y2 = min(h, by + bh + margin)
        mask[y1:y2, x1:x2] = 0

    corners = cv2.goodFeaturesToTrack(pu, maxCorners=max_c, qualityLevel=qual,
                                       minDistance=min_d, mask=mask)
    if corners is None or len(corners) < min_v:
        return None, 0

    nc, st, _ = cv2.calcOpticalFlowPyrLK(pu, cu, corners, None,
                                          winSize=(21,21), maxLevel=3,
                                          criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    if nc is None or st is None:
        return None, 0

    gp = corners[st.ravel() == 1]
    gc = nc[st.ravel() == 1]
    if len(gp) < min_v:
        return None, len(gp)

    try:
        A, inl = cv2.estimateAffinePartial2D(gp, gc, method=cv2.RANSAC,
                                              ransacReprojThreshold=ransac,
                                              maxIters=2000, confidence=0.99)
    except Exception:
        return None, 0

    if A is None:
        return None, 0
    n_inl = int(inl.sum()) if inl is not None else 0
    if n_inl < min_v:
        return None, n_inl
    return A.astype(np.float32), n_inl


def compensated_frame_diff(prev_gray, curr_gray, affine, cfg):
    """Warp prev_gray to curr_gray coords, return absdiff."""
    if affine is None:
        return cv2.absdiff(prev_gray, curr_gray)

    if prev_gray.dtype == np.float32 and prev_gray.max() <= 1.5:
        pu = prev_gray.copy()
        cu = curr_gray.copy()
    else:
        pu = prev_gray.astype(np.float32) / 255.0
        cu = curr_gray.astype(np.float32) / 255.0

    h, w = cu.shape[:2]
    affine_2x3 = affine.astype(np.float64)
    warped = cv2.warpAffine(pu, affine_2x3, (w, h), borderMode=cv2.BORDER_REPLICATE)
    diff = cv2.absdiff(cu, warped)
    return diff


def extract_motion_candidates(diff, cfg):
    """Threshold diff, morph, connected components → candidate bbox list."""
    use_adaptive = cfg.get("scene3_motion_diff_use_adaptive", True)
    thresh_val = cfg.get("scene3_motion_diff_threshold", 18)
    percentile = cfg.get("scene3_motion_diff_percentile", 97.5)

    if diff.dtype == np.float32 and diff.max() <= 1.5:
        du = (diff * 255).astype(np.uint8)
    else:
        du = diff.astype(np.uint8)

    blur = cv2.GaussianBlur(du, (5, 5), 0)

    if use_adaptive:
        pv = np.percentile(blur, percentile)
        thresh = max(5, pv * 0.5)
    else:
        thresh = thresh_val

    _, binary = cv2.threshold(blur, int(thresh), 255, cv2.THRESH_BINARY)

    morph_open = cfg.get("scene3_motion_morph_open", 1)
    morph_dilate = cfg.get("scene3_motion_morph_dilate", 2)
    kernel_o = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    if morph_open > 0:
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_o, iterations=morph_open)
    if morph_dilate > 0:
        kernel_d = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel_d, iterations=morph_dilate)

    num_lbl, lbls, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

    min_area = cfg.get("scene3_motion_min_area", 4)
    max_area = cfg.get("scene3_motion_max_area", 280)
    min_w = cfg.get("scene3_motion_min_w", 2)
    max_w = cfg.get("scene3_motion_max_w", 45)
    min_h = cfg.get("scene3_motion_min_h", 2)
    max_h = cfg.get("scene3_motion_max_h", 55)

    candidates = []
    for i in range(1, num_lbl):
        x, y, w, h, area = stats[i]
        if area < min_area or area > max_area:
            continue
        if w < min_w or w > max_w:
            continue
        if h < min_h or h > max_h:
            continue
        cx, cy = centroids[i]
        # motion_score = average diff intensity in region
        roi_diff = diff[y:y+h, x:x+w]
        motion_score = float(roi_diff.mean()) if roi_diff.size > 0 else 0.0
        candidates.append({
            "x": int(x), "y": int(y), "w": int(w), "h": int(h),
            "area": int(area), "cx": float(cx), "cy": float(cy),
            "motion_score": motion_score,
        })

    # Sort by motion_score descending
    candidates.sort(key=lambda c: c["motion_score"], reverse=True)
    topk = cfg.get("scene3_motion_topk", 20)
    return candidates[:topk]


def score_motion_candidate(cand, curr_gray, prev_center, cfg):
    """Compute direction_score and shape_score for a motion candidate."""
    w, h = cand["w"], cand["h"]
    area = cand["area"]

    # Shape score: prefers compact small targets (not long thin lines)
    aspect = max(w, max(h, 1)) / max(min(w, h), 1)
    shape_score = 1.0
    if aspect > 3.0:
        shape_score = 0.5
    elif aspect > 2.0:
        shape_score = 0.75
    if area < 10:
        shape_score *= 0.8  # very small → slightly lower

    # Direction score: how well does it match expected downward motion
    dir_score = 0.5  # neutral default
    if prev_center is not None:
        dx = cand["cx"] - prev_center[0]
        dy = cand["cy"] - prev_center[1]
        # For scene3, target mostly moves downward (dy positive)
        max_lat = cfg.get("scene3_max_lateral_step", 20)
        max_fwd = cfg.get("scene3_max_forward_step", 16)
        if abs(dx) < max_lat and 0 < dy < max_fwd:
            dir_score = 0.9
        elif abs(dx) < max_lat * 1.5 and dy >= 0:
            dir_score = 0.7
        elif dy < -8:
            dir_score = 0.2  # moving up strongly → unlikely
    else:
        dir_score = 0.5

    return {
        "shape_score": shape_score,
        "direction_score": dir_score,
    }
