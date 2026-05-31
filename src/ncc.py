"""Self-implemented Normalized Cross-Correlation (NCC) matching.

Two search modes:
- ncc_search: vectorized stride-trick NCC (faster for small search areas)
- ncc_search_integral: integral-image accelerated NCC (faster for large/full-frame)

Integral images are built with np.cumsum — NO cv2.matchTemplate or OpenCV integral.
No deep learning, no OpenCV tracking, no third-party matching.
"""

import numpy as np


# ---------------------------------------------------------------------------
#  Basic NCC score (single patch)
# ---------------------------------------------------------------------------

def ncc_score(patch, template):
    """Compute Normalized Cross-Correlation score for two same-size arrays.

    NCC = sum((I - mean_I) * (T - mean_T))
        / sqrt(sum((I - mean_I)^2) * sum((T - mean_T)^2))

    Returns float in [-1, 1]. Returns -1 if denominator is near zero.
    """
    if patch.shape != template.shape:
        raise ValueError(
            f"Shape mismatch: patch {patch.shape} vs template {template.shape}"
        )
    patch = patch.astype(np.float32)
    template = template.astype(np.float32)

    p_mean = patch.mean()
    t_mean = template.mean()

    p_diff = patch - p_mean
    t_diff = template - t_mean

    numerator = np.sum(p_diff * t_diff)
    denominator = np.sqrt(np.sum(p_diff ** 2) * np.sum(t_diff ** 2))

    if denominator < 1e-10:
        return -1.0
    return float(numerator / denominator)


# ---------------------------------------------------------------------------
#  Integral-image helpers (pure NumPy — NO cv2.integral)
# ---------------------------------------------------------------------------

def compute_integral_images_numpy(gray):
    """Build integral and squared-integral images using np.cumsum.

    Args:
        gray: 2D float32 grayscale array (H, W).

    Returns:
        integral: (H+1, W+1) float32 — padded so rect_sum uses (x,y,w,h).
        integral_sq: (H+1, W+1) float32 — integral of element-wise squares.
    """
    # Keep float64 to avoid catastrophic cancellation when computing
    # var_sum = sum_I2 - sum_I^2/N — for bright/low-variance patches
    # these two terms are large and nearly equal, needing double precision.
    integral = np.pad(
        np.cumsum(np.cumsum(gray.astype(np.float64), axis=0), axis=1),
        ((1, 0), (1, 0)),
    ).astype(np.float64)

    integral_sq = np.pad(
        np.cumsum(np.cumsum((gray.astype(np.float64)) ** 2, axis=0), axis=1),
        ((1, 0), (1, 0)),
    ).astype(np.float64)

    return integral, integral_sq


def rect_sum(integral, x, y, w, h):
    """O(1) sum of pixel values in rectangle (x, y, w, h).

    integral is (H+1, W+1), so the pixel at (x', y') in the original image
    corresponds to integral[y'+1, x'+1].

    Rectangle from (x, y) to (x+w-1, y+h-1):
      sum = I[y+h, x+w] - I[y, x+w] - I[y+h, x] + I[y, x]
    """
    return float(
        integral[y + h, x + w]
        - integral[y, x + w]
        - integral[y + h, x]
        + integral[y, x]
    )


# ---------------------------------------------------------------------------
#  Vectorized stride-trick NCC (original, good for small search areas)
# ---------------------------------------------------------------------------

def _ncc_scores_1d(strip, template, t_diff, t_std):
    """Compute NCC scores for a horizontal image strip across all x positions.

    Uses np.lib.stride_tricks.as_strided to create windows without copying.
    All x-positions for one y-row are evaluated in a single vectorized call.
    """
    tmpl_h, tmpl_w = template.shape
    strip_w = strip.shape[1]

    n_windows = strip_w - tmpl_w + 1
    if n_windows <= 0:
        return np.array([], dtype=np.float32), np.array([], dtype=np.int32)

    shape = (n_windows, tmpl_h, tmpl_w)
    strides = (strip.strides[1], strip.strides[0], strip.strides[1])
    windows = np.lib.stride_tricks.as_strided(
        strip, shape=shape, strides=strides
    )

    w_mean = windows.mean(axis=(1, 2))
    w_diff = windows - w_mean[:, None, None]
    w_std = np.sqrt(np.sum(w_diff ** 2, axis=(1, 2)))

    numerator = np.sum(w_diff * t_diff, axis=(1, 2))
    denominator = w_std * t_std

    valid = denominator > 1e-10
    scores = np.where(valid, numerator / denominator, -1.0)

    return scores.astype(np.float32), np.arange(n_windows, dtype=np.int32)


def ncc_search(search_img, template, step=2, verbose=False):
    """Stride-trick NCC search (vectorized over x-axis per row).

    Good for small-to-medium search areas. For full-frame search on large
    images, prefer ncc_search_integral.

    Returns (best_x, best_y, best_score).
    """
    img_h, img_w = search_img.shape
    tmpl_h, tmpl_w = template.shape

    if tmpl_h > img_h or tmpl_w > img_w:
        return -1, -1, -1.0

    t_mean = template.mean()
    t_diff = template - t_mean
    t_std = np.sqrt(np.sum(t_diff ** 2))

    if t_std < 1e-10:
        return -1, -1, -1.0

    best_score = -1.0
    best_x, best_y = 0, 0

    y_range = range(0, img_h - tmpl_h + 1, step)
    n_y = len(y_range)
    report_every = max(1, n_y // 10)

    for i, y in enumerate(y_range):
        strip = search_img[y:y + tmpl_h, :]
        scores, x_positions = _ncc_scores_1d(strip, template, t_diff, t_std)

        sub_idx = slice(0, len(scores), step)
        scores_sub = scores[sub_idx]
        x_sub = x_positions[sub_idx]

        if len(scores_sub) > 0:
            best_local = np.argmax(scores_sub)
            if scores_sub[best_local] > best_score:
                best_score = scores_sub[best_local]
                best_x, best_y = x_sub[best_local], y

        if verbose and i % report_every == 0:
            pct = 100.0 * i / n_y
            print(f"  NCC search progress: {pct:.0f}%")

    return best_x, best_y, float(best_score)


# ---------------------------------------------------------------------------
#  Integral-image NCC (fast for full-frame / large search areas)
# ---------------------------------------------------------------------------

def ncc_search_integral(search_img, template, step=2, verbose=False):
    """Integral-image accelerated NCC search with vectorized batch processing.

    For each y row, all x positions are evaluated in a single vectorized pass:
    - rect_sum computed via numpy array indexing (fast, no Python loop)
    - numerator computed via strided windows on that row only

    This avoids both the double Python loop and the huge memory allocations
    of full-frame stride-trick — it only allocates one row of windows at a time.

    Returns (best_x, best_y, best_score).
    """
    img_h, img_w = search_img.shape
    tmpl_h, tmpl_w = template.shape

    if tmpl_h > img_h or tmpl_w > img_w:
        return -1, -1, -1.0

    t_mean = template.mean()
    template_zero = template - t_mean
    template_norm = np.sqrt(np.sum(template_zero ** 2))

    if template_norm < 1e-10:
        return -1, -1, -1.0

    N = float(tmpl_w * tmpl_h)

    # Build integral images once
    integral, integral_sq = compute_integral_images_numpy(search_img)

    # Pre-compute all x positions
    x_positions = np.arange(0, img_w - tmpl_w + 1, step, dtype=np.int32)
    n_x = len(x_positions)

    best_score = -1.0
    best_x, best_y = 0, 0

    y_range = range(0, img_h - tmpl_h + 1, step)
    n_y = len(y_range)
    report_every = max(1, n_y // 10)

    BATCH_X = 32  # Process N x-positions at a time to limit memory

    for i, y in enumerate(y_range):
        # Vectorized integral sums for all x at this y (keep float64 for precision)
        xp = x_positions
        sum_I = (
            integral[y + tmpl_h, xp + tmpl_w]
            - integral[y, xp + tmpl_w]
            - integral[y + tmpl_h, xp]
            + integral[y, xp]
        )  # float64

        sum_I2 = (
            integral_sq[y + tmpl_h, xp + tmpl_w]
            - integral_sq[y, xp + tmpl_w]
            - integral_sq[y + tmpl_h, xp]
            + integral_sq[y, xp]
        )  # float64

        # Variance in float64 to avoid catastrophic cancellation
        var_sum = np.maximum(sum_I2 - (sum_I * sum_I) / N, 0.0)
        # Require meaningful patch variance: flat regions give undefined NCC
        valid = var_sum > 1e-6

        if not valid.any():
            continue

        # Numerator: compute in small batches to limit memory
        numerator = np.empty(n_x, dtype=np.float32)
        for b_start in range(0, n_x, BATCH_X):
            b_end = min(b_start + BATCH_X, n_x)
            batch = x_positions[b_start:b_end]
            batch_patches = np.array([
                search_img[y:y + tmpl_h, x:x + tmpl_w]
                for x in batch
            ], dtype=np.float32)
            numerator[b_start:b_end] = np.sum(
                batch_patches * template_zero, axis=(1, 2)
            )

        # Stable score computation
        patch_std = np.sqrt(np.maximum(var_sum, 0.0))
        denom = np.where(valid, patch_std * template_norm, 1.0)
        scores = np.where(valid, numerator / denom, -1.0)

        best_local = np.argmax(scores)
        if scores[best_local] > best_score:
            best_score = scores[best_local]
            best_x, best_y = x_positions[best_local], y

        if verbose and i % report_every == 0:
            pct = 100.0 * i / n_y
            print(f"  NCC integral search progress: {pct:.0f}%")

    return best_x, best_y, float(best_score)


# ---------------------------------------------------------------------------
#  Multi-template search (dispatcher)
# ---------------------------------------------------------------------------

def multi_template_search(search_img, templates, step=2, use_integral=True,
                          verbose=False, collect_all_scores=False):
    """Search multiple templates and return the best overall match.

    Args:
        search_img: 2D float32 grayscale image.
        templates: list of (template_array, scale, template_id) tuples.
        step: sliding window step.
        use_integral: if True, use ncc_search_integral (faster for large areas);
                      if False, use ncc_search (stride-trick, faster for small areas).
        verbose: if True, print progress per template.
        collect_all_scores: if True, also return per-candidate scores list.

    Returns:
        dict {x, y, w, h, score, scale, template_id} or None.
        If collect_all_scores, also returns list of per-candidate dicts.
    """
    search_func = ncc_search_integral if use_integral else ncc_search
    best_overall_score = -1.0
    best_result = None
    all_scores = [] if collect_all_scores else None

    for _, (tmpl_img, scale, tmpl_id) in enumerate(templates):
        if verbose:
            method = "integral" if use_integral else "stride"
            print(f"  [{method}] template {tmpl_id} scale {scale:.2f} "
                  f"({tmpl_img.shape[1]}x{tmpl_img.shape[0]})...")

        x, y, score = search_func(search_img, tmpl_img, step=step)

        if collect_all_scores:
            all_scores.append({
                "source_template_id": int(tmpl_id),
                "template_id": int(tmpl_id),
                "scale": float(scale),
                "angle": 0,
                "template_w": tmpl_img.shape[1],
                "template_h": tmpl_img.shape[0],
                "score": float(score),
                "match_x": int(x),
                "match_y": int(y),
                "accepted_as_best": False,
            })

        if score > best_overall_score:
            best_overall_score = score
            best_result = {
                "x": int(x),
                "y": int(y),
                "w": tmpl_img.shape[1],
                "h": tmpl_img.shape[0],
                "score": float(score),
                "scale": float(scale),
                "template_id": int(tmpl_id),
            }

    # Mark the winner
    if best_result is not None and all_scores is not None:
        for c in all_scores:
            if (c["template_id"] == best_result["template_id"]
                    and abs(c["scale"] - best_result["scale"]) < 0.001):
                c["accepted_as_best"] = True
                break

    if collect_all_scores:
        return best_result, all_scores
    return best_result
