"""自写归一化互相关(NCC)模板匹配 —— 纯NumPy实现。

NCC公式: score = Σ((I-μI)×(T-μT)) / (σI×σT)
减均值消除亮度差异，除标准差消除对比度差异，分数范围[-1,1]。

三种搜索模式：
- ncc_score(): 单窗口NCC计算
- ncc_search(): stride-trick矢量化（适合小范围搜索）
- ncc_search_integral(): 积分图加速（适合大范围/全图搜索）

积分图用np.cumsum自建（非cv2.integral），保持float64精度防止高均值低方差
区域的灾难性数值抵消。

multi_template_search()遍历所有模板/尺度候选，取最高分。支持collect_all_scores
（返回每个候选的分数）和return_topk（返回空间Top-K候选列表供后续约束筛选）。

完全避免cv2.matchTemplate —— 积分图、滑窗、NCC计算全部自写。
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

def ncc_search_integral(search_img, template, step=2, verbose=False, topk=None):
    """Integral-image accelerated NCC search.

    Args:
        search_img: 2D float32 grayscale image.
        template: 2D float32 template.
        step: pixel step for sliding window.
        verbose: print progress.
        topk: if int, return top-K candidates [(score,x,y),...] sorted descending.
              if None, return (best_x, best_y, best_score).

    Returns:
        If topk is None: (best_x, best_y, best_score).
        If topk is int: list of (score, x, y) tuples, length <= topk.
    """
    img_h, img_w = search_img.shape
    tmpl_h, tmpl_w = template.shape

    if tmpl_h > img_h or tmpl_w > img_w:
        if topk is not None:
            return []
        return -1, -1, -1.0

    t_mean = template.mean()
    template_zero = template - t_mean
    template_norm = np.sqrt(np.sum(template_zero ** 2))

    if template_norm < 1e-10:
        if topk is not None:
            return []
        return -1, -1, -1.0

    N = float(tmpl_w * tmpl_h)
    integral, integral_sq = compute_integral_images_numpy(search_img)
    x_positions = np.arange(0, img_w - tmpl_w + 1, step, dtype=np.int32)
    n_x = len(x_positions)

    want_topk = topk is not None and topk > 0
    best_score = -1.0
    best_x, best_y = 0, 0
    topk_heap = []  # list of (score, x, y) for top-K

    y_range = range(0, img_h - tmpl_h + 1, step)
    n_y = len(y_range)
    report_every = max(1, n_y // 10)
    BATCH_X = 32

    for i, y in enumerate(y_range):
        xp = x_positions
        sum_I = (
            integral[y + tmpl_h, xp + tmpl_w]
            - integral[y, xp + tmpl_w]
            - integral[y + tmpl_h, xp]
            + integral[y, xp]
        )
        sum_I2 = (
            integral_sq[y + tmpl_h, xp + tmpl_w]
            - integral_sq[y, xp + tmpl_w]
            - integral_sq[y + tmpl_h, xp]
            + integral_sq[y, xp]
        )
        var_sum = np.maximum(sum_I2 - (sum_I * sum_I) / N, 0.0)
        valid = var_sum > 1e-6

        if not valid.any():
            continue

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

        patch_std = np.sqrt(np.maximum(var_sum, 0.0))
        denom = np.where(valid, patch_std * template_norm, 1.0)
        scores = np.where(valid, numerator / denom, -1.0)

        # Single best
        best_local = np.argmax(scores)
        if scores[best_local] > best_score:
            best_score = scores[best_local]
            best_x, best_y = x_positions[best_local], y

        # Top-K collection
        if want_topk:
            k = min(topk, len(scores))
            row_top_indices = np.argpartition(scores, -k)[-k:]
            for idx in row_top_indices:
                s = float(scores[idx])
                if s > -0.5:
                    topk_heap.append((s, int(x_positions[idx]), y))
            # Keep only top-K globally
            if len(topk_heap) > topk * 2:
                topk_heap.sort(key=lambda t: t[0], reverse=True)
                topk_heap = topk_heap[:topk]

        if verbose and i % report_every == 0:
            pct = 100.0 * i / n_y
            print(f"  NCC integral search progress: {pct:.0f}%")

    if want_topk:
        topk_heap.sort(key=lambda t: t[0], reverse=True)
        return topk_heap[:topk]
    return best_x, best_y, float(best_score)


# ---------------------------------------------------------------------------
#  Multi-template search (dispatcher)
# ---------------------------------------------------------------------------

def multi_template_search(search_img, templates, step=2, use_integral=True,
                          verbose=False, collect_all_scores=False,
                          return_topk=False, topk=8):
    """Search multiple templates and return the best overall match.

    Args:
        search_img: 2D float32 grayscale image.
        templates: list of (template_array, scale, template_id) tuples.
        step: sliding window step.
        use_integral: if True, use ncc_search_integral.
        verbose: if True, print progress per template.
        collect_all_scores: if True, also return per-candidate scores list.
        return_topk: if True, return spatial top-K candidate list.
        topk: number of top-K candidates (default 8).

    Returns:
        - (best_result, all_scores) if collect_all_scores=True (no topk)
        - (best_result, all_scores, topk_candidates) if also return_topk=True
        - best_result if neither flag set
    """
    search_func = ncc_search_integral if use_integral else ncc_search
    best_overall_score = -1.0
    best_result = None
    all_scores = [] if collect_all_scores else None
    topk_candidates = [] if return_topk else None

    for _, (tmpl_img, scale, tmpl_id) in enumerate(templates):
        if verbose:
            method = "integral" if use_integral else "stride"
            print(f"  [{method}] template {tmpl_id} scale {scale:.2f} "
                  f"({tmpl_img.shape[1]}x{tmpl_img.shape[0]})...")

        if return_topk and use_integral:
            tk_list = search_func(search_img, tmpl_img, step=step, topk=topk)
            if tk_list:
                best_s, best_x_i, best_y_i = tk_list[0]
            else:
                best_s, best_x_i, best_y_i = -1.0, -1, -1
            # Add all top-K from this template/scale
            for s, px, py in tk_list:
                topk_candidates.append({
                    "x": int(px), "y": int(py),
                    "w": tmpl_img.shape[1], "h": tmpl_img.shape[0],
                    "score": float(s), "scale": float(scale),
                    "template_id": int(tmpl_id),
                })
        else:
            best_x_i, best_y_i, best_s = search_func(search_img, tmpl_img, step=step)

        if collect_all_scores:
            all_scores.append({
                "source_template_id": int(tmpl_id),
                "template_id": int(tmpl_id),
                "scale": float(scale),
                "angle": 0,
                "template_w": tmpl_img.shape[1],
                "template_h": tmpl_img.shape[0],
                "score": float(best_s),
                "match_x": int(best_x_i),
                "match_y": int(best_y_i),
                "accepted_as_best": False,
            })

        if best_s > best_overall_score:
            best_overall_score = best_s
            best_result = {
                "x": int(best_x_i), "y": int(best_y_i),
                "w": tmpl_img.shape[1], "h": tmpl_img.shape[0],
                "score": float(best_s), "scale": float(scale),
                "template_id": int(tmpl_id),
            }

    if best_result is not None and all_scores is not None:
        for c in all_scores:
            if (c["template_id"] == best_result["template_id"]
                    and abs(c["scale"] - best_result["scale"]) < 0.001):
                c["accepted_as_best"] = True
                break

    # Sort topk_candidates by score descending
    if topk_candidates:
        topk_candidates.sort(key=lambda c: c["score"], reverse=True)

    if return_topk:
        if collect_all_scores:
            return best_result, all_scores, topk_candidates
        return best_result, topk_candidates
    if collect_all_scores:
        return best_result, all_scores
    return best_result
