"""Self-implemented Normalized Cross-Correlation (NCC) matching.

This module contains only hand-written NCC algorithms using NumPy.
No cv2.matchTemplate, no OpenCV tracking, no deep learning.
"""

import numpy as np


def ncc_score(patch, template):
    """Compute Normalized Cross-Correlation score.

    NCC(x,y) = sum((I - mean(I)) * (T - mean(T)))
             / sqrt(sum((I - mean(I))^2) * sum((T - mean(T))^2))

    Args:
        patch: 2D float32 array, the image region to compare.
        template: 2D float32 array, same shape as patch.

    Returns:
        float in [-1, 1]. Returns -1 if denominator is near zero.
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


def ncc_search(search_img, template, step=2):
    """Sliding-window NCC search for the best template match location.

    Args:
        search_img: 2D float32 grayscale image to search within.
        template: 2D float32 grayscale template.
        step: pixel step for sliding window (2 = skip every other pixel).

    Returns:
        (best_x, best_y, best_score) coordinates in search_img.
    """
    img_h, img_w = search_img.shape
    tmpl_h, tmpl_w = template.shape

    if tmpl_h > img_h or tmpl_w > img_w:
        return -1, -1, -1.0

    best_score = -1.0
    best_x, best_y = 0, 0

    for y in range(0, img_h - tmpl_h + 1, step):
        for x in range(0, img_w - tmpl_w + 1, step):
            patch = search_img[y:y + tmpl_h, x:x + tmpl_w]
            score = ncc_score(patch, template)
            if score > best_score:
                best_score = score
                best_x, best_y = x, y

    return best_x, best_y, best_score


def multi_template_search(search_img, templates, step=2):
    """Search with multiple templates and return best overall match.

    Each template in `templates` is a (template_image, scale, template_id) tuple.

    Args:
        search_img: 2D float32 grayscale image.
        templates: list of (template_array, scale, template_id).
        step: sliding window step.

    Returns:
        dict with x, y, w, h, score, template_id, or None if no valid match.
    """
    best_overall_score = -1.0
    best_result = None

    for tmpl_img, scale, tmpl_id in templates:
        x, y, score = ncc_search(search_img, tmpl_img, step=step)
        if score > best_overall_score:
            best_overall_score = score
            best_result = {
                "x": int(x),
                "y": int(y),
                "w": tmpl_img.shape[1],
                "h": tmpl_img.shape[0],
                "score": float(score),
                "template_id": int(tmpl_id),
            }

    return best_result
