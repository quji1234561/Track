"""Image preprocessing: grayscale conversion, normalization, multi-scale templates.

OpenCV cv2.cvtColor and cv2.resize are the only OpenCV functions used here.
No matching or tracking functions are called.
"""

import cv2
import numpy as np


def to_gray(image):
    """Convert BGR or RGB image to grayscale."""
    if image.ndim == 2:
        return image
    if image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def normalize_gray(gray):
    """Convert grayscale to float32 [0, 1]."""
    gray = gray.astype(np.float32)
    mn, mx = gray.min(), gray.max()
    if mx > mn:
        gray = (gray - mn) / (mx - mn)
    return gray


def preprocess_frame(frame, resize_scale=1.0):
    """Convert frame to normalized grayscale.

    Args:
        frame: BGR image from cv2.VideoCapture.
        resize_scale: optional resize factor (1.0 = original size).

    Returns:
        normalized float32 grayscale [0, 1].
    """
    if frame is None:
        raise ValueError("Frame is None, possibly end of video or read error.")
    gray = to_gray(frame)
    if resize_scale != 1.0:
        h, w = gray.shape
        new_w, new_h = int(w * resize_scale), int(h * resize_scale)
        gray = cv2.resize(gray, (new_w, new_h))
    return normalize_gray(gray)


def preprocess_template(template_path, scales=None):
    """Read template image, convert to grayscale, generate multi-scale versions.

    Args:
        template_path: path to template image.
        scales: list of scale factors, e.g. [0.9, 1.0, 1.1]. Default [1.0].

    Returns:
        list of (template_array, scale) tuples.
    """
    if scales is None:
        scales = [1.0]
    img = cv2.imread(template_path)
    if img is None:
        raise FileNotFoundError(f"Template not found: {template_path}")
    gray = to_gray(img)
    gray_norm = normalize_gray(gray)
    templates = []
    for s in scales:
        if s == 1.0:
            t = gray_norm
        else:
            h, w = gray_norm.shape
            t = cv2.resize(gray_norm, (int(w * s), int(h * s)))
            t = normalize_gray(t)
        templates.append((t.astype(np.float32), s))
    return templates
