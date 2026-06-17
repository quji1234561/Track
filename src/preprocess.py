"""图像预处理: 灰度化、归一化、多尺度模板生成。

核心函数:
- to_gray(): BGR→灰度
- normalize_gray(): uint8[0,255]→float32[0,1] （除以255而非min-max，保持帧和模板
  在同一强度参考系下，min-max会独立拉伸导致NCC失效）
- preprocess_frame(): 灰度化+可选resize+归一化
- preprocess_template(): 读取模板→灰度→归一化→按multi_scale生成多个缩放版本
- _imread_unicode(): np.fromfile+cv2.imdecode解决Windows CJK路径问题

模板尺寸:
- 过大: 包含过多背景，NCC受背景主导，区分度下降
- 过小: 目标纹理不足，NCC可靠度下降
- 建议: 目标本体 + 少量周围背景（~20% padding）

OpenCV仅用于cvtColor和resize，不涉及任何匹配算法。
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
    """Convert grayscale to float32 [0, 1].

    Uses division by 255.0 (NOT min-max) so that frame and template share
    the same intensity reference. Min-max normalization would stretch each
    image independently, destroying NCC cross-correlation when frame and
    template have different contrast ranges.
    """
    gray = gray.astype(np.float32)
    if gray.max() > 1.5:
        # uint8 input [0, 255] -> float32 [0, 1]
        gray = gray / 255.0
    # else: already float32 [0, 1], leave as-is
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


def _imread_unicode(path):
    """Read image with Unicode path support (cv2.imread fails on Windows with CJK paths)."""
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img


def compute_gradient(gray_img):
    """Compute Sobel gradient magnitude for edge-enhanced NCC matching.

    Uses cv2.Sobel (traditional image processing) on grayscale image.
    Returns float32 [0, 1] normalized gradient magnitude.
    """
    if gray_img.dtype != np.uint8:
        gx = cv2.Sobel(gray_img, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray_img, cv2.CV_32F, 0, 1, ksize=3)
    else:
        gx = cv2.Sobel(gray_img.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray_img.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    if mag.max() > 1e-10:
        mag = mag / mag.max()
    return mag.astype(np.float32)


def preprocess_template_gradient(template_path, scales=None):
    """Read template, generate both grayscale and gradient versions.

    Returns:
        list of (gray_template, grad_template, scale) tuples for NCC matching.
    """
    if scales is None:
        scales = [1.0]
    img = _imread_unicode(template_path)
    gray = to_gray(img)
    gray_norm = normalize_gray(gray)
    grad_norm = normalize_gray(compute_gradient(gray))
    templates = []
    for s in scales:
        if s == 1.0:
            tg = gray_norm
            te = grad_norm
        else:
            h, w = gray_norm.shape
            tg = cv2.resize(gray_norm, (int(w * s), int(h * s)))
            te = cv2.resize(grad_norm, (int(w * s), int(h * s)))
        templates.append((tg.astype(np.float32), te.astype(np.float32), s))
    return templates


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
    img = _imread_unicode(template_path)
    gray = to_gray(img)
    gray_norm = normalize_gray(gray)
    templates = []
    for s in scales:
        if s == 1.0:
            t = gray_norm
        else:
            h, w = gray_norm.shape
            t = cv2.resize(gray_norm, (int(w * s), int(h * s)))
            # cv2.resize preserves [0,1] range; no re-normalization needed
        templates.append((t.astype(np.float32), s))
    return templates
