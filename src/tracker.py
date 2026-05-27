"""Traditional target tracker combining multi-template NCC, local search, and Kalman filter.

Core tracking algorithm:
1. First frame / re-initialization: full-frame multi-template NCC search.
2. Subsequent frames: Kalman predict center -> crop local search window -> NCC -> validate.
3. Lost tolerance: if target lost for too long, fall back to full-frame search.

All matching uses self-implemented NCC in src/ncc.py.
OpenCV is only used for video I/O and drawing — no cv2.matchTemplate or OpenCV Tracker.
"""

import math
import numpy as np

from .ncc import multi_template_search
from .kalman import KalmanFilter2D
from .preprocess import preprocess_template


class TraditionalTracker:
    """Traditional target tracker using multi-template NCC + Kalman prediction.

    Attributes:
        cfg: scene configuration dict.
        templates: list of (array, scale, id) template data.
        kalman: KalmanFilter2D instance or None before init.
        bbox: current bounding box [x, y, w, h] or None.
        center: current center point (cx, cy) or None.
        lost_count: consecutive frames where target was not confidently detected.
        prev_score: NCC score from last accepted detection.
    """

    def __init__(self, scene_config):
        self.cfg = scene_config
        self.templates = []
        self.kalman = None
        self.bbox = None
        self.center = None
        self.lost_count = 0
        self.prev_score = 0.0
        self.initialized = False
        self._load_templates()

    def _load_templates(self):
        """Load and preprocess all templates for this scene."""
        template_paths = self.cfg["templates"]
        scales = self.cfg.get("multi_scale", [1.0])
        for tid, tpath in enumerate(template_paths):
            tmpls = preprocess_template(tpath, scales=scales)
            for tmpl_img, sc in tmpls:
                self.templates.append((tmpl_img, sc, tid))

    def initialize(self, first_gray_frame):
        """Full-frame search to find initial target position.

        Args:
            first_gray_frame: normalized float32 grayscale frame.

        Returns:
            result dict or None if no target found.
        """
        result = multi_template_search(first_gray_frame, self.templates, step=2)
        if result is None or result["score"] < self.cfg["threshold"]:
            return None

        self.bbox = [result["x"], result["y"], result["w"], result["h"]]
        cx = result["x"] + result["w"] // 2
        cy = result["y"] + result["h"] // 2
        self.center = (cx, cy)
        self.kalman = KalmanFilter2D(cx, cy)
        self.initialized = True
        self.lost_count = 0
        self.prev_score = result["score"]

        return {
            "frame_id": 0,
            "bbox": self.bbox.copy(),
            "center": self.center,
            "score": result["score"],
            "detected": True,
            "predicted": False,
            "template_id": result["template_id"],
        }

    def track_frame(self, gray_frame, frame_id):
        """Track target in a single frame.

        Args:
            gray_frame: normalized float32 grayscale frame.
            frame_id: current frame number.

        Returns:
            result dict with keys: frame_id, bbox, center, score, detected, predicted, template_id.
        """
        if not self.initialized:
            result = self.initialize(gray_frame)
            if result is not None:
                result["frame_id"] = frame_id
            return result

        # Kalman predict
        pred_x, pred_y = self.kalman.predict()
        search_radius = self.cfg["search_radius"]

        # If lost for a while, expand search radius
        if self.lost_count > 0:
            search_radius = int(search_radius * (1.0 + self.lost_count * 0.5))

        # Crop local search window around prediction
        img_h, img_w = gray_frame.shape
        tmpl_h = self.bbox[3]
        tmpl_w = self.bbox[2]

        x1 = max(0, int(pred_x - tmpl_w // 2 - search_radius))
        y1 = max(0, int(pred_y - tmpl_h // 2 - search_radius))
        x2 = min(img_w, int(pred_x + tmpl_w // 2 + search_radius))
        y2 = min(img_h, int(pred_y + tmpl_h // 2 + search_radius))

        search_window = gray_frame[y1:y2, x1:x2]

        if search_window.shape[0] < tmpl_h or search_window.shape[1] < tmpl_w:
            # Search window too small, expand to full frame
            search_window = gray_frame
            x1, y1 = 0, 0

        # Multi-template NCC in local window
        result = multi_template_search(search_window, self.templates, step=2)

        threshold = self.cfg["threshold"]
        max_motion = max(50, search_radius * 0.75)

        if result is not None:
            global_x = result["x"] + x1
            global_y = result["y"] + y1
            score = result["score"]

            # Distance from prediction
            match_cx = global_x + result["w"] // 2
            match_cy = global_y + result["h"] // 2
            dist = math.hypot(match_cx - pred_x, match_cy - pred_y)

            accept = score >= threshold and dist <= max_motion

            if accept:
                # Update with detection
                self.bbox = [global_x, global_y, result["w"], result["h"]]
                up_x, up_y = self.kalman.update(match_cx, match_cy)
                self.center = (int(up_x), int(up_y))
                self.lost_count = 0
                self.prev_score = score
                return {
                    "frame_id": frame_id,
                    "bbox": self.bbox.copy(),
                    "center": self.center,
                    "score": score,
                    "detected": True,
                    "predicted": False,
                    "template_id": result["template_id"],
                }

        # Detection rejected or not found — use prediction
        self.lost_count += 1
        self.bbox = [
            int(pred_x - tmpl_w // 2),
            int(pred_y - tmpl_h // 2),
            tmpl_w,
            tmpl_h,
        ]
        self.center = (int(pred_x), int(pred_y))

        # Full re-search if lost too long
        if self.lost_count >= self.cfg["max_lost"]:
            full_result = multi_template_search(gray_frame, self.templates, step=2)
            if full_result is not None and full_result["score"] >= threshold:
                self.bbox = [
                    full_result["x"], full_result["y"],
                    full_result["w"], full_result["h"],
                ]
                cx = full_result["x"] + full_result["w"] // 2
                cy = full_result["y"] + full_result["h"] // 2
                self.kalman = KalmanFilter2D(cx, cy)
                self.center = (cx, cy)
                self.lost_count = 0
                self.prev_score = full_result["score"]
                return {
                    "frame_id": frame_id,
                    "bbox": self.bbox.copy(),
                    "center": self.center,
                    "score": full_result["score"],
                    "detected": True,
                    "predicted": False,
                    "template_id": full_result["template_id"],
                }

        return {
            "frame_id": frame_id,
            "bbox": self.bbox.copy(),
            "center": self.center,
            "score": 0.0,
            "detected": False,
            "predicted": True,
            "template_id": -1,
        }
