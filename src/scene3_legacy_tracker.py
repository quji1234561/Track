"""Scene3 legacy simple NCC tracker — isolated from scene2's complex state machine.

Pure traditional CV: self-implemented NCC + Kalman + basic threshold/jump check.
No AI/ML. No cv2.matchTemplate. No OpenCV Tracker.
"""

import math
import cv2
import numpy as np

from .ncc import multi_template_search
from .kalman import KalmanFilter2D
from .preprocess import preprocess_template

_SENTINEL = -1
_FULL_SEARCH_SCALE_MAX = 0.5
_MIN_TMPL_DIM = 60
_MIN_TOTAL_SCALE = 0.35


def _get_search_scale(templates, img_h, img_w):
    if not templates:
        return _FULL_SEARCH_SCALE_MAX
    min_dim = min(min(t.shape[0], t.shape[1]) for t, _, _ in templates)
    safe = _MIN_TMPL_DIM / max(min_dim, 1)
    return min(_FULL_SEARCH_SCALE_MAX, max(_MIN_TOTAL_SCALE, min(1.0, safe)))


def _to_original(x, y, w, h, scale):
    if scale >= 1.0:
        return x, y, w, h
    return int(x / scale), int(y / scale), int(w / scale), int(h / scale)


class Scene3LegacyTracker:
    """Simple NCC + Kalman tracker for scene3_bicycle only."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.templates = []
        self.kalman = None
        self.bbox = None
        self.center = None
        self.lost_count = 0
        self.prev_score = 0.0
        self.initialized = False
        self.trajectory = []
        self._last_all_scores = []  # main.py compat
        self.scale_usage = {}       # main.py compat
        self._load_templates()

    def print_scale_stats(self):
        pass  # legacy tracker doesn't track scale stats

    def _load_templates(self):
        scales = self.cfg.get("multi_scale", [1.0])
        for tid, tp in enumerate(self.cfg["templates"]):
            for img, s in preprocess_template(tp, scales=scales):
                self.templates.append((img.astype(np.float32), s, tid))
        n = len(self.templates)
        s = len(scales)
        print(f"  [Scene3Legacy] {len(self.cfg['templates'])} src x {s} scales = {n} candidates")

    def initialize(self, gray_frame):
        """Full-frame NCC at anchor frame."""
        rs = self.cfg.get("resize_scale", 1.0)
        if rs < 1.0:
            h, w = gray_frame.shape
            gray_frame = cv2.resize(gray_frame, (int(w * rs), int(h * rs)))
        img_h, img_w = gray_frame.shape
        ss = _get_search_scale(self.templates, img_h, img_w)
        ts = rs * ss

        if ss < 1.0:
            sh, sw = int(img_h * ss), int(img_w * ss)
            srch = cv2.resize(gray_frame, (sw, sh))
        else:
            srch = gray_frame
            sh, sw = img_h, img_w

        st = []
        for t, sc, tid in self.templates:
            th, tw = t.shape
            st.append((cv2.resize(t, (int(tw * ss), int(th * ss))).astype(np.float32), sc, tid))

        step = max(2, self.cfg.get("ncc_step", 2))
        print(f"  [Scene3Legacy] Full-frame init ({sw}x{sh}, scale={ts:.2f}, "
              f"{len(st)} candidates, step={step})...")
        result = multi_template_search(srch, st, step=step, use_integral=True, verbose=True)

        if result is None or result["score"] < self.cfg.get("threshold", 0.6):
            return None
        result["x"] = int(result["x"] / ss)
        result["y"] = int(result["y"] / ss)
        result["w"] = int(result["w"] / ss)
        result["h"] = int(result["h"] / ss)
        ox, oy, ow, oh = _to_original(result["x"], result["y"], result["w"], result["h"], rs)
        cx, cy = ox + ow // 2, oy + oh // 2
        self.bbox = [ox, oy, ow, oh]
        self.center = (cx, cy)
        self.kalman = KalmanFilter2D(cx, cy)
        self.initialized = True
        self.lost_count = 0
        self.prev_score = result["score"]
        print(f"  [Scene3Legacy] Init: bbox={self.bbox}, center=({cx},{cy}), "
              f"score={result['score']:.4f}")
        return {"frame_id": 0, "bbox": self.bbox, "center": self.center,
                "score": result["score"], "detected": True, "predicted": False,
                "template_id": result["template_id"], "scale": result.get("scale", 1.0)}

    def track_frame(self, gray_frame, frame_id):
        """Simple local NCC + Kalman tracking."""
        if not self.initialized:
            r = self.initialize(gray_frame)
            if r is not None:
                r["frame_id"] = frame_id
                return r
            return {"frame_id": frame_id, "bbox": [0,0,0,0], "center": (0,0),
                    "score": -1.0, "detected": False, "predicted": False,
                    "template_id": _SENTINEL, "reject_reason": "init_failed"}

        # Pre-scale
        rs = self.cfg.get("resize_scale", 1.0)
        if rs < 1.0:
            h, w = gray_frame.shape
            sf = cv2.resize(gray_frame, (int(w * rs), int(h * rs)))
        else:
            sf = gray_frame

        s_h, s_w = sf.shape
        pred_x, pred_y = self.kalman.predict()
        spx = pred_x * rs
        spy = pred_y * rs
        sbw = int(self.bbox[2] * rs)
        sbh = int(self.bbox[3] * rs)
        rad = int(self.cfg.get("search_radius", 90) * rs)
        if self.lost_count > 0:
            rad = int(rad * (1.0 + self.lost_count * 0.5))

        x1 = max(0, int(spx - sbw // 2 - rad))
        y1 = max(0, int(spy - sbh // 2 - rad))
        x2 = min(s_w, int(spx + sbw // 2 + rad))
        y2 = min(s_h, int(spy + sbh // 2 + rad))
        swin = sf[y1:y2, x1:x2]
        if swin.shape[0] < sbh or swin.shape[1] < sbw:
            swin = sf
            x1, y1 = 0, 0

        wt = []
        if rs < 1.0:
            for t, sc, tid in self.templates:
                th, tw = t.shape
                wt.append((cv2.resize(t, (int(tw * rs), int(th * rs))).astype(np.float32), sc, tid))
        else:
            wt = self.templates

        step = max(1, self.cfg.get("ncc_step", 2) - 1)
        result = multi_template_search(swin, wt, step=step, use_integral=True)
        threshold = self.cfg.get("threshold", 0.6)

        if result is not None and result["score"] >= threshold:
            gx = result["x"] + x1
            gy = result["y"] + y1
            gw, gh = result["w"], result["h"]
            mc = gx + gw // 2
            my = gy + gh // 2
            mc_orig = mc / rs
            my_orig = my / rs
            max_jump = self.cfg.get("scene3_simple_max_jump", 60)
            jump_ok = True
            if self.center is not None:
                if math.hypot(mc_orig - self.center[0], my_orig - self.center[1]) > max_jump:
                    jump_ok = False
            if jump_ok:
                up_x, up_y = self.kalman.update(mc_orig, my_orig)
                ox, oy, ow, oh = _to_original(gx, gy, gw, gh, rs)
                self.bbox = [ox, oy, ow, oh]
                self.center = (int(up_x), int(up_y))
                self.lost_count = 0
                self.prev_score = result["score"]
                return {"frame_id": frame_id, "bbox": self.bbox, "center": self.center,
                        "score": result["score"], "detected": True, "predicted": False,
                        "template_id": result["template_id"],
                        "scale": result.get("scale", 1.0)}
            else:
                self.lost_count += 1
                self.center = (int(pred_x), int(pred_y))
                return {"frame_id": frame_id, "bbox": self.bbox, "center": self.center,
                        "score": result["score"], "detected": False, "predicted": True,
                        "template_id": _SENTINEL, "reject_reason": "scene3_legacy_jump_too_large"}

        self.lost_count += 1
        self.center = (int(pred_x), int(pred_y))
        return {"frame_id": frame_id, "bbox": self.bbox, "center": self.center,
                "score": -1.0, "detected": False, "predicted": True,
                "template_id": _SENTINEL, "reject_reason": "scene3_legacy_score_below_threshold"}
