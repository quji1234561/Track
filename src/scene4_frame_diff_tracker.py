"""Scene4 frame-difference tracker — pure traditional CV for small drone target.

Uses three-frame/two-frame differencing + connected components + Kalman.
No AI/ML. No cv2.matchTemplate. No OpenCV Tracker.

Compatible with main.py's result dict format.
"""

import cv2
import numpy as np

from .kalman import KalmanFilter2D

_SENTINEL = -1


class Scene4FrameDiffTracker:
    """Frame-difference + Kalman tracker for scene4_drone."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.kalman = None
        self.bbox = None
        self.center = None
        self.lost_count = 0
        self.prev_score = 0.0
        self.initialized = False
        self.scene4_state = "INIT"

        # Frame buffers for diff
        self._prev_gray = None
        self._prev2_gray = None
        self._frame_count = 0

        # main.py compat
        self._last_all_scores = []
        self.templates = []  # not used, compat only
        self.scale_usage = {}

    def print_scale_stats(self):
        pass

    def initialize(self, gray_frame):
        """Not used by main.py for scene4; track_frame handles init internally."""
        r = self.track_frame(gray_frame, 0)
        if r and r.get("detected", False):
            self.initialized = True
        return r

    def track_frame(self, gray_frame, frame_id):
        rs = self.cfg.get("resize_scale", 0.5)
        orig_h, orig_w = gray_frame.shape

        # Resize for speed
        if rs < 1.0:
            sf = cv2.resize(gray_frame, (int(orig_w * rs), int(orig_h * rs)))
        else:
            sf = gray_frame
        sf = sf.astype(np.float32)

        self._frame_count += 1

        # --- Store frame history ---
        if self._prev_gray is None:
            self._prev_gray = sf.copy()
            return self._result(frame_id, None, "scene4_waiting_for_frames", "INIT")
        if self._prev2_gray is None:
            self._prev2_gray = self._prev_gray.copy()
            self._prev_gray = sf.copy()
            return self._result(frame_id, None, "scene4_waiting_for_frames", "INIT")

        # --- Frame differencing ---
        diff_method = self.cfg.get("scene4_diff_method", "three_frame")
        blur_k = self.cfg.get("scene4_gaussian_blur", 3)
        if blur_k > 0:
            sf_blur = cv2.GaussianBlur(sf, (blur_k, blur_k), 0)
            p1_blur = cv2.GaussianBlur(self._prev_gray, (blur_k, blur_k), 0)
            p2_blur = cv2.GaussianBlur(self._prev2_gray, (blur_k, blur_k), 0)
        else:
            sf_blur, p1_blur, p2_blur = sf, self._prev_gray, self._prev2_gray

        if diff_method == "three_frame":
            d1 = cv2.absdiff(sf_blur, p1_blur)
            d2 = cv2.absdiff(p1_blur, p2_blur)
            use_adaptive = self.cfg.get("scene4_diff_use_adaptive", True)
            if use_adaptive:
                pct = self.cfg.get("scene4_diff_percentile", 97.0)
                t1 = max(5.0, np.percentile(d1, pct) * 0.4)
                t2 = max(5.0, np.percentile(d2, pct) * 0.4)
            else:
                t1 = t2 = self.cfg.get("scene4_diff_threshold", 18)
            _, m1 = cv2.threshold(d1, t1, 255, cv2.THRESH_BINARY)
            _, m2 = cv2.threshold(d2, t2, 255, cv2.THRESH_BINARY)
            mask = cv2.bitwise_and(m1, m2).astype(np.uint8)
        else:
            diff = cv2.absdiff(sf_blur, p1_blur)
            use_adaptive = self.cfg.get("scene4_diff_use_adaptive", True)
            if use_adaptive:
                pct = self.cfg.get("scene4_diff_percentile", 97.0)
                thr = max(5.0, np.percentile(diff, pct) * 0.4)
            else:
                thr = self.cfg.get("scene4_diff_threshold", 18)
            _, mask = cv2.threshold(diff, thr, 255, cv2.THRESH_BINARY)
            mask = mask.astype(np.uint8)

        # Morphology
        mo = self.cfg.get("scene4_morph_open", 1)
        md = self.cfg.get("scene4_morph_dilate", 2)
        if mo > 0:
            k_o = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_o, iterations=mo)
        if md > 0:
            k_d = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, k_d, iterations=md)

        # Connected components
        nl, lbls, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        min_a = self.cfg.get("scene4_min_area", 3)
        max_a = self.cfg.get("scene4_max_area", 220)
        min_w = self.cfg.get("scene4_min_w", 2)
        max_w = self.cfg.get("scene4_max_w", 50)
        min_h = self.cfg.get("scene4_min_h", 2)
        max_h = self.cfg.get("scene4_max_h", 50)

        candidates = []
        for i in range(1, nl):
            x, y, w, h, area = stats[i]
            if area < min_a or area > max_a:
                continue
            if w < min_w or w > max_w or h < min_h or h > max_h:
                continue
            cx, cy = centroids[i]
            pad = self.cfg.get("scene4_bbox_padding", 4)
            bx = max(0, x - pad)
            by = max(0, y - pad)
            bw = min(sf.shape[1] - bx, w + 2 * pad)
            bh = min(sf.shape[0] - by, h + 2 * pad)
            # Motion score: mean diff in region
            roi_d = d1[by:by+bh, bx:bx+bw] if diff_method == "three_frame" else diff[by:by+bh, bx:bx+bw]
            motion_s = float(roi_d.mean()) / 255.0 if roi_d.size > 0 else 0.0
            candidates.append({
                "x": int(bx), "y": int(by), "w": int(bw), "h": int(bh),
                "center_x": int(cx), "center_y": int(cy),
                "area": int(area), "motion_score": motion_s,
            })

        dbg = {
            "scene4_state": self.scene4_state,
            "scene4_candidate_count": len(candidates),
            "reject_reason": "",
        }

        pred_cx, pred_cy = None, None
        pred_gate = self.cfg.get("scene4_prediction_gate", 120)
        if self.kalman is not None:
            pred_cx, pred_cy = self.kalman.predict()  # already in original coords

        # Score and filter candidates
        scored = []
        for c in candidates:
            # Area score: prefer ~30-80 px
            a = c["area"]
            if 20 < a < 100:
                area_s = 0.8
            elif 10 < a < 150:
                area_s = 0.6
            else:
                area_s = 0.3

            # Prediction score
            if pred_cx is not None:
                # candidate is in scaled coords, Kalman in original → divide by rs to compare
                dist = np.sqrt((c["center_x"] / rs - pred_cx) ** 2
                               + (c["center_y"] / rs - pred_cy) ** 2)
                pred_s = max(0.0, 1.0 - dist / pred_gate) if pred_gate > 0 else 0.5
            else:
                pred_s = 0.5

            score = 0.45 * c["motion_score"] + 0.25 * area_s + 0.20 * pred_s + 0.10 * 0.5
            c["score"] = score
            c["area_score"] = area_s
            c["prediction_score"] = pred_s
            scored.append((score, c))

        scored.sort(key=lambda x: x[0], reverse=True)
        min_score = self.cfg.get("scene4_min_candidate_score", 0.35)

        if scored and scored[0][0] >= min_score:
            best = scored[0][1]
            cx_o = int(best["center_x"] / rs)
            cy_o = int(best["center_y"] / rs)
            ox = int(best["x"] / rs)
            oy = int(best["y"] / rs)
            ow = int(best["w"] / rs)
            oh = int(best["h"] / rs)

            # Jump check
            max_jump = self.cfg.get("scene4_max_jump", 140)
            jump_ok = True
            if self.center is not None:
                if np.sqrt((cx_o - self.center[0]) ** 2 + (cy_o - self.center[1]) ** 2) > max_jump:
                    jump_ok = False

            if jump_ok:
                if self.kalman is None:
                    self.kalman = KalmanFilter2D(cx_o, cy_o)
                else:
                    self.kalman.update(cx_o, cy_o)
                self.bbox = [ox, oy, ow, oh]
                self.center = (cx_o, cy_o)
                self.lost_count = 0
                self.prev_score = best["score"]
                self.initialized = True
                self.scene4_state = "TRACKING"

                dbg["scene4_state"] = "TRACKING"
                dbg["scene4_best_score"] = best["score"]
                dbg["scene4_motion_score"] = best["motion_score"]
                dbg["scene4_area_score"] = best["area_score"]
                dbg["scene4_prediction_score"] = best["prediction_score"]
                dbg["scene4_lost_count"] = 0

                out = {"frame_id": frame_id, "bbox": self.bbox, "center": self.center,
                       "score": best["score"], "detected": True, "predicted": False,
                       "lost": False, "used_for_trajectory": True,
                       "template_id": _SENTINEL}
                out.update(dbg)
                self._shift_frames(sf)
                return out
            else:
                dbg["reject_reason"] = "scene4_jump_too_large"

        # No valid candidate — predict or lost
        self.lost_count += 1
        dbg["scene4_lost_count"] = self.lost_count
        max_lost = self.cfg.get("scene4_max_lost", 20)

        if self.lost_count > max_lost or not self.initialized:
            self.scene4_state = "LOST"
            dbg["reject_reason"] = dbg.get("reject_reason") or (
                "scene4_lost_too_long" if self.lost_count > max_lost
                else "scene4_no_motion_candidate")
            dbg["scene4_state"] = "LOST"
            out = {"frame_id": frame_id, "bbox": None, "center": None,
                   "score": -1.0, "detected": False, "predicted": False,
                   "lost": True, "used_for_trajectory": False,
                   "template_id": _SENTINEL}
            out.update(dbg)
            self._shift_frames(sf)
            return out

        # Kalman prediction (Kalman operates in original coords, no /rs needed)
        if self.kalman is not None:
            kx, ky = self.kalman.predict()  # original coords
            if self.bbox is not None:
                pw, ph = self.bbox[2], self.bbox[3]
                pb = [int(kx - pw // 2), int(ky - ph // 2), pw, ph]
                self.center = (int(kx), int(ky))
                self.bbox = pb
            self.scene4_state = "PREDICTING"
            dbg["reject_reason"] = dbg.get("reject_reason") or "scene4_kalman_prediction"
            dbg["scene4_state"] = "PREDICTING"
            out = {"frame_id": frame_id, "bbox": self.bbox, "center": self.center,
                   "score": -1.0, "detected": False, "predicted": True,
                   "lost": False, "used_for_trajectory": True,
                   "template_id": _SENTINEL}
            out.update(dbg)
        else:
            out = {"frame_id": frame_id, "bbox": None, "center": None,
                   "score": -1.0, "detected": False, "predicted": False,
                   "lost": False, "used_for_trajectory": False,
                   "template_id": _SENTINEL}
            out.update(dbg)

        self._shift_frames(sf)
        return out

    def _shift_frames(self, sf):
        self._prev2_gray = self._prev_gray
        self._prev_gray = sf.copy()

    def _result(self, frame_id, bbox, reason, state):
        return {"frame_id": frame_id,
                "bbox": bbox, "center": None,
                "score": -1.0, "detected": False, "predicted": False,
                "lost": False, "used_for_trajectory": False,
                "template_id": _SENTINEL,
                "reject_reason": reason, "scene4_state": state}
