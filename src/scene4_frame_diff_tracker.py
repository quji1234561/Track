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

        # Debug video writer
        self._debug_vw = None

        # main.py compat
        self._last_all_scores = []
        self.templates = []
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

        # --- Frame differencing (with diagnostics) ---
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
            diff = d1  # for diagnostic reference
            use_adaptive = self.cfg.get("scene4_diff_use_adaptive", True)
            if use_adaptive:
                pct = self.cfg.get("scene4_diff_percentile", 97.0)
                t1 = max(1.0, np.percentile(d1, pct) * 0.3)
                t2 = max(1.0, np.percentile(d2, pct) * 0.3)
            else:
                t1 = t2 = max(1.0, self.cfg.get("scene4_diff_threshold", 10))
            _, m1 = cv2.threshold(d1, t1, 255, cv2.THRESH_BINARY)
            _, m2 = cv2.threshold(d2, t2, 255, cv2.THRESH_BINARY)
            mask = cv2.bitwise_and(m1, m2).astype(np.uint8)
        else:
            diff = cv2.absdiff(sf_blur, p1_blur)
            use_adaptive = self.cfg.get("scene4_diff_use_adaptive", True)
            if use_adaptive:
                pct = self.cfg.get("scene4_diff_percentile", 97.0)
                thr = max(1.0, np.percentile(diff, pct) * 0.3)
            else:
                thr = max(1.0, self.cfg.get("scene4_diff_threshold", 10))
            _, mask = cv2.threshold(diff, thr, 255, cv2.THRESH_BINARY)
            mask = mask.astype(np.uint8)

        # Diagnostics: diff stats
        if diff.size > 0:
            diff_f = diff.astype(np.float32)
            mask_raw_nz = int(mask.sum() / 255)
        else:
            mask_raw_nz = 0

        # Morphology
        mo = self.cfg.get("scene4_morph_open", 1)
        md = self.cfg.get("scene4_morph_dilate", 2)
        if mo > 0:
            k_o = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_o, iterations=mo)
        if md > 0:
            k_d = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, k_d, iterations=md)
        mask_morph_nz = int(mask.sum() / 255)

        # Connected components — raw count before filtering
        nl, lbls, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        raw_cc = nl - 1  # exclude background
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
            "scene4_diff_max": float(diff.max()) if diff.size > 0 else 0.0,
            "scene4_diff_mean": float(diff.mean()) if diff.size > 0 else 0.0,
            "scene4_diff_p95": float(np.percentile(diff, 95)) if diff.size > 0 else 0.0,
            "scene4_mask_nonzero_raw": mask_raw_nz,
            "scene4_mask_nonzero_after_morph": mask_morph_nz,
            "scene4_component_count_raw": raw_cc,
            "scene4_component_count_after_filter": len(candidates),
            "reject_reason": "",
        }

        pred_cx, pred_cy = None, None
        pred_gate = self.cfg.get("scene4_prediction_gate", 120)
        if self.kalman is not None:
            pred_cx, pred_cy = self.kalman.predict()  # already in original coords

        # Score and filter candidates
        scored = []
        for c in candidates:
            a = c["area"]
            if 15 < a < 100:
                area_s = 0.8
            elif 5 < a < 200:
                area_s = 0.6
            else:
                area_s = 0.3
            if pred_cx is not None:
                dist = np.sqrt((c["center_x"] / rs - pred_cx) ** 2
                               + (c["center_y"] / rs - pred_cy) ** 2)
                pred_s = max(0.0, 1.0 - dist / pred_gate) if pred_gate > 0 else 0.5
            else:
                pred_s = 0.5

            if not self.initialized:
                score = 0.60 * c["motion_score"] + 0.30 * area_s + 0.10 * 0.5
            else:
                score = 0.40 * c["motion_score"] + 0.25 * area_s + 0.20 * pred_s + 0.15 * 0.5
            c["score"] = score
            c["area_score"] = area_s
            c["prediction_score"] = pred_s
            scored.append((score, c))

        scored.sort(key=lambda x: x[0], reverse=True)
        min_score = (self.cfg.get("scene4_init_min_score", 0.18) if not self.initialized
                     else self.cfg.get("scene4_min_candidate_score", 0.20))

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
                self._write_debug_frame(sf, mask, candidates, best, rs)
                return out
            else:
                dbg["reject_reason"] = "scene4_jump_too_large"

        # No valid candidate — predict or lost
        self.lost_count += 1
        dbg["scene4_lost_count"] = self.lost_count
        max_lost = self.cfg.get("scene4_max_lost", 20)

        if not self.initialized:
            self.scene4_state = "INIT"
            dbg["reject_reason"] = "scene4_no_motion_candidate_init"
            dbg["scene4_state"] = "INIT"
            dbg["scene4_candidate_count"] = len(candidates)
            dbg["scene4_best_score"] = scored[0][0] if scored else _SENTINEL
            out = {"frame_id": frame_id, "bbox": None, "center": None,
                   "score": -1.0, "detected": False, "predicted": False,
                   "lost": False, "used_for_trajectory": False,
                   "template_id": _SENTINEL}
            out.update(dbg)
            self._shift_frames(sf)
            self._write_debug_frame(sf, mask, candidates, None, rs)
            return out

        if self.lost_count > max_lost:
            self.scene4_state = "LOST"
            dbg["reject_reason"] = dbg.get("reject_reason") or "scene4_lost_too_long"
            dbg["scene4_state"] = "LOST"
            dbg["scene4_candidate_count"] = len(candidates)
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

    def _write_debug_frame(self, sf, mask, candidates, best_cand, rs):
        """Draw debug frame with mask, all candidates, best candidate."""
        try:
            from pathlib import Path
            out_dir = Path("outputs/videos")
            if self._debug_vw is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                dw = sf.shape[1]
                dh = sf.shape[0]
                self._debug_vw = cv2.VideoWriter(
                    str(out_dir / "scene4_drone_diff_debug.mp4"),
                    fourcc, 15.0, (dw, dh))
            # Draw mask as green overlay
            vis = (sf * 255).astype(np.uint8)
            vis_col = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
            if mask is not None and mask.any():
                vis_col[:, :, 1] = cv2.add(vis_col[:, :, 1], mask.astype(np.uint8))
            # All candidates: gray
            for c in candidates:
                cv2.rectangle(vis_col, (c["x"], c["y"]),
                              (c["x"] + c["w"], c["y"] + c["h"]), (128, 128, 128), 1)
            # Best: red
            if best_cand is not None:
                cv2.rectangle(vis_col, (best_cand["x"], best_cand["y"]),
                              (best_cand["x"] + best_cand["w"], best_cand["y"] + best_cand["h"]),
                              (0, 0, 255), 2)
            cv2.putText(vis_col, f"n={len(candidates)}", (5, 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            self._debug_vw.write(vis_col)
        except Exception:
            pass

    def _result(self, frame_id, bbox, reason, state):
        return {"frame_id": frame_id,
                "bbox": bbox, "center": None,
                "score": -1.0, "detected": False, "predicted": False,
                "lost": False, "used_for_trajectory": False,
                "template_id": _SENTINEL,
                "reject_reason": reason, "scene4_state": state}
