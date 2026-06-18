"""Scene4 hybrid tracker — manual/template init + local frame-diff + hover hold + Kalman.

No AI/ML. No cv2.matchTemplate. No OpenCV Tracker.
"""

import cv2
import numpy as np

from .kalman import KalmanFilter2D

_SENTINEL = -1


def _to_255(work):
    if work.size > 0 and work.max() <= 1.5:
        return np.clip(work.astype(np.float32) * 255.0, 0, 255)
    return np.clip(work.astype(np.float32), 0, 255)


class Scene4FrameDiffTracker:
    """Hybrid tracker: manual/template init + local diff + hover hold + Kalman."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.kalman = None
        self.bbox = None
        self.center = None
        self.lost_count = 0
        self.prev_score = 0.0
        self.initialized = False
        self.scene4_state = "INIT"
        self.hover_hold_count = 0
        self.init_frame_id = -1
        self.init_center = None
        self.fixed_box_w = 0
        self.fixed_box_h = 0
        self.initial_template = None

        # Reliable template for hover hold
        self.last_reliable_template = None
        self.last_reliable_bbox = None
        self.last_reliable_center = None
        self._frames_since_init = 0

        # Frame buffers
        self._prev_gray = None
        self._frame_count = 0

        # Templates
        self._tmpl_list = []
        self._tmpl_loaded = False

        # Debug video
        self._debug_vw = None

        # main.py compat
        self._last_all_scores = []
        self.templates = []
        self.scale_usage = {}

    def print_scale_stats(self):
        pass

    def initialize(self, gray_frame):
        r = self.track_frame(gray_frame, 0)
        return r

    # =========================================================================
    #  Template loading
    # =========================================================================

    def _load_templates(self):
        if self._tmpl_loaded:
            return
        try:
            from .preprocess import preprocess_template
            for tp in self.cfg.get("templates", []):
                scales = self.cfg.get("multi_scale", [1.0])
                for img, s in preprocess_template(tp, scales=scales):
                    self._tmpl_list.append((img.astype(np.float32), s))
            self._tmpl_loaded = True
        except Exception:
            pass

    # =========================================================================
    #  Manual init + template ROI init
    # =========================================================================

    def _try_init(self, gray_frame, frame_id):
        """Try manual bbox init or template ROI init. Returns result dict or None."""
        # Manual bbox init
        if (self.cfg.get("scene4_use_manual_init_bbox", False)
                and frame_id == self.cfg.get("scene4_manual_init_frame", 0)):
            bb = self.cfg.get("scene4_manual_init_bbox", None)
            if bb and len(bb) == 4:
                self.bbox = [int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])]
                self.center = (self.bbox[0] + self.bbox[2] // 2,
                               self.bbox[1] + self.bbox[3] // 2)
                self.kalman = KalmanFilter2D(self.center[0], self.center[1])
                self.initialized = True
                self.scene4_state = "TRACKING"
                self.init_frame_id = frame_id
                self.init_center = self.center
                self.fixed_box_w = self.bbox[2]
                self.fixed_box_h = self.bbox[3]
                self.last_reliable_bbox = self.bbox.copy()
                self.last_reliable_center = self.center
                self._update_template(gray_frame)
                self.initial_template = (self.last_reliable_template.copy()
                                          if self.last_reliable_template is not None else None)
                print(f"  [Scene4] Manual init: bbox={self.bbox}, center={self.center}")
                return {"frame_id": frame_id, "bbox": self.bbox, "center": self.center,
                        "score": 1.0, "detected": True, "predicted": False,
                        "lost": False, "used_for_trajectory": True,
                        "template_id": _SENTINEL,
                        "scene4_state": "TRACKING",
                        "reject_reason": "scene4_manual_init_bbox"}

        # Template ROI init
        if (self.cfg.get("scene4_use_template_init", False)
                and not self.initialized):
            roi = self.cfg.get("scene4_init_search_roi", None)
            thr = self.cfg.get("scene4_template_init_threshold", 0.34)
            if roi and len(roi) == 4:
                self._load_templates()
                if self._tmpl_list:
                    work = _to_255(gray_frame)
                    rx, ry, rw, rh = roi
                    x1, x2 = max(0, rx), min(work.shape[1], rx + rw)
                    y1, y2 = max(0, ry), min(work.shape[0], ry + rh)
                    srch = work[y1:y2, x1:x2]
                    if srch.size > 0 and self._tmpl_list:
                        from .ncc import multi_template_search
                        rs = self.cfg.get("resize_scale", 1.0)
                        if rs < 1.0:
                            h_s, w_s = srch.shape
                            srch = cv2.resize(srch, (int(w_s * rs), int(h_s * rs)))
                        st = []
                        for t, scl in self._tmpl_list:
                            th, tw = t.shape
                            if t.max() <= 1.5:
                                t = (t * 255).astype(np.float32)
                            if rs < 1.0:
                                t = cv2.resize(t, (int(tw * rs), int(th * rs))).astype(np.float32)
                            st.append((t, scl, 0))
                        r = multi_template_search(srch, st, step=2, use_integral=True)
                        if r and r["score"] >= thr:
                            gx = int(r["x"] / rs) + x1
                            gy = int(r["y"] / rs) + y1
                            gw = int(r["w"] / rs)
                            gh = int(r["h"] / rs)
                            ggx = gx + gw // 2
                            ggy = gy + gh // 2
                            self.bbox = [gx, gy, gw, gh]
                            self.center = (ggx, ggy)
                            self.kalman = KalmanFilter2D(ggx, ggy)
                            self.initialized = True
                            self.scene4_state = "TRACKING"
                            self.init_frame_id = frame_id
                            self.init_center = self.center
                            self.fixed_box_w = self.bbox[2]
                            self.fixed_box_h = self.bbox[3]
                            self.last_reliable_bbox = self.bbox.copy()
                            self.last_reliable_center = self.center
                            self._update_template(gray_frame)
                            self.initial_template = (self.last_reliable_template.copy()
                                                      if self.last_reliable_template is not None else None)
                            print(f"  [Scene4] Template init: bbox={self.bbox}, "
                                  f"center={self.center}, score={r['score']:.4f}")
                            return {"frame_id": frame_id, "bbox": self.bbox,
                                    "center": self.center, "score": r["score"],
                                    "detected": True, "predicted": False,
                                    "lost": False, "used_for_trajectory": True,
                                    "template_id": r["template_id"],
                                    "scene4_state": "TRACKING",
                                    "reject_reason": "scene4_template_init_roi"}

        return None

    def _update_template(self, gray_frame):
        """Save current crop as reliable template for hover hold."""
        try:
            if self.bbox is None:
                return
            work = _to_255(gray_frame)
            x, y, w, h = self.bbox
            x, y = max(0, x), max(0, y)
            w = min(w, work.shape[1] - x)
            h = min(h, work.shape[0] - y)
            if w > 5 and h > 5:
                self.last_reliable_template = work[y:y+h, x:x+w].astype(np.float32).copy()
        except Exception:
            pass

    # =========================================================================
    #  Track frame
    # =========================================================================

    def track_frame(self, gray_frame, frame_id):
        rs = self.cfg.get("resize_scale", 0.5)
        work = _to_255(gray_frame)
        orig_h, orig_w = work.shape
        if rs < 1.0:
            sf = cv2.resize(work, (int(orig_w * rs), int(orig_h * rs)))
        else:
            sf = work

        # --- Init ---
        if not self.initialized:
            ir = self._try_init(gray_frame, frame_id)
            if ir is not None:
                return ir
            return {"frame_id": frame_id, "bbox": None, "center": None,
                    "score": -1.0, "detected": False, "predicted": False,
                    "lost": False, "used_for_trajectory": False,
                    "template_id": _SENTINEL,
                    "scene4_state": "INIT",
                    "reject_reason": "scene4_waiting_for_init"}

        # --- Kalman predict ---
        kx, ky = self.kalman.predict()
        pred_cx, pred_cy = kx, ky
        pred_gate = self.cfg.get("scene4_prediction_gate", 160)

        # --- Local frame-diff ---
        if self._prev_gray is None:
            self._prev_gray = sf.copy()
        diff = cv2.absdiff(sf, self._prev_gray)
        thr = self.cfg.get("scene4_diff_threshold", 3)
        use_adaptive = self.cfg.get("scene4_diff_use_adaptive", False)
        if use_adaptive:
            pct = self.cfg.get("scene4_diff_percentile", 95.0)
            thr = max(1.0, np.percentile(diff, pct) * 0.3)
        _, mask = cv2.threshold(diff, thr, 255, cv2.THRESH_BINARY)
        mask = mask.astype(np.uint8)
        md = self.cfg.get("scene4_morph_dilate", 1)
        if md > 0:
            kd = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kd, iterations=md)

        nl, lbls, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        min_a = self.cfg.get("scene4_min_area", 8)
        max_a = self.cfg.get("scene4_max_area", 800)
        min_w = self.cfg.get("scene4_min_w", 4)
        max_w = self.cfg.get("scene4_max_w", 120)
        min_h = self.cfg.get("scene4_min_h", 4)
        max_h = self.cfg.get("scene4_max_h", 120)
        exp_a = self.cfg.get("scene4_expected_area", 60)
        a_tol = self.cfg.get("scene4_area_tolerance", 80)
        excl_rois = self.cfg.get("scene4_exclude_rois", [])

        scored = []
        for i in range(1, nl):
            x, y, w, h, area = stats[i]
            if area < min_a or area > max_a or w < min_w or w > max_w or h < min_h or h > max_h:
                continue
            cx, cy = centroids[i]
            cx_o, cy_o = cx / rs, cy / rs
            # Exclude ROI
            in_ex = False
            for er in excl_rois:
                ex, ey, ew, eh = er
                if ex <= cx_o <= ex + ew and ey <= cy_o <= ey + eh:
                    in_ex = True
                    break
            if in_ex:
                continue
            # Prediction gate
            dist = np.sqrt((cx_o - pred_cx) ** 2 + (cy_o - pred_cy) ** 2)
            if self.initialized and dist > pred_gate:
                continue
            # Area score
            area_s = max(0.0, 1.0 - abs(area - exp_a) / a_tol)
            motion_s = float(diff[y:y+h, x:x+w].mean()) / 255.0 if diff.size > 0 else 0.0
            pred_s = max(0.0, 1.0 - dist / pred_gate) if pred_gate > 0 else 0.5
            # Template verify
            tmpl_s = self._template_score(gray_frame, cx_o, cy_o,
                                           int(x/rs), int(y/rs), int(w/rs), int(h/rs))
            score = 0.35 * motion_s + 0.25 * area_s + 0.20 * pred_s + 0.20 * tmpl_s
            pad = self.cfg.get("scene4_bbox_padding", 4)
            bx, by = max(0, x-pad), max(0, y-pad)
            bw = min(sf.shape[1]-bx, w+2*pad)
            bh = min(sf.shape[0]-by, h+2*pad)
            scored.append((score, {"x": bx, "y": by, "w": bw, "h": bh,
                                    "center_x": cx, "center_y": cy,
                                    "area": area, "motion_score": motion_s,
                                    "area_score": area_s, "prediction_score": pred_s,
                                    "template_score": tmpl_s, "score": score}))
        scored.sort(key=lambda x: x[0], reverse=True)

        dbg = {"scene4_state": self.scene4_state,
               "scene4_candidate_count": len(scored),
               "scene4_motion_score": 0.0, "scene4_area_score": 0.0,
               "scene4_prediction_score": 0.0, "scene4_template_score": 0.0,
               "reject_reason": ""}
        best = scored[0][1] if scored else None
        if best:
            dbg["scene4_motion_score"] = best["motion_score"]
            dbg["scene4_area_score"] = best["area_score"]
            dbg["scene4_prediction_score"] = best["prediction_score"]
            dbg["scene4_template_score"] = best["template_score"]

        self._frames_since_init = (frame_id - self.init_frame_id
                                    if self.init_frame_id >= 0 else 999)
        # Strict gates
        stab_frames = self.cfg.get("scene4_stabilize_frames", 30)
        if self._frames_since_init <= stab_frames:
            pred_gate = self.cfg.get("scene4_stabilize_prediction_gate", 45)
            max_jump = self.cfg.get("scene4_stabilize_max_jump", 35)
        else:
            pred_gate = self.cfg.get("scene4_prediction_gate", 80)
            max_jump = self.cfg.get("scene4_max_jump", 70)

        tmpl_min = self.cfg.get("scene4_template_min_score", 0.30)
        min_score = self.cfg.get("scene4_min_candidate_score", 0.35)
        fixed_bbox = self.cfg.get("scene4_use_fixed_bbox_size", True)
        freeze_tmpl = (self.cfg.get("scene4_freeze_template_frames", 60)
                       and self._frames_since_init <= self.cfg.get("scene4_freeze_template_frames", 60))

        if best and best["score"] >= min_score and best.get("template_score", 0) >= tmpl_min:
            cx_o = int(best["center_x"] / rs)
            cy_o = int(best["center_y"] / rs)
            dist_pred = np.sqrt((cx_o - pred_cx)**2 + (cy_o - pred_cy)**2)
            dist_prev = (np.sqrt((cx_o - self.center[0])**2 + (cy_o - self.center[1])**2)
                         if self.center else 0)
            dbg["scene4_dist_to_pred"] = dist_pred
            dbg["scene4_dist_to_prev"] = dist_prev
            if self.init_center:
                dbg["scene4_dist_to_init"] = np.sqrt((cx_o - self.init_center[0])**2
                                                      + (cy_o - self.init_center[1])**2)

            if dist_pred > pred_gate:
                dbg["reject_reason"] = "scene4_strict_prediction_gate_reject"
            elif dist_prev > max_jump:
                dbg["reject_reason"] = "scene4_strict_jump_reject"
            elif (self._frames_since_init <= stab_frames and self.init_center
                  and np.sqrt((cx_o - self.init_center[0])**2
                              + (cy_o - self.init_center[1])**2) > 120):
                dbg["reject_reason"] = "scene4_stabilize_init_region_reject"
            else:
                self.kalman.update(cx_o, cy_o)
                ow_cand = int(best["w"] / rs)
                oh_cand = int(best["h"] / rs)
                if fixed_bbox and self.fixed_box_w > 0:
                    ow = self.fixed_box_w
                    oh = self.fixed_box_h
                else:
                    alpha = self.cfg.get("scene4_bbox_size_update_alpha", 0.05)
                    ow = int(self.fixed_box_w * (1 - alpha) + ow_cand * alpha)
                    oh = int(self.fixed_box_h * (1 - alpha) + oh_cand * alpha)
                ox = int(cx_o - ow // 2)
                oy = int(cy_o - oh // 2)
                self.bbox = [ox, oy, ow, oh]
                self.center = (cx_o, cy_o)
                self.lost_count = 0
                self.prev_score = best["score"]
                self.scene4_state = "TRACKING"
                self.hover_hold_count = 0
                self.last_reliable_bbox = self.bbox.copy()
                self.last_reliable_center = self.center
                ts = best.get("template_score", 0)
                if (not freeze_tmpl and ts >= self.cfg.get("scene4_template_update_threshold", 0.65)
                        and dist_pred < pred_gate * 0.5):
                    self._update_template(gray_frame)

                dbg["scene4_state"] = "TRACKING"
                dbg["scene4_best_score"] = best["score"]
                dbg["reject_reason"] = ""
                out = {"frame_id": frame_id, "bbox": self.bbox, "center": self.center,
                       "score": best["score"], "detected": True, "predicted": False,
                       "lost": False, "used_for_trajectory": True,
                       "template_id": _SENTINEL}
                out.update(dbg)
                self._prev_gray = sf.copy()
                return out

        # --- Hover template hold ---
        hover_thr = self.cfg.get("scene4_hover_template_threshold", 0.45)
        hover_rad = self.cfg.get("scene4_hover_search_radius", 45)
        hover_max_shift = self.cfg.get("scene4_hover_max_shift", 35)
        if (self.cfg.get("scene4_use_hover_template_hold", False)
                and self.last_reliable_template is not None
                and self.last_reliable_center is not None):
            lx, ly = self.last_reliable_center
            hx1 = max(0, int((lx - hover_rad) * rs))
            hy1 = max(0, int((ly - hover_rad) * rs))
            hx2 = min(sf.shape[1], int((lx + hover_rad) * rs))
            hy2 = min(sf.shape[0], int((ly + hover_rad) * rs))
            hs = sf[hy1:hy2, hx1:hx2]
            if hs.size > 0 and self.last_reliable_template is not None:
                try:
                    hss = hs.astype(np.float32)
                    lrt = self.last_reliable_template
                    if lrt.max() <= 1.5:
                        lrt = lrt * 255.0
                    lrt = lrt.astype(np.float32)
                    thr_s = hss.shape[0]
                    thw = hss.shape[1]
                    if lrt.shape[0] < thr_s and lrt.shape[1] < thw:
                        from .ncc import multi_template_search
                        hr = multi_template_search(
                            hss, [(lrt, 1.0, 0)], step=2, use_integral=True)
                        if hr and hr["score"] >= hover_thr:
                            hx_o = int((hr["x"] + hx1) / rs)
                            hy_o = int((hr["y"] + hy1) / rs)
                            hw_o = int(hr["w"] / rs)
                            hh_o = int(hr["h"] / rs)
                            hcx = hx_o + hw_o // 2
                            hcy = hy_o + hh_o // 2
                            hshift = np.sqrt((hcx - pred_cx)**2 + (hcy - pred_cy)**2)
                            dbg["scene4_hover_template_score"] = hr["score"]
                            dbg["scene4_hover_shift"] = hshift
                            if hshift > hover_max_shift:
                                dbg["reject_reason"] = "scene4_hover_shift_too_large"
                            else:
                                self.kalman.update(hcx, hcy)
                                self.bbox = [hx_o, hy_o, hw_o, hh_o]
                                self.center = (hcx, hcy)
                                self.lost_count = 0
                                self.prev_score = hr["score"]
                                self.scene4_state = "HOVER_HOLD"
                                self.hover_hold_count += 1
                                dbg["scene4_state"] = "HOVER_HOLD"
                                dbg["scene4_hover_hold_count"] = self.hover_hold_count
                                dbg["reject_reason"] = "scene4_hover_template_hold"
                                out = {"frame_id": frame_id, "bbox": self.bbox,
                                       "center": self.center, "score": hr["score"],
                                       "detected": True, "predicted": False,
                                       "lost": False, "used_for_trajectory": True,
                                       "template_id": _SENTINEL}
                                out.update(dbg)
                                self._prev_gray = sf.copy()
                                return out
                except Exception:
                    pass

        # --- Kalman fallback ---
        self.lost_count += 1
        dbg["scene4_lost_count"] = self.lost_count
        max_lost = self.cfg.get("scene4_max_lost", 20)

        if self.lost_count <= max_lost:
            pw, ph = (self.bbox[2], self.bbox[3]) if self.bbox else (40, 20)
            pb = [int(kx - pw // 2), int(ky - ph // 2), pw, ph]
            self.center = (int(kx), int(ky))
            self.bbox = pb
            self.scene4_state = "KALMAN_PREDICT"
            dbg["reject_reason"] = "scene4_kalman_prediction"
            out = {"frame_id": frame_id, "bbox": self.bbox, "center": self.center,
                   "score": -1.0, "detected": False, "predicted": True,
                   "lost": False, "used_for_trajectory": True,
                   "template_id": _SENTINEL}
            out.update(dbg)
        else:
            self.scene4_state = "LOST"
            dbg["reject_reason"] = "scene4_lost_too_long"
            out = {"frame_id": frame_id, "bbox": None, "center": None,
                   "score": -1.0, "detected": False, "predicted": False,
                   "lost": True, "used_for_trajectory": False,
                   "template_id": _SENTINEL}
            out.update(dbg)

        self._prev_gray = sf.copy()
        return out

    def _template_score(self, gray_frame, cx, cy, x, y, w, h):
        """Compute NCC template score for a candidate region (0~1)."""
        if not self.cfg.get("scene4_use_template_verify", False):
            return 0.5
        self._load_templates()
        if not self._tmpl_list:
            return 0.5
        try:
            work = _to_255(gray_frame)
            pw = max(10, w)
            ph = max(10, h)
            px = max(0, x - 2)
            py = max(0, y - 2)
            patch = work[py:py+ph+4, px:px+pw+4]
            if patch.size < 100:
                return 0.3
            best = -1.0
            from .ncc import ncc_score
            for tmpl_img, _ in self._tmpl_list:
                t = tmpl_img.astype(np.float32)
                if t.max() <= 1.5:
                    t = t * 255.0
                try:
                    st = cv2.resize(t, (patch.shape[1], patch.shape[0]))
                    ns = ncc_score(patch.astype(np.float32), st.astype(np.float32))
                    if ns > best:
                        best = ns
                except Exception:
                    pass
            return max(0.0, best) if best > -0.5 else 0.3
        except Exception:
            return 0.4
