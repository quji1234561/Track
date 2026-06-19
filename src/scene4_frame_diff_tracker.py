"""Scene4 motion-tracklet tracker — template init + tracklet scoring + state machine.

Core principle: single-frame frame-diff candidates NEVER directly update center.
Only multi-frame motion tracklets that meet ALL criteria (area, energy, direction,
continuity, anchor, template) are allowed to take over the target.

States: INIT → INIT_LOCKED → TRACKING ⇄ HOVER_HOLD ⇄ REACQUIRE → KALMAN_PREDICT → LOST

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


def _distance(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def _valid_point(p):
    """True if p is a valid 2D point (tuple/list of length >= 2)."""
    return p is not None and isinstance(p, (list, tuple)) and len(p) >= 2


def _safe_coord(p, idx):
    """Return coordinate value or '' if invalid."""
    if _valid_point(p):
        return int(p[idx])
    return ""


def _safe_dist(a, b):
    """Return Euclidean distance or -1 if either point invalid."""
    if not _valid_point(a) or not _valid_point(b):
        return -1.0
    return float(((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5)


def _safe_num(v, default=-1.0):
    """Return v or default if v is None or NaN."""
    if v is None:
        return default
    try:
        if v != v:  # NaN is not equal to itself
            return default
    except Exception:
        pass
    return v


class Scene4FrameDiffTracker:
    """Motion-tracklet tracker with state machine. No single-frame candidate takeover."""

    # ── States ──────────────────────────────────────────────────────────
    STATE_INIT = "INIT"
    STATE_INIT_LOCKED = "INIT_LOCKED"
    STATE_TRACKING = "TRACKING"
    STATE_HOVER_HOLD = "HOVER_HOLD"
    STATE_REACQUIRE = "REACQUIRE"
    STATE_KALMAN_PREDICT = "KALMAN_PREDICT"
    STATE_LOST = "LOST"
    STATE_PHASE_HOLD = "PHASE_HOLD"

    # ── State transition counters ───────────────────────────────────────
    MAX_KALMAN_PREDICT = 20   # fallback: use cfg scene4_max_lost
    MAX_REACQUIRE_FRAMES = 15

    def __init__(self, cfg):
        self.cfg = cfg

        # Core state
        self.state = self.STATE_INIT
        self.initialized = False
        self.kalman = None
        self.bbox = None
        self.center = None

        # Anchor / reliable state (set at init, updated only on reliable tracklet accept)
        self.init_frame_id = -1
        self.init_bbox = None
        self.init_center = None
        self.anchor_center = None
        self.last_reliable_center = None
        self.last_reliable_bbox = None
        self.fixed_box_w = 0
        self.fixed_box_h = 0
        self.initial_template = None
        self.last_reliable_template = None

        # Motion statistics (EMA-updated on reliable tracklet accept)
        self.expected_motion_area = cfg.get("scene4_expected_area", 60)
        self.expected_motion_energy = None
        self.last_motion_direction = None

        # Tracklets
        self.motion_tracklets = []
        self._tracklet_id_counter = 0

        # Frame buffers
        self._prev_gray = None
        self._prev_frame_gray_full = None  # full-res for candidate extraction
        self.scene4_prev_gray = None       # nearest_motion_contour mode
        self._frame_count = 0
        self._frames_since_init = 0

        # Predicted anchor (moves with phase, used for search / gate comparison)
        self.predicted_anchor_center = None

        # Counters
        self.lost_count = 0
        self.kalman_predict_count = 0
        self.hover_hold_count = 0
        self.reacquire_count = 0

        # Templates (NCC templates from config)
        self._tmpl_list = []
        self._tmpl_loaded = False

        # Debug
        self._debug_vw = None
        self._debug_candidates = []
        self._debug_tracklets = []

        # Interactive mode
        self._paused = False
        self._dragging = False
        self._drag_start = (0, 0)
        self._drag_end = (0, 0)
        self._manual_bbox = None
        self._mouse_set = False

        # main.py compat
        self._last_all_scores = []
        self.templates = []
        self.scale_usage = {}
        self.prev_score = 0.0

    def print_scale_stats(self):
        pass

    def initialize(self, gray_frame):
        r = self.track_frame(gray_frame, 0)
        return r

    # =========================================================================
    #  Template loading (unchanged)
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
        """Try manual bbox init or template ROI init. Sets all anchor/reliable state."""
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
                self._set_anchor_state(frame_id)
                self._update_template(gray_frame)
                self.initial_template = (self.last_reliable_template.copy()
                                         if self.last_reliable_template is not None else None)
                print(f"  [Scene4] Manual init: bbox={self.bbox}, center={self.center}")
                return self._init_locked_result(frame_id)

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
                            self._set_anchor_state(frame_id)
                            self._update_template(gray_frame)
                            self.initial_template = (self.last_reliable_template.copy()
                                                     if self.last_reliable_template is not None else None)
                            print(f"  [Scene4] Template init: bbox={self.bbox}, "
                                  f"center={self.center}, score={r['score']:.4f}")
                            return self._init_locked_result(frame_id)

        return None

    def _set_anchor_state(self, frame_id):
        """Record anchor/reliable state after successful init."""
        self.state = self.STATE_INIT_LOCKED
        self.init_frame_id = frame_id
        self.init_bbox = self.bbox.copy() if self.bbox else None
        self.init_center = self.center
        self.anchor_center = self.center
        self.last_reliable_center = self.center
        self.last_reliable_bbox = self.bbox.copy() if self.bbox else None
        self.fixed_box_w = self.bbox[2] if self.bbox else 0
        self.fixed_box_h = self.bbox[3] if self.bbox else 0
        self.predicted_anchor_center = self.center
        self.expected_motion_area = self.cfg.get("scene4_expected_area", 60)
        self.expected_motion_energy = None
        self.last_motion_direction = None
        self.lost_count = 0
        self.kalman_predict_count = 0
        self.hover_hold_count = 0
        self.reacquire_count = 0

    def _init_locked_result(self, frame_id):
        """Build result dict for successful init (INIT_LOCKED state)."""
        dbg = self._build_debug([], [], frame_id)
        dbg.update({
            "scene4_state": self.state,
            "scene4_reject_reason_top": "",
            "scene4_center_source": "init",
        })
        return self._make_result(
            frame_id, self.bbox, self.center, 1.0,
            True, False, False, True,
            self.state,
            "scene4_manual_init_bbox" if self.cfg.get("scene4_use_manual_init_bbox") else "scene4_template_init_roi",
            dbg)

    def _update_template(self, gray_frame):
        """Save current crop as reliable template."""
        try:
            if self.bbox is None:
                return
            work = _to_255(gray_frame)
            x, y, w, h = self.bbox
            x, y = max(0, x), max(0, y)
            w = min(w, work.shape[1] - x)
            h = min(h, work.shape[0] - y)
            if w > 5 and h > 5:
                self.last_reliable_template = work[y:y + h, x:x + w].astype(np.float32).copy()
        except Exception:
            pass

    # =========================================================================

    def _on_mouse(self, event, x, y, flags, param):
        """Mouse callback: click-drag to re-select target."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self._dragging = True
            self._drag_start = (x, y)
            self._drag_end = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self._dragging:
            self._drag_end = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self._dragging = False
            self._drag_end = (x, y)
            x1 = min(self._drag_start[0], self._drag_end[0])
            y1 = min(self._drag_start[1], self._drag_end[1])
            x2 = max(self._drag_start[0], self._drag_end[0])
            y2 = max(self._drag_start[1], self._drag_end[1])
            if x2 > x1 + 4 and y2 > y1 + 4:
                # Convert display coords back to original coords
                s = self._interactive_display_w / max(self._bgr_frame.shape[1], 1)
                ox1 = int(x1 / s)
                oy1 = int(y1 / s)
                ox2 = int(x2 / s)
                oy2 = int(y2 / s)
                self._manual_bbox = [ox1, oy1, ox2 - ox1, oy2 - oy1]
                cx = ox1 + (ox2 - ox1) // 2
                cy = oy1 + (oy2 - oy1) // 2
                self.center = (cx, cy)
                self.bbox = self._manual_bbox
                self.last_reliable_center = (cx, cy)
                self.last_reliable_bbox = self._manual_bbox
                self.predicted_anchor_center = (cx, cy)
                self.fixed_box_w = ox2 - ox1
                self.fixed_box_h = oy2 - oy1
                print(f"  [Scene4] Manual re-select: bbox={self._manual_bbox}, center=({cx},{cy})")

    def _draw_overlay(self, bgr_frame, result, frame_id):
        """Create display image with tracking overlay."""
        dh = 800
        dw = int(bgr_frame.shape[1] * dh / bgr_frame.shape[0])
        self._interactive_display_w = dw
        display = cv2.resize(bgr_frame, (dw, dh))

        if result.get("detected"):
            bb = result["bbox"]
            if bb:
                x, y, w, h = [int(v) for v in bb]
                s = dw / bgr_frame.shape[1]
                dx, dy = int(x * s), int(y * s)
                dw2, dh2 = int(w * s), int(h * s)
                cv2.rectangle(display, (dx, dy), (dx + dw2, dy + dh2), (0, 255, 255), 2)
                cv2.circle(display, (dx + dw2 // 2, dy + dh2 // 2), 5, (0, 0, 255), -1)

        if self._dragging:
            cv2.rectangle(display, self._drag_start, self._drag_end, (0, 255, 0), 2)

        state = result.get("scene4_state", "?")
        det = "DETECT" if result.get("detected") else "HOLD"
        cv2.putText(display, f"F{frame_id} {det} {state}  [SPACE:pause R:reset Q:quit]",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        if self._paused:
            cv2.putText(display, "PAUSED - drag to re-select", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        if not self._mouse_set:
            cv2.namedWindow("Scene4 Interactive")
            cv2.setMouseCallback("Scene4 Interactive", self._on_mouse)
            self._mouse_set = True
        cv2.imshow("Scene4 Interactive", display)

    def track_frame_interactive(self, gray_frame, bgr_frame, frame_id):
        """Run whatever detection mode is active, then add interactive overlay.
        Blocks video when paused. Mouse-drag re-selects target."""
        # Run normal detection (respects scene4_detection_mode)
        result = self.track_frame(gray_frame, frame_id)
        self._bgr_frame = bgr_frame

        # Show frame once, then enter pause loop if paused
        self._draw_overlay(bgr_frame, result, frame_id)

        while True:
            key = cv2.waitKey(30) & 0xFF

            # ── Check for manual bbox from mouse drag ──────────────
            if self._manual_bbox is not None:
                bb = self._manual_bbox
                cx = bb[0] + bb[2] // 2
                cy = bb[1] + bb[3] // 2
                self.center = (cx, cy)
                self.bbox = bb
                self.last_reliable_center = (cx, cy)
                self.last_reliable_bbox = bb
                self.predicted_anchor_center = (cx, cy)
                self.fixed_box_w = bb[2]
                self.fixed_box_h = bb[3]
                result["bbox"] = bb
                result["center"] = (cx, cy)
                result["detected"] = True
                result["predicted"] = False
                result["lost"] = False
                result["used_for_trajectory"] = True
                result["scene4_center_source"] = "manual_re_select"
                result["scene4_state"] = "COMPONENT_TRACK"
                result["reject_reason"] = "scene4_manual_re_select"
                self._manual_bbox = None
                self._draw_overlay(bgr_frame, result, frame_id)
                print(f"  [Scene4] Re-selected: bbox={bb}, center=({cx},{cy})")

            if key == ord(' '):
                self._paused = not self._paused
                if self._paused:
                    print(f"  [Scene4] Paused at F{frame_id} - drag to re-select, SPACE to resume")
                    self._draw_overlay(bgr_frame, result, frame_id)
                else:
                    print(f"  [Scene4] Resumed")
                    return result

            elif key == ord('r') and self._paused:
                self._manual_bbox = None
                print("  [Scene4] Reset: clear manual bbox")

            elif key == ord('q'):
                result["_user_quit"] = True
                return result

            if not self._paused:
                return result

    # =========================================================================
    #  ROI largest-component detection (simple mode)
    # =========================================================================

    def _detect_roi_largest_component(self, sf, work_full, rs, frame_id):
        """Simple detection: largest connected component in ROI around anchor.

        Returns result dict. No tracklets, no state machine.
        Only updates last_reliable when a valid component is found.
        """
        anchor = self.predicted_anchor_center or self.center or self.last_reliable_center
        if anchor is None:
            return self._make_result(frame_id, None, None, -1.0,
                                     False, True, False, False,
                                     "NO_COMPONENT_HOLD", "scene4_no_anchor")

        search_rad = self.cfg.get("scene4_roi_component_search_radius", 120)
        min_a = self.cfg.get("scene4_component_min_area", 5)
        max_a = self.cfg.get("scene4_component_max_area", 800)
        min_w = self.cfg.get("scene4_component_min_width", 2)
        max_w = self.cfg.get("scene4_component_max_width", 80)
        min_h = self.cfg.get("scene4_component_min_height", 2)
        max_h = self.cfg.get("scene4_component_max_height", 80)
        morph_k = self.cfg.get("scene4_component_morph_kernel", 3)
        diff_thr = self.cfg.get("scene4_component_diff_threshold", 8)

        # ── ROI in scaled frame ──────────────────────────────────────
        ax, ay = anchor
        rx1 = max(0, int((ax - search_rad) * rs))
        ry1 = max(0, int((ay - search_rad) * rs))
        rx2 = min(sf.shape[1], int((ax + search_rad) * rs))
        ry2 = min(sf.shape[0], int((ay + search_rad) * rs))
        if rx2 <= rx1 + 10 or ry2 <= ry1 + 10:
            return self._make_result(frame_id, None, None, -1.0,
                                     False, True, False, False,
                                     "NO_COMPONENT_HOLD", "scene4_roi_too_small")

        roi = sf[ry1:ry2, rx1:rx2]

        # ── Frame diff in ROI ────────────────────────────────────────
        if self._prev_gray is not None:
            prev_roi = self._prev_gray[ry1:ry2, rx1:rx2]
            diff = cv2.absdiff(roi, prev_roi)
        else:
            diff = np.zeros_like(roi)

        _, mask = cv2.threshold(diff, diff_thr, 255, cv2.THRESH_BINARY)
        mask = mask.astype(np.uint8)

        # Morphology
        if morph_k > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_k, morph_k))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # ── Connected components ─────────────────────────────────────
        nl, lbls, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

        dbg = {
            "scene4_component_count": nl - 1,
            "scene4_component_valid_count": 0,
            "scene4_component_best_area": 0,
            "scene4_component_best_w": 0, "scene4_component_best_h": 0,
            "scene4_component_best_x": 0, "scene4_component_best_y": 0,
            "scene4_component_reject_small_area_count": 0,
            "scene4_component_reject_large_area_count": 0,
            "scene4_component_reject_size_count": 0,
            "scene4_motion_phase": self._scene4_motion_phase(frame_id),
            "scene4_predicted_anchor_x": _safe_coord(self.predicted_anchor_center, 0),
            "scene4_predicted_anchor_y": _safe_coord(self.predicted_anchor_center, 1),
        }

        # ── Filter & pick largest ────────────────────────────────────
        best = None
        best_area = 0
        candidate_boxes_orig = []  # all valid candidates in original coords
        for i in range(1, nl):
            x, y, w, h, area = stats[i]
            if area < min_a:
                dbg["scene4_component_reject_small_area_count"] += 1
                continue
            if area > max_a:
                dbg["scene4_component_reject_large_area_count"] += 1
                continue
            if w < min_w or h < min_h or w > max_w or h > max_h:
                dbg["scene4_component_reject_size_count"] += 1
                continue
            # Store candidate box in original coords for debug drawing
            box_o = [int((rx1 + x) / rs), int((ry1 + y) / rs),
                     int(w / rs), int(h / rs)]
            candidate_boxes_orig.append(box_o)
            if area > best_area:
                best_area = area
                cx_s, cy_s = centroids[i]
                best = (i, x, y, w, h, area, cx_s, cy_s)

        # Store debug boxes for main.py to draw
        dbg["scene4_debug_candidate_boxes"] = candidate_boxes_orig
        dbg["scene4_debug_pred_bbox"] = None
        dbg["scene4_debug_best_box"] = None
        dbg["scene4_debug_roi"] = [int(rx1 / rs), int(ry1 / rs),
                                   int((rx2 - rx1) / rs), int((ry2 - ry1) / rs)]
        dbg["scene4_debug_anchor"] = list(anchor) if anchor else None

        dbg["scene4_component_valid_count"] = len(candidate_boxes_orig)
        if best:
            dbg["scene4_component_best_area"] = best_area
            dbg["scene4_component_best_w"] = best[3]
            dbg["scene4_component_best_h"] = best[4]
            dbg["scene4_component_best_x"] = best[1]
            dbg["scene4_component_best_y"] = best[2]

        # ── Build result ─────────────────────────────────────────────
        if best:
            _i, bx, by, bw, bh, barea, bcx_s, bcy_s = best
            # Convert back to original image coordinates
            cx_o = int((rx1 + bcx_s) / rs)
            cy_o = int((ry1 + bcy_s) / rs)
            bx_o = int((rx1 + bx) / rs)
            by_o = int((ry1 + by) / rs)
            bw_o = int(bw / rs)
            bh_o = int(bh / rs)

            # Use fixed bbox size around centroid
            fw, fh = self.fixed_box_w, self.fixed_box_h
            if fw <= 0:
                fw, fh = bw_o, bh_o
            bbox_o = [int(cx_o - fw // 2), int(cy_o - fh // 2), fw, fh]

            # Debug boxes for main.py drawing
            best_box_orig = [bx_o, by_o, bw_o, bh_o]
            dbg["scene4_debug_best_box"] = best_box_orig
            pred_box = [int(cx_o - fw // 2), int(cy_o - fh // 2), fw, fh]
            dbg["scene4_debug_pred_bbox"] = pred_box

            # Update state
            self.center = (cx_o, cy_o)
            self.bbox = bbox_o
            self.last_reliable_center = (cx_o, cy_o)
            self.last_reliable_bbox = bbox_o
            self.predicted_anchor_center = (cx_o, cy_o)
            self.kalman.update(cx_o, cy_o)
            self.lost_count = 0
            self.state = "COMPONENT_TRACK"

            dbg.update({
                "scene4_state": "COMPONENT_TRACK",
                "scene4_center_source": "roi_largest_component",
                "scene4_reject_reason_top": "",
            })
            return self._make_result(frame_id, bbox_o, (cx_o, cy_o),
                                     float(best_area), True, False, False, True,
                                     "COMPONENT_TRACK", "", dbg)
        else:
            # No valid component — hold at predicted_anchor
            pa = self.predicted_anchor_center or anchor
            fw, fh = self.fixed_box_w, self.fixed_box_h
            if fw <= 0:
                fw, fh = 60, 30
            hold_bbox = [int(pa[0] - fw // 2), int(pa[1] - fh // 2), fw, fh]
            self.center = pa
            self.bbox = hold_bbox
            self.state = "NO_COMPONENT_HOLD"
            dbg["scene4_debug_pred_bbox"] = list(hold_bbox)

            dbg.update({
                "scene4_state": "NO_COMPONENT_HOLD",
                "scene4_center_source": "component_hold",
                "scene4_reject_reason_top": "scene4_no_valid_component",
            })
            return self._make_result(frame_id, hold_bbox, pa, -1.0,
                                     False, True, False, False,
                                     "NO_COMPONENT_HOLD", "scene4_no_valid_component", dbg)

    # =========================================================================
    #  State machine
    # =========================================================================

    def track_frame(self, gray_frame, frame_id):
        rs = self.cfg.get("resize_scale", 0.5)
        work = _to_255(gray_frame)
        orig_h, orig_w = work.shape
        if rs < 1.0:
            sf = cv2.resize(work, (int(orig_w * rs), int(orig_h * rs)))
        else:
            sf = work

        # Store full-res gray for template scoring
        self._last_gray_full = work.copy()

        # ── Init ────────────────────────────────────────────────────
        if not self.initialized:
            ir = self._try_init(gray_frame, frame_id)
            if ir is not None:
                self._prev_gray = sf.copy()
                return ir
            return self._make_result(frame_id, None, None, -1.0,
                                     False, False, False, False,
                                     "INIT", "scene4_waiting_for_init")

        # ── Kalman predict ──────────────────────────────────────────
        kx, ky = self.kalman.predict()
        pred_cx, pred_cy = kx, ky

        # ── Detection mode dispatch ──────────────────────────────────
        detection_mode = self.cfg.get("scene4_detection_mode", "tracklet")

        if detection_mode == "nearest_motion_contour":
            # No Kalman, no phase, no predicted_anchor drift
            result = self._detect_nearest_motion_contour(gray_frame, frame_id)
            self._prev_gray = sf.copy()
            self._frame_count += 1
            self._frames_since_init = (frame_id - self.init_frame_id
                                       if self.init_frame_id >= 0 else 999)
            return result

        if detection_mode == "roi_largest_component":
            # Simple mode: largest component in ROI around anchor
            self._update_predicted_anchor(frame_id)
            result = self._detect_roi_largest_component(sf, work, rs, frame_id)
            self._prev_gray = sf.copy()
            self._frame_count += 1
            self._frames_since_init = (frame_id - self.init_frame_id
                                       if self.init_frame_id >= 0 else 999)
            return result


    # =========================================================================
    #  Motion phase detection
    # =========================================================================

    def _update_predicted_anchor(self, frame_id):
        """Set predicted_anchor to last_reliable_center (phase motion prior disabled)."""
        anchor = self.last_reliable_center or self.anchor_center
        self.predicted_anchor_center = anchor

    def _scene4_motion_phase(self, frame_id):
        """Return NONE — phase motion prior disabled."""
        return "NONE"

    def _make_result(self, frame_id, bbox, center, score,
                     detected, predicted, lost, used_for_trajectory,
                     state, reject_reason, extra_dbg=None):
        """Standardized result dict builder."""
        valid_c = (center is not None and isinstance(center, (list, tuple)) and len(center) >= 2)
        result = {
            "frame_id": frame_id,
            "bbox": bbox,
            "center": center,
            "center_x": center[0] if valid_c else "",
            "center_y": center[1] if valid_c else "",
            "score": score,
            "detected": detected,
            "predicted": predicted,
            "lost": lost,
            "used_for_trajectory": used_for_trajectory,
            "template_id": _SENTINEL,
            "scene4_state": state,
            "reject_reason": reject_reason,
        }
        if extra_dbg:
            result.update(extra_dbg)
        return result

    def _build_debug(self, scored_tracklets, raw_candidates, frame_id=0):
        """Build debug info dict from current tracklets/candidates.

        Defaults: coords → \"\", distances/scores → -1, text → \"\", counts → 0.
        Only overwritten with real values when a best tracklet exists.
        """
        best = scored_tracklets[0] if scored_tracklets else None
        dbg = {
            "scene4_state": self.state,
            "scene4_tracklet_count": len(self.motion_tracklets),
            "scene4_candidate_count": len(scored_tracklets),
            "scene4_raw_candidate_count": len(raw_candidates),
            "scene4_reject_reason_top": "",
            "scene4_center_source": "none",
            "scene4_motion_phase": self._scene4_motion_phase(frame_id),
            "scene4_predicted_anchor_x": _safe_coord(self.predicted_anchor_center, 0),
            "scene4_predicted_anchor_y": _safe_coord(self.predicted_anchor_center, 1),
        }

        # ── Defaults: coordinates → "", distances/scores → -1, text → "", counts → 0 ──
        _coord_fields = [
            "scene4_best_tracklet_start_x", "scene4_best_tracklet_start_y",
            "scene4_best_tracklet_last_x", "scene4_best_tracklet_last_y",
        ]
        _dist_score_fields = [
            "scene4_best_tracklet_score",
            "scene4_best_tracklet_area_score", "scene4_best_tracklet_energy_score",
            "scene4_best_tracklet_direction_score",
            "scene4_best_tracklet_continuity_score",
            "scene4_best_tracklet_anchor_score",
            "scene4_best_tracklet_template_score",
            "scene4_best_tracklet_net_displacement",
            "scene4_best_tracklet_dist_to_init_center",
            "scene4_best_tracklet_dist_to_anchor",
            "scene4_best_tracklet_dist_to_last_reliable",
            "scene4_best_tracklet_start_dist_to_last_reliable",
            "scene4_best_tracklet_last_dist_to_last_reliable",
            "scene4_best_tracklet_start_dist_to_init_center",
            "scene4_best_tracklet_last_dist_to_init_center",
            "scene4_best_tracklet_center_jump",
            "scene4_best_tracklet_start_dist_to_lrc",
            "scene4_best_tracklet_dx_to_last_reliable",
            "scene4_best_tracklet_dy_to_last_reliable",
            "scene4_best_tracklet_abs_dx_to_last_reliable",
            "scene4_best_tracklet_abs_dy_to_last_reliable",
            "scene4_best_tracklet_start_dx_to_last_reliable",
            "scene4_best_tracklet_start_dy_to_last_reliable",
            "scene4_best_tracklet_dist_to_predicted_anchor",
            "scene4_best_tracklet_start_dist_to_predicted_anchor",
            "scene4_best_tracklet_last_dist_to_predicted_anchor",
        ]
        _count_fields = [
            "scene4_best_tracklet_id",
            "scene4_best_tracklet_start_frame", "scene4_best_tracklet_end_frame",
            "scene4_best_tracklet_age",
        ]
        _text_fields = [
            "scene4_tracklet_accept_reason",
            "scene4_tracklet_reject_flags",
            "scene4_motion_phase",
            "scene4_phase_gate_pass",
            "scene4_phase_reject_flags",
        ]
        _center_jump_coords = [
            "scene4_center_before_x", "scene4_center_before_y",
            "scene4_center_after_x", "scene4_center_after_y",
        ]
        for k in _coord_fields + _center_jump_coords:
            dbg[k] = ""
        for k in _dist_score_fields:
            dbg[k] = -1.0
        for k in _count_fields:
            dbg[k] = 0 if k != "scene4_best_tracklet_id" else -1
        for k in _text_fields:
            dbg[k] = ""
        # scene4_tracklet_accept is 0/1 integer, default 0
        dbg["scene4_tracklet_accept"] = 0
        # center_jump is a distance, default -1
        dbg["scene4_center_jump"] = -1.0

        if best:
            trk = best["tracklet"]
            centers = trk["centers"]
            frames = trk["frames"]
            start_c = centers[0] if centers else None
            last_c = centers[-1] if centers else None
            lrc = self.last_reliable_center
            ic = self.init_center

            dbg.update({
                "scene4_best_tracklet_id": int(trk["id"]),
                "scene4_best_tracklet_start_frame": int(frames[0]) if frames else 0,
                "scene4_best_tracklet_end_frame": int(frames[-1]) if frames else 0,
                "scene4_best_tracklet_start_x": _safe_coord(start_c, 0),
                "scene4_best_tracklet_start_y": _safe_coord(start_c, 1),
                "scene4_best_tracklet_last_x": _safe_coord(last_c, 0),
                "scene4_best_tracklet_last_y": _safe_coord(last_c, 1),
                # scores
                "scene4_best_tracklet_age": trk["age"],
                "scene4_best_tracklet_score": round(best["total_score"], 4),
                "scene4_best_tracklet_area_score": round(best["area_score"], 4),
                "scene4_best_tracklet_energy_score": round(best["energy_score"], 4),
                "scene4_best_tracklet_direction_score": round(best["direction_score"], 4),
                "scene4_best_tracklet_continuity_score": round(best["continuity_score"], 4),
                "scene4_best_tracklet_anchor_score": round(best["anchor_score"], 4),
                "scene4_best_tracklet_template_score": round(best["template_score"], 4),
                "scene4_best_tracklet_net_displacement": round(best["net_displacement"], 2),
                # distance diagnostics
                "scene4_best_tracklet_dist_to_init_center": _safe_dist(last_c, ic),
                "scene4_best_tracklet_dist_to_anchor": round(best["dist_anchor"], 2),
                "scene4_best_tracklet_dist_to_last_reliable": _safe_dist(last_c, lrc),
                "scene4_best_tracklet_start_dist_to_last_reliable": _safe_dist(start_c, lrc),
                "scene4_best_tracklet_last_dist_to_last_reliable": _safe_dist(last_c, lrc),
                "scene4_best_tracklet_start_dist_to_init_center": _safe_dist(start_c, ic),
                "scene4_best_tracklet_last_dist_to_init_center": _safe_dist(last_c, ic),
                "scene4_best_tracklet_center_jump": _safe_num(best.get("center_jump"), -1.0),
                "scene4_best_tracklet_start_dist_to_lrc": _safe_num(best.get("start_dist_to_lrc"), -1.0),
                # dx/dy diagnostics
                "scene4_best_tracklet_dx_to_last_reliable": _safe_num(best.get("dx_to_lrc"), -1.0),
                "scene4_best_tracklet_dy_to_last_reliable": _safe_num(best.get("dy_to_lrc"), -1.0),
                "scene4_best_tracklet_abs_dx_to_last_reliable": abs(_safe_num(best.get("dx_to_lrc"), 0.0)) if best.get("dx_to_lrc") is not None else -1.0,
                "scene4_best_tracklet_abs_dy_to_last_reliable": abs(_safe_num(best.get("dy_to_lrc"), 0.0)) if best.get("dy_to_lrc") is not None else -1.0,
                "scene4_best_tracklet_start_dx_to_last_reliable": _safe_num(best.get("start_dx_to_lrc"), -1.0),
                "scene4_best_tracklet_start_dy_to_last_reliable": _safe_num(best.get("start_dy_to_lrc"), -1.0),
                "scene4_best_tracklet_dist_to_predicted_anchor": _safe_num(best.get("last_dist_to_pac"), -1.0),
                "scene4_best_tracklet_start_dist_to_predicted_anchor": _safe_num(best.get("start_dist_to_pac"), -1.0),
                "scene4_best_tracklet_last_dist_to_predicted_anchor": _safe_num(best.get("last_dist_to_pac"), -1.0),
                # phase gates
                "scene4_motion_phase": best.get("phase", ""),
                "scene4_phase_gate_pass": "1" if best.get("phase_pass", False) else "0",
                "scene4_phase_reject_flags": "|".join(best["phase_reject_flags"]) if best.get("phase_reject_flags") else "",
                # accept / reject
                "scene4_tracklet_accept": 1 if best["_accept"] else 0,
                "scene4_tracklet_accept_reason": "passed_tracklet_gates" if best["_accept"] else "",
                "scene4_tracklet_reject_flags": "|".join(best["_reject_flags"]) if best["_reject_flags"] else "",
            })
        return dbg
