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
    #  Frame-diff candidate extraction
    # =========================================================================

    def _extract_candidates(self, sf, work_full, rs, frame_id):
        """Extract frame-diff connected components as raw candidates.

        Returns list of candidate dicts with center/bbox in ORIGINAL image coords.
        """
        if self._prev_gray is None:
            self._prev_gray = sf.copy()
            return []

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
        excl_rois = self.cfg.get("scene4_exclude_rois", [])

        candidates = []
        for i in range(1, nl):
            x, y, w, h, area = stats[i]
            if area < min_a or area > max_a or w < min_w or w > max_w or h < min_h or h > max_h:
                continue
            cx_s, cy_s = centroids[i]
            cx_o, cy_o = int(cx_s / rs), int(cy_s / rs)

            # Exclude ROIs (in original coords)
            in_ex = False
            for er in excl_rois:
                ex, ey, ew, eh = er
                if ex <= cx_o <= ex + ew and ey <= cy_o <= ey + eh:
                    in_ex = True
                    break
            if in_ex:
                continue

            # Motion energy = sum of diff values inside the component mask
            patch_diff = diff[y:y + h, x:x + w]
            motion_energy = float(np.sum(patch_diff)) if patch_diff.size > 0 else 0.0
            mean_diff = motion_energy / max(area, 1)

            # ── Gate: candidate must be near predicted_anchor ──────
            cand_gate = self.cfg.get("scene4_candidate_gate_to_pred_anchor", 90)
            if self.predicted_anchor_center and cand_gate > 0:
                d_pa = _distance((cx_o, cy_o), self.predicted_anchor_center)
                if d_pa > cand_gate:
                    continue  # far from predicted anchor → skip

            bbox_o = [int(x / rs), int(y / rs), int(w / rs), int(h / rs)]

            candidates.append({
                "center": (cx_o, cy_o),
                "bbox": bbox_o,
                "area": int(area),
                "motion_energy": motion_energy,
                "mean_diff": mean_diff,
                "frame_id": frame_id,
            })

        self._debug_candidates = candidates
        return candidates

    # =========================================================================
    #  Motion tracklet management
    # =========================================================================

    def _update_tracklets(self, candidates, frame_id):
        """Associate new candidates to existing tracklets; create new tracklets; age/cleanup."""
        assoc_dist = self.cfg.get("scene4_tracklet_assoc_dist", 45)

        matched_tracklet_ids = set()

        for cand in candidates:
            cc = cand["center"]
            best_dist = assoc_dist + 1
            best_tid = -1
            # Find closest tracklet
            for trk in self.motion_tracklets:
                d = _distance(cc, trk["last_center"])
                if d < best_dist:
                    best_dist = d
                    best_tid = trk["id"]
            if best_tid >= 0:
                # Associate to existing tracklet
                for trk in self.motion_tracklets:
                    if trk["id"] == best_tid:
                        trk["centers"].append(cc)
                        trk["areas"].append(cand["area"])
                        trk["energies"].append(cand["motion_energy"])
                        trk["frames"].append(frame_id)
                        trk["age"] += 1
                        trk["missed"] = 0
                        trk["last_center"] = cc
                        trk["last_area"] = cand["area"]
                        trk["last_energy"] = cand["motion_energy"]
                        matched_tracklet_ids.add(best_tid)
                        break
            else:
                # New tracklet
                self._tracklet_id_counter += 1
                self.motion_tracklets.append({
                    "id": self._tracklet_id_counter,
                    "centers": [cc],
                    "areas": [cand["area"]],
                    "energies": [cand["motion_energy"]],
                    "frames": [frame_id],
                    "age": 1,
                    "missed": 0,
                    "last_center": cc,
                    "last_area": cand["area"],
                    "last_energy": cand["motion_energy"],
                })

        # Mark unmatched tracklets as missed
        max_missed = self.cfg.get("scene4_tracklet_max_missed", 3)
        for trk in self.motion_tracklets:
            if trk["id"] not in matched_tracklet_ids:
                trk["missed"] += 1
                trk["age"] += 1  # age keeps ticking for continuity

        # Remove dead tracklets
        self.motion_tracklets = [
            trk for trk in self.motion_tracklets
            if trk["missed"] <= max_missed
        ]

        # ── Prune: remove far-away tracklets and limit total count ──
        self._prune_tracklets()

        self._debug_tracklets = list(self.motion_tracklets)

    def _prune_tracklets(self):
        """Remove tracklets far from anchor and limit total count."""
        anchor = self.last_reliable_center or self.anchor_center
        if anchor is None:
            return
        prune_radius = self.cfg.get("scene4_tracklet_prune_by_anchor_radius", 220)
        max_tracklets = self.cfg.get("scene4_max_tracklets", 80)

        # Drop tracklets whose last_center is too far from anchor
        self.motion_tracklets = [
            trk for trk in self.motion_tracklets
            if _distance(trk["last_center"], anchor) <= prune_radius
        ]

        # If still too many, keep the top N by age (oldest = most consistent)
        if len(self.motion_tracklets) > max_tracklets:
            self.motion_tracklets.sort(key=lambda trk: trk["age"], reverse=True)
            self.motion_tracklets = self.motion_tracklets[:max_tracklets]

    # =========================================================================
    #  Tracklet scoring
    # =========================================================================

    def _score_tracklets(self, frame_id):
        """Score all eligible tracklets (age >= min_age). Returns sorted list (best first)."""
        min_age = self.cfg.get("scene4_tracklet_min_age", 3)
        window = self.cfg.get("scene4_tracklet_window", 8)

        scored = []
        for trk in self.motion_tracklets:
            if trk["age"] < min_age:
                continue

            # ── 1. Area score ──────────────────────────────────────
            recent_areas = trk["areas"][-window:]
            mean_area = np.mean(recent_areas)
            exp_area = self.expected_motion_area
            area_tol = self.cfg.get("scene4_area_tolerance", 80)
            area_score = max(0.0, 1.0 - abs(mean_area - exp_area) / max(area_tol, 1))

            # ── 2. Energy score ────────────────────────────────────
            recent_energies = trk["energies"][-window:]
            mean_energy = np.mean(recent_energies)
            if self.expected_motion_energy is not None and self.expected_motion_energy > 0:
                energy_score = max(0.0, 1.0 - abs(
                    mean_energy - self.expected_motion_energy) / max(self.expected_motion_energy, 1.0))
            else:
                energy_norm = self.cfg.get("scene4_energy_norm", 500.0)
                energy_score = min(1.0, mean_energy / max(energy_norm, 1.0))

            # ── 3. Direction score ─────────────────────────────────
            centers = trk["centers"][-window:]
            if len(centers) >= 3:
                net_disp = _distance(centers[-1], centers[0])
                path_len = sum(
                    _distance(centers[i], centers[i - 1]) for i in range(1, len(centers)))
                direction_score = net_disp / max(path_len, 1e-6)
            else:
                direction_score = 0.0

            # ── 4. Continuity score ─────────────────────────────────
            continuity_score = min(1.0, trk["age"] / max(min_age, 1))

            # ── 5. Anchor score ─────────────────────────────────────
            anchor = self.last_reliable_center or self.anchor_center
            if anchor is not None:
                dist_anchor = _distance(trk["last_center"], anchor)
                reacq_radius = self.cfg.get("scene4_reacquire_radius", 220)
                anchor_score = max(0.0, 1.0 - dist_anchor / max(reacq_radius, 1))
            else:
                dist_anchor = 0.0
                anchor_score = 1.0

            # ── 6. Template score ───────────────────────────────────
            template_score = self._tracklet_template_score(trk)

            # ── Net displacement ────────────────────────────────────
            net_displacement = _distance(trk["centers"][-1], trk["centers"][0]) if len(trk["centers"]) >= 2 else 0.0

            # ── Total score ─────────────────────────────────────────
            total_score = (
                0.20 * area_score
                + 0.15 * energy_score
                + 0.25 * direction_score
                + 0.15 * continuity_score
                + 0.10 * anchor_score
                + 0.15 * template_score
            )

            # ── Hard acceptance check ───────────────────────────────
            min_dir = self.cfg.get("scene4_tracklet_min_direction_score", 0.65)
            min_area = self.cfg.get("scene4_tracklet_min_area_score", 0.45)
            min_energy = self.cfg.get("scene4_tracklet_min_energy_score", 0.35)
            min_tmpl = self.cfg.get("scene4_tracklet_min_template_score", 0.25)
            min_total = self.cfg.get("scene4_tracklet_min_total_score", 0.55)
            min_net_disp = self.cfg.get("scene4_tracklet_min_net_displacement", 10)
            reacq_radius = self.cfg.get("scene4_reacquire_radius", 220)

            # Detailed rejection tracking
            reject_flags = []
            phase = self._scene4_motion_phase(frame_id)  # compute early for phase-adaptive gates
            if trk["age"] < min_age:
                reject_flags.append("age")
            # Phase-adaptive direction threshold (hover = little movement expected)
            _min_dir = min_dir
            if phase == "HOVER":
                _min_dir = 0.20  # hovering drone has weak direction
            if direction_score < _min_dir:
                reject_flags.append(f"dir({direction_score:.2f}<{_min_dir})")
            if area_score < min_area:
                reject_flags.append(f"area({area_score:.2f}<{min_area})")
            if energy_score < min_energy:
                reject_flags.append(f"energy({energy_score:.2f}<{min_energy})")
            if template_score < min_tmpl:
                reject_flags.append(f"tmpl({template_score:.2f}<{min_tmpl})")
            if total_score < min_total:
                reject_flags.append(f"total({total_score:.2f}<{min_total})")
            if net_displacement < min_net_disp:
                reject_flags.append(f"netdisp({net_displacement:.1f}<{min_net_disp})")
            if dist_anchor > reacq_radius:
                reject_flags.append(f"dist({dist_anchor:.0f}>{reacq_radius})")

            # ── Hard spatial + phase gates (use predicted_anchor as primary reference) ──
            lrc = self.last_reliable_center
            pac = self.predicted_anchor_center
            start_c = centers[0] if len(centers) > 0 else None
            last_c_from_centers = centers[-1] if len(centers) > 0 else None
            start_dist_to_lrc = -1.0
            start_dist_to_pac = -1.0
            last_dist_to_pac = -1.0
            center_jump = -1.0
            dx_to_lrc = 0.0
            dy_to_lrc = 0.0
            start_dx_to_lrc = 0.0
            start_dy_to_lrc = 0.0

            # Compute dx/dy from last_reliable_center (for debug/reference)
            if lrc and last_c_from_centers:
                dx_to_lrc = last_c_from_centers[0] - lrc[0]
                dy_to_lrc = last_c_from_centers[1] - lrc[1]
                center_jump = _distance(last_c_from_centers, lrc)
            if lrc and start_c:
                start_dx_to_lrc = start_c[0] - lrc[0]
                start_dy_to_lrc = start_c[1] - lrc[1]
                start_dist_to_lrc = _distance(start_c, lrc)

            # Compute distances to predicted_anchor (primary gate reference)
            if pac and last_c_from_centers:
                last_dist_to_pac = _distance(last_c_from_centers, pac)
            if pac and start_c:
                start_dist_to_pac = _distance(start_c, pac)

            # ── Primary gate: tracklet must be near predicted_anchor ──
            trk_gate = self.cfg.get("scene4_tracklet_gate_to_pred_anchor", 90)
            stabilize_frames = self.cfg.get("scene4_stabilize_frames", 30)
            frames_since_init = frame_id - self.init_frame_id if self.init_frame_id >= 0 else 999
            if frames_since_init <= stabilize_frames:
                trk_gate = self.cfg.get("scene4_stabilize_tracklet_gate_to_pred_anchor", 60)

            if last_dist_to_pac > trk_gate:
                reject_flags.append(
                    f"tracklet_far_from_predicted_anchor({last_dist_to_pac:.0f}>{trk_gate})")

            # Tracklet start must be near predicted_anchor too
            max_start_dist_pa = self.cfg.get("scene4_tracklet_max_start_dist_to_pred_anchor", 70)
            if start_dist_to_pac > max_start_dist_pa:
                reject_flags.append(
                    f"tracklet_start_not_near_predicted_anchor({start_dist_to_pac:.0f}>{max_start_dist_pa})")

            # ── Phase-aware motion gates (relative to predicted_anchor) ──
            # phase already computed above
            phase_pass = True
            phase_reject = []

            # Use dx/dy relative to predicted_anchor (pac)
            dx_pa = (last_c_from_centers[0] - pac[0]) if (pac and last_c_from_centers) else 999
            dy_pa = (last_c_from_centers[1] - pac[1]) if (pac and last_c_from_centers) else 999

            # Proximity bonus: when tracklet is very close to predicted_anchor,
            # relax phase direction gates (strong positional signal outweighs weak motion check)
            close_to_pa = (last_dist_to_pac > 0 and last_dist_to_pac <= 35)
            very_close_to_pa = (last_dist_to_pac > 0 and last_dist_to_pac <= 20)

            if phase == "RISE":
                stabilize_max_dx = self.cfg.get("scene4_stabilize_max_dx", 35)
                if abs(dx_pa) > stabilize_max_dx:
                    phase_pass = False
                    phase_reject.append(f"rise_x_jump_too_large({abs(dx_pa):.0f}>{stabilize_max_dx})")
                # Relax vertical direction gate when close to predicted_anchor
                dy_allow = 10
                if close_to_pa:
                    dy_allow = 30
                if very_close_to_pa:
                    dy_allow = 60  # very close → almost any vertical motion OK
                if dy_pa > dy_allow:
                    phase_pass = False
                    phase_reject.append(f"rise_wrong_vertical_direction(dy={dy_pa:.0f}>{dy_allow})")

            elif phase == "HOVER":
                hover_max_dx = self.cfg.get("scene4_hover_max_dx", 12)
                hover_max_dy = self.cfg.get("scene4_hover_max_dy", 12)
                if close_to_pa:
                    hover_max_dx = 25
                    hover_max_dy = 25
                if very_close_to_pa:
                    hover_max_dx = 40
                    hover_max_dy = 40
                if abs(dx_pa) > hover_max_dx or abs(dy_pa) > hover_max_dy:
                    phase_pass = False
                    phase_reject.append(
                        f"hover_motion_too_large(dx={abs(dx_pa):.0f}>{hover_max_dx},dy={abs(dy_pa):.0f}>{hover_max_dy})")

            elif phase == "DESCEND":
                max_dx_frame = self.cfg.get("scene4_max_dx_per_frame", 25)
                if abs(dx_pa) > max_dx_frame:
                    phase_pass = False
                    phase_reject.append(f"descend_x_jump_too_large({abs(dx_pa):.0f}>{max_dx_frame})")
                dy_allow = -10
                if close_to_pa:
                    dy_allow = -30
                if very_close_to_pa:
                    dy_allow = -60
                if dy_pa < dy_allow:
                    phase_pass = False
                    phase_reject.append(f"descend_wrong_vertical_direction(dy={dy_pa:.0f}<{dy_allow})")

            if not phase_pass:
                reject_flags.extend(phase_reject)

            # ── Euclidean dist_anchor (to last_reliable, broad safety filter) ──
            if frames_since_init <= stabilize_frames:
                max_dist_anchor = self.cfg.get("scene4_stabilize_max_dist_anchor", 45)
            else:
                max_dist_anchor = self.cfg.get("scene4_tracklet_max_dist_anchor_after_stable", 90)
            if dist_anchor > max_dist_anchor:
                reject_flags.append(f"too_far_from_anchor({dist_anchor:.0f}>{max_dist_anchor})")

            accept = len(reject_flags) == 0

            scored.append({
                "tracklet": trk,
                "area_score": area_score,
                "energy_score": energy_score,
                "direction_score": direction_score,
                "continuity_score": continuity_score,
                "anchor_score": anchor_score,
                "template_score": template_score,
                "total_score": total_score,
                "net_displacement": net_displacement,
                "dist_anchor": dist_anchor,
                "mean_area": mean_area,
                "mean_energy": mean_energy,
                "center_jump": center_jump,
                "start_dist_to_lrc": start_dist_to_lrc,
                "last_dist_to_pac": last_dist_to_pac,
                "start_dist_to_pac": start_dist_to_pac,
                "dx_to_lrc": dx_to_lrc,
                "dy_to_lrc": dy_to_lrc,
                "start_dx_to_lrc": start_dx_to_lrc,
                "start_dy_to_lrc": start_dy_to_lrc,
                "phase": phase,
                "phase_pass": phase_pass,
                "phase_reject_flags": phase_reject,
                "_accept": accept,
                "_reject_flags": reject_flags,
            })

        scored.sort(key=lambda x: x["total_score"], reverse=True)
        return scored

    def _tracklet_template_score(self, trk):
        """Compute NCC template score at tracklet's last_center."""
        if not self.cfg.get("scene4_use_template_verify", True):
            return 0.5
        self._load_templates()
        if not self._tmpl_list:
            return 0.5
        try:
            # We need a gray frame — use the stored last_reliable_template's shape
            # as reference. We can't access gray_frame here, so use a simpler approach:
            # Score against initial_template if available, using the available templates.
            from .ncc import ncc_score
            best = -1.0
            # We need the actual frame to crop. Store last gray frame.
            if not hasattr(self, '_last_gray_full') or self._last_gray_full is None:
                return 0.4
            work = self._last_gray_full
            cx, cy = trk["last_center"]
            fw, fh = self.fixed_box_w, self.fixed_box_h
            if fw <= 0 or fh <= 0:
                return 0.4
            px = max(0, int(cx - fw // 2) - 4)
            py = max(0, int(cy - fh // 2) - 4)
            pw = min(fw + 8, work.shape[1] - px)
            ph = min(fh + 8, work.shape[0] - py)
            if pw < 10 or ph < 10:
                return 0.3
            patch = work[py:py + ph, px:px + pw]
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

    # =========================================================================
    # =========================================================================
    #  Nearest motion contour detection
    # =========================================================================

    def _detect_nearest_motion_contour(self, gray_frame, frame_id):
        """Frame-diff → threshold → morph → contours → nearest to prev center.

        Works on FULL-RES frame (matching classmate's video4.py algorithm).
        No Kalman, no phase prior, no coordinate scaling confusion.
        On failure: hold at last position, don't extrapolate.
        """
        blur_k = self.cfg.get("scene4_blur_kernel", 5)
        if blur_k % 2 == 0:
            blur_k += 1
        morph_k = self.cfg.get("scene4_morph_kernel", 5)
        if morph_k % 2 == 0:
            morph_k += 1
        diff_thr = self.cfg.get("scene4_diff_threshold", 30)
        min_area = self.cfg.get("scene4_min_motion_area", 100)
        search_win = self.cfg.get("scene4_search_window", 150)
        smooth = self.cfg.get("scene4_smooth_factor", 0.7)

        prev_center = self.last_reliable_center or self.center
        fw, fh = self.fixed_box_w, self.fixed_box_h
        if fw <= 0:
            fw, fh = 60, 30

        dbg = {
            "scene4_nearest_contour_count": 0,
            "scene4_nearest_contour_valid_count": 0,
            "scene4_nearest_contour_best_area": 0,
            "scene4_nearest_contour_best_dist": 0,
            "scene4_nearest_contour_search_window": search_win,
            "scene4_nearest_contour_selected_by": "none",
            "scene4_motion_phase": "NONE",
            "scene4_predicted_anchor_x": _safe_coord(prev_center, 0),
            "scene4_predicted_anchor_y": _safe_coord(prev_center, 1),
        }

        # ── Full-res grayscale + blur (matching classmate's video4.py) ──
        gray = _to_255(gray_frame)
        gray = cv2.GaussianBlur(gray, (blur_k, blur_k), 0)

        if self.scene4_prev_gray is None:
            self.scene4_prev_gray = gray.copy()
            return self._make_result(frame_id, self.bbox, self.center, -1.0,
                                     False, True, False, False,
                                     "NO_MOTION_HOLD", "scene4_no_prev_frame", dbg)

        # ── Frame diff + threshold ──────────────────────────────────
        diff = cv2.absdiff(gray, self.scene4_prev_gray)
        _, mask = cv2.threshold(diff, diff_thr, 255, cv2.THRESH_BINARY)
        mask = mask.astype(np.uint8)
        self.scene4_prev_gray = gray.copy()

        # ── Morphology: open then close ─────────────────────────────
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_k, morph_k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # ── Find contours (all in full-res coords, no scaling) ──────
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # ── Filter by area ──────────────────────────────────────────
        valid = []
        candidate_boxes = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            bx, by, bw, bh = cv2.boundingRect(cnt)
            # All coords already in full-res original frame
            valid.append({"center": (cx, cy), "area": area,
                          "bbox": [bx, by, bw, bh]})
            candidate_boxes.append([bx, by, bw, bh])

        dbg["scene4_nearest_contour_count"] = len(valid)
        dbg["scene4_debug_candidate_boxes"] = candidate_boxes

        # ── Selection: nearest to prev_center, fallback to largest (matching video4.py) ──
        best = None
        if prev_center and valid:
            # Try nearest within search window
            best_dist = search_win
            for c in valid:
                d = _distance(c["center"], prev_center)
                if d < best_dist:
                    best_dist = d
                    best = c
            if best:
                dbg["scene4_nearest_contour_selected_by"] = "nearest"
                dbg["scene4_nearest_contour_best_dist"] = round(best_dist, 1)
        if best is None and valid:
            # Fallback: largest contour (reacquisition after lost / init)
            best = max(valid, key=lambda c: c["area"])
            dbg["scene4_nearest_contour_selected_by"] = "largest_fallback" if prev_center else "largest_init"

        dbg["scene4_nearest_contour_valid_count"] = 1 if best else 0
        dbg["scene4_debug_best_box"] = best["bbox"] if best else None
        dbg["scene4_debug_pred_bbox"] = ([int(prev_center[0] - fw // 2),
                                           int(prev_center[1] - fh // 2), fw, fh]
                                          if prev_center else None)
        dbg["scene4_debug_roi"] = ([int(prev_center[0] - search_win),
                                     int(prev_center[1] - search_win),
                                     search_win * 2, search_win * 2]
                                    if prev_center else None)
        dbg["scene4_debug_anchor"] = list(prev_center) if prev_center else None

        # ── Build result ─────────────────────────────────────────────
        if best:
            dbg["scene4_nearest_contour_best_area"] = best["area"]
            detected_center = best["center"]
            # Smooth (matching video4.py: old*0.7 + new*0.3)
            if prev_center:
                new_x = int(prev_center[0] * smooth + detected_center[0] * (1 - smooth))
                new_y = int(prev_center[1] * smooth + detected_center[1] * (1 - smooth))
                new_center = (new_x, new_y)
            else:
                new_center = detected_center

            bbox_o = [int(new_center[0] - fw // 2), int(new_center[1] - fh // 2), fw, fh]

            self.center = new_center
            self.bbox = bbox_o
            self.last_reliable_center = new_center
            self.last_reliable_bbox = bbox_o
            self.predicted_anchor_center = new_center
            self.lost_count = 0
            self.state = "CONTOUR_TRACK"

            dbg.update({"scene4_state": "CONTOUR_TRACK",
                        "scene4_center_source": "nearest_motion_contour",
                        "scene4_reject_reason_top": ""})
            return self._make_result(frame_id, bbox_o, new_center, float(best["area"]),
                                     True, False, False, True,
                                     "CONTOUR_TRACK", "", dbg)
        else:
            # Hold: no valid contour → keep last position, no extrapolation
            hold_center = prev_center or (self.bbox[0] + self.bbox[2] // 2,
                                          self.bbox[1] + self.bbox[3] // 2) if self.bbox else (0, 0)
            hold_bbox = self.bbox or [0, 0, fw, fh]
            self.center = hold_center
            self.state = "NO_MOTION_HOLD"

            dbg.update({"scene4_state": "NO_MOTION_HOLD",
                        "scene4_center_source": "nearest_contour_hold",
                        "scene4_reject_reason_top": "scene4_no_motion_contour"})
            return self._make_result(frame_id, hold_bbox, hold_center, 0.0,
                                     False, False, True, False,
                                     "NO_MOTION_HOLD", "scene4_no_motion_contour", dbg)

    # =========================================================================
    #  Interactive mode: mouse correction + nearest motion contour
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

    def _interactive_display(self, bgr_frame, result, frame_id):
        """Show frame; if paused, block until user unpauses.
        Returns False if user quit, True to continue."""
        self._bgr_frame = bgr_frame
        self._draw_overlay(bgr_frame, result, frame_id)

        while True:
            key = cv2.waitKey(30) & 0xFF

            if key == ord(' '):
                self._paused = not self._paused
                if not self._paused:
                    self._cached_result = None
                    print(f"  [Scene4] Resumed")
                    return True  # resume processing
                else:
                    print(f"  [Scene4] Paused - drag to re-select, SPACE to resume")
                    self._draw_overlay(bgr_frame, result, frame_id)  # update PAUSED text
            elif key == ord('r'):
                self._manual_bbox = None
                print("  [Scene4] Reset: clear manual bbox")
            elif key == ord('q'):
                return False  # quit

            # If not paused, return immediately so video advances
            if not self._paused:
                return True

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

        # ── Tracklet mode (original) ────────────────────────────────
        # ── Update predicted anchor (moves with phase) ───────────────
        self._update_predicted_anchor(frame_id)

        # ── Extract frame-diff candidates (filtered by predicted_anchor) ──
        raw_candidates = self._extract_candidates(sf, work, rs, frame_id)

        # ── Update tracklets ────────────────────────────────────────
        self._update_tracklets(raw_candidates, frame_id)

        # ── Score tracklets ─────────────────────────────────────────
        scored_tracklets = self._score_tracklets(frame_id)

        # ── Update counters ─────────────────────────────────────────
        self._frames_since_init = (frame_id - self.init_frame_id
                                   if self.init_frame_id >= 0 else 999)

        # ── State machine decision ──────────────────────────────────
        result = self._run_state_machine(frame_id, pred_cx, pred_cy,
                                         raw_candidates, scored_tracklets, rs, work)

        self._prev_gray = sf.copy()
        self._frame_count += 1
        return result

    def _run_state_machine(self, frame_id, pred_cx, pred_cy,
                           raw_candidates, scored_tracklets, rs, work_full):
        """Core state machine. Returns result dict.

        Key rules:
        - Noise tracklets far from anchor do NOT trigger REACQUIRE → HOVER_HOLD instead.
        - Only tracklets within reacquire_radius of anchor can trigger REACQUIRE.
        - HOVER_HOLD keeps last_reliable_center, not Kalman prediction.
        - Spatial pre-filter: in anchored states, only tracklets near anchor are considered.
        """

        init_locked_frames = self.cfg.get("scene4_init_locked_frames", 30)
        init_lock_radius = self.cfg.get("scene4_init_lock_radius", 80)
        reacq_radius = self.cfg.get("scene4_reacquire_radius", 220)
        anchor = self.last_reliable_center or self.anchor_center

        # ══ Spatial pre-filter: in all tracking states, only consider near-anchor tracklets ══
        if anchor and self.state in (self.STATE_INIT_LOCKED, self.STATE_TRACKING,
                                      self.STATE_HOVER_HOLD, self.STATE_REACQUIRE):
            radius = init_lock_radius if self.state == self.STATE_INIT_LOCKED else reacq_radius
            scored_tracklets = [s for s in scored_tracklets
                                if s.get("dist_anchor", 9999) <= radius]

        best = scored_tracklets[0] if scored_tracklets else None
        bt_accepts = best["_accept"] if best else False
        best_tracklet = best["tracklet"] if best else None

        # Build base debug info
        dbg = self._build_debug(scored_tracklets, raw_candidates, frame_id)

        # ── Accept tracklet ─────────────────────────────────────────
        if bt_accepts and best_tracklet:
            return self._accept_tracklet_result(best, best_tracklet, frame_id, dbg)

        # ── Phase-guided fallback (replaces free Kalman drift) ──────
        if self._scene4_motion_phase(frame_id) != "NONE":
            return self._phase_hold_result(frame_id, dbg)

        # ── No acceptable tracklet — state-dependent fallback ───────
        has_candidates = len(raw_candidates) > 0

        # Determine if ANY tracklet (even non-accepted) is near anchor
        any_near_anchor = False
        best_near_dist = 9999.0
        if anchor and scored_tracklets:
            for s in scored_tracklets:
                d = s.get("dist_anchor", 9999)
                if d < best_near_dist:
                    best_near_dist = d
                if d <= reacq_radius:
                    any_near_anchor = True

        # Noise gate: candidates that are ALL far from anchor → treat as no candidates
        has_relevant_candidates = has_candidates and any_near_anchor
        dbg["scene4_has_relevant_candidates"] = 1 if has_relevant_candidates else 0
        dbg["scene4_best_near_dist"] = round(best_near_dist, 1) if best_near_dist < 9998 else -1

        # ── Hover check ─────────────────────────────────────────────
        def _do_hover():
            """Try hover NCC, fallback to hover anchor position."""
            self._transition_to(self.STATE_HOVER_HOLD)
            self.hover_hold_count += 1
            hover_r = self._try_hover_ncc(frame_id, pred_cx, pred_cy, rs, work_full)
            if hover_r:
                hover_r.update(dbg)
                return hover_r
            return self._hover_kalman_result(frame_id, pred_cx, pred_cy, dbg)

        # ── Reacquire check ─────────────────────────────────────────
        def _do_reacquire():
            """Enter reacquire: kalman predict at anchor, waiting for reliable tracklet."""
            self._transition_to(self.STATE_REACQUIRE)
            self.reacquire_count += 1
            return self._handle_reacquire_or_kalman(frame_id, pred_cx, pred_cy, dbg)

        if self.state in (self.STATE_INIT_LOCKED, self.STATE_TRACKING):
            if not has_relevant_candidates:
                return _do_hover()
            else:
                return _do_reacquire()

        elif self.state == self.STATE_HOVER_HOLD:
            if not has_relevant_candidates:
                self.hover_hold_count += 1
                hover_r = self._try_hover_ncc(frame_id, pred_cx, pred_cy, rs, work_full)
                if hover_r:
                    hover_r.update(dbg)
                    return hover_r
                return self._hover_kalman_result(frame_id, pred_cx, pred_cy, dbg)
            else:
                return _do_reacquire()

        elif self.state == self.STATE_REACQUIRE:
            if not has_relevant_candidates:
                # Lost the promising tracklet → back to hover
                return _do_hover()
            self.reacquire_count += 1
            return self._handle_reacquire_or_kalman(frame_id, pred_cx, pred_cy, dbg)

        elif self.state == self.STATE_KALMAN_PREDICT:
            self.kalman_predict_count += 1
            max_lost = self.cfg.get("scene4_max_lost", 20)
            if self.kalman_predict_count > max_lost:
                return self._make_lost_result(frame_id, dbg)
            if has_relevant_candidates:
                self._transition_to(self.STATE_REACQUIRE)
                self.reacquire_count = 0
            return self._kalman_result(frame_id, pred_cx, pred_cy, dbg)

        elif self.state == self.STATE_LOST:
            if has_relevant_candidates:
                self._transition_to(self.STATE_REACQUIRE)
                self.reacquire_count = 0
                return self._kalman_result(frame_id, pred_cx, pred_cy, dbg)
            return self._make_lost_result(frame_id, dbg)

        # Fallback — keep at anchor
        return self._hover_kalman_result(frame_id, pred_cx, pred_cy, dbg)

    def _transition_to(self, new_state):
        """Transition state and reset relevant counters."""
        if self.state == new_state:
            return
        self.state = new_state
        if new_state == self.STATE_HOVER_HOLD:
            self.hover_hold_count = 0
        elif new_state == self.STATE_REACQUIRE:
            self.reacquire_count = 0
        elif new_state == self.STATE_KALMAN_PREDICT:
            self.kalman_predict_count = 0
        elif new_state == self.STATE_PHASE_HOLD:
            pass

    # =========================================================================
    #  Motion phase detection
    # =========================================================================

    def _update_predicted_anchor(self, frame_id):
        """Update predicted_anchor_center based on phase prior or last accept.

        - After a tracklet accept: predicted_anchor = last_reliable_center.
        - RISE phase: predicted_anchor moves up each frame from its previous position.
        - HOVER phase: predicted_anchor = last_reliable_center.
        - DESCEND phase: predicted_anchor moves down each frame from its previous position.
        - Otherwise: stays at last_reliable_center.

        This is the current search center, NOT the last confirmed position.
        """
        anchor = self.last_reliable_center or self.anchor_center
        if anchor is None:
            self.predicted_anchor_center = self.center
            return

        phase = self._scene4_motion_phase(frame_id)
        prev_pa = self.predicted_anchor_center

        if phase == "RISE":
            step_y = self.cfg.get("scene4_rise_hold_step_y", -3)
            max_dy = self.cfg.get("scene4_stabilize_max_dy", 45)
            if prev_pa:
                new_y = prev_pa[1] + step_y
            else:
                new_y = anchor[1] + step_y
            new_y = max(new_y, anchor[1] - max_dy)
            self.predicted_anchor_center = (anchor[0], int(new_y))

        elif phase == "DESCEND":
            step_y = self.cfg.get("scene4_descend_hold_step_y", 3)
            descend_max = self.cfg.get("scene4_descend_max_downward_dy_from_anchor", 180)
            if prev_pa:
                new_y = prev_pa[1] + step_y
            else:
                new_y = anchor[1] + step_y
            new_y = min(new_y, anchor[1] + descend_max)
            self.predicted_anchor_center = (anchor[0], int(new_y))

        else:  # HOVER / INIT / NONE
            self.predicted_anchor_center = anchor

    def _scene4_motion_phase(self, frame_id):
        """Return RISE / HOVER / DESCEND / INIT based on known drone motion."""
        if not self.cfg.get("scene4_use_phase_motion_prior", False):
            return "NONE"
        rise_start, rise_end = self.cfg.get("scene4_rise_frame_range", [1, 15])
        hover_start, hover_end = self.cfg.get("scene4_hover_frame_range", [16, 31])
        descend_start = self.cfg.get("scene4_descend_start_frame", 32)
        if rise_start <= frame_id <= rise_end:
            return "RISE"
        if hover_start <= frame_id <= hover_end:
            return "HOVER"
        if frame_id >= descend_start:
            return "DESCEND"
        return "INIT"

    def _phase_hold_result(self, frame_id, dbg):
        """Phase-guided fallback prediction — does NOT update last_reliable.

        Accumulates from current center each frame (not fixed anchor offset),
        so the predicted position progressively moves in the expected direction.
        Bounded by phase max_dy from anchor to prevent runaway.
        """
        phase = self._scene4_motion_phase(frame_id)
        anchor = self.last_reliable_center or self.anchor_center
        if anchor is None:
            return self._kalman_result(frame_id, 0, 0, dbg)

        lx, ly = anchor
        # Base from current center (progressive), fallback to anchor
        cur_cx, cur_cy = self.center if self.center else anchor
        fw, fh = self.fixed_box_w, self.fixed_box_h
        if fw <= 0:
            fw, fh = 60, 30

        if phase == "RISE":
            step_y = self.cfg.get("scene4_rise_hold_step_y", -3)
            max_dy = self.cfg.get("scene4_stabilize_max_dy", 45)
            cx = lx  # keep x at anchor
            cy = cur_cy + step_y
            # Clamp: don't go beyond max_dy upward from anchor
            if cy < ly - max_dy:
                cy = ly - max_dy
            source = "rise_phase_hold"
        elif phase == "DESCEND":
            step_y = self.cfg.get("scene4_descend_hold_step_y", 3)
            max_dy = self.cfg.get("scene4_max_dy_per_frame", 35)
            cx = lx
            cy = cur_cy + step_y
            # Clamp: don't go beyond reasonable distance from anchor
            descend_max = self.cfg.get("scene4_descend_max_downward_dy_from_anchor", 180)
            if cy > ly + descend_max * 0.5:  # conservative: half of max
                cy = ly + descend_max * 0.5
            source = "descend_phase_hold"
        else:  # HOVER or INIT or NONE
            cx, cy = lx, ly
            source = "hover_hold_last_reliable"

        bbox = [int(cx - fw // 2), int(cy - fh // 2), fw, fh]
        self.center = (cx, cy)
        self.bbox = bbox
        self.predicted_anchor_center = (cx, cy)  # keep in sync
        self.state = self.STATE_PHASE_HOLD

        dbg.update({
            "scene4_state": "PHASE_HOLD",
            "scene4_motion_phase": phase,
            "scene4_center_source": source,
            "scene4_reject_reason_top": f"scene4_phase_hold_{phase.lower()}",
        })
        return self._make_result(frame_id, bbox, (cx, cy), -1.0,
                                 False, True, False, False,
                                 "PHASE_HOLD",
                                 f"scene4_phase_hold_{phase.lower()}", dbg)

    def _handle_reacquire_or_kalman(self, frame_id, pred_cx, pred_cy, dbg):
        """During reacquire: use Kalman prediction at anchor.
        If reacquire takes too long → fall back to HOVER_HOLD (anchor position).
        """
        max_reacq = self.cfg.get("scene4_max_reacquire_frames", 15)
        if self.reacquire_count > max_reacq:
            # Too long without reliable tracklet → return to hover at anchor
            self._transition_to(self.STATE_HOVER_HOLD)
            self.hover_hold_count += 1
            return self._hover_kalman_result(frame_id, pred_cx, pred_cy, dbg)
        # Still in reacquire window: use Kalman prediction (short-term)
        return self._kalman_result(frame_id, pred_cx, pred_cy, dbg)

    # =========================================================================
    #  Result builders
    # =========================================================================

    def _accept_tracklet_result(self, scored, tracklet, frame_id, dbg):
        """Accept a reliable tracklet: update center, bbox, Kalman, anchor, motion stats."""
        new_center = tracklet["last_center"]

        # ── Center jump diagnostics (before updating) ───────────────
        before_center = self.center or self.last_reliable_center
        center_jump = _safe_dist(before_center, new_center)
        dbg["scene4_center_before_x"] = _safe_coord(before_center, 0)
        dbg["scene4_center_before_y"] = _safe_coord(before_center, 1)
        dbg["scene4_center_after_x"] = _safe_coord(new_center, 0)
        dbg["scene4_center_after_y"] = _safe_coord(new_center, 1)
        dbg["scene4_center_jump"] = center_jump
        dbg["scene4_tracklet_accept_reason"] = "passed_tracklet_gates"


        # Fixed bbox around new center
        fw, fh = self.fixed_box_w, self.fixed_box_h
        if fw <= 0 or fh <= 0:
            fw, fh = 60, 30  # fallback
        new_bbox = [int(new_center[0] - fw // 2), int(new_center[1] - fh // 2), fw, fh]

        # Update core state
        self.center = new_center
        self.bbox = new_bbox
        self.kalman.update(new_center[0], new_center[1])
        self.last_reliable_center = new_center
        self.last_reliable_bbox = new_bbox
        self.predicted_anchor_center = new_center  # reset to accepted position

        # Update motion statistics (EMA)
        self.expected_motion_area = (
            0.8 * self.expected_motion_area + 0.2 * scored["mean_area"]
        )
        if self.expected_motion_energy is None:
            self.expected_motion_energy = scored["mean_energy"]
        else:
            self.expected_motion_energy = (
                0.8 * self.expected_motion_energy + 0.2 * scored["mean_energy"]
            )

        # Update direction
        centers = tracklet["centers"]
        if len(centers) >= 2:
            dx = centers[-1][0] - centers[0][0]
            dy = centers[-1][1] - centers[0][1]
            norm = np.sqrt(dx * dx + dy * dy)
            if norm > 1e-6:
                self.last_motion_direction = (dx / norm, dy / norm)

        # Template update (only if conditions met)
        freeze_frames = self.cfg.get("scene4_freeze_template_frames", 120)
        if (self._frames_since_init > freeze_frames
                and scored.get("template_score", 0) >= self.cfg.get("scene4_template_update_threshold", 0.70)):
            # We'd need the gray frame here; defer by setting a flag
            pass

        # Reset counters
        self.lost_count = 0
        self.kalman_predict_count = 0
        self.reacquire_count = 0
        self.prev_score = scored["total_score"]

        # Transition to TRACKING
        self.state = self.STATE_TRACKING

        # Determine if init-locked period is over
        init_locked_frames = self.cfg.get("scene4_init_locked_frames", 30)
        if self._frames_since_init <= init_locked_frames:
            # Still in init-locked window but tracklet is reliable AND near init
            self.state = self.STATE_TRACKING  # Allow tracking

        dbg.update({
            "scene4_state": self.state,
            "scene4_tracklet_accept": 1,
            "scene4_reject_reason_top": "scene4_tracklet_accept",
            "scene4_center_source": "tracklet",
        })

        return self._make_result(frame_id, new_bbox, new_center,
                                 scored["total_score"], True, False, False, True,
                                 self.state, "scene4_tracklet_accept", dbg)

    def _try_hover_ncc(self, frame_id, pred_cx, pred_cy, rs, work_full):
        """Try local NCC at last_reliable_center to confirm hover position."""
        if not self.cfg.get("scene4_use_hover_template_hold", True):
            return None
        if self.last_reliable_template is None or self.last_reliable_center is None:
            return None

        hover_rad = self.cfg.get("scene4_hover_search_radius", 45)
        hover_thr = self.cfg.get("scene4_hover_template_threshold", 0.45)
        hover_max_shift = self.cfg.get("scene4_hover_max_shift", 35)

        lx, ly = self.last_reliable_center
        fw, fh = self.fixed_box_w, self.fixed_box_h
        if fw <= 0 or fh <= 0:
            return None

        # Search in scaled frame around last_reliable_center
        hx1 = max(0, int((lx - hover_rad) * rs))
        hy1 = max(0, int((ly - hover_rad) * rs))
        hx2 = min(work_full.shape[1] if rs < 1.0 else work_full.shape[1],
                  int((lx + hover_rad) * rs))
        hy2 = min(work_full.shape[0] if rs < 1.0 else work_full.shape[0],
                  int((ly + hover_rad) * rs))

        if rs < 1.0:
            sf = cv2.resize(work_full, (int(work_full.shape[1] * rs), int(work_full.shape[0] * rs)))
        else:
            sf = work_full
        hx1_s = max(0, hx1)
        hy1_s = max(0, hy1)
        hx2_s = min(sf.shape[1], hx2)
        hy2_s = min(sf.shape[0], hy2)
        hs = sf[hy1_s:hy2_s, hx1_s:hx2_s]

        if hs.size < 100 or self.last_reliable_template is None:
            return None
        if hs.shape[0] < 10 or hs.shape[1] < 10:
            return None

        try:
            hss = hs.astype(np.float32)
            lrt = self.last_reliable_template
            if lrt.max() <= 1.5:
                lrt = lrt * 255.0
            lrt = lrt.astype(np.float32)

            # Resize lrt to fit in search area if needed
            if lrt.shape[0] >= hss.shape[0] or lrt.shape[1] >= hss.shape[1]:
                scale_tmpl = min(hss.shape[0] / max(lrt.shape[0], 1),
                                 hss.shape[1] / max(lrt.shape[1], 1)) * 0.9
                new_h = max(6, int(lrt.shape[0] * scale_tmpl))
                new_w = max(6, int(lrt.shape[1] * scale_tmpl))
                lrt = cv2.resize(lrt, (new_w, new_h)).astype(np.float32)

            from .ncc import multi_template_search
            hr = multi_template_search(hss, [(lrt, 1.0, 0)], step=2, use_integral=True)
            if hr and hr["score"] >= hover_thr:
                hx_o = int((hr["x"] + hx1_s) / rs)
                hy_o = int((hr["y"] + hy1_s) / rs)
                hw_o = int(hr["w"] / rs)
                hh_o = int(hr["h"] / rs)
                hcx = hx_o + hw_o // 2
                hcy = hy_o + hh_o // 2
                hshift = np.sqrt((hcx - pred_cx) ** 2 + (hcy - pred_cy) ** 2)

                if hshift <= hover_max_shift:
                    self.kalman.update(hcx, hcy)
                    new_bbox = [hx_o, hy_o, self.fixed_box_w, self.fixed_box_h]
                    self.center = (hcx, hcy)
                    self.bbox = new_bbox
                    self.lost_count = 0
                    self.prev_score = hr["score"]
                    self.hover_hold_count += 1
                    self.kalman_predict_count = 0

                    return self._make_result(
                        frame_id, self.bbox, self.center, hr["score"],
                        True, False, False, True,
                        "HOVER_HOLD", "scene4_hover_template_hold",
                        {"scene4_hover_template_score": hr["score"],
                         "scene4_hover_shift": hshift,
                         "scene4_hover_hold_count": self.hover_hold_count,
                         "scene4_center_source": "hover_ncc"}
                    )
        except Exception:
            pass
        return None

    def _hover_kalman_result(self, frame_id, pred_cx, pred_cy, dbg):
        """When hovering and no NCC match → predict at last_reliable center."""
        anchor = self.last_reliable_center or self.anchor_center
        if anchor is None:
            anchor = (pred_cx, pred_cy)
        cx, cy = anchor
        fw, fh = self.fixed_box_w, self.fixed_box_h
        if fw <= 0:
            fw, fh = 60, 30
        bbox = [int(cx - fw // 2), int(cy - fh // 2), fw, fh]
        self.center = (cx, cy)
        self.bbox = bbox
        self.hover_hold_count += 1

        dbg.update({
            "scene4_state": self.state,
            "scene4_reject_reason_top": "scene4_hover_wait_for_reliable_motion",
            "scene4_center_source": "hover_anchor",
            "scene4_hover_hold_count": self.hover_hold_count,
        })
        return self._make_result(frame_id, bbox, (cx, cy), -1.0,
                                 False, True, False, False,
                                 self.state, "scene4_hover_wait_for_reliable_motion", dbg)

    def _kalman_result(self, frame_id, pred_cx, pred_cy, dbg):
        """Kalman prediction result (used during REACQUIRE / KALMAN_PREDICT states)."""
        fw, fh = self.fixed_box_w, self.fixed_box_h
        if fw <= 0:
            fw, fh = 60, 30
        bbox = [int(pred_cx - fw // 2), int(pred_cy - fh // 2), fw, fh]
        self.center = (int(pred_cx), int(pred_cy))
        self.bbox = bbox
        self.lost_count += 1
        self.kalman_predict_count += 1

        dbg.update({
            "scene4_state": self.state,
            "scene4_reject_reason_top": dbg.get("scene4_reject_reason_top", "") or "scene4_kalman_prediction",
            "scene4_center_source": "kalman",
            "scene4_lost_count": self.lost_count,
            "scene4_kalman_predict_count": self.kalman_predict_count,
        })
        return self._make_result(frame_id, bbox, self.center, -1.0,
                                 False, True, False, True,
                                 self.state, "scene4_kalman_prediction", dbg)

    def _make_lost_result(self, frame_id, dbg):
        """Lost result — no bbox, no center, no trajectory."""
        self.state = self.STATE_LOST
        self.bbox = None
        self.center = None
        dbg.update({
            "scene4_state": "LOST",
            "scene4_reject_reason_top": "scene4_lost_too_long",
            "scene4_center_source": "none",
        })
        return self._make_result(frame_id, None, None, -1.0,
                                 False, False, True, False,
                                 "LOST", "scene4_lost_too_long", dbg)

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
