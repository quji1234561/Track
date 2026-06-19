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
        self._frame_count = 0
        self._frames_since_init = 0

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
        self.expected_motion_area = self.cfg.get("scene4_expected_area", 60)
        self.expected_motion_energy = None
        self.last_motion_direction = None
        self.lost_count = 0
        self.kalman_predict_count = 0
        self.hover_hold_count = 0
        self.reacquire_count = 0

    def _init_locked_result(self, frame_id):
        """Build result dict for successful init (INIT_LOCKED state)."""
        dbg = self._build_debug([], [])
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

        self._debug_tracklets = list(self.motion_tracklets)

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
            if trk["age"] < min_age:
                reject_flags.append("age")
            if direction_score < min_dir:
                reject_flags.append(f"dir({direction_score:.2f}<{min_dir})")
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

        # ── Extract frame-diff candidates ───────────────────────────
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
        dbg = self._build_debug(scored_tracklets, raw_candidates)

        # ── Accept tracklet ─────────────────────────────────────────
        if bt_accepts and best_tracklet:
            return self._accept_tracklet_result(best, best_tracklet, frame_id, dbg)

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
        if before_center is not None:
            center_jump = _distance(before_center, new_center)
        else:
            center_jump = -1.0
        dbg["scene4_center_before_x"] = int(before_center[0]) if before_center else 0
        dbg["scene4_center_before_y"] = int(before_center[1]) if before_center else 0
        dbg["scene4_center_after_x"] = int(new_center[0])
        dbg["scene4_center_after_y"] = int(new_center[1])
        dbg["scene4_center_jump"] = round(center_jump, 2)
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

    def _build_debug(self, scored_tracklets, raw_candidates):
        """Build debug info dict from current tracklets/candidates."""
        best = scored_tracklets[0] if scored_tracklets else None
        dbg = {
            "scene4_state": self.state,
            "scene4_tracklet_count": len(self.motion_tracklets),
            "scene4_candidate_count": len(scored_tracklets),
            "scene4_raw_candidate_count": len(raw_candidates),
            "scene4_reject_reason_top": "",
            "scene4_center_source": "none",
        }

        # ── All best-tracklet fields (zeroed by default) ──────────
        _zero_fields = [
            # identity
            "scene4_best_tracklet_id",
            "scene4_best_tracklet_start_frame", "scene4_best_tracklet_end_frame",
            # positions
            "scene4_best_tracklet_start_x", "scene4_best_tracklet_start_y",
            "scene4_best_tracklet_last_x", "scene4_best_tracklet_last_y",
            # scores
            "scene4_best_tracklet_age", "scene4_best_tracklet_score",
            "scene4_best_tracklet_area_score", "scene4_best_tracklet_energy_score",
            "scene4_best_tracklet_direction_score",
            "scene4_best_tracklet_continuity_score",
            "scene4_best_tracklet_anchor_score",
            "scene4_best_tracklet_template_score",
            "scene4_best_tracklet_net_displacement",
            # distances
            "scene4_best_tracklet_dist_to_init_center",
            "scene4_best_tracklet_dist_to_anchor",
            "scene4_best_tracklet_dist_to_last_reliable",
            "scene4_best_tracklet_start_dist_to_last_reliable",
            "scene4_best_tracklet_last_dist_to_last_reliable",
            "scene4_best_tracklet_start_dist_to_init_center",
            "scene4_best_tracklet_last_dist_to_init_center",
            # accept / reject
            "scene4_tracklet_accept", "scene4_tracklet_accept_reason",
            "scene4_tracklet_reject_flags",
        ]
        for k in _zero_fields:
            if k.endswith(("_x", "_y")) or k.endswith(("_frame", "_id")):
                dbg[k] = 0
            elif k in ("scene4_tracklet_accept_reason", "scene4_tracklet_reject_flags"):
                dbg[k] = ""
            else:
                dbg[k] = 0.0

        if best:
            trk = best["tracklet"]
            centers = trk["centers"]
            frames = trk["frames"]
            start_c = centers[0] if centers else (0, 0)
            last_c = centers[-1] if centers else (0, 0)
            lrc = self.last_reliable_center
            ic = self.init_center
            anchor = lrc or ic

            dbg.update({
                "scene4_best_tracklet_id": trk["id"],
                "scene4_best_tracklet_start_frame": int(frames[0]) if frames else 0,
                "scene4_best_tracklet_end_frame": int(frames[-1]) if frames else 0,
                "scene4_best_tracklet_start_x": int(start_c[0]),
                "scene4_best_tracklet_start_y": int(start_c[1]),
                "scene4_best_tracklet_last_x": int(last_c[0]),
                "scene4_best_tracklet_last_y": int(last_c[1]),
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
                "scene4_best_tracklet_dist_to_init_center": round(_distance(last_c, ic), 2) if ic else -1.0,
                "scene4_best_tracklet_dist_to_anchor": round(best["dist_anchor"], 2),
                "scene4_best_tracklet_dist_to_last_reliable": round(_distance(last_c, lrc), 2) if lrc else -1.0,
                "scene4_best_tracklet_start_dist_to_last_reliable": round(_distance(start_c, lrc), 2) if lrc else -1.0,
                "scene4_best_tracklet_last_dist_to_last_reliable": round(_distance(last_c, lrc), 2) if lrc else -1.0,
                "scene4_best_tracklet_start_dist_to_init_center": round(_distance(start_c, ic), 2) if ic else -1.0,
                "scene4_best_tracklet_last_dist_to_init_center": round(_distance(last_c, ic), 2) if ic else -1.0,
                # accept / reject
                "scene4_tracklet_accept": 1 if best["_accept"] else 0,
                "scene4_tracklet_accept_reason": "passed_tracklet_gates" if best["_accept"] else "",
                "scene4_tracklet_reject_flags": "|".join(best["_reject_flags"]) if best["_reject_flags"] else "",
            })
        return dbg
