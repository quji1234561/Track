"""Traditional target tracker combining multi-template NCC, local search, and Kalman filter.

Core tracking algorithm:
1. First frame / re-initialization: full-frame multi-template NCC search.
2. Subsequent frames: Kalman predict → local search window → NCC → validate.
3. Lost tolerance: fall back to full-frame search after max_lost failures.

Every track_frame() / initialize() call returns a dict with both tracking results
AND per-frame debug fields for score_debug.csv.
"""

import math
import cv2
import numpy as np

from .ncc import multi_template_search
from .kalman import KalmanFilter2D
from .preprocess import preprocess_template

# --- 全局搜索策略常量（一般不需要改动） ---
# 初始搜索 / 全图重搜索时的最大缩放比例（0.5 = 缩小到一半分辨率）。
# 调小 = 初始化更快，但模板可能缩到认不出来。
FULL_SEARCH_SCALE_MAX = 0.5
# 全图搜索缩放后模板至少保留的像素尺寸。
# 调大 = 搜索分辨率更高、更精确、更慢。
# 调小 = 搜索更快，但模板太小可能漏掉目标。
MIN_TMPL_DIM_AFTER_SCALE = 60
# 全图搜索时的 NCC 滑窗步长（比局部搜索更粗）。
FULL_SEARCH_STEP = 3
# 全图搜索的最小总缩放比例（resize_scale × search_scale）。
# 防止层层缩放后模板只剩几个像素。
MIN_TOTAL_SCALE = 0.35

# Sentinel for missing / not-applicable numeric fields
_SENTINEL = -1


def _make_debug_base():
    """Return a dict with all debug fields set to sentinel defaults."""
    return {
        "best_score": _SENTINEL,
        "gray_score": None,
        "edge_score": None,
        "final_score": _SENTINEL,
        "threshold": 0.0,
        "initialized": False,
        "detected": False,
        "predicted": False,
        "lost": False,
        "lost_count": 0,
        "template_id": _SENTINEL,
        "source_template_id": _SENTINEL,
        "scale": None,
        "angle": None,
        "template_w": _SENTINEL,
        "template_h": _SENTINEL,
        "search_x": _SENTINEL,
        "search_y": _SENTINEL,
        "search_w": _SENTINEL,
        "search_h": _SENTINEL,
        "match_x": _SENTINEL,
        "match_y": _SENTINEL,
        "match_w": _SENTINEL,
        "match_h": _SENTINEL,
        "center_x": _SENTINEL,
        "center_y": _SENTINEL,
        "predicted_x": _SENTINEL,
        "predicted_y": _SENTINEL,
        "distance_to_prediction": _SENTINEL,
        "accept_result": False,
        "reject_reason": "",
    }


class TraditionalTracker:
    """Traditional target tracker using multi-template NCC + Kalman prediction."""

    def __init__(self, scene_config):
        self.cfg = scene_config
        self.templates = []          # list of (img, scale, source_tid) tuples
        self.kalman = None
        self.bbox = None
        self.center = None
        self.lost_count = 0
        self.prev_score = 0.0
        self.initialized = False

        self.use_integral = self.cfg.get("use_integral_ncc", True)
        self.ncc_step = self.cfg.get("ncc_step", 2)
        self.resize_scale = self.cfg.get("resize_scale", 1.0)

        # --- Init confirmation state ---
        self._init_pending = False
        self._init_candidate_pos = None   # (cx, cy) of first good detection
        self._init_confirm_count = 0

        # Per-scale statistics for debug
        self.scale_usage = {}       # key: "tid_s{scale}" → count
        self._last_all_scores = []  # per-candidate scores from last track_frame

        self._load_templates()
        self._print_template_summary()

    # ------------------------------------------------------------------
    #  Scale statistics
    # ------------------------------------------------------------------

    @property
    def scale_stats(self):
        """Return sorted scale usage statistics."""
        return dict(sorted(self.scale_usage.items()))

    def print_scale_stats(self):
        """Print per-scale usage summary."""
        if not self.scale_usage:
            print("  Scale usage: (no accepted detections yet)")
            return
        print(f"  Scale usage statistics:")
        for key, count in sorted(self.scale_usage.items()):
            print(f"    {key}: {count} frames")
        scales_seen = set()
        for key in self.scale_usage:
            parts = key.split("_s")
            if len(parts) == 2:
                scales_seen.add(float(parts[1]))
        print(f"  Unique scales used: {sorted(scales_seen)}")
        if len(scales_seen) <= 1 and list(scales_seen) == [1.0]:
            print(f"  NOTE: Only scale=1.0 was used. Multi-scale may not be "
                  f"effective for this scene, or config.multi_scale = [1.0].")

    # ------------------------------------------------------------------
    #  Template loading
    # ------------------------------------------------------------------

    def _load_templates(self):
        """Load templates and generate multi-scale candidates."""
        template_paths = self.cfg["templates"]
        scales = self.cfg.get("multi_scale", [1.0])
        for tid, tpath in enumerate(template_paths):
            tmpls = preprocess_template(tpath, scales=scales)
            for tmpl_img, sc in tmpls:
                # (image, scale, source_template_id)
                self.templates.append((tmpl_img, sc, tid))

    def _print_template_summary(self):
        """Print how many template candidates were generated."""
        src_count = len(self.cfg["templates"])
        scales = self.cfg.get("multi_scale", [1.0])
        n_total = len(self.templates)
        print(f"  Templates: {src_count} source × {len(scales)} scales "
              f"= {n_total} total candidates")
        for tmpl_img, sc, tid in self.templates:
            label = f"    source={tid} scale={sc:.2f}: {tmpl_img.shape[1]}×{tmpl_img.shape[0]}"
            if tmpl_img.shape[0] < 5 or tmpl_img.shape[1] < 5:
                label += " [WARNING: very small]"
            print(label)

    # ------------------------------------------------------------------
    #  Adaptive search scaling
    # ------------------------------------------------------------------

    def _get_search_scale(self, img_h, img_w):
        if not self.templates:
            return FULL_SEARCH_SCALE_MAX
        min_dim = min(min(t.shape[0], t.shape[1]) for t, _, _ in self.templates)
        safe_scale = MIN_TMPL_DIM_AFTER_SCALE / min_dim if min_dim > 0 else 1.0
        clamped = max(MIN_TOTAL_SCALE, min(1.0, safe_scale))
        return min(FULL_SEARCH_SCALE_MAX, clamped)

    @staticmethod
    def _scale_templates(templates, scale):
        scaled = []
        for tmpl_img, sc, tid in templates:
            th, tw = tmpl_img.shape
            small_t = cv2.resize(tmpl_img, (int(tw * scale), int(th * scale)))
            scaled.append((small_t.astype(np.float32), sc, tid))
        return scaled

    # ------------------------------------------------------------------
    #  Frame pre-scaling
    # ------------------------------------------------------------------

    def _record_scale_usage(self, result):
        """Track which scale won for per-scale statistics."""
        if result is None:
            return
        tid = result.get("template_id", -1)
        sc = result.get("scale", 1.0)
        key = f"tid{tid}_s{sc:.2f}"
        self.scale_usage[key] = self.scale_usage.get(key, 0) + 1

    # ------------------------------------------------------------------
    #  Stability strategy helpers
    # ------------------------------------------------------------------

    def _check_jump(self, new_cx, new_cy, new_w, new_h):
        """Return (is_jump, reason) based on jump detection config."""
        if not self.cfg.get("enable_jump_detection", False):
            return False, ""
        if self.center is None:
            return False, ""

        prev_cx, prev_cy = self.center
        dist = math.hypot(new_cx - prev_cx, new_cy - prev_cy)
        max_dist = self.cfg.get("jump_max_distance", 150)
        if dist > max_dist:
            return True, f"jump_distance({dist:.0f}>{max_dist})"

        max_area = self.cfg.get("jump_max_area_change", 0)
        if max_area > 0 and self.bbox is not None:
            prev_area = max(self.bbox[2] * self.bbox[3], 1)
            new_area = max(new_w * new_h, 1)
            ratio = max(new_area / prev_area, prev_area / new_area)
            if ratio > max_area:
                return True, f"jump_area({ratio:.1f}>{max_area})"

        return False, ""

    def _try_confirm_init(self, result, dbg, frame_id):
        """Init confirmation state machine.

        Returns (accepted, result_dict).
        If accepted, sets self.initialized and returns the final init result.
        If still pending, returns a placeholder result.
        """
        enable = self.cfg.get("enable_init_confirmation", False)
        if not enable:
            # No confirmation needed — accept immediately
            return True, self._accept_initialization(result, dbg)

        confirm_frames = self.cfg.get("init_confirm_frames", 3)
        min_score = self.cfg.get("init_confirm_min_score", 0.45)
        max_dist = self.cfg.get("init_confirm_max_distance", 30)
        score = result["score"]

        # Must meet score threshold to count as a valid detection
        if result is None or score < min_score:
            self._init_pending = False
            self._init_confirm_count = 0
            self._init_candidate_pos = None
            dbg["reject_reason"] = "init_confirm_failed"
            dbg["lost"] = True
            dbg["lost_count"] = 1
            self.lost_count = 1
            out = {
                "frame_id": frame_id,
                "bbox": [0, 0, 50, 50],
                "center": (25, 25),
                "score": 0.0,
                "detected": False,
                "predicted": False,
                "template_id": _SENTINEL,
            }
            out.update(dbg)
            return False, out

        cx = result.get("center_x", _SENTINEL)
        cy = result.get("center_y", _SENTINEL)
        if cx == _SENTINEL:
            cx = result.get("match_x", 0) + result.get("match_w", 0) // 2
            cy = result.get("match_y", 0) + result.get("match_h", 0) // 2

        if not self._init_pending:
            # First good detection → start confirmation
            self._init_pending = True
            self._init_candidate_pos = (cx, cy)
            self._init_confirm_count = 1
            print(f"  Init confirm [{self._init_confirm_count}/{confirm_frames}]: "
                  f"pos=({cx},{cy}), score={score:.4f}")
        else:
            # Check distance to candidate
            px, py = self._init_candidate_pos
            dist = math.hypot(cx - px, cy - py)
            if dist <= max_dist and score >= min_score:
                self._init_confirm_count += 1
                # Update candidate position (moving average to track small drift)
                self._init_candidate_pos = (
                    (px + cx) / 2, (py + cy) / 2
                )
                print(f"  Init confirm [{self._init_confirm_count}/{confirm_frames}]: "
                      f"pos=({cx},{cy}), dist={dist:.1f}, score={score:.4f}")
            else:
                # Check failed — reset
                print(f"  Init confirm RESET: dist={dist:.1f} > {max_dist} or "
                      f"score={score:.4f} < {min_score}")
                self._init_pending = False
                self._init_confirm_count = 0
                self._init_candidate_pos = None

        if self._init_confirm_count >= confirm_frames:
            print(f"  Init CONFIRMED after {confirm_frames} frames!")
            self._init_pending = False
            self._init_confirm_count = 0
            self._init_candidate_pos = None
            return True, self._accept_initialization(result, dbg)

        # Still pending — return placeholder
        dbg["reject_reason"] = "init_confirm_pending"
        dbg["lost"] = True
        dbg["lost_count"] = self._init_confirm_count
        out = {
            "frame_id": frame_id,
            "bbox": [int(cx - 25), int(cy - 25), 50, 50],
            "center": (int(cx), int(cy)),
            "score": score,
            "detected": False,
            "predicted": False,
            "template_id": result.get("template_id", _SENTINEL),
        }
        out.update(dbg)
        return False, out

    def _accept_initialization(self, result, dbg):
        """Finalize initialization: set up bbox, center, Kalman."""
        orig_x = result.get("match_x", 0)
        orig_y = result.get("match_y", 0)
        orig_w = result.get("match_w", 0)
        orig_h = result.get("match_h", 0)
        cx = orig_x + orig_w // 2
        cy = orig_y + orig_h // 2

        self.bbox = [orig_x, orig_y, orig_w, orig_h]
        self.center = (cx, cy)
        self.kalman = KalmanFilter2D(cx, cy)
        self.initialized = True
        self.lost_count = 0
        self.prev_score = result.get("final_score", result.get("best_score", 0))

        self._record_scale_usage(result)

        print(f"  Initialized: bbox={self.bbox}, center=({cx},{cy}), "
              f"score={self.prev_score:.4f}")

        dbg["initialized"] = True
        dbg["detected"] = True
        dbg["accept_result"] = True
        dbg["center_x"] = cx
        dbg["center_y"] = cy
        dbg["lost_count"] = 0
        dbg["lost"] = False

        out = {
            "frame_id": 0,
            "bbox": self.bbox.copy(),
            "center": self.center,
            "score": self.prev_score,
            "detected": True,
            "predicted": False,
            "template_id": result.get("template_id", _SENTINEL),
        }
        out.update(dbg)
        return out

    # ------------------------------------------------------------------
    #  Unified prediction validation
    # ------------------------------------------------------------------

    def _validate_prediction(self, pred_cx, pred_cy, pred_w, pred_h, frame_w, frame_h):
        """Validate Kalman prediction against unified motion constraints.

        Returns (ok, reason). If ok=False, the prediction should be rejected
        and the tracker should keep the previous bbox/center.
        """
        if not self.cfg.get("constrain_predictions", False):
            return True, ""

        if self.center is None or self.bbox is None:
            return True, ""

        prev_cx, prev_cy = self.center
        prev_w, prev_h = self.bbox[2], self.bbox[3]

        # 1) Center jump
        max_jump = self.cfg.get("prediction_max_center_jump", 60)
        dist = math.hypot(pred_cx - prev_cx, pred_cy - prev_cy)
        if dist > max_jump:
            return False, f"pred_jump({dist:.0f}>{max_jump})"

        # 2) Y direction reversal
        max_y_reverse = self.cfg.get("prediction_max_y_reverse", 5)
        dir_y = self.cfg.get("motion_direction_y", 0)
        if dir_y != 0 and max_y_reverse > 0:
            dy = pred_cy - prev_cy  # positive = moving down
            if dir_y == -1 and dy > max_y_reverse:
                return False, f"pred_y_reverse(dy={dy:.0f}>{max_y_reverse})"
            elif dir_y == 1 and dy < -max_y_reverse:
                return False, f"pred_y_reverse(dy={dy:.0f}<{-max_y_reverse})"

        # 3) Area / width / height sudden change
        if prev_w > 0 and prev_h > 0 and pred_w > 0 and pred_h > 0:
            max_area = self.cfg.get("prediction_max_area_change_ratio", 1.3)
            prev_area = prev_w * prev_h
            pred_area = pred_w * pred_h
            if max(pred_area / max(prev_area, 1), prev_area / max(pred_area, 1)) > max_area:
                return False, "pred_area_change"

            max_w = self.cfg.get("prediction_max_width_change_ratio", 1.3)
            if max(pred_w / max(prev_w, 1), prev_w / max(pred_w, 1)) > max_w:
                return False, "pred_width_change"

            max_h = self.cfg.get("prediction_max_height_change_ratio", 1.3)
            if max(pred_h / max(prev_h, 1), prev_h / max(pred_h, 1)) > max_h:
                return False, "pred_height_change"

        # 4) Out of bounds
        policy = self.cfg.get("prediction_out_of_bounds_policy", "reject")
        if policy == "reject":
            bx = pred_cx - pred_w // 2
            by = pred_cy - pred_h // 2
            if bx < 0 or by < 0 or bx + pred_w > frame_w or by + pred_h > frame_h:
                return False, "pred_out_of_bounds"
        # "clamp" policy is handled by caller

        return True, ""

    def _pre_scale_frame(self, gray_frame):
        if self.resize_scale >= 1.0:
            return gray_frame, 1.0
        h, w = gray_frame.shape
        new_w, new_h = int(w * self.resize_scale), int(h * self.resize_scale)
        return cv2.resize(gray_frame, (new_w, new_h)), self.resize_scale

    def _to_original_coords(self, x, y, w, h, scale):
        if scale >= 1.0:
            return x, y, w, h
        return int(x / scale), int(y / scale), int(w / scale), int(h / scale)

    # ------------------------------------------------------------------
    #  Initialization (full-frame search)
    # ------------------------------------------------------------------

    def initialize(self, first_gray_frame):
        dbg = _make_debug_base()
        dbg["threshold"] = self.cfg["threshold"]
        dbg["initialized"] = False

        scaled_frame, pre_scale = self._pre_scale_frame(first_gray_frame)
        img_h, img_w = scaled_frame.shape

        raw_search_scale = self._get_search_scale(img_h, img_w)
        if pre_scale < 1.0 and raw_search_scale < 1.0:
            if pre_scale * raw_search_scale < MIN_TOTAL_SCALE:
                search_scale = min(1.0, MIN_TOTAL_SCALE / pre_scale)
            else:
                search_scale = raw_search_scale
        else:
            search_scale = raw_search_scale

        total_scale = pre_scale * search_scale
        if search_scale < 1.0:
            small_h, small_w = int(img_h * search_scale), int(img_w * search_scale)
            search_img = cv2.resize(scaled_frame, (small_w, small_h))
        else:
            small_h, small_w = img_h, img_w
            search_img = scaled_frame

        search_templates = self._scale_templates(self.templates, search_scale)
        step = max(2, self.ncc_step + (1 if search_scale < 0.5 else 0))

        print(f"  Full-frame NCC ({small_w}x{small_h}, "
              f"total_scale={total_scale:.2f}, "
              f"{len(search_templates)} templates, step={step}, "
              f"integral={self.use_integral})...")

        result = multi_template_search(
            search_img, search_templates, step=step,
            use_integral=self.use_integral, verbose=True,
        )

        dbg["final_score"] = result["score"] if result else _SENTINEL
        dbg["gray_score"] = dbg["final_score"] if result else _SENTINEL
        dbg["best_score"] = dbg["final_score"]
        dbg["template_id"] = result["template_id"] if result else _SENTINEL
        # Use the ACTUAL winning scale from multi_template_search result
        dbg["scale"] = result["scale"] if result else None

        # Search region in original coords
        dbg["search_x"] = 0
        dbg["search_y"] = 0
        dbg["search_w"] = int(small_w / pre_scale) if pre_scale > 0 else small_w
        dbg["search_h"] = int(small_h / pre_scale) if pre_scale > 0 else small_h

        if result is not None:
            raw_mx = int(result["x"] / pre_scale) if pre_scale > 0 else result["x"]
            raw_my = int(result["y"] / pre_scale) if pre_scale > 0 else result["y"]
            raw_mw = int(result["w"] / pre_scale) if pre_scale > 0 else result["w"]
            raw_mh = int(result["h"] / pre_scale) if pre_scale > 0 else result["h"]
            dbg["match_x"] = raw_mx
            dbg["match_y"] = raw_my
            dbg["match_w"] = raw_mw
            dbg["match_h"] = raw_mh

        if result is None or result["score"] < self.cfg["threshold"]:
            print(f"  WARNING: Target not found (best score: "
                  f"{result['score'] if result else 'N/A'})")
            dbg["reject_reason"] = ("score_below_threshold" if result
                                    else "ncc_failed")
            dbg["lost"] = True
            dbg["lost_count"] = 1
            self.lost_count = 1
            out = {
                "frame_id": 0,
                "bbox": [0, 0, 50, 50],
                "center": (25, 25),
                "score": 0.0,
                "detected": False,
                "predicted": False,
                "template_id": _SENTINEL,
            }
            out.update(dbg)
            return out

        # Map back to scaled_frame coords
        result["x"] = int(result["x"] / search_scale)
        result["y"] = int(result["y"] / search_scale)
        result["w"] = int(result["w"] / search_scale)
        result["h"] = int(result["h"] / search_scale)

        # Refine at scaled-frame resolution
        margin = int(max(result["w"], result["h"]) * 0.5)
        rx1 = max(0, result["x"] - margin)
        ry1 = max(0, result["y"] - margin)
        rx2 = min(img_w, result["x"] + result["w"] + margin)
        ry2 = min(img_h, result["y"] + result["h"] + margin)
        refine_roi = scaled_frame[ry1:ry2, rx1:rx2]
        refine_templates = (
            self._scale_templates(self.templates, pre_scale)
            if pre_scale < 1.0 else self.templates
        )
        refine_result = multi_template_search(
            refine_roi, refine_templates, step=1,
            use_integral=False,
        )
        if refine_result is not None and refine_result["score"] > result["score"]:
            result["x"] = refine_result["x"] + rx1
            result["y"] = refine_result["y"] + ry1
            result["score"] = refine_result["score"]

        # Store refined match in original coords for init confirmation
        orig_x, orig_y, orig_w, orig_h = self._to_original_coords(
            result["x"], result["y"], result["w"], result["h"], pre_scale
        )

        dbg["match_x"] = orig_x
        dbg["match_y"] = orig_y
        dbg["match_w"] = orig_w
        dbg["match_h"] = orig_h
        dbg["final_score"] = result["score"]
        dbg["best_score"] = result["score"]
        dbg["gray_score"] = result["score"]
        dbg["source_template_id"] = result["template_id"]
        dbg["template_w"] = orig_w
        dbg["template_h"] = orig_h
        dbg["center_x"] = orig_x + orig_w // 2
        dbg["center_y"] = orig_y + orig_h // 2

        # Return raw detection for confirmation flow
        out = {
            "frame_id": 0,
            "bbox": [orig_x, orig_y, orig_w, orig_h],
            "center": (orig_x + orig_w // 2, orig_y + orig_h // 2),
            "score": result["score"],
            "detected": True,
            "predicted": False,
            "template_id": result["template_id"],
        }
        out.update(dbg)
        return out

    # ------------------------------------------------------------------
    #  Per-frame tracking
    # ------------------------------------------------------------------

    def track_frame(self, gray_frame, frame_id):
        dbg = _make_debug_base()
        dbg["threshold"] = self.cfg["threshold"]
        dbg["initialized"] = self.initialized

        if not self.initialized:
            init_result = self.initialize(gray_frame)
            if init_result is None:
                # initialize returned None (unexpected)
                dbg["reject_reason"] = "init_failed"
                out = {
                    "frame_id": frame_id, "bbox": [0, 0, 50, 50],
                    "center": (25, 25), "score": 0.0,
                    "detected": False, "predicted": False, "template_id": _SENTINEL,
                }
                out.update(dbg)
                return out
            init_result["frame_id"] = frame_id
            accepted, confirmed = self._try_confirm_init(init_result, dbg, frame_id)
            confirmed["frame_id"] = frame_id
            return confirmed

        scaled_frame, pre_scale = self._pre_scale_frame(gray_frame)
        s_img_h, s_img_w = scaled_frame.shape

        # Kalman predict (original coords)
        pred_x, pred_y = self.kalman.predict()
        dbg["predicted_x"] = int(pred_x)
        dbg["predicted_y"] = int(pred_y)

        sp_x = pred_x * pre_scale
        sp_y = pred_y * pre_scale
        s_bbox_w = int(self.bbox[2] * pre_scale)
        s_bbox_h = int(self.bbox[3] * pre_scale)

        search_radius = int(self.cfg["search_radius"] * pre_scale)
        if self.lost_count > 0:
            search_radius = int(search_radius * (1.0 + self.lost_count * 0.5))

        x1 = max(0, int(sp_x - s_bbox_w // 2 - search_radius))
        y1 = max(0, int(sp_y - s_bbox_h // 2 - search_radius))
        x2 = min(s_img_w, int(sp_x + s_bbox_w // 2 + search_radius))
        y2 = min(s_img_h, int(sp_y + s_bbox_h // 2 + search_radius))

        # Search region in original coords
        dbg["search_x"] = int(x1 / pre_scale) if pre_scale > 0 else x1
        dbg["search_y"] = int(y1 / pre_scale) if pre_scale > 0 else y1
        dbg["search_w"] = int((x2 - x1) / pre_scale) if pre_scale > 0 else (x2 - x1)
        dbg["search_h"] = int((y2 - y1) / pre_scale) if pre_scale > 0 else (y2 - y1)

        search_window = scaled_frame[y1:y2, x1:x2]
        local_step = max(1, self.ncc_step - 1)

        if search_window.shape[0] < s_bbox_h or search_window.shape[1] < s_bbox_w:
            search_window = scaled_frame
            x1, y1 = 0, 0
            local_step = self.ncc_step
            dbg["search_x"] = 0
            dbg["search_y"] = 0
            dbg["search_w"] = int(s_img_w / pre_scale) if pre_scale > 0 else s_img_w
            dbg["search_h"] = int(s_img_h / pre_scale) if pre_scale > 0 else s_img_h

        working_templates = (
            self._scale_templates(self.templates, pre_scale)
            if pre_scale < 1.0 else self.templates
        )

        result, all_scores = multi_template_search(
            search_window, working_templates, step=local_step,
            use_integral=self.use_integral, collect_all_scores=True,
        )
        self._last_all_scores = all_scores if all_scores else []

        dbg["best_score"] = result["score"] if result else _SENTINEL
        dbg["gray_score"] = dbg["best_score"]
        dbg["final_score"] = dbg["best_score"]
        dbg["template_id"] = result["template_id"] if result else _SENTINEL
        # Use the ACTUAL winning scale from multi_template_search result
        dbg["scale"] = result["scale"] if result else None

        threshold = self.cfg["threshold"]
        max_motion = max(50 * pre_scale, search_radius * 0.75)

        if result is not None:
            global_x = result["x"] + x1
            global_y = result["y"] + y1
            score = result["score"]
            match_cx = global_x + result["w"] // 2
            match_cy = global_y + result["h"] // 2
            dist = math.hypot(match_cx - sp_x, match_cy - sp_y)

            # Raw match in original coords
            rm_x, rm_y, rm_w, rm_h = self._to_original_coords(
                global_x, global_y, result["w"], result["h"], pre_scale
            )
            dbg["match_x"] = rm_x
            dbg["match_y"] = rm_y
            dbg["match_w"] = rm_w
            dbg["match_h"] = rm_h
            dbg["distance_to_prediction"] = dist / max(pre_scale, 1e-6)
            # Record which template candidate was tested (even if rejected)
            dbg["source_template_id"] = result["template_id"]
            dbg["template_w"] = rm_w
            dbg["template_h"] = rm_h

            # Pre-compute match center in original coords
            match_cx_orig = match_cx / pre_scale
            match_cy_orig = match_cy / pre_scale

            if score < threshold:
                dbg["reject_reason"] = "score_below_threshold"
            elif dist > max_motion:
                dbg["reject_reason"] = "distance_too_large"
            else:
                # --- Motion direction check ---
                motion_violation = False
                if self.cfg.get("enable_motion_direction", False) and self.center:
                    dir_y = self.cfg.get("motion_direction_y", 0)
                    tolerance = self.cfg.get("motion_direction_tolerance", 10)
                    prev_cy = self.center[1]
                    dy = match_cy_orig - prev_cy  # positive = moving down
                    if dir_y == -1 and dy > tolerance:
                        motion_violation = True
                    elif dir_y == 1 and dy < -tolerance:
                        motion_violation = True

                if motion_violation:
                    dbg["reject_reason"] = "motion_direction_violation"
                else:
                    # --- Jump detection check ---
                    _, _, rw, rh = self._to_original_coords(
                        0, 0, result["w"], result["h"], pre_scale
                    )
                    is_jump, jump_reason = self._check_jump(
                        match_cx_orig, match_cy_orig, rw, rh
                    )

                    if is_jump:
                        dbg["reject_reason"] = jump_reason
                    else:
                        # === ACCEPT ===
                        up_x, up_y = self.kalman.update(match_cx_orig, match_cy_orig)

                        orig_x, orig_y, orig_w, orig_h = self._to_original_coords(
                            global_x, global_y, result["w"], result["h"], pre_scale,
                        )

                        self.bbox = [orig_x, orig_y, orig_w, orig_h]
                        self.center = (int(up_x), int(up_y))
                        self.lost_count = 0
                        self.prev_score = score

                        self._record_scale_usage(result)

                        dbg["detected"] = True
                        dbg["accept_result"] = True
                        dbg["center_x"] = int(up_x)
                        dbg["center_y"] = int(up_y)
                        dbg["lost_count"] = 0
                        dbg["lost"] = False
                        dbg["final_score"] = score
                        dbg["source_template_id"] = result["template_id"]
                        dbg["template_w"] = orig_w
                        dbg["template_h"] = orig_h

                        out = {
                            "frame_id": frame_id,
                            "bbox": self.bbox.copy(),
                            "center": self.center,
                            "score": score,
                            "detected": True,
                            "predicted": False,
                            "template_id": result["template_id"],
                        }
                        out.update(dbg)
                        return out
        else:
            dbg["reject_reason"] = "ncc_failed"

        # --- PREDICTION PATH (no valid detection) ---
        self.lost_count += 1
        orig_w = self.bbox[2]
        orig_h = self.bbox[3]

        # Validate Kalman prediction against unified constraints
        # pred coords are in original-frame, so pass original-frame dimensions
        orig_frame_w = int(s_img_w / pre_scale) if pre_scale > 0 else s_img_w
        orig_frame_h = int(s_img_h / pre_scale) if pre_scale > 0 else s_img_h
        pred_ok, pred_reason = self._validate_prediction(
            int(pred_x), int(pred_y), orig_w, orig_h, orig_frame_w, orig_frame_h
        )

        if pred_ok:
            self.bbox = [
                int(pred_x - orig_w // 2), int(pred_y - orig_h // 2),
                orig_w, orig_h,
            ]
            self.center = (int(pred_x), int(pred_y))
        else:
            # Prediction rejected — keep previous bbox/center
            if self.cfg.get("prediction_reject_adds_lost", True):
                pass  # lost_count already incremented above

        dbg["detected"] = False
        dbg["predicted"] = True
        dbg["lost"] = True
        dbg["lost_count"] = self.lost_count
        dbg["center_x"] = int(self.center[0])
        dbg["center_y"] = int(self.center[1])
        if not dbg["reject_reason"]:
            base_reason = "score_below_threshold"
            dbg["reject_reason"] = f"{base_reason};{pred_reason}" if not pred_ok else base_reason

        # Full re-search if lost too long
        if self.lost_count >= self.cfg["max_lost"]:
            print(f"  Frame {frame_id}: Lost for {self.lost_count} frames, "
                  f"full re-search...")
            search_scale = self._get_search_scale(s_img_h, s_img_w)
            if search_scale < 1.0:
                rs_h, rs_w = int(s_img_h * search_scale), int(s_img_w * search_scale)
                rs_img = cv2.resize(scaled_frame, (rs_w, rs_h))
            else:
                rs_img = scaled_frame
            rs_templates = self._scale_templates(self.templates, search_scale)
            full_result, _ = multi_template_search(
                rs_img, rs_templates, step=max(2, self.ncc_step),
                use_integral=self.use_integral, collect_all_scores=True,
            )
            if full_result is not None:
                full_result["x"] = int(full_result["x"] / search_scale)
                full_result["y"] = int(full_result["y"] / search_scale)
                full_result["w"] = int(full_result["w"] / search_scale)
                full_result["h"] = int(full_result["h"] / search_scale)

                if full_result["score"] >= threshold:
                    orig_x, orig_y, orig_w, orig_h = self._to_original_coords(
                        full_result["x"], full_result["y"],
                        full_result["w"], full_result["h"], pre_scale,
                    )
                    cx = orig_x + orig_w // 2
                    cy = orig_y + orig_h // 2
                    self.bbox = [orig_x, orig_y, orig_w, orig_h]
                    self.center = (cx, cy)
                    self.kalman = KalmanFilter2D(cx, cy)
                    self.lost_count = 0
                    self.prev_score = full_result["score"]

                    self._record_scale_usage(full_result)

                    dbg["detected"] = True
                    dbg["predicted"] = False
                    dbg["accept_result"] = True
                    dbg["lost"] = False
                    dbg["lost_count"] = 0
                    dbg["reject_reason"] = ""
                    dbg["final_score"] = full_result["score"]
                    dbg["best_score"] = full_result["score"]
                    dbg["center_x"] = cx
                    dbg["center_y"] = cy
                    dbg["template_id"] = full_result["template_id"]
                    dbg["scale"] = full_result.get("scale", None)
                    dbg["source_template_id"] = full_result["template_id"]
                    dbg["template_w"] = orig_w
                    dbg["template_h"] = orig_h
                    dbg["match_x"] = orig_x
                    dbg["match_y"] = orig_y
                    dbg["match_w"] = orig_w
                    dbg["match_h"] = orig_h

                    out = {
                        "frame_id": frame_id,
                        "bbox": self.bbox.copy(),
                        "center": self.center,
                        "score": full_result["score"],
                        "detected": True,
                        "predicted": False,
                        "template_id": full_result["template_id"],
                    }
                    out.update(dbg)
                    return out

        out = {
            "frame_id": frame_id,
            "bbox": self.bbox.copy(),
            "center": self.center,
            "score": 0.0,
            "detected": False,
            "predicted": True,
            "template_id": _SENTINEL,
        }
        out.update(dbg)
        return out
