"""传统目标跟踪器 —— NCC多模板匹配 + 卡尔曼预测 + 多种运动约束。

跟踪流程：
1. 初始化：全图多模板多尺度NCC搜索 → 可选init_search_roi限制区域 →
   可选连续确认 → 设置bbox/center/Kalman
2. 每帧跟踪：卡尔曼预测位置 → 局部搜索窗口裁剪 → 多模板NCC →
   Top-K候选逐个通过约束检查（阈值/距离/方向/跳变/y-speed）→ 接受或拒绝
3. 预测补偿：NCC失败时卡尔曼预测填充 → 可选预测约束限制漂移
4. 丢失处理：连续丢失≥max_lost → 全图重搜索（同样经过约束检查）

每个track_frame()/initialize()返回的dict同时包含跟踪结果和debug字段，
供score_debug.csv使用。

关键设计：
- last_accepted_center/bbox保存最后一次成功检测位置，
  所有约束检查参照此而非self.center（避免卡尔曼预测污染参考点）
- _validate_detection_candidate()统一验证局部搜索和全图重搜索的结果
- _check_y_forward_speed()限制候选不能冲太快也不能落后太多
"""

import math
import cv2
import numpy as np

from .ncc import multi_template_search
from .kalman import KalmanFilter2D
from .preprocess import preprocess_template
try:
    from .global_motion import estimate_global_affine, apply_affine
    _HAS_GMC = True
except ImportError:
    _HAS_GMC = False

# --- Global search strategy constants ---
FULL_SEARCH_SCALE_MAX = 0.5
MIN_TMPL_DIM_AFTER_SCALE = 60
FULL_SEARCH_STEP = 3
MIN_TOTAL_SCALE = 0.35

_SENTINEL = -1


def _make_debug_base():
    """Return a dict with all debug fields set to sentinel defaults."""
    return {
        "best_score": _SENTINEL,
        "gray_score": None,
        "edge_score": None,
        "final_score": _SENTINEL,
        "threshold": 0.0,
        "init_gate": 0.0,
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
        "max_motion_used": _SENTINEL,
        "accept_result": False,
        "reject_reason": "",
        "prediction_valid": False,
        "prediction_rejected": False,
        "prediction_reject_reason": "",
        "init_confirm_count": 0,
        "init_confirm_min_score": 0.0,
        "init_confirm_max_distance": 0,
        # Top-K + y-speed fields
        "topk_enabled": False,
        "topk_candidates_count": 0,
        "best_raw_score": _SENTINEL,
        "best_raw_center_x": _SENTINEL,
        "best_raw_center_y": _SENTINEL,
        "accepted_candidate_rank": _SENTINEL,
        "accepted_candidate_score": _SENTINEL,
        "accepted_candidate_center_x": _SENTINEL,
        "accepted_candidate_center_y": _SENTINEL,
        "accepted_candidate_reject_history": "",
        "y_delta_from_prev": _SENTINEL,
        "y_delta_from_prediction": _SENTINEL,
        "max_y_decrease_per_frame": _SENTINEL,
        "max_candidate_ahead_of_prediction_y": _SENTINEL,
        "y_forward_speed_valid": False,
        "last_accepted_center_x": _SENTINEL,
        "last_accepted_center_y": _SENTINEL,
        "last_accepted_frame_id": _SENTINEL,
    }


class TraditionalTracker:
    """Traditional target tracker using multi-template NCC + Kalman prediction."""

    def __init__(self, scene_config):
        self.cfg = scene_config
        self.templates = []
        self.kalman = None
        self.bbox = None
        self.center = None
        self.lost_count = 0
        self.prev_score = 0.0
        self.initialized = False

        self.use_integral = self.cfg.get("use_integral_ncc", True)
        self.ncc_step = self.cfg.get("ncc_step", 2)
        self.resize_scale = self.cfg.get("resize_scale", 1.0)

        # --- Scene-specific strategy flags ---
        self.scene3_use_gradient = self.cfg.get("scene3_use_gradient_ncc", False)
        self.gradient_templates = []  # (gray, grad, scale, tid) tuples for scene3
        self.scene4_visibility_gate = self.cfg.get("scene4_use_visibility_gate", False)
        self.scene4_occluded = False  # visibility state for scene4

        # --- Last accepted detection state (never polluted by predictions) ---
        self.last_accepted_bbox = None
        self.last_accepted_center = None
        self.last_accepted_frame_id = -1

        # --- Scene2 stable detect counter (for delayed gate) ---
        self.stable_detect_count = 0
        self.last_reliable_center = None    # only updated on confident accept
        self.last_reliable_bbox = None
        self.last_reliable_frame = -1
        self.reliable_history = []          # [(frame_id, cx, cy, w, h), ...] last ~20

        # --- Scene2 occlusion state machine ---
        self.scene2_state = "TRACKING"       # TRACKING / OCCLUDED / RECOVERY / RECOVERY_FAILED
        self.scene2_occlusion_count = 0
        self.scene2_recovery_confirm_count = 0
        self.scene2_recovery_frame_count = 0  # total frames in RECOVERY
        self.scene2_post_occlusion_lock = False

        # --- GMC (global motion compensation) state ---
        self.prev_gray_for_gmc = None
        self.last_global_affine = None
        self.gmc_residual_vx = 0.0
        self.gmc_residual_vy = 0.0
        self.gmc_residual_history = []         # list of (dx, dy) from last 20 reliable frames
        self.scene2_last_comp_pred = None       # iterative prediction during occlusion

        # --- Init confirmation state ---
        self._init_pending = False
        self._init_candidate_pos = None
        self._init_confirm_count = 0

        # Per-scale statistics for debug
        self.scale_usage = {}
        self._last_all_scores = []

        self._load_templates()
        self._print_template_summary()

    # ------------------------------------------------------------------
    #  Scale statistics
    # ------------------------------------------------------------------

    @property
    def scale_stats(self):
        return dict(sorted(self.scale_usage.items()))

    def print_scale_stats(self):
        if not self.scale_usage:
            print("  Scale usage: (no accepted detections yet)")
            return
        print("  Scale usage statistics:")
        for key, count in sorted(self.scale_usage.items()):
            print(f"    {key}: {count} frames")
        scales_seen = set()
        for key in self.scale_usage:
            parts = key.split("_s")
            if len(parts) == 2:
                scales_seen.add(float(parts[1]))
        print(f"  Unique scales used: {sorted(scales_seen)}")
        if len(scales_seen) <= 1 and list(scales_seen) == [1.0]:
            print("  NOTE: Only scale=1.0 was used. Multi-scale may not be "
                  "effective for this scene, or config.multi_scale = [1.0].")

    # ------------------------------------------------------------------
    #  Template loading
    # ------------------------------------------------------------------

    def _load_templates(self):
        template_paths = self.cfg["templates"]
        scales = self.cfg.get("multi_scale", [1.0])
        for tid, tpath in enumerate(template_paths):
            # Gradient NCC (scene3): generate (gray, grad, scale) tuples
            if self.scene3_use_gradient:
                from .preprocess import preprocess_template_gradient
                tmpls = preprocess_template_gradient(tpath, scales=scales)
                for gray_t, grad_t, sc in tmpls:
                    self.templates.append((gray_t, sc, tid))
                    self.gradient_templates.append((grad_t, sc, tid))
            else:
                tmpls = preprocess_template(tpath, scales=scales)
                for tmpl_img, sc in tmpls:
                    self.templates.append((tmpl_img, sc, tid))

    def _print_template_summary(self):
        src_count = len(self.cfg["templates"])
        scales = self.cfg.get("multi_scale", [1.0])
        n_total = len(self.templates)
        print(f"  Templates: {src_count} source x {len(scales)} scales "
              f"= {n_total} total candidates")
        for tmpl_img, sc, tid in self.templates:
            label = (f"    source={tid} scale={sc:.2f}: "
                     f"{tmpl_img.shape[1]}x{tmpl_img.shape[0]}")
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

    def _record_scale_usage(self, result):
        if result is None:
            return
        tid = result.get("template_id", -1)
        sc = result.get("scale", 1.0)
        key = f"tid{tid}_s{sc:.2f}"
        self.scale_usage[key] = self.scale_usage.get(key, 0) + 1

    def _update_last_accepted(self, bbox, center, frame_id, dbg=None):
        """Record last accepted detection — never polluted by Kalman predictions."""
        self.last_accepted_bbox = bbox.copy()
        self.last_accepted_center = (center[0], center[1])
        self.last_accepted_frame_id = frame_id
        if dbg is not None:
            dbg["last_accepted_center_x"] = center[0]
            dbg["last_accepted_center_y"] = center[1]
            dbg["last_accepted_frame_id"] = frame_id

    # ------------------------------------------------------------------
    #  Stability strategies
    # ------------------------------------------------------------------

    def _score_candidate_vehicle_prior(self, cand_cx, cand_cy, cand_w, cand_h,
                                        cand_score, pred_x, pred_y,
                                        prev_cx, prev_cy, prev_w, prev_h):
        """Scene2 vehicle prior: compute weighted final_score for a candidate.

        final = ncc_w * norm_ncc - dist_w * norm_dist - dir_w * dir_penalty
                - scale_w * scale_penalty - fwd_w * forward_speed_penalty
        """
        ncc_w = self.cfg.get("scene2_ncc_weight", 0.50)
        dist_w = self.cfg.get("scene2_distance_penalty_weight", 0.20)
        dir_w = self.cfg.get("scene2_direction_penalty_weight", 0.15)
        scale_w = self.cfg.get("scene2_scale_penalty_weight", 0.10)
        fwd_w = self.cfg.get("scene2_forward_speed_penalty_weight", 0.20)

        norm_ncc = max(0.0, cand_score)

        max_dist = max(1.0, self.cfg.get("max_motion_distance", 50))
        dist = math.hypot(cand_cx - pred_x, cand_cy - pred_y)
        norm_dist = min(1.0, dist / max_dist)

        dir_y = self.cfg.get("motion_direction_y", 0)
        tolerance = self.cfg.get("motion_direction_tolerance", 10)
        dy = cand_cy - prev_cy
        dir_penalty = 0.0
        if dir_y == -1 and dy > tolerance:
            dir_penalty = min(1.0, (dy - tolerance) / 30.0)
        elif dir_y == 1 and dy < -tolerance:
            dir_penalty = min(1.0, (-dy - tolerance) / 30.0)

        max_lat = self.cfg.get("scene2_max_lateral_shift", 25)
        dx = abs(cand_cx - prev_cx)
        lat_penalty = min(1.0, dx / max_lat) if dx > max_lat else 0.0

        scale_penalty = 0.0
        if prev_w > 0 and prev_h > 0 and cand_w > 0 and cand_h > 0:
            prev_area = prev_w * prev_h
            cand_area = cand_w * cand_h
            ratio = max(cand_area / prev_area, prev_area / cand_area)
            scale_penalty = min(1.0, (ratio - 1.0) / 1.0)

        # Forward speed penalty
        max_fwd = self.cfg.get("scene2_max_forward_step", 8)
        back_tol = self.cfg.get("scene2_backward_tolerance", 4)
        fwd_penalty = 0.0
        if dy < -max_fwd:
            fwd_penalty = min(1.0, abs(dy + max_fwd) / max_fwd)
        if dy > back_tol:
            fwd_penalty = max(fwd_penalty, min(1.0, (dy - back_tol) / back_tol))

        final = (ncc_w * norm_ncc - dist_w * norm_dist
                 - dir_w * dir_penalty - scale_w * (scale_penalty + lat_penalty * 0.5)
                 - fwd_w * fwd_penalty)
        return final, {
            "norm_ncc": norm_ncc, "norm_dist": norm_dist,
            "dir_penalty": dir_penalty, "scale_penalty": scale_penalty,
            "lat_penalty": lat_penalty, "forward_speed_penalty": fwd_penalty,
            "final_score": final, "dy": dy,
        }

    def _check_jump(self, new_cx, new_cy, new_w, new_h):
        if not self.cfg.get("enable_jump_detection", False):
            return False, ""
        if self.center is None:
            return False, ""
        prev_cx, prev_cy = self.last_accepted_center
        dist = math.hypot(new_cx - prev_cx, new_cy - prev_cy)
        max_dist = self.cfg.get("jump_max_distance", 150)
        if dist > max_dist:
            return True, f"jump_distance({dist:.0f}>{max_dist})"
        max_area = self.cfg.get("jump_max_area_change", 0)
        if max_area > 0 and self.last_accepted_bbox is not None:
            prev_area = max(self.last_accepted_bbox[2] * self.last_accepted_bbox[3], 1)
            new_area = max(new_w * new_h, 1)
            ratio = max(new_area / prev_area, prev_area / new_area)
            if ratio > max_area:
                return True, f"jump_area({ratio:.1f}>{max_area})"
        return False, ""

    def _check_motion_direction(self, match_cy_orig, prefix=""):
        """Return (violation, reason). Uses last_accepted as reference."""
        if not self.cfg.get("enable_motion_direction", False) or self.last_accepted_center is None:
            return False, ""
        dir_y = self.cfg.get("motion_direction_y", 0)
        tolerance = self.cfg.get("motion_direction_tolerance", 10)
        prev_cy = self.last_accepted_center[1]
        dy = match_cy_orig - prev_cy
        if dir_y == -1 and dy > tolerance:
            return True, f"{prefix}motion_direction_violation(dy={dy:.0f}>{tolerance})"
        elif dir_y == 1 and dy < -tolerance:
            return True, f"{prefix}motion_direction_violation(dy={dy:.0f}<{-tolerance})"
        return False, ""

    def _validate_detection_candidate(self, cx, cy, w, h, score,
                                       pred_x, pred_y, pre_scale,
                                       source="local"):
        """Unified validation for both local search and full re-search.

        Returns (accepted, reject_reason).
        Used by track_frame() for both normal and full-re-search paths.
        """
        threshold = self.cfg["threshold"]
        if score < threshold:
            reason = ("full_research_score_below_threshold"
                      if source == "full_re_search"
                      else "score_below_threshold")
            return False, reason

        # max_motion
        max_motion = 50 * pre_scale
        max_motion_cfg = self.cfg.get("max_motion_distance", 0)
        if max_motion_cfg and max_motion_cfg > 0:
            max_motion = max_motion_cfg * pre_scale
        else:
            max_motion = max(50 * pre_scale, (self.cfg["search_radius"] * pre_scale) * 0.75)

        dist = math.hypot(cx - pred_x, cy - pred_y)
        if dist > max_motion:
            reason = ("full_research_distance_too_large"
                      if source == "full_re_search"
                      else "distance_too_large")
            return False, reason

        # Motion direction
        violated, m_reason = self._check_motion_direction(
            cy, prefix="full_research_" if source == "full_re_search" else ""
        )
        if violated:
            return False, m_reason

        # Jump detection
        is_jump, j_reason = self._check_jump(cx, cy, w, h)
        if is_jump:
            if source == "full_re_search":
                return False, f"full_research_{j_reason}"
            return False, j_reason

        # y-forward speed limit
        y_fwd_violated, y_fwd_reason = self._check_y_forward_speed(cy, pred_y, source)
        if y_fwd_violated:
            return False, y_fwd_reason

        return True, ""

    def _check_y_forward_speed(self, candidate_cy, pred_y, source=""):
        """Check that candidate hasn't overshot ahead of target direction.

        For motion_direction_y=-1 (target moving up): candidate y must not
        decrease too much relative to prev accepted center or Kalman prediction.
        """
        if not self.cfg.get("enable_y_forward_speed_limit", False):
            return False, ""
        if self.last_accepted_center is None:
            return False, ""

        dir_y = self.cfg.get("motion_direction_y", 0)
        if dir_y != -1 and dir_y != 1:
            return False, ""

        max_decrease = self.cfg.get("max_y_decrease_per_frame", 18)
        max_ahead = self.cfg.get("max_candidate_ahead_of_prediction_y", 18)
        max_behind = self.cfg.get("max_candidate_behind_prediction_y", 0)
        prev_cy = self.last_accepted_center[1]

        dy_prev = candidate_cy - prev_cy        # positive = moving down
        dy_pred = candidate_cy - pred_y

        prefix = f"{source}_" if source else ""

        if dir_y == -1:  # target should only move up (y decreases)
            if dy_prev < -max_decrease:
                return True, f"{prefix}y_forward_too_fast(dy={dy_prev:.0f}<{-max_decrease})"
            if dy_pred < -max_ahead:
                return True, f"{prefix}candidate_too_far_ahead_of_prediction(dy={dy_pred:.0f}<{-max_ahead})"
            # Candidate too far behind: y hasn't decreased enough vs prediction
            if max_behind > 0 and dy_pred > max_behind:
                return True, f"{prefix}candidate_too_far_behind_prediction(dy={dy_pred:.0f}>{max_behind})"
        elif dir_y == 1:  # target should only move down (y increases)
            if dy_prev > max_decrease:
                return True, f"{prefix}y_forward_too_fast(dy={dy_prev:.0f}>{max_decrease})"
            if dy_pred > max_ahead:
                return True, f"{prefix}candidate_too_far_ahead_of_prediction(dy={dy_pred:.0f}>{max_ahead})"
            # Candidate too far behind: y hasn't increased enough vs prediction
            if max_behind > 0 and dy_pred < -max_behind:
                return True, f"{prefix}candidate_too_far_behind_prediction(dy={dy_pred:.0f}<{-max_behind})"

        return False, ""

    # ------------------------------------------------------------------
    #  Init confirmation helpers
    # ------------------------------------------------------------------

    def _try_confirm_init(self, result, dbg, frame_id):
        enable = self.cfg.get("enable_init_confirmation", False)

        # Track init confirm count in debug even before acceptance
        dbg["init_confirm_count"] = self._init_confirm_count
        dbg["init_confirm_min_score"] = self.cfg.get("init_confirm_min_score", 0.0)
        dbg["init_confirm_max_distance"] = self.cfg.get("init_confirm_max_distance", 0)

        if not enable:
            accepted_result = self._accept_initialization(result, dbg)
            if accepted_result.get("detected", False):
                return True, accepted_result
            else:
                return False, accepted_result

        # result must not be None
        if result is None:
            dbg["reject_reason"] = "init_confirm_failed"
            dbg["lost"] = True
            dbg["lost_count"] = 1
            self.lost_count = 1
            out = {"frame_id": frame_id, "bbox": [0, 0, 50, 50],
                   "center": (25, 25), "score": 0.0,
                   "detected": False, "predicted": False,
                   "template_id": _SENTINEL}
            out.update(dbg)
            return False, out

        confirm_frames = self.cfg.get("init_confirm_frames", 3)
        min_score = self.cfg.get("init_confirm_min_score", 0.45)
        max_dist = self.cfg.get("init_confirm_max_distance", 30)
        score = result.get("score", 0.0)

        if score < min_score:
            self._init_pending = False
            self._init_confirm_count = 0
            self._init_candidate_pos = None
            dbg["reject_reason"] = "init_confirm_failed_score"
            dbg["lost"] = True
            dbg["lost_count"] = 1
            self.lost_count = 1
            out = {"frame_id": frame_id, "bbox": [0, 0, 50, 50],
                   "center": (25, 25), "score": float(score),
                   "detected": False, "predicted": False,
                   "template_id": _SENTINEL}
            out.update(dbg)
            return False, out

        # Get center from result
        cx = result.get("center_x", _SENTINEL)
        cy = result.get("center_y", _SENTINEL)
        if cx == _SENTINEL:
            cx = result.get("match_x", 0) + result.get("match_w", 0) // 2
            cy = result.get("match_y", 0) + result.get("match_h", 0) // 2

        if not self._init_pending:
            self._init_pending = True
            self._init_candidate_pos = (cx, cy)
            self._init_confirm_count = 1
            print(f"  Init confirm [1/{confirm_frames}]: "
                  f"pos=({cx},{cy}), score={score:.4f}")
        else:
            px, py = self._init_candidate_pos
            dist = math.hypot(cx - px, cy - py)
            if dist <= max_dist and score >= min_score:
                self._init_confirm_count += 1
                self._init_candidate_pos = ((px + cx) / 2, (py + cy) / 2)
                print(f"  Init confirm [{self._init_confirm_count}/{confirm_frames}]: "
                      f"pos=({cx},{cy}), dist={dist:.1f}, score={score:.4f}")
            else:
                detail = f"dist={dist:.1f}" if dist > max_dist else f"score={score:.4f}"
                print(f"  Init confirm RESET: {detail} > limits")
                self._init_pending = False
                self._init_confirm_count = 0
                self._init_candidate_pos = None

        dbg["init_confirm_count"] = self._init_confirm_count

        if self._init_confirm_count >= confirm_frames:
            print(f"  Init CONFIRMED after {confirm_frames} frames!")
            self._init_pending = False
            self._init_confirm_count = 0
            self._init_candidate_pos = None
            return True, self._accept_initialization(result, dbg)

        # Still pending — preserve real score/match info in debug
        dbg["reject_reason"] = "init_confirm_pending"
        dbg["lost"] = True
        dbg["lost_count"] = self._init_confirm_count
        # Don't overwrite dbg fields — init already set match_x/y, score, scale etc.
        out = {
            "frame_id": frame_id,
            "bbox": [int(cx - 25), int(cy - 25), 50, 50],
            "center": (int(cx), int(cy)),
            "score": float(score),
            "detected": False,
            "predicted": False,
            "template_id": result.get("template_id", _SENTINEL),
        }
        out.update(dbg)
        return False, out

    @staticmethod
    def _init_rejected_result(result, dbg, reason):
        dbg["detected"] = False
        dbg["predicted"] = False
        dbg["accept_result"] = False
        dbg["reject_reason"] = reason
        dbg["lost"] = True
        dbg["lost_count"] = 1
        out = {
            "frame_id": 0,
            "bbox": [0, 0, 50, 50],
            "center": (25, 25),
            "score": result.get("score", result.get("final_score", 0)),
            "detected": False,
            "predicted": False,
            "template_id": _SENTINEL,
        }
        out.update(dbg)
        return out

    def _accept_initialization(self, result, dbg):
        threshold = self.cfg["threshold"]
        score = result.get("score", result.get("final_score", 0))
        if score < threshold:
            print(f"  Init REJECTED: score={score:.4f} < threshold={threshold}")
            return self._init_rejected_result(result, dbg, "score_below_threshold")

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
        self.prev_score = score
        self._update_last_accepted(self.bbox, self.center, 0, dbg)

        self._record_scale_usage(result)

        print(f"  Initialized: bbox={self.bbox}, center=({cx},{cy}), "
              f"score={score:.4f}")

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
            "score": score,
            "detected": True,
            "predicted": False,
            "template_id": result.get("template_id", _SENTINEL),
        }
        out.update(dbg)
        # Preserve real values from initialize() result — dbg from track_frame
        # has _SENTINEL defaults that would overwrite them.
        for key in ("final_score", "best_score", "gray_score", "match_x",
                     "match_y", "match_w", "match_h", "source_template_id",
                     "template_w", "template_h", "scale"):
            if key in result and result[key] is not None and result[key] != _SENTINEL:
                out[key] = result[key]
        return out

    # ------------------------------------------------------------------
    #  Prediction validation
    # ------------------------------------------------------------------

    def _validate_prediction(self, pred_cx, pred_cy, pred_w, pred_h,
                              frame_w, frame_h):
        if not self.cfg.get("constrain_predictions", False):
            return True, ""
        if self.last_accepted_center is None or self.last_accepted_bbox is None:
            return True, ""

        prev_cx, prev_cy = self.last_accepted_center
        prev_w, prev_h = self.last_accepted_bbox[2], self.last_accepted_bbox[3]

        max_jump = self.cfg.get("prediction_max_center_jump", 60)
        dist = math.hypot(pred_cx - prev_cx, pred_cy - prev_cy)
        if dist > max_jump:
            return False, f"pred_jump({dist:.0f}>{max_jump})"

        max_y_reverse = self.cfg.get("prediction_max_y_reverse", 5)
        dir_y = self.cfg.get("motion_direction_y", 0)
        if dir_y != 0 and max_y_reverse > 0:
            dy = pred_cy - prev_cy
            if dir_y == -1 and dy > max_y_reverse:
                return False, f"pred_y_reverse(dy={dy:.0f}>{max_y_reverse})"
            elif dir_y == 1 and dy < -max_y_reverse:
                return False, f"pred_y_reverse(dy={dy:.0f}<{-max_y_reverse})"

        if prev_w > 0 and prev_h > 0 and pred_w > 0 and pred_h > 0:
            max_area = self.cfg.get("prediction_max_area_change_ratio", 1.3)
            prev_area = prev_w * prev_h
            pred_area = pred_w * pred_h
            if max(pred_area / max(prev_area, 1),
                   prev_area / max(pred_area, 1)) > max_area:
                return False, "pred_area_change"
            max_w = self.cfg.get("prediction_max_width_change_ratio", 1.3)
            if max(pred_w / max(prev_w, 1), prev_w / max(pred_w, 1)) > max_w:
                return False, "pred_width_change"
            max_h = self.cfg.get("prediction_max_height_change_ratio", 1.3)
            if max(pred_h / max(prev_h, 1), prev_h / max(pred_h, 1)) > max_h:
                return False, "pred_height_change"

        policy = self.cfg.get("prediction_out_of_bounds_policy", "reject")
        if policy == "reject":
            bx = pred_cx - pred_w // 2
            by = pred_cy - pred_h // 2
            if bx < 0 or by < 0 or bx + pred_w > frame_w or by + pred_h > frame_h:
                return False, "pred_out_of_bounds"

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
    #  Initialization
    # ------------------------------------------------------------------

    def initialize(self, first_gray_frame):
        dbg = _make_debug_base()
        dbg["threshold"] = self.cfg["threshold"]
        dbg["initialized"] = False

        # When init_confirmation is on, use init_confirm_min_score as the
        # initialization gate threshold (instead of the harder threshold).
        init_gate = self.cfg["threshold"]
        if self.cfg.get("enable_init_confirmation", False):
            init_gate = self.cfg.get("init_confirm_min_score", self.cfg["threshold"])
        dbg["init_gate"] = init_gate

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

        # --- Init search ROI ---
        search_offset_x = 0
        search_offset_y = 0
        search_img_for_ncc = search_img
        init_roi = self.cfg.get("init_search_roi", None)
        if init_roi is not None and len(init_roi) == 4:
            rx, ry, rw, rh = init_roi
            # total_scale maps original frame coords → search_img coords
            x1 = max(0, int(rx * total_scale))
            y1 = max(0, int(ry * total_scale))
            x2 = min(search_img.shape[1], int((rx + rw) * total_scale))
            y2 = min(search_img.shape[0], int((ry + rh) * total_scale))
            if x2 > x1 and y2 > y1:
                search_offset_x = x1
                search_offset_y = y1
                search_img_for_ncc = search_img[y1:y2, x1:x2]
                print(f"  Using init_search_roi: original={init_roi}, "
                      f"scaled=[{x1},{y1},{x2-x1},{y2-y1}] "
                      f"(total_scale={total_scale:.3f})")
            else:
                print(f"  Invalid init_search_roi (empty after clamp), "
                      f"fallback to full-frame: {init_roi}")

        print(f"  Full-frame NCC ({search_img_for_ncc.shape[1]}x"
              f"{search_img_for_ncc.shape[0]}, "
              f"total_scale={total_scale:.2f}, "
              f"{len(search_templates)} templates, step={step}, "
              f"integral={self.use_integral})...")

        result = multi_template_search(
            search_img_for_ncc, search_templates, step=step,
            use_integral=self.use_integral, verbose=True,
        )

        # Add ROI offset back
        if result is not None and (search_offset_x > 0 or search_offset_y > 0):
            result["x"] += search_offset_x
            result["y"] += search_offset_y

        dbg["final_score"] = result["score"] if result else _SENTINEL
        dbg["gray_score"] = dbg["final_score"] if result else _SENTINEL
        dbg["best_score"] = dbg["final_score"]
        dbg["template_id"] = result["template_id"] if result else _SENTINEL
        dbg["scale"] = result["scale"] if result else None

        # Report search ROI origin in original coords
        dbg["search_x"] = int(search_offset_x / total_scale) if total_scale > 0 else 0
        dbg["search_y"] = int(search_offset_y / total_scale) if total_scale > 0 else 0
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

        # Use init_gate (not threshold) for the initial filter when confirmation is on
        if result is None or result["score"] < init_gate:
            real_score = result["score"] if result else _SENTINEL
            print(f"  WARNING: Target not found (best score: "
                  f"{real_score if real_score != _SENTINEL else 'N/A'})")
            dbg["reject_reason"] = ("score_below_threshold" if result
                                    else "ncc_failed")
            dbg["lost"] = True
            dbg["lost_count"] = 1
            self.lost_count = 1
            out = {
                "frame_id": 0,
                "bbox": [0, 0, 50, 50],
                "center": (25, 25),
                "score": float(real_score) if real_score != _SENTINEL else 0.0,
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
            # Full update of all match fields
            result["x"] = refine_result["x"] + rx1
            result["y"] = refine_result["y"] + ry1
            result["w"] = refine_result["w"]
            result["h"] = refine_result["h"]
            result["score"] = refine_result["score"]
            result["template_id"] = refine_result["template_id"]
            result["scale"] = refine_result.get("scale", result.get("scale"))

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

        # --- GMC: estimate global affine EVERY frame (not just on accept) ---
        use_gmc_global = (self.cfg.get("scene2_use_global_motion_compensation", False)
                          and _HAS_GMC)
        if use_gmc_global and self.prev_gray_for_gmc is not None:
            gmc_exclude = self.bbox if self.bbox and self.bbox[2] > 0 and self.bbox[3] > 0 else None
            A_new, n_inliers = estimate_global_affine(
                self.prev_gray_for_gmc, scaled_frame,
                exclude_bbox=gmc_exclude, cfg=self.cfg)
            dbg["gmc_inlier_count"] = n_inliers
            if A_new is not None:
                self.last_global_affine = A_new
                dbg["gmc_valid"] = True
            else:
                dbg["gmc_valid"] = False
        else:
            dbg["gmc_valid"] = False

        # --- Scene3 exit detection ---
        if self.cfg.get("scene3_stop_after_exit", False) and self.center is not None:
            exit_y = self.cfg.get("scene3_exit_y", 500)
            exit_lost = self.cfg.get("scene3_exit_lost_frames", 5)
            if self.center[1] >= exit_y and self.lost_count >= exit_lost:
                dbg["reject_reason"] = "scene3_target_exited"
                dbg["detected"] = False
                dbg["predicted"] = False
                dbg["lost"] = True
                dbg["lost_count"] = self.lost_count
                dbg["center_x"] = int(self.center[0])
                dbg["center_y"] = int(self.center[1])
                out = {"frame_id": frame_id, "bbox": self.bbox.copy(),
                       "center": self.center, "score": self.prev_score,
                       "detected": False, "predicted": False,
                       "template_id": _SENTINEL}
                out.update(dbg)
                return out

        # --- Scene4 visibility gate ---
        if self.scene4_visibility_gate and not self.scene4_occluded:
            patience = self.cfg.get("scene4_low_score_patience", 5)
            min_vis = self.cfg.get("scene4_min_score_for_visible", 0.32)
            min_contrast = self.cfg.get("scene4_min_local_contrast", 8.0)
            # Check if we've been in low-score state
            if self.lost_count >= patience and self.prev_score < min_vis:
                self.scene4_occluded = True
                print(f"  Frame {frame_id}: Scene4 visibility gate — OCCLUDED")
            elif self.lost_count >= patience:
                # Check local contrast at last accepted position
                if (self.last_accepted_center is not None
                        and self.last_accepted_bbox is not None):
                    bx, by = self.last_accepted_bbox[0], self.last_accepted_bbox[1]
                    bw, bh = self.last_accepted_bbox[2], self.last_accepted_bbox[3]
                    roi = gray_frame[by:by+bh, bx:bx+bw] if bx+bw <= gray_frame.shape[1] and by+bh <= gray_frame.shape[0] else gray_frame
                    if roi.size > 0 and roi.std() * 255 < min_contrast:
                        self.scene4_occluded = True
                        print(f"  Frame {frame_id}: Scene4 visibility gate — low contrast OCCLUDED")
        if self.scene4_occluded:
            dbg["reject_reason"] = "scene4_occluded_visibility_gate"
            dbg["detected"] = False
            dbg["predicted"] = False
            dbg["lost"] = True
            dbg["lost_count"] = self.lost_count
            dbg["center_x"] = int(self.center[0]) if self.center else 0
            dbg["center_y"] = int(self.center[1]) if self.center else 0
            out = {"frame_id": frame_id,
                   "bbox": self.bbox.copy() if self.bbox else [0,0,0,0],
                   "center": self.center if self.center else (0,0),
                   "score": self.prev_score, "detected": False,
                   "predicted": False, "template_id": _SENTINEL}
            out.update(dbg)
            return out

        # --- Scene2 occlusion state machine ---
        occ_bridge = self.cfg.get("scene2_occlusion_bridge_enabled", False)
        occ_range = self.cfg.get("scene2_occlusion_frame_range", [])
        if occ_bridge and occ_range and len(occ_range) == 2:
            occ_start, occ_end = occ_range
            max_predict = self.cfg.get("scene2_max_occlusion_predict_frames", 20)
            if occ_start <= frame_id <= occ_end:
                self.scene2_state = "OCCLUDED"
                self.scene2_occlusion_count = frame_id - occ_start + 1
                self.scene2_post_occlusion_lock = True
                if self.scene2_occlusion_count > max_predict:
                    self.scene2_state = "OCCLUDED"  # stay occluded, max reached
            elif frame_id > occ_end and self.scene2_state in ("OCCLUDED", "RECOVERY"):
                if self.scene2_state == "OCCLUDED":
                    self.scene2_state = "RECOVERY"
                    self.scene2_recovery_confirm_count = 0
                    self.scene2_recovery_frame_count = 0
                else:
                    self.scene2_recovery_frame_count += 1
            elif frame_id < occ_start:
                self.scene2_state = "TRACKING"
                self.scene2_last_comp_pred = None
                self.scene2_post_occlusion_lock = False

        # --- Handle OCCLUDED state ---
        if self.scene2_state == "OCCLUDED":
            use_gmc = (self.cfg.get("scene2_use_global_motion_compensation", False)
                       and _HAS_GMC)
            kf_cx, kf_cy = self.kalman.predict()
            dbg["kalman_pred_x"] = int(kf_cx)
            dbg["kalman_pred_y"] = int(kf_cy)

            # GMC prediction: apply affine + residual velocity
            gmc_ok = False
            comp_cx, comp_cy = kf_cx, kf_cy
            if use_gmc and self.last_global_affine is not None and self.last_reliable_center:
                # Iterative: start from last reliable on first occlusion frame,
                # then project the previous compensated prediction forward.
                if self.scene2_last_comp_pred is not None:
                    base_x, base_y = self.scene2_last_comp_pred
                else:
                    base_x, base_y = self.last_reliable_center
                bg_x, bg_y = apply_affine(self.last_global_affine, base_x, base_y)
                comp_cx = bg_x + self.gmc_residual_vx
                comp_cy = bg_y + self.gmc_residual_vy
                self.scene2_last_comp_pred = (comp_cx, comp_cy)
                gmc_ok = True
                dbg["gmc_valid"] = True
                dbg["comp_pred_x"] = int(comp_cx)
                dbg["comp_pred_y"] = int(comp_cy)
            else:
                dbg["gmc_valid"] = False

            # Blend: GMC + Kalman
            if use_gmc and gmc_ok:
                gmc_w = self.cfg.get("scene2_compensated_prediction_weight", 0.75)
                kf_w = self.cfg.get("scene2_kalman_prediction_weight", 0.25)
                pred_cx = gmc_w * comp_cx + kf_w * kf_cx
                pred_cy = gmc_w * comp_cy + kf_w * kf_cy
                dbg["reject_reason"] = "scene2_occlusion_bridge_prediction_gmc"
            else:
                pred_cx, pred_cy = kf_cx, kf_cy
                # Trajectory median velocity fallback
                if len(self.reliable_history) >= 3:
                    vxs = [self.reliable_history[i][1] - self.reliable_history[i-1][1]
                           for i in range(1, len(self.reliable_history))]
                    vys = [self.reliable_history[i][2] - self.reliable_history[i-1][2]
                           for i in range(1, len(self.reliable_history))]
                    med_vx = sorted(vxs)[len(vxs)//2]
                    med_vy = sorted(vys)[len(vys)//2]
                    lr_cx2, lr_cy2 = self.reliable_history[-1][1], self.reliable_history[-1][2]
                    pred_cx = 0.5 * kf_cx + 0.5 * (lr_cx2 + med_vx * self.scene2_occlusion_count)
                    pred_cy = 0.5 * kf_cy + 0.5 * (lr_cy2 + med_vy * self.scene2_occlusion_count)
                dbg["reject_reason"] = "scene2_occlusion_bridge_prediction"

            prev_w = self.last_reliable_bbox[2] if self.last_reliable_bbox else 40
            prev_h = self.last_reliable_bbox[3] if self.last_reliable_bbox else 40
            occ_bbox = [int(pred_cx - prev_w//2), int(pred_cy - prev_h//2), prev_w, prev_h]
            self.center = (int(pred_cx), int(pred_cy))
            self.bbox = occ_bbox
            dbg["scene2_state"] = "OCCLUDED"
            dbg["scene2_occlusion_count"] = self.scene2_occlusion_count
            dbg["detected"] = False
            dbg["predicted"] = True
            dbg["lost"] = False
            dbg["center_x"] = int(pred_cx)
            dbg["center_y"] = int(pred_cy)
            dbg["used_for_trajectory"] = True
            out = {"frame_id": frame_id, "bbox": occ_bbox,
                   "center": (int(pred_cx), int(pred_cy)),
                   "score": 0.0, "detected": False,
                   "predicted": True, "template_id": _SENTINEL}
            out.update(dbg)
            if _HAS_GMC:
                self.prev_gray_for_gmc = scaled_frame.copy()
            return out

        # Kalman predict (original coords)
        pred_x, pred_y = self.kalman.predict()
        dbg["predicted_x"] = int(pred_x)
        dbg["predicted_y"] = int(pred_y)

        # --- Use GMC compensated prediction for search center in RECOVERY ---
        if (self.scene2_state in ("RECOVERY",)
                and use_gmc_global and self.last_global_affine is not None
                and self.last_reliable_center is not None):
            if self.scene2_last_comp_pred is not None:
                base_x, base_y = self.scene2_last_comp_pred
            else:
                base_x, base_y = self.last_reliable_center
            bg_rx, bg_ry = apply_affine(self.last_global_affine, base_x, base_y)
            gmc_pred_x = bg_rx + self.gmc_residual_vx
            gmc_pred_y = bg_ry + self.gmc_residual_vy
            pred_x = 0.5 * pred_x + 0.5 * gmc_pred_x
            pred_y = 0.5 * pred_y + 0.5 * gmc_pred_y

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

        # --- Scene3 gradient NCC: dual-channel matching ---
        if self.scene3_use_gradient and self.gradient_templates:
            from .preprocess import compute_gradient
            grad_scaled = compute_gradient(scaled_frame)
            grad_search = grad_scaled[y1:y2, x1:x2]
            if search_window.shape[0] < s_bbox_h or search_window.shape[1] < s_bbox_w:
                grad_search = grad_scaled
            grad_working = (
                self._scale_templates(self.gradient_templates, pre_scale)
                if pre_scale < 1.0 else self.gradient_templates
            )
            gray_w = self.cfg.get("scene3_gray_weight", 0.7)
            grad_w = self.cfg.get("scene3_grad_weight", 0.3)

            gray_result, gray_scores = multi_template_search(
                search_window, working_templates, step=local_step,
                use_integral=self.use_integral, collect_all_scores=True,
            )
            grad_result, grad_scores = multi_template_search(
                grad_search, grad_working, step=local_step,
                use_integral=self.use_integral, collect_all_scores=True,
            )

            # Fuse per-candidate scores: same template_id+scale → weighted sum
            fused_scores = []
            for gs in (gray_scores or []):
                gs_score = gs["score"] if gs["score"] > _SENTINEL else 0.0
                gs_match = None
                for es in (grad_scores or []):
                    if (es["template_id"] == gs["template_id"]
                            and abs(es["scale"] - gs["scale"]) < 0.01
                            and es["match_x"] == gs["match_x"]
                            and es["match_y"] == gs["match_y"]):
                        gs_match = es
                        break
                edge_s = gs_match["score"] if gs_match and gs_match["score"] > _SENTINEL else 0.0
                fused = gray_w * gs_score + grad_w * edge_s
                gs["edge_score"] = edge_s if gs_match else None
                gs["fused_score"] = fused
                fused_scores.append(gs)

            all_scores = fused_scores
            self._last_all_scores = all_scores if all_scores else []
            # Use fused best for single-result fallback
            if fused_scores:
                best_fused = max(fused_scores, key=lambda x: x.get("fused_score", -1.0))
                result = best_fused
                dbg["gray_score"] = gray_result["score"] if gray_result else _SENTINEL
                dbg["edge_score"] = grad_result["score"] if grad_result else _SENTINEL
            else:
                result = None
            topk_candidates = [result] if result else []
            self._last_all_scores = all_scores
        else:
            pass  # gradient path already set topk_candidates above

        # --- Top-K candidate selection (standard, non-gradient) ---
        if not self.scene3_use_gradient:
            use_topk = self.cfg.get("enable_topk_candidate_selection", False)
            topk_count = self.cfg.get("topk_candidates", 8)
            topk_enabled = use_topk and topk_count > 0
        else:
            use_topk = False
            topk_count = 8
            topk_enabled = False
        dbg["topk_enabled"] = topk_enabled

        if not self.scene3_use_gradient:
            if topk_enabled:
                result, all_scores, topk_candidates = multi_template_search(
                    search_window, working_templates, step=local_step,
                    use_integral=self.use_integral, collect_all_scores=True,
                    return_topk=True, topk=topk_count,
                )
            else:
                result, all_scores = multi_template_search(
                    search_window, working_templates, step=local_step,
                    use_integral=self.use_integral, collect_all_scores=True,
                )
                topk_candidates = [result] if result else []
            self._last_all_scores = all_scores if all_scores else []

        # Compute max_motion
        max_motion_cfg = self.cfg.get("max_motion_distance", 0)
        if max_motion_cfg and max_motion_cfg > 0:
            max_motion = max_motion_cfg * pre_scale
        else:
            max_motion = max(50 * pre_scale, search_radius * 0.75)
        dbg["max_motion_used"] = max_motion

        dbg["topk_candidates_count"] = len(topk_candidates) if topk_candidates else 0
        dbg["best_raw_score"] = topk_candidates[0]["score"] if topk_candidates else _SENTINEL
        dbg["best_raw_center_x"] = (topk_candidates[0]["x"] + topk_candidates[0]["w"] // 2) if topk_candidates else _SENTINEL
        dbg["best_raw_center_y"] = (topk_candidates[0]["y"] + topk_candidates[0]["h"] // 2) if topk_candidates else _SENTINEL

        # --- Scene2 vehicle prior: re-score candidates ---
        use_vehicle_prior = self.cfg.get("scene2_use_vehicle_prior", False)
        if use_vehicle_prior and topk_candidates and self.last_accepted_center:
            prev_cx, prev_cy = self.last_accepted_center
            prev_w = self.last_accepted_bbox[2] if self.last_accepted_bbox else 50
            prev_h = self.last_accepted_bbox[3] if self.last_accepted_bbox else 50
            scored = []
            for cand in topk_candidates:
                if cand is None:
                    continue
                cx_sc = (cand["x"] + cand["w"] // 2 + x1) / pre_scale
                cy_sc = (cand["y"] + cand["h"] // 2 + y1) / pre_scale
                cw, ch = cand["w"] / pre_scale, cand["h"] / pre_scale
                fscore, details = self._score_candidate_vehicle_prior(
                    cx_sc, cy_sc, cw, ch, cand["score"],
                    pred_x, pred_y, prev_cx, prev_cy, prev_w, prev_h)
                cand["_vehicle_final"] = fscore
                cand["_vehicle_details"] = details
                scored.append((fscore, cand))
            scored.sort(key=lambda x: x[0], reverse=True)
            topk_candidates = [c for _, c in scored]

        # --- Filter None/invalid candidates ---
        safe_candidates = []
        for cand in (topk_candidates or []):
            if cand is None or not isinstance(cand, dict):
                continue
            if "score" not in cand and "final_score" not in cand:
                continue
            safe_candidates.append(cand)
        topk_candidates = safe_candidates

        # --- Loop through top-K candidates ---
        accepted_candidate = None
        reject_history = []
        topk_min_score = self.cfg.get("topk_min_score", 0.0)

        for rank, cand in enumerate(topk_candidates or [], start=1):
            if cand is None or not isinstance(cand, dict):
                reject_history.append("candidate_none")
                continue
            cand_score = cand.get("score", cand.get("final_score", -1.0))
            if cand_score is None:
                cand_score = -1.0
            if cand_score < topk_min_score:
                reject_history.append(f"rank{rank}:score_below_topk_min")
                continue

            global_x = cand["x"] + x1
            global_y = cand["y"] + y1
            match_cx = global_x + cand["w"] // 2
            match_cy = global_y + cand["h"] // 2
            match_cx_orig = match_cx / pre_scale
            match_cy_orig = match_cy / pre_scale

            # --- Scene2 forward/lateral hard gate (delayed enable) ---
            if use_vehicle_prior and self.last_accepted_center is not None:
                min_stable = self.cfg.get("scene2_forward_gate_min_stable_frames", 25)
                gate_enabled = (self.stable_detect_count >= min_stable
                                and self.lost_count == 0)
                if gate_enabled:
                    max_fwd = self.cfg.get("scene2_max_forward_step", 8)
                    back_tol = self.cfg.get("scene2_backward_tolerance", 4)
                    max_lat = self.cfg.get("scene2_max_lateral_step", 12)
                    dy_hard = match_cy_orig - self.last_accepted_center[1]
                    dx_hard = match_cx_orig - self.last_accepted_center[0]
                    rejected = False
                    if dy_hard < -max_fwd:
                        reason = f"scene2_forward_jump_too_large(dy={dy_hard:.0f}<{-max_fwd})"
                        rejected = True
                    elif dy_hard > back_tol:
                        reason = f"scene2_backward_motion(dy={dy_hard:.0f}>{back_tol})"
                        rejected = True
                    elif abs(dx_hard) > max_lat:
                        reason = f"scene2_lateral_jump_too_large(dx={dx_hard:.0f}>{max_lat})"
                        rejected = True
                    if rejected:
                        reject_history.append(f"rank{rank}:{reason}")
                        continue
                dbg["forward_gate_enabled"] = gate_enabled
                dbg["stable_detect_count"] = self.stable_detect_count
                dbg["dx"] = (match_cx_orig - self.last_accepted_center[0]) if self.last_accepted_center else _SENTINEL

            # --- Scene2 recovery gate: lost后恢复必须靠近预测位置 ---
            is_recovery = (use_vehicle_prior
                           and self.cfg.get("scene2_recovery_gate_enabled", False)
                           and self.lost_count > 0
                           and self.last_reliable_center is not None)
            if is_recovery:
                rec_max_dx = self.cfg.get("scene2_recovery_max_pred_x_error", 12)
                rec_max_dy = self.cfg.get("scene2_recovery_max_pred_y_error", 8)
                rec_max_d = self.cfg.get("scene2_recovery_max_total_distance", 18)
                rec_dx = match_cx_orig - pred_x
                rec_dy = match_cy_orig - pred_y
                rec_dist = math.hypot(rec_dx, rec_dy)
                dbg["dx_pred"] = int(rec_dx)
                dbg["dy_pred"] = int(rec_dy)
                dbg["dist_pred"] = rec_dist
                dbg["last_reliable_center_x"] = int(self.last_reliable_center[0])
                dbg["last_reliable_center_y"] = int(self.last_reliable_center[1])
                dbg["scene2_recovery_gate_enabled"] = True
                rejected = False
                if abs(rec_dx) > rec_max_dx:
                    reason = f"scene2_recovery_x_too_far_from_prediction(dx={rec_dx:.0f}>{rec_max_dx})"
                    rejected = True
                elif abs(rec_dy) > rec_max_dy:
                    reason = f"scene2_recovery_y_too_far_from_prediction(dy={rec_dy:.0f}>{rec_max_dy})"
                    rejected = True
                elif rec_dist > rec_max_d:
                    reason = f"scene2_recovery_too_far_from_prediction(dist={rec_dist:.0f}>{rec_max_d})"
                    rejected = True
                if rejected:
                    reject_history.append(f"rank{rank}:{reason}")
                    continue

            accepted, reason = self._validate_detection_candidate(
                match_cx, match_cy, cand["w"], cand["h"],
                cand_score, sp_x, sp_y, pre_scale, source="local"
            )
            if accepted:
                accepted_candidate = cand
                accepted_candidate["rank"] = rank
                accepted_candidate["global_x"] = global_x
                accepted_candidate["global_y"] = global_y
                accepted_candidate["match_cx_orig"] = match_cx_orig
                accepted_candidate["match_cy_orig"] = match_cy_orig
                break
            else:
                reject_history.append(f"rank{rank}:{reason}")

        if accepted_candidate is not None:
            # --- Scene2 RECOVERY confirmation ---
            if self.scene2_state == "RECOVERY":
                rec_min_score = self.cfg.get("scene2_recovery_min_score", 0.36)
                rec_ahead = self.cfg.get("scene2_recovery_max_ahead_y", 6)
                cand_cx = accepted_candidate["match_cx_orig"]
                cand_cy = accepted_candidate["match_cy_orig"]

                # Use GMC-compensated prediction as reference if available
                use_gmc_rec = (self.cfg.get("scene2_recovery_use_compensated_prediction", False)
                               and _HAS_GMC and self.last_global_affine is not None
                               and self.last_reliable_center is not None)
                if use_gmc_rec:
                    lr_cx, lr_cy = self.last_reliable_center
                    ref_x, ref_y = apply_affine(self.last_global_affine, lr_cx, lr_cy)
                    ref_x += self.gmc_residual_vx
                    ref_y += self.gmc_residual_vy
                    dbg["comp_pred_x"] = int(ref_x)
                    dbg["comp_pred_y"] = int(ref_y)
                else:
                    _, ref_y2 = self.kalman.predict()
                    ref_x, ref_y = cand_cx, ref_y2  # fallback: use Kalman for y, current for x

                dx_comp = cand_cx - ref_x
                dy_comp = cand_cy - ref_y
                dist_comp = math.hypot(dx_comp, dy_comp)
                dbg["dx_comp"] = dx_comp
                dbg["dy_comp"] = dy_comp
                dbg["dist_comp"] = dist_comp

                rejected = False
                if abs(dx_comp) > self.cfg.get("scene2_recovery_max_pred_x_error", 12):
                    rejected = True
                    reject_history.append(f"rank{accepted_candidate.get('rank',1)}:scene2_recovery_x_too_far_from_gmc_prediction")
                elif abs(dy_comp) > self.cfg.get("scene2_recovery_max_pred_y_error", 8):
                    rejected = True
                    reject_history.append(f"rank{accepted_candidate.get('rank',1)}:scene2_recovery_y_too_far_from_gmc_prediction")
                elif dist_comp > self.cfg.get("scene2_recovery_max_total_distance", 18):
                    rejected = True
                    reject_history.append(f"rank{accepted_candidate.get('rank',1)}:scene2_recovery_too_far_from_gmc_prediction")
                elif cand_cy < ref_y - rec_ahead:
                    rejected = True
                    reject_history.append(f"rank{accepted_candidate.get('rank',1)}:scene2_recovery_ahead_of_gmc_prediction")
                elif accepted_candidate["score"] < rec_min_score:
                    rejected = True
                    reject_history.append(f"rank{accepted_candidate.get('rank',1)}:scene2_recovery_score_below_min")

                if rejected:
                    accepted_candidate = None
                    # Falls through to ACCEPT guard which checks for None
                else:
                    self.scene2_recovery_confirm_count += 1
                    conf_frames = self.cfg.get("scene2_recovery_confirm_frames", 3)
                    if self.scene2_recovery_confirm_count < conf_frames:
                        # Still confirming — mark as predicted, not detected
                        cand = accepted_candidate
                        dbg["scene2_state"] = "RECOVERY"
                        dbg["scene2_recovery_confirm_count"] = self.scene2_recovery_confirm_count
                        dbg["detected"] = False
                        dbg["predicted"] = True
                        dbg["center_x"] = int(cand["match_cx_orig"])
                        dbg["center_y"] = int(cand["match_cy_orig"])
                        dbg["reject_reason"] = "scene2_recovery_confirm_pending"
                        dbg["used_for_trajectory"] = False
                        out = {"frame_id": frame_id,
                               "bbox": [0,0,0,0], "center": (0,0),
                               "score": cand["score"], "detected": False,
                               "predicted": True,
                               "template_id": cand.get("template_id", _SENTINEL)}
                        out.update(dbg)
                        return out
                    else:
                        # Confirmed — switch to TRACKING
                        self.scene2_state = "TRACKING"
                        self.scene2_recovery_confirm_count = 0
                        self.scene2_recovery_frame_count = 0
                        self.scene2_last_comp_pred = None
                        self.scene2_post_occlusion_lock = False

            # --- ACCEPT (only if still valid after RECOVERY checks) ---
            if accepted_candidate is None:
                # RECOVERY rejected — skip accept, go to next candidate or prediction
                reject_history.append("rankNA:recovery_rejected_candidate_none")
                # continue the outer Top-K loop — use a flag
                pass
            else:
                # fall through to accept logic below
                pass

        # --- Scene2 post-occlusion lock: block normal accept ---
        is_post_occ = (self.scene2_post_occlusion_lock
                       and self.scene2_state not in ("OCCLUDED",))
        if is_post_occ and accepted_candidate is not None:
            # Only allow accept through RECOVERY confirmation path.
            # Normal accept is blocked — force prediction instead.
            dbg["reject_reason"] = "scene2_post_occlusion_lock_active"
            dbg["scene2_post_occlusion_lock"] = True
            dbg["scene2_state"] = self.scene2_state
            accepted_candidate = None

        # --- Scene2 RECOVERY max duration check ---
        if (self.scene2_state == "RECOVERY"
                and self.scene2_recovery_frame_count >= self.cfg.get("scene2_max_recovery_frames", 40)):
            self.scene2_state = "RECOVERY_FAILED"
            dbg["reject_reason"] = "scene2_recovery_failed_keep_hidden"
        if self.scene2_state == "RECOVERY_FAILED":
            dbg["detected"] = False
            dbg["predicted"] = True
            dbg["lost"] = False
            dbg["scene2_state"] = "RECOVERY_FAILED"
            dbg["reject_reason"] = dbg.get("reject_reason") or "scene2_recovery_failed_keep_hidden"
            dbg["used_for_trajectory"] = False
            dbg["center_x"] = int(self.center[0]) if self.center else 0
            dbg["center_y"] = int(self.center[1]) if self.center else 0
            out = {"frame_id": frame_id, "bbox": self.bbox.copy() if self.bbox else [0,0,0,0],
                   "center": self.center if self.center else (0,0),
                   "score": -1.0, "detected": False, "predicted": True,
                   "template_id": _SENTINEL}
            out.update(dbg)
            return out

        if accepted_candidate is not None:
            # --- ACCEPT ---
            cand = accepted_candidate
            score = cand["score"]
            up_x, up_y = self.kalman.update(cand["match_cx_orig"], cand["match_cy_orig"])
            orig_x, orig_y, orig_w, orig_h = self._to_original_coords(
                cand["global_x"], cand["global_y"], cand["w"], cand["h"], pre_scale,
            )
            self.bbox = [orig_x, orig_y, orig_w, orig_h]
            self.center = (int(up_x), int(up_y))
            self.lost_count = 0
            self.prev_score = score
            self.stable_detect_count += 1
            self._update_last_accepted(self.bbox, self.center, frame_id, dbg)
            self.last_reliable_center = self.center
            self.last_reliable_bbox = self.bbox.copy()
            self.last_reliable_frame = frame_id
            self.reliable_history.append((frame_id, self.center[0], self.center[1],
                                           self.bbox[2], self.bbox[3]))
            if len(self.reliable_history) > 20:
                self.reliable_history.pop(0)

            # GMC residual update: compare actual vs background-mapped prev center
            if use_gmc_global and self.last_global_affine is not None:
                if self.last_reliable_center is not None and len(self.reliable_history) >= 2:
                    prev_cx2, prev_cy2 = self.reliable_history[-2][1], self.reliable_history[-2][2]
                    bgx, bgy = apply_affine(self.last_global_affine, prev_cx2, prev_cy2)
                    rx = self.center[0] - bgx
                    ry = self.center[1] - bgy
                    self.gmc_residual_history.append((rx, ry))
                    if len(self.gmc_residual_history) > 20:
                        self.gmc_residual_history.pop(0)
                    if len(self.gmc_residual_history) >= 3:
                        rxs = sorted([r[0] for r in self.gmc_residual_history])
                        rys = sorted([r[1] for r in self.gmc_residual_history])
                        self.gmc_residual_vx = rxs[len(rxs)//2]
                        self.gmc_residual_vy = rys[len(rys)//2]
                dbg["residual_vx"] = self.gmc_residual_vx
                dbg["residual_vy"] = self.gmc_residual_vy

            self._record_scale_usage(cand)

            dbg["best_score"] = score
            dbg["gray_score"] = score
            dbg["final_score"] = score
            dbg["template_id"] = cand.get("template_id", _SENTINEL)
            dbg["scale"] = cand.get("scale", None)
            dbg["detected"] = True
            dbg["accept_result"] = True
            dbg["center_x"] = int(up_x)
            dbg["center_y"] = int(up_y)
            dbg["lost_count"] = 0
            dbg["lost"] = False
            dbg["source_template_id"] = cand.get("template_id", _SENTINEL)
            dbg["template_w"] = orig_w
            dbg["template_h"] = orig_h
            dbg["match_x"] = orig_x
            dbg["match_y"] = orig_y
            dbg["match_w"] = orig_w
            dbg["match_h"] = orig_h
            dbg["distance_to_prediction"] = math.hypot(
                cand["global_x"] + cand["w"] // 2 - sp_x,
                cand["global_y"] + cand["h"] // 2 - sp_y,
            ) / max(pre_scale, 1e-6)
            dbg["accepted_candidate_rank"] = cand.get("rank", 1)
            dbg["accepted_candidate_score"] = score
            dbg["accepted_candidate_center_x"] = int(up_x)
            dbg["accepted_candidate_center_y"] = int(up_y)
            dbg["accepted_candidate_reject_history"] = ";".join(reject_history) if reject_history else ""
            dbg["scene2_recovery_confirm_count"] = self.scene2_recovery_confirm_count
            # Vehicle prior debug
            vd = cand.get("_vehicle_details", {})
            if vd:
                dbg["candidate_final_score"] = vd.get("final_score", _SENTINEL)
                dbg["direction_penalty"] = vd.get("dir_penalty", _SENTINEL)
                dbg["scale_penalty"] = vd.get("scale_penalty", _SENTINEL)
                dbg["forward_speed_penalty"] = vd.get("forward_speed_penalty", _SENTINEL)
                dbg["dy"] = vd.get("dy", _SENTINEL)
            dbg["scene2_state"] = self.scene2_state
            dbg["scene2_post_occlusion_lock"] = self.scene2_post_occlusion_lock
            dbg["scene2_recovery_frame_count"] = self.scene2_recovery_frame_count
            dbg["scene_strategy"] = "vehicle_prior" if use_vehicle_prior else ""
            dbg["scene2_max_forward_step"] = self.cfg.get("scene2_max_forward_step", _SENTINEL)

            out = {
                "frame_id": frame_id,
                "bbox": self.bbox.copy(),
                "center": self.center,
                "score": score,
                "detected": True,
                "predicted": False,
                "template_id": cand.get("template_id", _SENTINEL),
            }
            out.update(dbg)
            if _HAS_GMC:
                self.prev_gray_for_gmc = scaled_frame.copy()
            return out

        # All candidates rejected
        dbg["accepted_candidate_rank"] = -1
        dbg["accepted_candidate_reject_history"] = ";".join(reject_history) if reject_history else ""
        if reject_history:
            dbg["reject_reason"] = f"all_topk_candidates_rejected({len(reject_history)})"
        else:
            dbg["reject_reason"] = "ncc_failed"
        if topk_candidates:
            dbg["best_score"] = topk_candidates[0]["score"]
            dbg["template_id"] = topk_candidates[0].get("template_id", _SENTINEL)
            dbg["scale"] = topk_candidates[0].get("scale", None)

        # --- PREDICTION PATH ---
        self.lost_count += 1
        self.stable_detect_count = 0  # reset on any lost frame
        orig_w = self.bbox[2]
        orig_h = self.bbox[3]
        orig_frame_w = int(s_img_w / pre_scale) if pre_scale > 0 else s_img_w
        orig_frame_h = int(s_img_h / pre_scale) if pre_scale > 0 else s_img_h

        pred_ok, pred_reason = self._validate_prediction(
            int(pred_x), int(pred_y), orig_w, orig_h,
            orig_frame_w, orig_frame_h
        )

        dbg["prediction_valid"] = pred_ok
        dbg["prediction_rejected"] = not pred_ok

        if pred_ok:
            dbg["prediction_reject_reason"] = ""
            self.bbox = [
                int(pred_x - orig_w // 2), int(pred_y - orig_h // 2),
                orig_w, orig_h,
            ]
            self.center = (int(pred_x), int(pred_y))
            dbg["predicted"] = True
            dbg["lost"] = True
        else:
            dbg["prediction_reject_reason"] = pred_reason
            dbg["predicted"] = False
            # keep previous bbox/center — don't update
            if self.cfg.get("prediction_reject_adds_lost", True):
                pass

        dbg["detected"] = False
        dbg["lost_count"] = self.lost_count
        dbg["center_x"] = int(self.center[0])
        dbg["center_y"] = int(self.center[1])
        dbg["scene2_state"] = self.scene2_state
        if not dbg["reject_reason"]:
            dbg["reject_reason"] = ("score_below_threshold" if not pred_ok
                                    else "score_below_threshold")

        # --- FULL RE-SEARCH ---
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

            # Use top-K if enabled
            if topk_enabled:
                full_result, _, fr_topk = multi_template_search(
                    rs_img, rs_templates, step=max(2, self.ncc_step),
                    use_integral=self.use_integral, collect_all_scores=True,
                    return_topk=True, topk=topk_count,
                )
            else:
                full_result, _ = multi_template_search(
                    rs_img, rs_templates, step=max(2, self.ncc_step),
                    use_integral=self.use_integral, collect_all_scores=True,
                )
                fr_topk = [full_result] if full_result else []

            fr_accepted = None
            fr_reject_history = []

            for rank, fc in enumerate(fr_topk or [], start=1):
                if fc is None:
                    continue
                fc_score = fc.get("score", -1.0)
                if fc_score < topk_min_score:
                    fr_reject_history.append(f"fr_rank{rank}:score_below_topk_min")
                    continue

                fc["x"] = int(fc["x"] / search_scale)
                fc["y"] = int(fc["y"] / search_scale)
                fc["w"] = int(fc["w"] / search_scale)
                fc["h"] = int(fc["h"] / search_scale)

                fc_cx = fc["x"] + fc["w"] // 2
                fc_cy = fc["y"] + fc["h"] // 2

                ok, reason = self._validate_detection_candidate(
                    fc_cx, fc_cy, fc["w"], fc["h"],
                    fc_score,
                    pred_x * pre_scale, pred_y * pre_scale,
                    pre_scale, source="full_re_search"
                )
                if ok:
                    fr_accepted = fc
                    fr_accepted["rank"] = rank
                    break
                else:
                    fr_reject_history.append(f"fr_rank{rank}:{reason}")

            if fr_accepted is not None:
                fc = fr_accepted
                orig_x, orig_y, orig_w, orig_h = self._to_original_coords(
                    fc["x"], fc["y"], fc["w"], fc["h"], pre_scale,
                )
                cx = orig_x + orig_w // 2
                cy = orig_y + orig_h // 2
                self.bbox = [orig_x, orig_y, orig_w, orig_h]
                self.center = (cx, cy)
                self.kalman = KalmanFilter2D(cx, cy)
                self.lost_count = 0
                self.prev_score = fc["score"]
                self.stable_detect_count += 1
                self._update_last_accepted(self.bbox, self.center, frame_id, dbg)
                self.last_reliable_center = self.center
                self.last_reliable_bbox = self.bbox.copy()
                self.last_reliable_frame = frame_id

                self._record_scale_usage(fc)

                dbg["detected"] = True
                dbg["predicted"] = False
                dbg["accept_result"] = True
                dbg["lost"] = False
                dbg["lost_count"] = 0
                dbg["reject_reason"] = ""
                dbg["final_score"] = fc["score"]
                dbg["best_score"] = fc["score"]
                dbg["center_x"] = cx
                dbg["center_y"] = cy
                dbg["template_id"] = fc.get("template_id", _SENTINEL)
                dbg["scale"] = fc.get("scale", None)
                dbg["source_template_id"] = fc.get("template_id", _SENTINEL)
                dbg["template_w"] = orig_w
                dbg["template_h"] = orig_h
                dbg["match_x"] = orig_x
                dbg["match_y"] = orig_y
                dbg["match_w"] = orig_w
                dbg["match_h"] = orig_h
                dbg["prediction_valid"] = False
                dbg["prediction_rejected"] = False
                dbg["accepted_candidate_rank"] = fc.get("rank", 1)
                dbg["accepted_candidate_score"] = fc["score"]
                dbg["accepted_candidate_reject_history"] = ";".join(fr_reject_history) if fr_reject_history else ""

                out = {
                    "frame_id": frame_id,
                    "bbox": self.bbox.copy(),
                    "center": self.center,
                    "score": fc["score"],
                    "detected": True,
                    "predicted": False,
                    "template_id": fc.get("template_id", _SENTINEL),
                }
                out.update(dbg)
                return out
            else:
                if fr_reject_history:
                    dbg["reject_reason"] = f"full_re_search_all_rejected({len(fr_reject_history)})"
                print(f"  Full re-search REJECTED: all {len(fr_reject_history)} candidates failed")

        # Save for next GMC frame
        if _HAS_GMC:
            self.prev_gray_for_gmc = scaled_frame.copy()

        out = {
            "frame_id": frame_id,
            "bbox": self.bbox.copy(),
            "center": self.center,
            "score": 0.0,
            "detected": False,
            "predicted": dbg["predicted"],
            "template_id": _SENTINEL,
        }
        out.update(dbg)
        return out
