"""Track 项目主入口 —— 传统图像处理目标跟踪系统。

用法:
    python -m src.main --scene scene1_animation
    python -m src.main --scene all
    python -m src.main --scene scene4_drone --max-frames 600 --debug

主循环流程:
1. 读取config获取场景参数
2. 打开视频，创建Tracker
3. 第一帧: tracker.initialize() 全图NCC搜索 → 初始化确认 → 设置初态
4. 逐帧循环:
   - 若frame_id > tracking_stop_frame: 跳过NCC，仅绘制历史轨迹
   - 否则: preprocess_frame → tracker.track_frame() → 约束检查 →
     accept/predict → 写结果
5. 运行结束: 保存debug CSV、轨迹CSV、指标CSV、跟踪视频

OpenCV仅用于: VideoCapture, VideoWriter, cvtColor, imwrite, 绘图函数。
NCC匹配完全由src/ncc.py自写，跟踪逻辑由src/tracker.py自主实现。
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import cv2

from .config import SCENES, ensure_output_dirs, get_scene_config, OUTPUT_SUBDIRS
from .preprocess import preprocess_frame
from .tracker import TraditionalTracker
try:
    from .scene3_legacy_tracker import Scene3LegacyTracker
except ImportError:
    Scene3LegacyTracker = None
try:
    from .scene4_frame_diff_tracker import Scene4FrameDiffTracker
except ImportError:
    Scene4FrameDiffTracker = None
from .visualize import draw_tracking_result
from .metrics import (
    save_trajectory_csv,
    compute_basic_metrics,
    save_basic_metrics_csv,
    save_summary_metrics,
)

# Debug CSV fields — every frame gets a row
DEBUG_CSV_FIELDS = [
    "frame_id",
    "best_score", "gray_score", "edge_score", "final_score",
    "threshold", "detected", "predicted", "lost", "initialized",
    "template_id", "source_template_id", "scale", "angle",
    "template_w", "template_h",
    "search_x", "search_y", "search_w", "search_h",
    "match_x", "match_y", "match_w", "match_h",
    "center_x", "center_y",
    "predicted_x", "predicted_y",
    "distance_to_prediction",
    "accept_result", "reject_reason", "lost_count",
    "used_for_trajectory", "show_predicted_bbox", "draw_predicted_trajectory",
    # Top-K
    "topk_enabled", "topk_candidates_count",
    "best_raw_score", "best_raw_center_x", "best_raw_center_y",
    "accepted_candidate_rank", "accepted_candidate_score",
    "accepted_candidate_center_x", "accepted_candidate_center_y",
    "accepted_candidate_reject_history",
    "y_delta_from_prev", "y_delta_from_prediction",
    "max_y_decrease_per_frame", "max_candidate_ahead_of_prediction_y",
    "y_forward_speed_valid",
    # last_accepted
    "last_accepted_center_x", "last_accepted_center_y",
    "last_accepted_frame_id",
    # scene2 state machine
    "scene2_state", "scene2_occlusion_count",
    "scene2_recovery_confirm_count",
    # GMC
    "gmc_valid", "gmc_inlier_count",
    "comp_pred_x", "comp_pred_y",
    "kalman_pred_x", "kalman_pred_y",
    "residual_vx", "residual_vy",
    # recovery validation
    "dx_comp", "dy_comp", "dist_comp",
    # scene2 vehicle prior details
    "candidate_final_score", "direction_penalty", "scale_penalty",
    "forward_speed_penalty", "dy", "dx", "forward_gate_enabled",
    "stable_detect_count", "scene_strategy", "scene2_max_forward_step",
    # affine
    "affine_a00", "affine_a01", "affine_a02",
    "affine_a10", "affine_a11", "affine_a12",
    # candidate raw position
    "candidate_x", "candidate_y",
    # scene2 recovery lock
    "scene2_post_occlusion_lock", "scene2_recovery_frame_count",
    # scene4 motion-tracklet tracker — identity & position
    "scene4_state", "scene4_tracklet_count",
    "scene4_best_tracklet_id",
    "scene4_best_tracklet_start_frame", "scene4_best_tracklet_end_frame",
    "scene4_best_tracklet_start_x", "scene4_best_tracklet_start_y",
    "scene4_best_tracklet_last_x", "scene4_best_tracklet_last_y",
    # scene4 tracklet scores
    "scene4_best_tracklet_age", "scene4_best_tracklet_score",
    "scene4_best_tracklet_area_score", "scene4_best_tracklet_energy_score",
    "scene4_best_tracklet_direction_score",
    "scene4_best_tracklet_continuity_score",
    "scene4_best_tracklet_anchor_score",
    "scene4_best_tracklet_template_score",
    "scene4_best_tracklet_net_displacement",
    # scene4 distance diagnostics
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
    # scene4 predicted anchor
    "scene4_predicted_anchor_x", "scene4_predicted_anchor_y",
    # scene4 nearest_motion_contour
    "scene4_nearest_contour_count", "scene4_nearest_contour_valid_count",
    "scene4_nearest_contour_best_area", "scene4_nearest_contour_best_dist",
    "scene4_nearest_contour_search_window", "scene4_nearest_contour_selected_by",
    # scene4 roi_largest_component
    "scene4_component_count", "scene4_component_valid_count",
    "scene4_component_best_area", "scene4_component_best_w",
    "scene4_component_best_h", "scene4_component_best_x",
    "scene4_component_best_y",
    "scene4_component_reject_small_area_count",
    "scene4_component_reject_large_area_count",
    "scene4_component_reject_size_count",
    # scene4 phase
    "scene4_motion_phase", "scene4_phase_gate_pass",
    "scene4_phase_reject_flags",
    # scene4 accept / reject
    "scene4_tracklet_accept", "scene4_tracklet_accept_reason",
    "scene4_tracklet_reject_flags",
    # scene4 center jump
    "scene4_center_before_x", "scene4_center_before_y",
    "scene4_center_after_x", "scene4_center_after_y",
    "scene4_center_jump",
    # scene4 state & source
    "scene4_reject_reason_top", "scene4_candidate_count",
    "scene4_raw_candidate_count", "scene4_center_source",
    "scene4_lost_count", "scene4_kalman_predict_count",
    "scene4_hover_hold_count", "scene4_hover_template_score",
    "scene4_hover_shift",
    "scene4_has_relevant_candidates", "scene4_best_near_dist",
    # legacy scene4 fields (keep for backward compat)
    "scene4_best_score", "scene4_motion_score",
    "scene4_area_score", "scene4_prediction_score",
    "scene4_contrast_score",
    "scene4_diff_max", "scene4_diff_mean", "scene4_diff_p95",
    "scene4_mask_nonzero_raw", "scene4_mask_nonzero_after_morph",
    "scene4_component_count_raw", "scene4_component_count_after_filter",
    "scene4_init_confirm_count", "scene4_template_score",
    "lost",
]

# Per-scale score CSV fields — one row per candidate template per frame
SCALE_SCORE_FIELDS = [
    "frame_id",
    "source_template_id", "scale", "angle",
    "template_w", "template_h",
    "score", "match_x", "match_y",
    "accepted_as_best",
]

DEBUG_DIR = OUTPUT_SUBDIRS.get("logs", Path("outputs/logs")).parent / "debug"


def _safe_center_str(r):
    """Safe center string for debug printing."""
    c = r.get("center", None)
    if c is not None and isinstance(c, (list, tuple)) and len(c) >= 2:
        return f"({c[0]}, {c[1]})"
    return "(None, None)"

def _safe_center(c):
    """Return (cx,cy) or (0,0) for trajectory/tool use."""
    if c is not None and isinstance(c, (list, tuple)) and len(c) >= 2:
        return (int(c[0]), int(c[1]))
    return None

# Fields that must always be integer 0/1 in the debug CSV
_BOOL_INT_FIELDS = {
    "scene4_tracklet_accept",
    "detected",
    "predicted",
    "lost",
    "used_for_trajectory",
    "initialized",
}


def _debug_row(result, frame_id):
    """Extract debug fields from a tracker result dict, filling missing with ''."""
    if result is None:
        return {f: "" for f in DEBUG_CSV_FIELDS}
    row = {}
    for f in DEBUG_CSV_FIELDS:
        val = result.get(f, "")
        if val is None:
            val = ""
        # Normalise boolean-ish fields to 0/1
        if f in _BOOL_INT_FIELDS and val != "":
            if isinstance(val, bool):
                val = 1 if val else 0
            elif isinstance(val, str) and val.lower() in ("true", "yes", "1"):
                val = 1
            elif isinstance(val, str) and val.lower() in ("false", "no", "0"):
                val = 0
            else:
                try:
                    val = int(val)
                except (ValueError, TypeError):
                    val = 0
        row[f] = val
    row["frame_id"] = frame_id

    # ── Ensure center_x / center_y are always extracted from center tuple ──
    center = result.get("center", None)
    if (isinstance(center, (list, tuple))
            and len(center) >= 2
            and center[0] is not None
            and center[1] is not None):
        if row.get("center_x", "") in ("", None):
            row["center_x"] = center[0]
        if row.get("center_y", "") in ("", None):
            row["center_y"] = center[1]
    return row


# Scene4 diff debug writer state (module-level so it persists across frames)
_scene4_debug_vw = None


def _put_label(img, text, org, color, scale=0.45, thickness=1):
    """Draw text with black shadow for readability."""
    x, y = int(org[0]), max(12, int(org[1]))
    # shadow
    cv2.putText(img, text, (x + 1, y + 1),
                cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0),
                thickness + 1, cv2.LINE_AA)
    # colored foreground
    cv2.putText(img, text, (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, scale, color,
                thickness, cv2.LINE_AA)


def _write_scene4_diff_debug(result, frame, prefix, frame_w, frame_h, fps):
    """If result has scene4 component debug boxes, draw them on a copy and
    write to a separate debug video. Uses same codec as tracking video."""
    global _scene4_debug_vw
    if prefix != "scene4_drone":
        return
    boxes = result.get("scene4_debug_candidate_boxes", None)
    pred_bbox = result.get("scene4_debug_pred_bbox", None)
    best_box = result.get("scene4_debug_best_box", None)
    roi = result.get("scene4_debug_roi", None)
    if boxes is None and pred_bbox is None:
        return

    try:
        # Lazy-init writer at half resolution for reliable encoding
        if _scene4_debug_vw is None:
            dw, dh = frame_w // 2, frame_h // 2
            path = str(OUTPUT_SUBDIRS["videos"] / "scene4_drone_diff_debug.avi")
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            _scene4_debug_vw = cv2.VideoWriter(path, fourcc, fps, (dw, dh))
            if _scene4_debug_vw.isOpened():
                print(f"  [Scene4] Diff debug video: {path} ({dw}x{dh})")
            else:
                _scene4_debug_vw = None
                return

        if _scene4_debug_vw is None or not _scene4_debug_vw.isOpened():
            return

        # Draw on a copy of the original color frame
        vis = frame.copy()
        dw, dh = frame_w // 2, frame_h // 2
        BLUE = (255, 0, 0)
        RED = (0, 0, 255)
        YELLOW = (0, 255, 255)

        # ── Yellow: ROI search rectangle ───────────────────────────
        if roi is not None:
            rx, ry, rw, rh = [int(v) for v in roi]
            cv2.rectangle(vis, (rx, ry), (rx + rw, ry + rh), YELLOW, 1)
            label_y = ry - 6 if ry > 18 else ry + rh + 14
            _put_label(vis, f"ROI {rw}x{rh}", (rx, label_y), YELLOW, scale=0.5, thickness=1)

        # ── Blue thin: candidate component boxes ───────────────────
        if boxes:
            max_label = 20  # limit labels to avoid clutter
            for i, box in enumerate(boxes):
                x, y, w, h = [max(0, int(v)) for v in box]
                cv2.rectangle(vis, (x, y), (x + w, y + h), BLUE, 1)
                if i < max_label:
                    label_y = y - 4 if y > 14 else y + h + 12
                    _put_label(vis, f"{w}x{h}", (x, label_y), BLUE)

        # ── Blue thick: best component box ─────────────────────────
        if best_box is not None:
            x, y, w, h = [max(0, int(v)) for v in best_box]
            cv2.rectangle(vis, (x, y), (x + w, y + h), BLUE, 2)
            label_y = y - 6 if y > 18 else y + h + 14
            _put_label(vis, f"best {w}x{h}", (x, label_y), BLUE, scale=0.5, thickness=1)

        # ── Red: predicted anchor bbox ─────────────────────────────
        if pred_bbox is not None:
            x, y, w, h = [max(0, int(v)) for v in pred_bbox]
            cv2.rectangle(vis, (x, y), (x + w, y + h), RED, 2)
            label_y = y - 6 if y > 18 else y + h + 14
            _put_label(vis, f"pred {w}x{h}", (x, label_y), RED, scale=0.5, thickness=1)

        # ── Info overlay: top-left ─────────────────────────────────
        src = result.get("scene4_center_source", "")
        cnt = result.get("scene4_component_count", 0)
        vcnt = result.get("scene4_component_valid_count", 0)
        area = result.get("scene4_component_best_area", 0)
        _put_label(vis, f"comp:{cnt} valid:{vcnt} area:{area} src:{src}",
                   (10, 25), YELLOW, scale=0.5, thickness=1)

        # Resize and write
        vis_small = cv2.resize(vis, (dw, dh))
        _scene4_debug_vw.write(vis_small)
    except Exception:
        pass


def run_scene(scene_key, args):
    """Run tracking for a single scene."""
    cfg = get_scene_config(scene_key)
    scene_name = cfg["name"]
    prefix = cfg["output_prefix"]
    max_frames = args.max_frames
    debug = args.debug
    save_frames = args.save_frames

    # Visualization config
    show_predicted_bbox = cfg.get("show_predicted_bbox", True)
    draw_predicted_trajectory = cfg.get("draw_predicted_trajectory", True)

    print(f"\n{'='*60}")
    print(f"Scene: {scene_key} — {scene_name}")
    print(f"Video: {cfg['video']}")
    print(f"Templates: {[Path(t).name for t in cfg['templates']]}")
    print(f"show_predicted_bbox={show_predicted_bbox}, "
          f"draw_predicted_trajectory={draw_predicted_trajectory}")
    print(f"{'='*60}\n")

    cap = cv2.VideoCapture(cfg["video"])
    if not cap.isOpened():
        print(f"ERROR: Cannot open video: {cfg['video']}", file=sys.stderr)
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0:
        fps = 30.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if max_frames and max_frames < total_frames:
        total_frames = max_frames
        print(f"Processing limited to {total_frames} frames.")

    print(f"Resolution: {frame_w}x{frame_h}, FPS: {fps:.2f}, "
          f"Total frames: {total_frames}")

    ensure_output_dirs()
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    out_video_path = OUTPUT_SUBDIRS["videos"] / f"{prefix}_tracking.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_writer = cv2.VideoWriter(
        str(out_video_path), fourcc, fps, (frame_w, frame_h)
    )

    # Dispatch scene-specific trackers
    if (scene_key == "scene3_bicycle"
            and cfg.get("use_scene3_legacy_tracker", False)
            and Scene3LegacyTracker is not None):
        tracker = Scene3LegacyTracker(cfg)
    elif (scene_key == "scene4_drone"
            and cfg.get("use_scene4_frame_diff_tracker", False)
            and Scene4FrameDiffTracker is not None):
        tracker = Scene4FrameDiffTracker(cfg)
    else:
        tracker = TraditionalTracker(cfg)

    start_frame = cfg.get("start_frame", 0)
    tracking_stop_frame = cfg.get("tracking_stop_frame", None)
    if tracking_stop_frame is not None:
        print(f"Tracking will stop after frame {tracking_stop_frame} "
              f"(no NCC, no boxes, no new trajectory)")

    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        print(f"Skipping to frame {start_frame}...")

    results = []
    debug_rows = []
    scale_score_rows = []
    trajectory = []

    # --- Read first frame ---
    ret, first_frame = cap.read()
    if not ret:
        print("ERROR: Cannot read video frame.", file=sys.stderr)
        cap.release()
        out_writer.release()
        return

    first_gray = preprocess_frame(first_frame)
    init_result = tracker.initialize(first_gray)
    frame_id = start_frame

    if init_result is None:
        init_result = {
            "frame_id": frame_id,
            "bbox": [0, 0, 50, 50],
            "center": (25, 25),
            "score": 0.0,
            "detected": False,
            "predicted": True,
            "template_id": -1,
        }

    # Determine if first frame participates in trajectory
    init_detected = init_result.get("detected", False)
    init_predicted = init_result.get("predicted", False)
    init_used = init_detected or (init_predicted and draw_predicted_trajectory)
    init_result["used_for_trajectory"] = init_used

    init_result["frame_id"] = frame_id
    results.append(init_result)
    if init_used:
        sc = _safe_center(init_result.get("center"))
        if sc is not None:
            trajectory.append(sc)
    debug_rows.append(_debug_row(init_result, frame_id))

    if debug:
        r = init_result
        print(f"Frame {frame_id:05d}: score={r.get('final_score', r['score']):.4f}, "
              f"center={_safe_center_str(r)}, "
              f"detected={r['detected']}, "
              f"reject={r.get('reject_reason', '')}")

    vis_frame = first_frame.copy()
    draw_tracking_result(vis_frame, init_result, trajectory, scene_name,
                         show_predicted_bbox=show_predicted_bbox,
                         draw_predicted_trajectory=draw_predicted_trajectory)
    out_writer.write(vis_frame)

    if save_frames:
        kf_path = OUTPUT_SUBDIRS["frames"] / f"{prefix}_frame_{frame_id:06d}.png"
        cv2.imwrite(str(kf_path), vis_frame)

    # --- Process remaining frames ---
    start_time = time.time()
    for i in range(1, total_frames):
        ret, frame = cap.read()
        if not ret:
            print(f"End of video at frame {start_frame + i}.")
            break

        frame_id = start_frame + i

        # --- Stop tracking after tracking_stop_frame ---
        if tracking_stop_frame is not None and frame_id > tracking_stop_frame:
            result = {
                "frame_id": frame_id,
                "bbox": [0, 0, 0, 0],
                "center": (0, 0),
                "score": -1.0,
                "detected": False,
                "predicted": False,
                "template_id": -1,
                "reject_reason": "tracking_stopped_after_target_obscured",
            }
            # Draw historical trajectory but NO box/center, then write
            vis_frame = frame.copy()
            draw_tracking_result(vis_frame, result, trajectory, scene_name,
                                 show_predicted_bbox=False,
                                 draw_predicted_trajectory=False)
            out_writer.write(vis_frame)
            results.append(result)
            debug_rows.append(_debug_row(result, frame_id))
            if total_frames > 20 and i % max(1, total_frames // 10) == 0:
                pct = 100.0 * i / total_frames
                print(f"  Progress: {pct:.0f}% (stopped)")
            continue

        gray = preprocess_frame(frame)

        # Scene4 interactive mode (independent of detection_mode)
        if (scene_key == "scene4_drone"
                and cfg.get("scene4_interactive", False)
                and hasattr(tracker, "track_frame_interactive")):
            result = tracker.track_frame_interactive(gray, frame, frame_id)
        else:
            result = tracker.track_frame(gray, frame_id)

        # Handle user quit from interactive mode
        if result and result.get("_user_quit"):
            print("  User quit interactive mode.")
            break

        if result is None:
            result = {
                "frame_id": frame_id,
                "bbox": [0, 0, 50, 50],
                "center": (25, 25),
                "score": 0.0,
                "detected": False,
                "predicted": True,
                "template_id": -1,
            }

        result["frame_id"] = frame_id

        # --- Compute used_for_trajectory ---
        # scene2 OCCLUDED: always honor tracker's used_for_trajectory (bridge trajectory)
        if result.get("scene2_state") == "OCCLUDED":
            used = bool(result.get("used_for_trajectory", True))
        else:
            is_detected = result.get("detected", False)
            is_predicted = result.get("predicted", False)
            if is_detected:
                used = True
            elif is_predicted and draw_predicted_trajectory:
                used = True
            else:
                used = False
        result["used_for_trajectory"] = used

        results.append(result)
        # Collect per-scale scores from tracker (before trajectory append)
        for cs in tracker._last_all_scores:
            cs["frame_id"] = frame_id
            scale_score_rows.append(cs)

        # Compute actual drawing config (scene2 OCCLUDED overrides)
        scene_show_bbox = show_predicted_bbox
        scene_draw_traj = draw_predicted_trajectory
        if result.get("scene2_state") == "OCCLUDED":
            scene_show_bbox = cfg.get("scene2_draw_prediction_during_occlusion", True)
            scene_draw_traj = True

        # Inject actual config into result for debug CSV (after override)
        result["show_predicted_bbox"] = scene_show_bbox
        result["draw_predicted_trajectory"] = scene_draw_traj

        debug_rows.append(_debug_row(result, frame_id))
        if used:
            sc = _safe_center(result.get("center"))
            if sc is not None:
                trajectory.append(sc)

        if debug:
            r = result
            print(f"Frame {frame_id:05d}: score={r.get('final_score', r['score']):.4f}, "
                  f"center={_safe_center_str(r)}, "
                  f"detected={r['detected']}, predicted={r.get('predicted', False)}, "
                  f"reject={r.get('reject_reason', '')}, "
                  f"lost_count={r.get('lost_count', '')}, "
                  f"used_traj={used}")

        vis_frame = frame.copy()
        draw_tracking_result(vis_frame, result, trajectory, scene_name,
                             show_predicted_bbox=scene_show_bbox,
                             draw_predicted_trajectory=scene_draw_traj)
        out_writer.write(vis_frame)

        # ── Scene4 diff debug video (blue=candidates, red=pred, separate from tracking) ──
        _write_scene4_diff_debug(result, frame, prefix, frame_w, frame_h, fps)

        # Keyframes at intervals (not every frame)
        if save_frames:
            interval = max(1, total_frames // 5)
            if i % interval == 0 or i == total_frames - 1:
                kf_path = OUTPUT_SUBDIRS["frames"] / f"{prefix}_frame_{frame_id:06d}.png"
                cv2.imwrite(str(kf_path), vis_frame)

        if total_frames > 20 and i % max(1, total_frames // 10) == 0:
            pct = 100.0 * i / total_frames
            print(f"  Progress: {pct:.0f}%")

    elapsed = time.time() - start_time

    cap.release()
    out_writer.release()

    processed = len(results)
    print(f"\nProcessing complete: {processed} frames in {elapsed:.2f}s "
          f"({processed / elapsed:.1f} FPS)")

    # --- Scale usage statistics ---
    tracker.print_scale_stats()

    # --- Release scene4 diff debug writer ---
    global _scene4_debug_vw
    if _scene4_debug_vw is not None:
        _scene4_debug_vw.release()
        _scene4_debug_vw = None

    # --- Cleanup interactive window ---
    cv2.destroyWindow("Scene4 Interactive")
    for _ in range(3):
        cv2.waitKey(1)

    # --- Save debug CSV (every frame) ---
    debug_csv_path = DEBUG_DIR / f"{prefix}_score_debug.csv"
    with open(str(debug_csv_path), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=DEBUG_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(debug_rows)
    n_debug = len(debug_rows)
    print(f"Score debug CSV: {debug_csv_path}")
    print(f"  processed_frames = {processed}")
    print(f"  debug_score_rows = {n_debug}")
    if n_debug != processed:
        print(f"  WARNING: score_debug.csv row count ({n_debug}) does not match "
              f"processed frame count ({processed}).")

    # --- Save per-scale scores CSV (one row per candidate template per frame) ---
    scale_csv_path = DEBUG_DIR / f"{prefix}_scale_scores_debug.csv"
    if scale_score_rows:
        with open(str(scale_csv_path), "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=SCALE_SCORE_FIELDS,
                                    extrasaction="ignore")
            writer.writeheader()
            # Only write fields that exist in SCALE_SCORE_FIELDS
            clean_rows = [{k: r.get(k, "") for k in SCALE_SCORE_FIELDS}
                          for r in scale_score_rows]
            writer.writerows(clean_rows)
        n_scale = len(scale_score_rows)
        n_candidates = len(tracker.templates)
        expected_per_frame = 1 + n_candidates  # +1 because frame 0 may not have all_scores
        print(f"Per-scale scores CSV: {scale_csv_path}  ({n_scale} rows)")
    else:
        print(f"Per-scale scores CSV: (no data — check if collect_all_scores is enabled)")

    # --- Save trajectory CSV ---
    traj_path = OUTPUT_SUBDIRS["trajectories"] / f"{prefix}_trajectory.csv"
    save_trajectory_csv(results, str(traj_path))
    print(f"Trajectory saved: {traj_path}")

    # --- Metrics ---
    basic = compute_basic_metrics(results, elapsed, scene_name)
    metrics_path = OUTPUT_SUBDIRS["metrics"] / f"{prefix}_metrics.csv"
    save_basic_metrics_csv(basic, str(metrics_path))
    print(f"Metrics saved: {metrics_path}")
    print(f"  Detection rate: {basic['detection_rate']:.2%}, "
          f"Avg score: {basic['average_score']:.4f}, "
          f"FPS: {basic['average_fps']:.1f}")

    if save_frames:
        print(f"Keyframes saved in: {OUTPUT_SUBDIRS['frames']}")

    # --- Scene3 backward fill ---
    if (scene_key == "scene3_bicycle"
            and cfg.get("scene3_enable_backward_fill", False)):
        from .scene3_backward_fill import run_backward_fill
        # Find anchor bbox from results (first good detection after start_frame)
        anchor_fid = cfg.get("scene3_backward_anchor_frame", 84)
        anchor_bbox = None
        for r in results:
            if r["frame_id"] >= anchor_fid and r.get("detected", False):
                anchor_bbox = r["bbox"]
                break
        if anchor_bbox is None and results:
            # Fallback: use the first result
            anchor_bbox = results[-1].get("bbox", None)
        if anchor_bbox is not None and anchor_bbox[2] > 0 and anchor_bbox[3] > 0:
            bw_results = run_backward_fill(
                cfg["video"], cfg, anchor_fid, anchor_bbox,
                output_dir=str(OUTPUT_SUBDIRS["trajectories"]))
            if bw_results:
                # Merge backward results into results list
                bw_ids = {r["frame_id"] for r in bw_results}
                results = [r for r in results if r["frame_id"] not in bw_ids]
                # Convert backward results to tracker-compatible dicts
                for br in bw_results:
                    results.append({
                        "frame_id": br["frame_id"],
                        "bbox": [br["x"], br["y"], br["w"], br["h"]],
                        "center": (br["center_x"], br["center_y"]),
                        "score": br["score"],
                        "detected": True,
                        "predicted": False,
                        "used_for_trajectory": True,
                        "template_id": -1,
                        "reject_reason": "scene3_backward_fill",
                    })
                results.sort(key=lambda r: r["frame_id"])
                # Re-save trajectory with merged data
                traj_path2 = OUTPUT_SUBDIRS["trajectories"] / f"{prefix}_trajectory.csv"
                save_trajectory_csv(results, str(traj_path2))
                # Save merged CSV
                merged_path = OUTPUT_SUBDIRS["trajectories"] / f"{prefix}_merged.csv"
                save_trajectory_csv(results, str(merged_path))
                print(f"Merged trajectory saved: {merged_path}")
                # Auto-render merged video from frame 0
                try:
                    from .scene3_render_merged import run_render
                    run_render(scene_key, max_frames or 0)
                except Exception as e:
                    print(f"  Merged render skipped: {e}")
        else:
            print("  Backward fill skipped: no valid anchor bbox")

    print(f"Output video: {out_video_path}")
    return basic


def main():
    parser = argparse.ArgumentParser(
        description="Traditional Image Processing Target Tracking System"
    )
    parser.add_argument("--scene", type=str, required=True,
                        help="Scene key or 'all' to run all scenes")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="Process at most N frames (0 = all frames)")
    parser.add_argument("--debug", action="store_true",
                        help="Print per-frame NCC score, center, and detection status")
    parser.add_argument("--save-frames", action="store_true",
                        help="Save keyframe screenshots at regular intervals")
    args = parser.parse_args()

    if args.scene == "all":
        all_metrics = []
        for key in SCENES:
            result = run_scene(key, args)
            if result:
                all_metrics.append(result)
        if all_metrics:
            summary_path = OUTPUT_SUBDIRS["metrics"] / "summary_metrics.csv"
            save_summary_metrics(all_metrics, str(summary_path))
            print(f"\nSummary metrics saved: {summary_path}")
    elif args.scene in SCENES:
        run_scene(args.scene, args)
    else:
        print(f"Unknown scene: {args.scene}", file=sys.stderr)
        print(f"Available: {', '.join(SCENES.keys())}, all", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
