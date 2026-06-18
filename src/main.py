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


def _debug_row(result, frame_id):
    """Extract debug fields from a tracker result dict, filling missing with ''."""
    if result is None:
        return {f: "" for f in DEBUG_CSV_FIELDS}
    row = {}
    for f in DEBUG_CSV_FIELDS:
        val = result.get(f, "")
        if val is None:
            val = ""
        row[f] = val
    row["frame_id"] = frame_id
    return row


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
    if init_used and init_result.get("center") is not None:
        trajectory.append(init_result["center"])
    debug_rows.append(_debug_row(init_result, frame_id))

    if debug:
        r = init_result
        print(f"Frame {frame_id:05d}: score={r.get('final_score', r['score']):.4f}, "
              f"center=({r['center'][0]}, {r['center'][1]}), "
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
        result = tracker.track_frame(gray, frame_id)

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
        if used and result.get("center") is not None:
            trajectory.append(result["center"])

        if debug:
            r = result
            print(f"Frame {frame_id:05d}: score={r.get('final_score', r['score']):.4f}, "
                  f"center=({r['center'][0]}, {r['center'][1]}), "
                  f"detected={r['detected']}, predicted={r.get('predicted', False)}, "
                  f"reject={r.get('reject_reason', '')}, "
                  f"lost_count={r.get('lost_count', '')}, "
                  f"used_traj={used}")

        vis_frame = frame.copy()
        draw_tracking_result(vis_frame, result, trajectory, scene_name,
                             show_predicted_bbox=scene_show_bbox,
                             draw_predicted_trajectory=scene_draw_traj)
        out_writer.write(vis_frame)

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
