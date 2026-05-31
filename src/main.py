"""Main entry point for the traditional image processing target tracking system.

Usage:
    python -m src.main --scene scene1_animation
    python -m src.main --scene all
    python -m src.main --scene scene2_car --max-frames 300 --debug
    python -m src.main --scene scene4_drone --save-frames

OpenCV is used only for: VideoCapture, VideoWriter, cvtColor, imwrite, and drawing.
Core NCC matching: self-implemented in src/ncc.py.
Tracking: multi-template local search + Kalman filter in src/tracker.py.
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
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        print(f"Skipping to frame {start_frame}...")

    results = []
    debug_rows = []
    scale_score_rows = []  # per-candidate scores (one row per template per frame)
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
        is_detected = result.get("detected", False)
        is_predicted = result.get("predicted", False)
        if is_detected:
            used = True
        elif is_predicted and draw_predicted_trajectory:
            used = True
        else:
            used = False
        result["used_for_trajectory"] = used

        # Inject config flags into result for debug CSV
        result["show_predicted_bbox"] = show_predicted_bbox
        result["draw_predicted_trajectory"] = draw_predicted_trajectory

        results.append(result)
        debug_rows.append(_debug_row(result, frame_id))
        # Collect per-scale scores from tracker
        for cs in tracker._last_all_scores:
            cs["frame_id"] = frame_id
            scale_score_rows.append(cs)
        if used:
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
                             show_predicted_bbox=show_predicted_bbox,
                             draw_predicted_trajectory=draw_predicted_trajectory)
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
