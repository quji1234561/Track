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


def run_scene(scene_key, args):
    """Run tracking for a single scene."""
    cfg = get_scene_config(scene_key)
    scene_name = cfg["name"]
    prefix = cfg["output_prefix"]
    max_frames = args.max_frames
    debug = args.debug
    save_frames = args.save_frames
    max_lost = cfg.get("max_lost", 15)

    print(f"\n{'='*60}")
    print(f"Scene: {scene_key} — {scene_name}")
    print(f"Video: {cfg['video']}")
    print(f"Templates: {[Path(t).name for t in cfg['templates']]}")
    print(f"{'='*60}\n")

    # Open video
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

    print(f"Resolution: {frame_w}x{frame_h}, FPS: {fps:.2f}, Total frames: {total_frames}")

    # Output video writer
    ensure_output_dirs()
    out_video_path = OUTPUT_SUBDIRS["videos"] / f"{prefix}_tracking.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_writer = cv2.VideoWriter(
        str(out_video_path), fourcc, fps, (frame_w, frame_h)
    )

    # Initialize tracker
    tracker = TraditionalTracker(cfg)

    # Skip to start_frame if configured
    start_frame = cfg.get("start_frame", 0)
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        print(f"Skipping to frame {start_frame}...")

    results = []
    trajectory = []

    # Read first frame
    ret, first_frame = cap.read()
    if not ret:
        print(f"ERROR: Cannot read video frames.", file=sys.stderr)
        cap.release()
        out_writer.release()
        return

    # Process first frame for initialization
    first_gray = preprocess_frame(first_frame)
    init_result = tracker.initialize(first_gray)
    frame_id = start_frame

    if init_result is None:
        print(f"WARNING: Target not found in frame {frame_id}. "
              f"Try adjusting start_frame or threshold.")
        # Still track with empty result
        init_result = {
            "frame_id": frame_id,
            "bbox": [0, 0, 50, 50],
            "center": (25, 25),
            "score": 0.0,
            "detected": False,
            "predicted": True,
            "template_id": -1,
        }

    init_result["frame_id"] = frame_id
    results.append(init_result)
    trajectory.append(init_result["center"])

    if debug:
        r = init_result
        print(f"Frame {frame_id:05d}: score={r['score']:.4f}, "
              f"center=({r['center'][0]}, {r['center'][1]}), "
              f"detected={r['detected']}")

    # Draw and write first frame
    vis_frame = first_frame.copy()
    draw_tracking_result(vis_frame, init_result, trajectory, scene_name)
    out_writer.write(vis_frame)

    # Save first frame as keyframe
    if save_frames:
        keyframe_path = OUTPUT_SUBDIRS["frames"] / f"{prefix}_frame_{frame_id:06d}.png"
        cv2.imwrite(str(keyframe_path), vis_frame)

    # Process remaining frames
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

        results.append(result)
        if result["detected"]:
            trajectory.append(result["center"])

        if debug:
            print(f"Frame {frame_id:05d}: score={result['score']:.4f}, "
                  f"center=({result['center'][0]}, {result['center'][1]}), "
                  f"detected={result['detected']}, predicted={result['predicted']}")

        # Visualize and write
        vis_frame = frame.copy()
        draw_tracking_result(vis_frame, result, trajectory, scene_name)
        out_writer.write(vis_frame)

        # Save keyframes at regular intervals
        if save_frames and (i % (total_frames // 5) == 0 or i == total_frames - 1):
            keyframe_path = OUTPUT_SUBDIRS["frames"] / f"{prefix}_frame_{frame_id:06d}.png"
            cv2.imwrite(str(keyframe_path), vis_frame)

        # Progress
        if total_frames > 20 and i % max(1, total_frames // 10) == 0:
            pct = 100.0 * i / total_frames
            print(f"  Progress: {pct:.0f}%")

    elapsed = time.time() - start_time

    # Cleanup
    cap.release()
    out_writer.release()

    print(f"\nProcessing complete: {len(results)} frames in {elapsed:.2f}s "
          f"({len(results)/elapsed:.1f} FPS)")

    # Save trajectory CSV
    traj_path = OUTPUT_SUBDIRS["trajectories"] / f"{prefix}_trajectory.csv"
    save_trajectory_csv(results, str(traj_path))
    print(f"Trajectory saved: {traj_path}")

    # Compute and save metrics
    basic = compute_basic_metrics(results, elapsed, scene_name)
    metrics_path = OUTPUT_SUBDIRS["metrics"] / f"{prefix}_metrics.csv"
    save_basic_metrics_csv(basic, str(metrics_path))
    print(f"Metrics saved: {metrics_path}")
    print(f"  Detection rate: {basic['detection_rate']:.2%}, "
          f"Avg score: {basic['average_score']:.4f}, "
          f"FPS: {basic['average_fps']:.1f}")

    # Save keyframes if not already done
    if save_frames:
        print(f"Keyframes saved in: {OUTPUT_SUBDIRS['frames']}")

    print(f"Output video: {out_video_path}")

    return basic


def main():
    parser = argparse.ArgumentParser(
        description="Traditional Image Processing Target Tracking System"
    )
    parser.add_argument(
        "--scene", type=str, required=True,
        help="Scene key or 'all' to run all scenes",
    )
    parser.add_argument(
        "--max-frames", type=int, default=0,
        help="Process at most N frames (0 = all frames)",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Print per-frame NCC score, center, and detection status",
    )
    parser.add_argument(
        "--save-frames", action="store_true",
        help="Save keyframe screenshots",
    )

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
        print(f"Available scenes: {', '.join(SCENES.keys())}, all", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
