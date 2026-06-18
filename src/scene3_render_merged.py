"""Scene3 merged video renderer — standalone post-processing script.

Reads forward + backward trajectory CSVs, merges them, re-renders video from frame 0.
Does NOT modify main.py's tracking loop or any other scene.

Usage:
    python -m src.scene3_render_merged --scene scene3_bicycle
    python -m src.scene3_render_merged --scene scene3_bicycle --max-frames 180
"""

import argparse
import csv
import sys
from pathlib import Path

import cv2

from .config import get_scene_config, OUTPUT_SUBDIRS, PROJECT_ROOT


def _row_to_bbox(row):
    """Safe bbox from CSV row. Returns (x,y,w,h) or None."""
    x = row.get("x", row.get("bbox_x", None))
    y = row.get("y", row.get("bbox_y", None))
    w = row.get("w", row.get("bbox_w", None))
    h = row.get("h", row.get("bbox_h", None))
    if None in (x, y, w, h):
        return None
    try:
        return int(float(x)), int(float(y)), int(float(w)), int(float(h))
    except (ValueError, TypeError):
        return None


def _row_to_center(row, bbox=None):
    """Safe center from CSV row. Falls back to bbox mid-point."""
    cx = row.get("center_x", None)
    cy = row.get("center_y", None)
    if cx is not None and cy is not None:
        try:
            return int(float(cx)), int(float(cy))
        except (ValueError, TypeError):
            pass
    if bbox:
        x, y, w, h = bbox
        return x + w // 2, y + h // 2
    return None


def _load_csv(path):
    """Load trajectory CSV, return {frame_id: row} dict."""
    rows = {}
    p = Path(path)
    if not p.exists():
        return rows
    with open(str(p), newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            fid = int(row["frame_id"])
            rows[fid] = row
    return rows


def run_render(scene_key, max_frames=0):
    cfg = get_scene_config(scene_key)
    prefix = cfg["output_prefix"]
    traj_dir = OUTPUT_SUBDIRS["trajectories"]
    video_out = OUTPUT_SUBDIRS["videos"] / f"{prefix}_tracking_merged.mp4"

    # Load forward trajectory
    fwd = _load_csv(traj_dir / f"{prefix}_trajectory.csv")
    # Load backward
    bwd = _load_csv(traj_dir / f"{prefix}_backward.csv")
    anchor_frame = cfg.get("scene3_backward_anchor_frame", 84)

    # Merge: backward wins for frame < anchor, forward for frame >= anchor
    merged = {}
    for fid, row in bwd.items():
        if fid < anchor_frame:
            merged[fid] = row
    for fid, row in fwd.items():
        if fid >= anchor_frame:
            merged[fid] = row

    print(f"  Forward rows: {len(fwd)}, Backward rows: {len(bwd)}, "
          f"Merged: {len(merged)}")

    # Save merged CSV
    if merged:
        mpath = traj_dir / f"{prefix}_merged.csv"
        fields = ["frame_id", "x", "y", "w", "h", "center_x", "center_y",
                   "score", "detected", "used_for_trajectory",
                   "track_direction", "reject_reason"]
        with open(str(mpath), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            for fid in sorted(merged):
                w.writerow(merged[fid])
        print(f"  Merged CSV: {mpath}")

    # Render video from frame 0
    cap = cv2.VideoCapture(cfg["video"])
    if not cap.isOpened():
        print(f"  ERROR: Cannot open {cfg['video']}", file=sys.stderr)
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0:
        fps = 30.0
    if max_frames and max_frames < total:
        total = max_frames

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_out), fourcc, fps, (fw, fh))

    trajectory = []
    for fid in range(total):
        ret, frame = cap.read()
        if not ret:
            break

        if fid in merged:
            row = merged[fid]
            bb = _row_to_bbox(row)
            center = _row_to_center(row, bb)
            is_backward = row.get("track_direction", "") == "backward"

            if bb and center:
                x, y, w, h = bb
                # Color: backward=yellow, forward=green
                color = (0, 255, 255) if is_backward else (0, 255, 0)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.circle(frame, center, 3, (0, 0, 255), -1)
                trajectory.append(center)

            # Status text
            status = "BACKWARD" if is_backward else "TRACK"
            cv2.putText(frame, f"scene3 f{fid} {status}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Draw historical trajectory
        for i in range(1, len(trajectory)):
            p1 = trajectory[i - 1]
            p2 = trajectory[i]
            if p1 and p2:
                cv2.line(frame, p1, p2, (255, 0, 0), 1)

        writer.write(frame)

        if total > 20 and fid % max(1, total // 10) == 0:
            print(f"  Rendering: {100 * fid // total}%")

    cap.release()
    writer.release()
    print(f"  Output video: {video_out}")
    print(f"  Frames: {total}, Trajectory points: {len(trajectory)}")


def main():
    parser = argparse.ArgumentParser(description="Scene3 merged video renderer")
    parser.add_argument("--scene", default="scene3_bicycle",
                        help="Scene key (default: scene3_bicycle)")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="Max frames to render (0=all)")
    args = parser.parse_args()
    run_render(args.scene, args.max_frames)


if __name__ == "__main__":
    main()
