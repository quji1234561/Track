"""Manual ground-truth annotation tool for tracking evaluation.

Usage:
    python -m src.annotation_tool --scene scene1_animation --interval 20

Displays frames at regular intervals; user clicks the target center.
 - Mouse click: record (x, y)
 - 's': save current annotation
 - 'n': skip to next frame
 - 'q': quit
 - 'd': delete last annotation

Output: outputs/metrics/{prefix}_ground_truth.csv
"""

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

from .config import get_scene_config, OUTPUT_SUBDIRS, ensure_output_dirs

GT_FIELDS = ["frame_id", "gt_x", "gt_y", "visible"]

_click_point = None
_annotations = []
_current_visible = True


def _mouse_callback(event, x, y, flags, param):
    global _click_point
    if event == cv2.EVENT_LBUTTONDOWN:
        _click_point = (x, y)


def run_annotation(scene_key, interval=20):
    """Run interactive annotation tool."""
    global _click_point, _annotations, _current_visible

    cfg = get_scene_config(scene_key)
    prefix = cfg["output_prefix"]

    cap = cv2.VideoCapture(cfg["video"])
    if not cap.isOpened():
        print(f"ERROR: Cannot open video: {cfg['video']}", file=sys.stderr)
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Annotation tool for: {cfg['name']}")
    print(f"Total frames: {total_frames}, interval: {interval}")
    print("Controls: click=mark center | s=save | n=skip | d=delete last | v=toggle visible | q=quit")

    cv2.namedWindow("Annotation", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Annotation", _mouse_callback)

    frame_idx = 0
    while frame_idx < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break

        _click_point = None
        display = frame.copy()

        # Show existing annotations near this frame
        nearby = [a for a in _annotations if abs(a[0] - frame_idx) <= interval]
        cv2.putText(
            display, f"Frame: {frame_idx} / {total_frames}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
        )
        cv2.putText(
            display, f"Visible: {'YES' if _current_visible else 'NO'}",
            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
        )
        cv2.putText(
            display, "Click target center, then s/n/d/v/q",
            (10, display.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            (255, 255, 255), 1,
        )

        cv2.imshow("Annotation", display)
        key = cv2.waitKey(0) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("s") and _click_point is not None:
            _annotations.append((frame_idx, _click_point[0], _click_point[1], int(_current_visible)))
            print(f"  Saved: frame={frame_idx}, pos={_click_point}, visible={_current_visible}")
            frame_idx += interval
        elif key == ord("n"):
            frame_idx += interval
        elif key == ord("d"):
            if _annotations:
                removed = _annotations.pop()
                print(f"  Deleted: frame={removed[0]}, pos=({removed[1]}, {removed[2]})")
        elif key == ord("v"):
            _current_visible = not _current_visible
        elif _click_point is not None:
            _annotations.append((frame_idx, _click_point[0], _click_point[1], int(_current_visible)))
            print(f"  Saved: frame={frame_idx}, pos={_click_point}, visible={_current_visible}")
            frame_idx += interval

    cap.release()
    cv2.destroyAllWindows()

    # Save annotations
    if _annotations:
        ensure_output_dirs()
        output_path = OUTPUT_SUBDIRS["metrics"] / f"{prefix}_ground_truth.csv"
        with open(str(output_path), "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(GT_FIELDS)
            writer.writerows(_annotations)
        print(f"\nAnnotations saved to: {output_path}")
        print(f"Total annotations: {len(_annotations)}")
    else:
        print("\nNo annotations recorded.")


def main():
    parser = argparse.ArgumentParser(
        description="Manual annotation tool for tracking ground truth"
    )
    parser.add_argument(
        "--scene", type=str, required=True,
        help="Scene key (e.g. scene1_animation)",
    )
    parser.add_argument(
        "--interval", type=int, default=20,
        help="Annotation interval in frames (default: 20)",
    )
    args = parser.parse_args()
    run_annotation(args.scene, args.interval)


if __name__ == "__main__":
    main()
