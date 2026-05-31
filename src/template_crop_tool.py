"""Template cropping tool — manually select ROI from original video frames.

Usage:
    python -m src.template_crop_tool --scene scene2_car --frame 30 --name my_template.png

This tool uses cv2.selectROI for manual bounding-box selection only.
No matchTemplate, no tracking, no AI/ML — just video read + crop + save.
"""

import argparse
import sys
import os
from pathlib import Path

import cv2
import numpy as np

from .config import SCENES, get_scene_config, PROJECT_ROOT

DEBUG_DIR = PROJECT_ROOT / "outputs" / "debug" / "templates"


# ---------------------------------------------------------------------------
#  Fallback: manual ROI selection via mouse callback
#  (avoid cv2.selectROI Qt issues on Windows)
# ---------------------------------------------------------------------------

def _manual_roi_select(win_name, frame):
    """Manual ROI selection using raw mouse callbacks.

    Returns (x, y, w, h) tuple, or (0,0,0,0) if cancelled.
    """
    state = {"drawing": False, "start": (0, 0), "end": (0, 0), "done": False}

    def _on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            state["drawing"] = True
            state["start"] = (x, y)
            state["end"] = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and state["drawing"]:
            state["end"] = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            state["drawing"] = False
            state["end"] = (x, y)

    cv2.setMouseCallback(win_name, _on_mouse)
    clone = frame.copy()
    print("  Draw ROI with mouse, then press ENTER to confirm or C to cancel.")

    while True:
        display = clone.copy()
        if state["drawing"] or state["start"] != state["end"]:
            x1, y1 = state["start"]
            x2, y2 = state["end"]
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imshow(win_name, display)
        key = cv2.waitKey(20) & 0xFF

        if key == 13:  # Enter
            x1, y1 = state["start"]
            x2, y2 = state["end"]
            x = min(x1, x2)
            y = min(y1, y2)
            w = abs(x2 - x1)
            h = abs(y2 - y1)
            if w > 0 and h > 0:
                return (x, y, w, h)
            print("  ROI too small, draw again.")
        elif key == ord("c") or key == 27:  # c or Esc
            return (0, 0, 0, 0)


def run_crop(scene_key, frame_id, output_name, output_dir, preview_dir):
    """Open video, jump to frame, let user select ROI, save template."""

    cfg = get_scene_config(scene_key)
    video_path = cfg["video"]

    print(f"Scene:         {scene_key} — {cfg['name']}")
    print(f"Video:         {video_path}")
    print(f"Target frame:  {frame_id}")

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video: {video_path}", file=sys.stderr)
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_id >= total_frames:
        print(f"ERROR: frame {frame_id} >= total frames {total_frames}",
              file=sys.stderr)
        cap.release()
        sys.exit(1)

    # Seek to frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print(f"ERROR: Could not read frame {frame_id}", file=sys.stderr)
        sys.exit(1)

    fh, fw = frame.shape[:2]
    print(f"Frame shape:   {fw}x{fh}")

    # --- Scale frame for display only (original is kept for cropping) ---
    # Large portrait frames (e.g. 1080×1920) don't fit on screen;
    # OpenCV can't scroll, so the bottom gets clipped.
    # We scale the display image to fit max_display_height, then map
    # ROI coordinates back to the original frame.
    max_display_h = 900
    if fh > max_display_h:
        disp_scale = max_display_h / fh
        disp_w = int(fw * disp_scale)
        disp_h = max_display_h
        display_frame = cv2.resize(frame, (disp_w, disp_h))
        print(f"Display scale: {disp_scale:.2f} → display {disp_w}x{disp_h}")
    else:
        disp_scale = 1.0
        display_frame = frame

    # --- Manual ROI selection ---
    win_name = f"ROI: {scene_key} frame {frame_id}  ENTER=confirm  C=cancel"
    cv2.imshow(win_name, display_frame)
    cv2.waitKey(1)  # let Qt event loop initialize the window
    try:
        roi = cv2.selectROI(win_name, display_frame, showCrosshair=True,
                            fromCenter=False)
    except cv2.error:
        print("cv2.selectROI failed — using manual mouse selection fallback.")
        roi = _manual_roi_select(win_name, display_frame)
    cv2.destroyAllWindows()

    dx, dy, dw, dh = roi

    if dw == 0 or dh == 0:
        print("ERROR: ROI is empty (w=0 or h=0). Selection cancelled or invalid.",
              file=sys.stderr)
        sys.exit(1)

    # Map display-space ROI back to original-frame coords
    if disp_scale != 1.0:
        x = int(dx / disp_scale)
        y = int(dy / disp_scale)
        w = int(dw / disp_scale)
        h = int(dh / disp_scale)
    else:
        x, y, w, h = dx, dy, dw, dh

    # Crop from original frame (no resize)
    template = frame[y:y + h, x:x + w]

    # Save template
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmpl_path = out_dir / output_name
    cv2.imwrite(str(tmpl_path), template)

    # Save preview with red rectangle
    prev_dir = Path(preview_dir)
    prev_dir.mkdir(parents=True, exist_ok=True)
    preview = frame.copy()
    cv2.rectangle(preview, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.putText(preview, f"ROI: {w}x{h}", (x, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
    prev_name = Path(output_name).stem + "_preview.png"
    prev_path = prev_dir / prev_name
    cv2.imwrite(str(prev_path), preview)

    # Summary
    print(f"\n{'='*50}")
    print(f"ROI:           x={x}, y={y}, w={w}, h={h}")
    print(f"Template:      {tmpl_path}  ({w}x{h})")
    print(f"Preview:       {prev_path}")
    print(f"\nAdd to config.py templates list:")
    print(f'  "document/{output_name}",')
    print(f"{'='*50}")


def main():
    parser = argparse.ArgumentParser(
        description="Crop template from original video frame"
    )
    parser.add_argument("--scene", type=str, required=True,
                        help="Scene key (e.g. scene2_car)")
    parser.add_argument("--frame", type=int, required=True,
                        help="Frame number to crop from")
    parser.add_argument("--name", type=str, required=True,
                        help="Output filename (e.g. my_template.png)")
    parser.add_argument("--output-dir", type=str, default="document",
                        help="Output directory (default: document/)")
    parser.add_argument("--preview-dir", type=str,
                        default=str(DEBUG_DIR),
                        help="Preview image directory")
    args = parser.parse_args()

    if args.scene not in SCENES:
        print(f"Unknown scene: {args.scene}", file=sys.stderr)
        print(f"Available: {', '.join(SCENES.keys())}", file=sys.stderr)
        sys.exit(1)

    run_crop(args.scene, args.frame, args.name,
             args.output_dir, args.preview_dir)


if __name__ == "__main__":
    main()
