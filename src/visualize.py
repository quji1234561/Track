"""Visualization: draw bounding boxes, center points, trajectories, and info text.

OpenCV drawing functions (cv2.rectangle, cv2.circle, cv2.line, cv2.putText)
are the only OpenCV functions used here — they are permitted per project constraints.
"""

import cv2
import numpy as np


def draw_tracking_result(frame, result, trajectory, scene_name):
    """Draw tracking result overlay on a video frame.

    Args:
        frame: BGR color frame (will be modified in-place).
        result: tracking result dict from tracker.track_frame().
        trajectory: list of (cx, cy) points for trail drawing.
        scene_name: scene display name for top-left label.

    Returns:
        The modified frame.
    """
    bbox = result["bbox"]
    x, y, w, h = bbox
    score = result["score"]
    detected = result["detected"]
    predicted = result["predicted"]
    frame_id = result["frame_id"]
    cx, cy = x + w // 2, y + h // 2

    # Box color: green = detected, yellow = predicted
    color = (0, 255, 0) if detected else (0, 255, 255)
    thickness = 2
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)

    # Center point
    cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

    # Trajectory trail
    pts = trajectory + [(cx, cy)]
    for i in range(1, len(pts)):
        p1 = pts[i - 1]
        p2 = pts[i]
        cv2.line(frame, p1, p2, (255, 0, 0), 1)

    # Info overlay - top left
    label_lines = [
        f"{scene_name}",
        f"Frame: {frame_id}",
        f"Score: {score:.3f}",
        f"Center: ({cx}, {cy})",
        f"Status: {'DETECT' if detected else 'PREDICT' if predicted else 'NONE'}",
    ]
    y0 = 25
    for i, text in enumerate(label_lines):
        cv2.putText(
            frame, text, (10, y0 + i * 22),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
            cv2.LINE_AA,
        )

    return frame
