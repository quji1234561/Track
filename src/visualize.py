"""可视化绘制: 检测框、预测框、中心点、轨迹线、信息文字。

绘制规则:
- detected=True: 绿色实线框 + 红色中心点（NCC匹配成功）
- predicted=True 且 show_predicted_bbox=True: 黄色实线框 + 红色中心点（卡尔曼预测）
- detected=False 且 predicted=False: 不画框不画点
- 轨迹线: 蓝色，根据传入的trajectory列表绘制（由调用者控制哪些点加入轨迹）

tracking_stop_frame后的行为:
- bbox=[0,0,0,0]→w=0,h=0→矩形不可见
- detected=False,predicted=False→不画框不画点
- 但trajectory列表非空→历史轨迹线仍正常绘制
- 左上角显示tracking stopped提示

OpenCV仅用于rectangle/circle/line/putText基础绘图函数。
"""

import cv2
import numpy as np


def draw_tracking_result(frame, result, trajectory, scene_name,
                         show_predicted_bbox=True,
                         draw_predicted_trajectory=True):
    """Draw tracking result overlay on a video frame.

    Args:
        frame: BGR color frame (modified in-place).
        result: tracking result dict from tracker.track_frame().
        trajectory: list of (cx, cy) points already selected for trail drawing.
        scene_name: scene display name for top-left label.
        show_predicted_bbox: if False, skip drawing yellow predicted box
                             (Kalman still runs internally).
        draw_predicted_trajectory: if False, trajectory only uses detected points
                                   (already handled by caller; this flag is for
                                   the status label).

    Returns:
        The modified frame.
    """
    bbox = result.get("bbox")
    if bbox is None or (isinstance(bbox, (list, tuple)) and len(bbox) == 4 and bbox[2] <= 0 and bbox[3] <= 0):
        # No valid bbox — skip drawing boxes/centers, only draw trajectory
        x = y = w = h = 0
    else:
        x, y, w, h = bbox
    score = result["score"]
    detected = result["detected"]
    predicted = result.get("predicted", False)
    frame_id = result["frame_id"]
    lost_count = result.get("lost_count", 0)
    cx, cy = x + w // 2, y + h // 2

    # --- Bounding box ---
    if detected:
        # Green box for accepted NCC detection
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
    elif predicted and show_predicted_bbox:
        # Yellow box only if configured to show predictions
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
    # else: predicted but show_predicted_bbox=False → no box, no center dot

    # --- Trajectory trail ---
    # trajectory list is already filtered by the caller based on
    # draw_predicted_trajectory — we just draw whatever points are provided.
    for i in range(1, len(trajectory)):
        p1 = trajectory[i - 1]
        p2 = trajectory[i]
        cv2.line(frame, p1, p2, (255, 0, 0), 1)

    # --- Info overlay (top-left) ---
    if detected:
        status = "DETECT"
    elif predicted:
        status = "PREDICT"
    else:
        status = "LOST"

    label_lines = [
        f"{scene_name}",
        f"Frame: {frame_id}  Lost: {lost_count}",
        f"Score: {score:.3f}  Status: {status}",
        f"Center: ({cx}, {cy})",
    ]
    if predicted and not show_predicted_bbox:
        label_lines.append("(pred bbox hidden)")
    if predicted and not draw_predicted_trajectory:
        label_lines.append("(pred traj hidden)")

    y0 = 25
    for i, text in enumerate(label_lines):
        cv2.putText(
            frame, text, (10, y0 + i * 22),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
            cv2.LINE_AA,
        )

    return frame
