"""指标统计与CSV输出。

输出文件:
- trajectory.csv: 每帧一行，含bbox位置、中心点、NCC分数、检测/预测状态、
  used_for_trajectory标志、template_id
- metrics.csv: 单场景总体指标（总帧数、检测帧数、预测帧数、检测率、
  平均NCC分数、处理速度FPS）
- summary_metrics.csv: 四个场景的指标汇总对比

基础指标计算(无需真值): detection_rate = detected_frames / total_frames
                         average_score = mean of non-zero NCC scores
                         average_fps = total_frames / elapsed_time

像素误差计算(需真值): compute_metrics_with_ground_truth()接收人工标注CSV后
                     计算mean/max pixel error, miss_rate, false_rate

不依赖OpenCV，仅使用标准库+pandas。
"""

import csv
import time
from pathlib import Path

import pandas as pd


TRAJECTORY_FIELDS = [
    "frame_id", "x", "y", "w", "h",
    "center_x", "center_y", "score",
    "detected", "predicted", "used_for_trajectory",
    "template_id",
]

BASIC_METRIC_FIELDS = [
    "scene", "total_frames", "detected_frames", "predicted_frames",
    "lost_frames", "detection_rate", "prediction_rate",
    "average_score", "average_fps",
]

GT_FIELDS = ["frame_id", "gt_x", "gt_y", "visible"]


def save_trajectory_csv(results, output_path):
    """Save per-frame tracking results as CSV."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=TRAJECTORY_FIELDS)
        writer.writeheader()
        for r in results:
            row = {
                "frame_id": r["frame_id"],
                "x": r["bbox"][0],
                "y": r["bbox"][1],
                "w": r["bbox"][2],
                "h": r["bbox"][3],
                "center_x": r["center"][0],
                "center_y": r["center"][1],
                "score": r["score"],
                "detected": int(r["detected"]),
                "predicted": int(r.get("predicted", False)),
                "used_for_trajectory": int(r.get("used_for_trajectory", False)),
                "template_id": r["template_id"],
            }
            writer.writerow(row)


def compute_basic_metrics(results, total_time, scene_name):
    """Compute basic tracking metrics without ground truth.

    Args:
        results: list of tracking result dicts.
        total_time: total processing time in seconds.
        scene_name: display name of the scene.

    Returns:
        dict of basic metrics.
    """
    valid = [r for r in results if r is not None]
    total_frames = len(valid)
    detected_frames = sum(1 for r in valid if r["detected"])
    predicted_frames = sum(1 for r in valid if r["predicted"])
    lost_frames = total_frames - detected_frames - sum(
        1 for r in valid if not r["detected"] and not r["predicted"]
    )
    # Actually, lost_frames = total - detected - predicted
    # But predicted frames mean we lost target and used prediction
    # Let's define: detected=target found, predicted=kalman only, lost=neither
    really_lost = sum(
        1 for r in valid if not r["detected"] and not r["predicted"]
    )
    # Actually, for frames where target is not detected but predicted=True, those are "predicted"
    # Frames where both False would be true lost frames
    scores = [r["score"] for r in valid if r["score"] > 0]
    avg_score = sum(scores) / len(scores) if scores else 0.0

    detection_rate = detected_frames / total_frames if total_frames > 0 else 0.0
    prediction_rate = predicted_frames / total_frames if total_frames > 0 else 0.0
    avg_fps = total_frames / total_time if total_time > 0 else 0.0

    return {
        "scene": scene_name,
        "total_frames": total_frames,
        "detected_frames": detected_frames,
        "predicted_frames": predicted_frames,
        "lost_frames": really_lost,
        "detection_rate": detection_rate,
        "prediction_rate": prediction_rate,
        "average_score": avg_score,
        "average_fps": avg_fps,
    }


def save_basic_metrics_csv(metrics_dict, output_path):
    """Save single-scene basic metrics as CSV."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=BASIC_METRIC_FIELDS)
        writer.writeheader()
        writer.writerow(metrics_dict)


def save_summary_metrics(all_metrics, output_path):
    """Save combined summary metrics for all scenes."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(all_metrics)
    # Add pixel-error columns as N/A when no ground truth
    for col in ["mean_pixel_error", "max_pixel_error", "miss_rate", "false_rate"]:
        if col not in df.columns:
            df[col] = "N/A"
    cols = BASIC_METRIC_FIELDS + ["mean_pixel_error", "max_pixel_error", "miss_rate", "false_rate"]
    df = df.reindex(columns=[c for c in cols if c in df.columns])
    df.to_csv(output_path, index=False, encoding="utf-8")


def compute_metrics_with_ground_truth(results, gt_df):
    """Compute pixel-error metrics when ground truth is available.

    Args:
        results: list of tracking result dicts.
        gt_df: pandas DataFrame with frame_id, gt_x, gt_y, visible columns.

    Returns:
        dict with mean_pixel_error, max_pixel_error, miss_rate, false_rate.
    """
    gt_map = {}
    for _, row in gt_df.iterrows():
        fid = int(row["frame_id"])
        gt_map[fid] = (float(row["gt_x"]), float(row["gt_y"]), int(row.get("visible", 1)))

    errors = []
    misses = 0
    false_detects = 0
    total_gt = 0

    for r in results:
        if r is None:
            continue
        fid = r["frame_id"]
        if fid in gt_map:
            gx, gy, vis = gt_map[fid]
            if vis:
                total_gt += 1
                cx = r["center"][0]
                cy = r["center"][1]
                err = ((cx - gx) ** 2 + (cy - gy) ** 2) ** 0.5
                errors.append(err)
                if not r["detected"]:
                    misses += 1
            elif r["detected"]:
                false_detects += 1

    return {
        "mean_pixel_error": sum(errors) / len(errors) if errors else None,
        "max_pixel_error": max(errors) if errors else None,
        "miss_rate": misses / total_gt if total_gt > 0 else None,
        "false_rate": false_detects / len(results) if results else None,
    }
