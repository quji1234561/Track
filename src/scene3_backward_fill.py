"""Scene3 backward trajectory fill — standalone NCC tracking from anchor backward.

Runs AFTER forward tracking completes. Uses local NCC search on earlier frames
starting from the anchor bbox, gradually stepping backward until tracking fails.
"""

import csv
from pathlib import Path

import cv2
import numpy as np

from .ncc import multi_template_search
from .preprocess import preprocess_template


def run_backward_fill(video_path, cfg, anchor_frame, anchor_bbox, output_dir=None):
    """Run backward NCC search from anchor_frame-1 down to 0.

    Returns list of dicts: [{frame_id, x, y, w, h, center_x, center_y, score, ...}, ...]
    Saves backward.csv to output_dir.
    """
    backward_end = cfg.get("scene3_backward_end_frame", 0)
    search_radius = cfg.get("scene3_backward_search_radius", 75)
    threshold = cfg.get("scene3_backward_threshold", 0.55)
    max_lost = cfg.get("scene3_backward_max_lost", 5)
    max_jump = cfg.get("scene3_backward_max_jump", 45)
    rs = cfg.get("resize_scale", 1.0)

    # Load templates
    templates = []
    scales = cfg.get("multi_scale", [1.0])
    for tid, tp in enumerate(cfg["templates"]):
        for img, s in preprocess_template(tp, scales=scales):
            templates.append((img.astype(np.float32), s, tid))

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [Backward] Cannot open video: {video_path}")
        return []

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  [Backward] Anchor frame={anchor_frame}, bbox={anchor_bbox}, "
          f"searching back to {backward_end}...")

    prev_bbox = anchor_bbox
    results = []
    lost_count = 0

    for fid in range(anchor_frame - 1, backward_end - 1, -1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        if rs < 1.0:
            h, w = gray.shape
            gray = cv2.resize(gray, (int(w * rs), int(h * rs)))

        # Search window around prev_bbox
        prev_x, prev_y, prev_w, prev_h = prev_bbox
        prev_cx = int((prev_x + prev_w // 2) * rs)
        prev_cy = int((prev_y + prev_h // 2) * rs)
        rad = int(search_radius * rs)

        x1 = max(0, prev_cx - rad)
        y1 = max(0, prev_cy - rad)
        x2 = min(gray.shape[1], prev_cx + rad)
        y2 = min(gray.shape[0], prev_cy + rad)
        sw = gray[y1:y2, x1:x2]
        if sw.shape[0] < 10 or sw.shape[1] < 10:
            lost_count += 1
            if lost_count >= max_lost:
                break
            continue

        # Scale templates to match resize_scale
        wt = []
        if rs < 1.0:
            for t, sc, tid in templates:
                th, tw = t.shape
                wt.append((cv2.resize(t, (int(tw * rs), int(th * rs))).astype(np.float32), sc, tid))
        else:
            wt = templates

        step = max(1, cfg.get("ncc_step", 2) - 1)
        result = multi_template_search(sw, wt, step=step, use_integral=True)

        if result is None or result["score"] < threshold:
            lost_count += 1
            if lost_count >= max_lost:
                print(f"  [Backward] Lost at frame {fid} (score="
                      f"{result['score'] if result else 'N/A'})")
                break
            continue

        gx = result["x"] + x1
        gy = result["y"] + y1
        gw, gh = result["w"], result["h"]
        mc = gx + gw // 2
        my = gy + gh // 2
        ox = int(gx / rs)
        oy = int(gy / rs)
        ow = int(gw / rs)
        oh = int(gh / rs)
        ocx = int(mc / rs)
        ocy = int(my / rs)

        # Jump check
        prev_ocx = prev_bbox[0] + prev_bbox[2] // 2
        prev_ocy = prev_bbox[1] + prev_bbox[3] // 2
        dist = np.sqrt((ocx - prev_ocx)**2 + (ocy - prev_ocy)**2)
        if dist > max_jump:
            lost_count += 1
            if lost_count >= max_lost:
                print(f"  [Backward] Jump too large at frame {fid} (dist={dist:.0f})")
                break
            continue

        lost_count = 0
        prev_bbox = [ox, oy, ow, oh]
        row = {
            "frame_id": fid,
            "x": ox, "y": oy, "w": ow, "h": oh,
            "center_x": ocx, "center_y": ocy,
            "score": result["score"],
            "detected": 1,
            "used_for_trajectory": 1,
            "track_direction": "backward",
            "reject_reason": "",
        }
        results.append(row)

    cap.release()

    # Sort ascending
    results.sort(key=lambda r: r["frame_id"])
    print(f"  [Backward] Filled {len(results)} frames "
          f"({results[0]['frame_id'] if results else 'N/A'} → "
          f"{results[-1]['frame_id'] if results else 'N/A'})")

    # Save CSV
    if output_dir:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        fields = ["frame_id", "x", "y", "w", "h", "center_x", "center_y",
                   "score", "detected", "used_for_trajectory",
                   "track_direction", "reject_reason"]
        bw_path = out_dir / "scene3_bicycle_backward.csv"
        with open(str(bw_path), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(results)
        print(f"  [Backward] Saved: {bw_path}")

    return results
