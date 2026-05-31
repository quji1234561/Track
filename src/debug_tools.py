"""Diagnostic tools for tracking system debugging.

Checks: NCC self-match, template/frame consistency, full-frame score distribution,
coordinate mapping sanity, and debug visualization.

Usage:
    python -m src.debug_tools --scene scene1_animation
    python -m src.debug_tools --scene all
"""

import argparse
import csv
import sys
import os
from pathlib import Path

import cv2
import numpy as np

from .config import SCENES, get_scene_config, PROJECT_ROOT, ensure_output_dirs
from .preprocess import preprocess_template, preprocess_frame, _imread_unicode
from .ncc import ncc_score, ncc_search_integral, multi_template_search
from .tracker import TraditionalTracker

DEBUG_DIR = PROJECT_ROOT / "outputs" / "debug"


def _p(s):
    """Print with flush for reliable output in piped environments."""
    print(s, flush=True)


# ====================================================================
#  Step 1: Template verification
# ====================================================================

def check_template_loading(scene_key):
    """Verify all templates load correctly and print their properties."""
    cfg = get_scene_config(scene_key)
    _p(f"\n{'='*60}")
    _p(f"TEMPLATE CHECK: {scene_key} - {cfg['name']}")
    _p(f"{'='*60}")
    _p(f"Video: {cfg['video']}")
    _p(f"resize_scale: {cfg.get('resize_scale', 1.0)}")
    _p(f"threshold: {cfg['threshold']}")
    _p(f"ncc_step: {cfg.get('ncc_step', 2)}")

    all_ok = True
    for tid, tpath in enumerate(cfg["templates"]):
        _p(f"\n  Template {tid}: {tpath}")
        if not os.path.exists(tpath):
            _p(f"    ERROR: File does not exist!")
            all_ok = False
            continue

        # Read raw
        try:
            raw = _imread_unicode(tpath)
        except Exception as e:
            _p(f"    ERROR reading file: {e}")
            all_ok = False
            continue

        if raw is None:
            _p(f"    ERROR: cv2.imdecode returned None")
            all_ok = False
            continue

        h, w = raw.shape[:2]
        channels = raw.shape[2] if raw.ndim == 3 else 1
        _p(f"    Raw: {w}x{h}, channels={channels}")

        # Check for annotation marks (blue/yellow circles, etc.)
        if channels == 3:
            # Check if template has unusual colored marks
            b, g, r = raw[:, :, 0].mean(), raw[:, :, 1].mean(), raw[:, :, 2].mean()
            _p(f"    Mean BGR: ({b:.1f}, {g:.1f}, {r:.1f})")
            # Very blue/red/green regions might indicate annotation
            blue_ratio = b / max(g, r, 1.0)
            if blue_ratio > 1.5:
                _p(f"    WARNING: Template has unusually high blue channel (annotation mark?)")

        # Preprocess (as tracker would)
        try:
            tmpls = preprocess_template(tpath, scales=cfg.get("multi_scale", [1.0]))
        except Exception as e:
            _p(f"    ERROR in preprocess_template: {e}")
            all_ok = False
            continue

        for t, s in tmpls:
            th, tw = t.shape
            t_min, t_max = t.min(), t.max()
            t_std = t.std()
            _p(f"    Preprocessed scale={s:.2f}: {tw}x{th}, "
              f"range=[{t_min:.4f}, {t_max:.4f}], std={t_std:.4f}")

            if t_std < 1e-6:
                _p(f"    WARNING: Template std near zero (uniform) - NCC will fail!")
                all_ok = False

            if th < 5 or tw < 5:
                _p(f"    WARNING: Template too small ({tw}x{th})")
                all_ok = False

    # Check video opens
    cap = cv2.VideoCapture(cfg["video"])
    if not cap.isOpened():
        _p(f"\n  ERROR: Cannot open video!")
        all_ok = False
    else:
        ret, frame = cap.read()
        if ret:
            fh, fw = frame.shape[:2]
            _p(f"\n  Video frame: {fw}x{fh}")
            if cfg.get("resize_scale", 1.0) < 1.0:
                sfh, sfw = int(fh * cfg["resize_scale"]), int(fw * cfg["resize_scale"])
                _p(f"  Scaled frame: {sfw}x{sfh}")

            gray = preprocess_frame(frame)
            _p(f"  Gray frame range: [{gray.min():.4f}, {gray.max():.4f}]")

            # Check: is any template larger than frame?
            for tid, tpath in enumerate(cfg["templates"]):
                raw = _imread_unicode(tpath)
                th, tw = raw.shape[:2]
                if th > fh or tw > fw:
                    _p(f"  WARNING: Template {tid} ({tw}x{th}) larger than frame ({fw}x{fh})!")
                    all_ok = False
                rs = cfg.get("resize_scale", 1.0)
                if th * rs > fh * rs or tw * rs > fw * rs:
                    pass  # same ratio

        cap.release()

    return all_ok


# ====================================================================
#  Step 2: NCC self-match test
# ====================================================================

def test_ncc_self_match(scene_key):
    """Template self-match: template matched against itself should give ~1.0."""
    cfg = get_scene_config(scene_key)
    _p(f"\n{'='*60}")
    _p(f"NCC SELF-MATCH TEST: {scene_key}")
    _p(f"{'='*60}")

    all_ok = True
    for tid, tpath in enumerate(cfg["templates"]):
        tmpls = preprocess_template(tpath, scales=cfg.get("multi_scale", [1.0]))
        for t, s in tmpls:
            score = ncc_score(t.copy(), t.copy())
            status = "OK" if score > 0.95 else "FAIL"
            if score <= 0.95:
                all_ok = False
            _p(f"  Template {tid} scale={s:.2f} ({t.shape[1]}x{t.shape[0]}): "
              f"self-NCC={score:.6f} [{status}]")

            if score <= 0.95:
                _p(f"    -> NCC self-match FAILED! Check ncc_score implementation.")
                _p(f"       Template mean={t.mean():.6f}, std={t.std():.6f}")
                t_diff = t - t.mean()
                num = np.sum(t_diff * t_diff)
                denom = np.sqrt(np.sum(t_diff**2) * np.sum(t_diff**2))
                _p(f"       num={num:.10f}, denom={denom:.10f}")

    return all_ok


# ====================================================================
#  Step 3: Full-frame search on first N frames
# ====================================================================

def full_frame_score_scan(scene_key, num_frames=20, save_debug=True):
    """Run full-frame NCC on first N frames, collect score distributions."""
    cfg = get_scene_config(scene_key)
    _p(f"\n{'='*60}")
    _p(f"FULL-FRAME SCORE SCAN: {scene_key} - first {num_frames} frames")
    _p(f"{'='*60}")

    cap = cv2.VideoCapture(cfg["video"])
    if not cap.isOpened():
        _p(f"ERROR: Cannot open video")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    num_frames = min(num_frames, total_frames)

    # Load templates (same as tracker)
    templates = []
    for tid, tpath in enumerate(cfg["templates"]):
        tmpls = preprocess_template(tpath, scales=cfg.get("multi_scale", [1.0]))
        for tmpl_img, sc in tmpls:
            templates.append((tmpl_img, sc, tid))

    rs = cfg.get("resize_scale", 1.0)
    all_scores = []
    best_per_frame = []

    DEBUG_DIR.mkdir(parents=True, exist_ok=True)

    for fi in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break

        gray = preprocess_frame(frame)
        if rs < 1.0:
            h, w = gray.shape
            gray = cv2.resize(gray, (int(w * rs), int(h * rs)))

        # Use same search scaling as tracker
        search_scale = _compute_search_scale(templates, gray.shape[0], gray.shape[1], rs)
        if search_scale < 1.0:
            sh, sw = int(gray.shape[0] * search_scale), int(gray.shape[1] * search_scale)
            search_img = cv2.resize(gray, (sw, sh))
        else:
            search_img = gray

        scaled_templates = _scale_templates(templates, search_scale)
        step = max(2, cfg.get("ncc_step", 3))

        result = multi_template_search(
            search_img, scaled_templates, step=step,
            use_integral=True, verbose=False,
        )

        if result is not None:
            score = result["score"]
            tmpl_id = result["template_id"]
            # Map coords back to original
            ox = int(result["x"] / search_scale / max(rs, 1e-6))
            oy = int(result["y"] / search_scale / max(rs, 1e-6))
            _p(f"  Frame {fi:4d}: best_score={score:.4f}, template={tmpl_id}, "
              f"pos=({ox}, {oy})")
            all_scores.append(score)
            best_per_frame.append({
                "frame_id": fi, "score": score, "template_id": tmpl_id,
                "x": ox, "y": oy,
            })
        else:
            _p(f"  Frame {fi:4d}: NO MATCH")
            best_per_frame.append({"frame_id": fi, "score": -1, "template_id": -1,
                                    "x": -1, "y": -1})

        # Save debug images for first frame and every 5th frame
        if save_debug and (fi == 0 or fi % 5 == 0 or fi == num_frames - 1):
            _save_debug_frame(cfg, frame, gray, templates, fi, search_img,
                            scaled_templates, result, search_scale, rs)

    cap.release()

    # Score statistics
    if all_scores:
        all_scores = np.array(all_scores)
        _p(f"\n  Score statistics (N={len(all_scores)}):")
        _p(f"    Max:    {all_scores.max():.4f}")
        _p(f"    Min:    {all_scores.min():.4f}")
        _p(f"    Mean:   {all_scores.mean():.4f}")
        _p(f"    Median: {np.median(all_scores):.4f}")
        _p(f"    Std:    {all_scores.std():.4f}")
        _p(f"    Top 10: {sorted(all_scores, reverse=True)[:10]}")

        # Count template usage
        from collections import Counter
        tmpl_counts = Counter(r["template_id"] for r in best_per_frame if r["score"] >= 0)
        _p(f"    Template usage: {dict(tmpl_counts)}")

        # Suggest threshold
        suggested = max(0.10, all_scores.mean() - all_scores.std())
        _p(f"    Suggested threshold: {suggested:.4f} (mean - 1 std)")
        _p(f"    Current threshold: {cfg['threshold']}")

    # Save score CSV
    if save_debug and best_per_frame:
        csv_path = DEBUG_DIR / f"{cfg['output_prefix']}_score_debug.csv"
        with open(str(csv_path), "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["frame_id", "score", "template_id", "x", "y"])
            writer.writeheader()
            writer.writerows(best_per_frame)
        _p(f"\n  Score debug CSV: {csv_path}")

    return best_per_frame


def _compute_search_scale(templates, img_h, img_w, pre_scale):
    """Replicate tracker's _get_search_scale logic for standalone debug."""
    if not templates:
        return 0.5
    MIN_TMPL_DIM = 60
    min_dim = min(min(t.shape[0], t.shape[1]) for t, _, _ in templates)
    safe_scale = MIN_TMPL_DIM / min_dim if min_dim > 0 else 1.0
    clamped = max(0.35, min(1.0, safe_scale))
    scale = min(0.5, clamped)
    if pre_scale < 1.0 and scale < 1.0:
        if pre_scale * scale < 0.35:
            scale = min(1.0, 0.35 / pre_scale)
    return scale


def _scale_templates(templates, scale):
    """Scale templates — mirror of tracker._scale_templates."""
    scaled = []
    for tmpl_img, sc, tid in templates:
        th, tw = tmpl_img.shape
        small_t = cv2.resize(tmpl_img, (int(tw * scale), int(th * scale)))
        scaled.append((small_t.astype(np.float32), sc, tid))
    return scaled


def _save_debug_frame(cfg, raw_frame, gray_frame, templates, fi,
                      search_img, search_templates, result, search_scale, rs):
    """Save debug visualization: raw frame, preprocessed, search image, match result."""
    prefix = cfg["output_prefix"]
    base = DEBUG_DIR / f"{prefix}_frame_{fi:04d}"

    # 1) Raw frame
    cv2.imwrite(str(base) + "_01_raw.png", raw_frame)

    # 2) Preprocessed search image (scale back for visualization)
    vis_gray = (gray_frame * 255).astype(np.uint8)
    cv2.imwrite(str(base) + "_02_preprocessed.png", vis_gray)

    # 3) Search image at search scale
    vis_search = (search_img * 255).astype(np.uint8)
    cv2.imwrite(str(base) + "_03_search_img.png", vis_search)

    # 4) Match result on raw frame
    if result is not None and result["score"] >= 0:
        vis_result = raw_frame.copy()
        ox = int(result["x"] / search_scale / max(rs, 1e-6))
        oy = int(result["y"] / search_scale / max(rs, 1e-6))
        ow = int(result["w"] / search_scale / max(rs, 1e-6))
        oh = int(result["h"] / search_scale / max(rs, 1e-6))
        cv2.rectangle(vis_result, (ox, oy), (ox + ow, oy + oh), (0, 255, 0), 2)
        cv2.putText(vis_result, f"score={result['score']:.3f}", (ox, oy - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        cv2.imwrite(str(base) + "_04_match.png", vis_result)

    # 5) Template at search scale
    if search_templates:
        t0 = search_templates[0][0]
        vis_tmpl = (t0 * 255).astype(np.uint8)
        cv2.imwrite(str(base) + "_05_template.png", vis_tmpl)


# ====================================================================
#  Step 4: Check preprocessing consistency
# ====================================================================

def check_preprocess_consistency(scene_key):
    """Verify frame and template preprocessing are consistent."""
    cfg = get_scene_config(scene_key)
    _p(f"\n{'='*60}")
    _p(f"PREPROCESS CONSISTENCY CHECK: {scene_key}")
    _p(f"{'='*60}")

    cap = cv2.VideoCapture(cfg["video"])
    ret, frame = cap.read()
    cap.release()
    if not ret:
        _p("ERROR: Cannot read video frame")
        return

    # Frame preprocessing (as done in main.py)
    gray_frame = preprocess_frame(frame)
    rs = cfg.get("resize_scale", 1.0)
    if rs < 1.0:
        h, w = gray_frame.shape
        gray_frame = cv2.resize(gray_frame, (int(w * rs), int(h * rs)))

    _p(f"  Frame preprocessed: {gray_frame.shape}, "
      f"dtype={gray_frame.dtype}, "
      f"range=[{gray_frame.min():.4f},{gray_frame.max():.4f}], "
      f"mean={gray_frame.mean():.4f}, std={gray_frame.std():.4f}")

    for tid, tpath in enumerate(cfg["templates"]):
        tmpls = preprocess_template(tpath, scales=cfg.get("multi_scale", [1.0]))
        for t, s in tmpls:
            # Resize template to match frame's resize_scale
            if rs < 1.0:
                th, tw = t.shape
                t_rs = cv2.resize(t, (int(tw * rs), int(th * rs)))
                t_rs = (t_rs - t_rs.min()) / max(t_rs.max() - t_rs.min(), 1e-10)
            else:
                t_rs = t

            _p(f"  Template {tid} scale={s:.2f} (after frame-scale match): {t_rs.shape}, "
              f"dtype={t_rs.dtype}, "
              f"range=[{t_rs.min():.4f},{t_rs.max():.4f}], "
              f"mean={t_rs.mean():.4f}, std={t_rs.std():.4f}")

            if t_rs.shape[0] > gray_frame.shape[0] or t_rs.shape[1] > gray_frame.shape[1]:
                _p(f"    WARNING: Template larger than frame!")


# ====================================================================
#  Step 5: Check template scales
# ====================================================================

def debug_check_template_scales(scene_key):
    """Verify multi-scale templates are generated correctly and save debug images."""
    cfg = get_scene_config(scene_key)
    _p(f"\n{'='*60}")
    _p(f"TEMPLATE SCALE CHECK: {scene_key}")
    _p(f"{'='*60}")

    scales = cfg.get("multi_scale", [1.0])
    src_count = len(cfg["templates"])
    expected = src_count * len(scales)
    _p(f"  Source templates: {src_count}")
    _p(f"  Scales: {scales}")
    _p(f"  Expected candidates: {expected}")

    tmpl_dir = DEBUG_DIR / "templates" / scene_key
    tmpl_dir.mkdir(parents=True, exist_ok=True)

    actual = 0
    all_sizes = []

    for tid, tpath in enumerate(cfg["templates"]):
        try:
            tmpls = preprocess_template(tpath, scales=scales)
        except Exception as e:
            _p(f"  ERROR loading template {tid} ({tpath}): {e}")
            continue

        for tmpl_img, s in tmpls:
            actual += 1
            th, tw = tmpl_img.shape
            all_sizes.append((tw, th))
            name = Path(tpath).stem
            status = ""
            if th < 5 or tw < 5:
                status = " [WARNING: too small]"
            _p(f"  Candidate: source={tid} scale={s:.2f} -> {tw}×{th}{status}")

            # Save scaled template image for visual inspection
            vis = (tmpl_img * 255).astype(np.uint8)
            out_name = f"{name}_s{s:.2f}.png"
            cv2.imwrite(str(tmpl_dir / out_name), vis)

    _p(f"\n  Actual candidates: {actual}")
    if actual != expected:
        _p(f"  WARNING: expected {expected} but got {actual}!")

    if all_sizes:
        unique_sizes = len(set((w, h) for w, h in all_sizes))
        if unique_sizes <= 1 and len(scales) > 1:
            _p(f"  WARNING: All templates have the same size despite "
              f"{len(scales)} scales — multi-scale may not be effective.")
        _p(f"  Unique sizes: {unique_sizes} (from {actual} candidates)")

    _p(f"  Debug images saved to: {tmpl_dir}")


# ====================================================================
#  Main diagnostic entry
# ====================================================================

def run_diagnostics(scene_key, num_frames=20):
    """Run full diagnostics on a scene."""
    _p(f"\n{'#'*60}")
    _p(f"# DIAGNOSTICS: {scene_key}")
    _p(f"{'#'*60}")

    # Step 1
    templates_ok = check_template_loading(scene_key)

    # Step 2
    ncc_ok = test_ncc_self_match(scene_key)

    # Step 3
    preprocess_ok = check_preprocess_consistency(scene_key)

    # Step 4 — template scale check
    debug_check_template_scales(scene_key)

    # Step 5 (must be last — most expensive)
    scores = full_frame_score_scan(scene_key, num_frames=num_frames)

    _p(f"\n  SUMMARY: templates_ok={templates_ok}, ncc_self_ok={ncc_ok}")

    return templates_ok and ncc_ok


def main():
    parser = argparse.ArgumentParser(description="Tracking system diagnostics")
    parser.add_argument("--scene", type=str, required=True,
                        help="Scene key or 'all'")
    parser.add_argument("--num-frames", type=int, default=20,
                        help="Frames for score scan (default 20)")
    args = parser.parse_args()

    DEBUG_DIR.mkdir(parents=True, exist_ok=True)

    if args.scene == "all":
        for key in SCENES:
            run_diagnostics(key, num_frames=args.num_frames)
    elif args.scene in SCENES:
        run_diagnostics(args.scene, num_frames=args.num_frames)
    else:
        _p(f"Unknown scene: {args.scene}")
        _p(f"Available: {', '.join(SCENES.keys())}, all")
        sys.exit(1)


if __name__ == "__main__":
    main()
