"""Scene configurations for four tracking experiments.

OpenCV is used only for video I/O, basic image conversion, and result drawing.
Core matching uses self-implemented NCC in src/ncc.py.
Tracking uses multi-template search, local search window, and Kalman prediction in src/tracker.py.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCUMENT_DIR = PROJECT_ROOT / "document"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

OUTPUT_SUBDIRS = {
    "videos": OUTPUTS_DIR / "videos",
    "frames": OUTPUTS_DIR / "frames",
    "trajectories": OUTPUTS_DIR / "trajectories",
    "metrics": OUTPUTS_DIR / "metrics",
    "logs": OUTPUTS_DIR / "logs",
}

SCENES = {
    "scene1_animation": {
        "name": "动画表情视频跟踪",
        "video": "document/动画表情视频.mp4",
        "templates": [
            "document/动画表情视频1template.png",
            "document/动画表情视频2template.png",
            "document/动画表情视频3template.png",
        ],
        "output_prefix": "scene1_animation",
        "threshold": 0.55,
        "search_radius": 120,
        "max_lost": 15,
        "multi_scale": [1.0],
        "start_frame": 0,
    },
    "scene2_car": {
        "name": "大疆无人机航拍视频-车辆目标跟踪",
        "video": "document/大疆无人机航拍视频.mp4",
        "templates": [
            "document/大疆无人机航拍视频template.png",
        ],
        "output_prefix": "scene2_car",
        "threshold": 0.50,
        "search_radius": 160,
        "max_lost": 20,
        "multi_scale": [0.9, 1.0, 1.1],
        "start_frame": 0,
    },
    "scene3_bicycle": {
        "name": "大疆无人机航拍视频-骑车人目标跟踪",
        "video": "document/大疆无人机航拍骑车人.mp4",
        "templates": [
            "document/大疆无人机航拍骑车人template.png",
        ],
        "output_prefix": "scene3_bicycle",
        "threshold": 0.48,
        "search_radius": 180,
        "max_lost": 20,
        "multi_scale": [0.8, 0.9, 1.0, 1.1, 1.2],
        "start_frame": 0,
    },
    "scene4_drone": {
        "name": "地面光学站跟踪无人机",
        "video": "document/地面光学站跟踪无人机.avi",
        "templates": [
            "document/地面光学站跟踪无人机template.png",
        ],
        "output_prefix": "scene4_drone",
        "threshold": 0.45,
        "search_radius": 220,
        "max_lost": 25,
        "multi_scale": [0.8, 0.9, 1.0, 1.1, 1.2],
        "start_frame": 0,
    },
}


def ensure_output_dirs():
    """Create all output directories if they don't exist."""
    for d in OUTPUT_SUBDIRS.values():
        d.mkdir(parents=True, exist_ok=True)


def get_scene_config(scene_key: str) -> dict:
    """Get scene configuration, resolving relative paths to PROJECT_ROOT."""
    if scene_key not in SCENES:
        available = ", ".join(SCENES.keys())
        raise ValueError(f"Unknown scene '{scene_key}'. Available: {available}")
    cfg = SCENES[scene_key].copy()
    cfg["video"] = str(PROJECT_ROOT / cfg["video"])
    cfg["templates"] = [str(PROJECT_ROOT / t) for t in cfg["templates"]]
    return cfg
