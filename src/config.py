"""Scene configurations for four tracking experiments.

OpenCV is used only for video I/O, basic image conversion, and result drawing.
Core matching uses self-implemented NCC in src/ncc.py (with optional integral image).
Tracking uses multi-template search, local search window, and Kalman prediction.
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

# ===========================================================================
#  场景参数说明 —— 每个场景独立调参，调整后直接影响跟踪效果。
#
#  threshold       : NCC 匹配阈值 (0~1)。调高 = 更严格，误检少但容易丢目标。
#                    调低 = 更容易锁定目标，但可能匹配到错误区域。
#                    建议设为 debug_tools 输出的 (均值 - 1×标准差)。
#  search_radius   : 局部搜索窗口半径（原始帧像素）。调大 = 目标快速移动时
#                    不易跑出搜索范围，但每帧计算量按半径平方增长。
#                    丢失时会自动按 (1 + lost_count×0.5) 倍率扩大。
#  max_lost        : 连续丢失多少帧后触发全图重搜索。调大 = 更依赖卡尔曼预测
#                    （更快，能扛短时遮挡）。调小 = 更快触发重搜索找回目标
#                    （更慢，但恢复更快）。
#  multi_scale     : 模板多尺度因子列表。每增加一个尺度，搜索时间约翻倍。
#                    目标大小稳定时用 [1.0]；目标靠近/远离镜头时加更多尺度。
#  start_frame     : 起始帧号（0 = 第一帧）。如果目标在视频后面才出现，设 >0。
#  use_integral_ncc: True = float64 积分图加速 NCC（大范围搜索更快）。
#                    False = stride-trick 矢量化 NCC（精调步骤使用）。
#  ncc_step        : NCC 滑窗步长（像素）。1 = 逐像素（最精确最慢）。
#                    3~4 = 跳着搜（快很多，局部搜索会精调，精度损失很小）。
#  resize_scale    : 帧缩放比例。1.0 = 原始分辨率（最精确，最慢）。
#                    0.5 = 宽高各缩一半（面积 1/4，约 4 倍速）。
#                    0.4 = 缩到 40%（面积约 1/6，约 6 倍速）。
#                    输出坐标始终自动映射回原始帧，不影响 CSV 和画框。
#  show_predicted_bbox     : 是否在输出视频中绘制预测黄框。
#                            False = 预测帧不画黄色预测框（卡尔曼仍正常运行）。
#  draw_predicted_trajectory: 是否将预测中心点加入轨迹线。
#                            False = 轨迹只用 detected 帧的中心点，避免预测不准时污染轨迹。
#
#  --- 初始化连续确认（防止提前锁到错误目标）---
#  enable_init_confirmation : True = 需要连续 N 帧确认后才正式初始化。
#  init_confirm_frames      : 连续确认帧数。例如 3 = 需连续 3 帧在附近检测到目标。
#  init_confirm_min_score   : 确认期间每帧至少需要的 NCC 分数。
#  init_confirm_max_distance: 确认期间相邻帧匹配中心的最大允许距离（像素）。
#
#  --- 运动方向约束（限制目标只能沿某一方向移动）---
#  enable_motion_direction  : True = 仅接受符合预期运动方向的匹配。
#  motion_direction_y       : 0=不限制, -1=目标只能向上(y减小), 1=目标只能向下(y增大)。
#  motion_direction_tolerance: 允许的逆向运动容差（像素）。例如 10 = 允许 y 反向 10px。
#
#  --- 跳变检测（防止检测框突然跳到远处）---
#  enable_jump_detection    : True = 检测到跳变时拒绝该匹配。
#  jump_max_distance        : 允许的最大帧间中心位移（像素）。超过则视为跳变。
#  jump_max_area_change     : 允许的最大 bbox 面积变化倍数。例如 3.0 = 面积变化不超过 3 倍。
#
#  --- 预测约束（让卡尔曼预测框也受方向/跳变/尺寸限制）---
#  constrain_predictions             : True = 卡尔曼预测框也要通过方向/跳变/尺寸检查。
#                                      False = 预测框不受限制, 跟随卡尔曼自由移动。
#  prediction_max_center_jump        : 预测框中心相对上一帧的最大位移（像素）。
#  prediction_max_y_reverse          : 预测框 y 方向最大逆向位移（像素）。
#  prediction_max_area_change_ratio  : 预测框面积相对上一帧的最大变化倍数。
#  prediction_max_width_change_ratio : 预测框宽度相对上一帧的最大变化倍数。
#  prediction_max_height_change_ratio: 预测框高度相对上一帧的最大变化倍数。
#  prediction_out_of_bounds_policy   : "reject"=超出画面就拒绝预测, "clamp"=裁切到画面内。
#  prediction_reject_adds_lost       : True = 预测被拒时也增加 lost_count。
# ===========================================================================

SCENES = {
    # =========================================================================
    # 场景1: 动画表情视频跟踪（竖屏 1080×1920, 5个模板, 单尺度）
    # 特点: 画面大部分为浅色背景, NCC 分数普遍偏低(0.18~0.32)
    # =========================================================================
    "scene1_animation": {
        "name": "动画表情视频跟踪",                     # 场景显示名称（用于视频标注和 CSV）
        "video": "document/动画表情视频.mp4",            # 视频文件路径（相对于项目根目录）
        "templates": [                                 # 模板图片列表（可多个, 逐一参与 NCC 搜索）
            "document/scene1_animation_template_1.png",
            "document/scene1_animation_template_2.png",
            "document/scene1_animation_template_3.png",
            "document/scene1_animation_template_4.png",
            "document/scene1_animation_template_5.png",
        ],
        "output_prefix": "scene1_animation",           # 输出文件前缀（视频/CSV/截图均以此为名）
        # --- 核心匹配 ---
        "threshold": 0.29,            # NCC 匹配阈值（调高=更严/误检少, 调低=易锁定/可能误匹配）
        "search_radius": 200,         # 局部搜索窗口半径(px)（丢失时自动按 lost_count×0.5 扩大）
        "max_lost": 10,               # 连续丢失多少帧后触发全图重搜索（调大=靠卡尔曼死撑, 调小=快找回）
        "multi_scale": [1.0],         # 模板多尺度因子（每增一个尺度, 搜索时间翻倍；此处目标大小稳定）
        "start_frame": 0,             # 起始帧号（0=第一帧开始；>0 则跳过前 N 帧再初始化）
        "use_integral_ncc": True,     # True=float64积分图加速NCC（大范围快）, False=stride-trick（小范围精调用）
        "ncc_step": 4,                # NCC滑窗步长(px)（1=逐像素最精确, 3~4=跳着搜快很多）
        "resize_scale": 0.4,          # 帧缩放比例（1080×1920→432×768, 像素量降至1/6, 大幅提速）
        # --- 可视化 ---
        "show_predicted_bbox": True,          # 是否在输出视频中绘制黄色预测框
        "draw_predicted_trajectory": False,    # 预测中心点是否加入轨迹线（False=只用 detected 点）
        # --- 初始化连续确认 ---
        "enable_init_confirmation": False,     # 是否启用连续确认（动画表情切换频繁, 不适合）
        "init_confirm_frames": 3,              # 需连续确认的帧数
        "init_confirm_min_score": 0.45,        # 确认期间最低 NCC 分数
        "init_confirm_max_distance": 30,       # 确认期间相邻帧中心最大距离(px)
        # --- 运动方向约束 ---
        "enable_motion_direction": False,      # 是否限制目标运动方向（动画目标移动方向不固定, 不启用）
        "motion_direction_y": 0,               # 0=不限, -1=只能向上(y减小), 1=只能向下(y增大)
        "motion_direction_tolerance": 10,      # 允许的逆向运动容差(px)
        # --- 跳变检测 ---
        "enable_jump_detection": False,        # 是否检测帧间跳变（防止检测框突然跳到远处）
        "jump_max_distance": 150,              # 最大帧间中心位移(px)（超过视为跳变拒绝）
        "jump_max_area_change": 3.0,           # 最大 bbox 面积变化倍数（超过视为跳变拒绝）
        # --- 预测约束 ---
        "constrain_predictions": False,              # 预测框是否受方向/跳变/尺寸限制（动画场景不启用）
        "prediction_max_center_jump": 60,             # 预测框中心相对上一帧的最大位移(px)
        "prediction_max_y_reverse": 5,                # 预测框 y 方向最大逆向位移(px)
        "prediction_max_area_change_ratio": 1.3,      # 预测框面积相对上一帧最大变化倍数
        "prediction_max_width_change_ratio": 1.3,     # 预测框宽度相对上一帧最大变化倍数
        "prediction_max_height_change_ratio": 1.3,    # 预测框高度相对上一帧最大变化倍数
        "prediction_out_of_bounds_policy": "reject",  # 预测出界策略: "reject"=拒绝预测, "clamp"=裁切到画面内
        "prediction_reject_adds_lost": True,          # 预测被拒时是否也增加 lost_count
    },

    # =========================================================================
    # 场景2: 大疆无人机航拍-车辆目标跟踪（640×512, 2个模板, 5个尺度）
    # 特点: 黑白航拍, 目标从下方驶向远处(y减小), 尺度逐渐变小
    # =========================================================================
    "scene2_car": {
        "name": "大疆无人机航拍视频-车辆目标跟踪",         # 场景显示名称
        "video": "document/大疆无人机航拍视频.mp4",        # 视频文件路径
        "templates": [                                    # 模板列表（注释掉的为备用模板, 需要时取消注释）
            "document/scene2_car_template_1.png",
            "document/scene2_car_template_2.png",
            # "document/scene2_car_template_3.png",
            # "document/scene2_car_template_4.png",
        ],
        "output_prefix": "scene2_car",                     # 输出文件前缀
        # --- 核心匹配 ---
        "threshold": 0.45,            # NCC 匹配阈值（降低到 0.45 以适应远距离车辆变小后的低分区域）
        "search_radius": 35,          # 局部搜索窗口半径(px)（缩小搜索范围, 防止跳到远处其他车辆）
        "max_lost": 30,               # 连续丢失容忍帧数（提高以减少全图重搜索, 避免匹配到其他车）
        "multi_scale": [0.7,0.85,0.95,1.0,1.1],  # 多尺度（0.70~1.10 覆盖车辆从近到远的尺度变化）
        "start_frame": 0,             # 起始帧号（从第0帧开始, 配合 init_confirmation 防止误初始化）
        "use_integral_ncc": True,     # 积分图加速 NCC
        "ncc_step": 2,                # 滑窗步长（模板小~76px, 用较细步长保证精度）
        "resize_scale": 1.0,          # 帧缩放（640×512 已够小, 无需缩放）
        # --- 可视化 ---
        "show_predicted_bbox": True,           # 显示预测黄框（卡尔曼漂移时参考用）
        "draw_predicted_trajectory": False,     # 预测点不参与轨迹（避免污染运动轨迹）
        # --- 初始化连续确认 ---
        "enable_init_confirmation": True,      # 启用连续确认（防止初始锁到车道旁的反向车辆）
        "init_confirm_frames": 4,              # 需连续 4 帧确认（比默认 3 更严格）
        "init_confirm_min_score": 0.65,        # 确认期间最低 NCC 分数（提高门槛, 防止低分误初始化）
        "init_confirm_max_distance": 25,       # 确认期间最大帧间位移(px)
        # --- 运动方向约束 ---
        "enable_motion_direction": True,       # 启用方向约束（车从画面下方驶向远处, y 应持续减小）
        "motion_direction_y": -1,              # -1=只能向上移动（y 减小）
        "motion_direction_tolerance": 15,      # 逆向容差(px)（允许偶尔帧间 y 微增 15px 以内）
        # --- 跳变检测 ---
        "enable_jump_detection": True,         # 启用跳变检测（防止检测框跳到其他车道车辆）
        "jump_max_distance": 50,               # 最大帧间位移(px)（车辆帧间移动有限, 超过 50px 视为跳变）
        "jump_max_area_change": 3.0,           # 最大面积变化倍数（尺度突变 >3x 视为跳变）
        # --- 预测约束 ---
        "constrain_predictions": True,               # 预测框也受方向/跳变/尺寸限制
        "prediction_max_center_jump": 40,             # 预测框帧间最大位移(px)
        "prediction_max_y_reverse": 3,                # 预测框 y 方向最大逆向位移(px)（车不应倒退）
        "prediction_max_area_change_ratio": 1.3,      # 预测框面积变化上限
        "prediction_max_width_change_ratio": 1.3,     # 预测框宽度变化上限
        "prediction_max_height_change_ratio": 1.3,    # 预测框高度变化上限
        "prediction_out_of_bounds_policy": "reject",  # 预测出界策略: 拒绝
        "prediction_reject_adds_lost": True,          # 预测被拒也计入连续丢失
    },

    # =========================================================================
    # 场景3: 大疆无人机航拍-骑车人目标跟踪（640×512, 3个模板, 5个尺度）
    # 特点: 骑车人目标小(~27x36px), 移动快, 与路面背景对比度低
    # =========================================================================
    "scene3_bicycle": {
        "name": "大疆无人机航拍视频-骑车人目标跟踪",       # 场景显示名称
        "video": "document/大疆无人机航拍骑车人.mp4",      # 视频文件路径
        "templates": [                                    # 模板列表
            # "document/scene3_bicycle_template_2.png",
            "document/scene3_bicycle_template_3.png",
            "document/scene3_bicycle_template9.png",
            "document/scene3_bicycle_template10.png",
        ],
        "output_prefix": "scene3_bicycle",                 # 输出文件前缀
        # --- 核心匹配 ---
        "threshold": 0.81,             # NCC 匹配阈值（提高以保证小目标的匹配置信度）
        "search_radius": 100,         # 局部搜索窗口半径(px)（骑车人移动快, 需较大搜索范围）
        "max_lost": 15,               # 连续丢失容忍帧数
        "multi_scale": [0.8, 0.9, 1.0, 1.1, 1.2],  # 多尺度（骑车人尺度变化范围大）
        "start_frame": 50,             # 起始帧号
        "use_integral_ncc": True,     # 积分图加速 NCC
        "ncc_step": 2,                # 滑窗步长（模板很小~27px, 必须用细步长）
        "resize_scale": 1.0,          # 帧缩放（640×512 已够小）
        # --- 可视化 ---
        "show_predicted_bbox": False,          # 隐藏预测黄框（避免错误预测框干扰观察）
        "draw_predicted_trajectory": False,     # 预测点不参与轨迹
        # --- 初始化连续确认 ---
        "enable_init_confirmation": True,      # 启用连续确认
        "init_confirm_frames": 3,              # 需连续 3 帧确认
        "init_confirm_min_score": 0.8,         # 确认期间最低 NCC 分数（高阈值保证初始化质量）
        "init_confirm_max_distance": 30,       # 确认期间最大帧间位移(px)
        # --- 运动方向约束 ---
        "enable_motion_direction": True,       # 启用方向约束
        "motion_direction_y": 1,              # 1=只能向下移动（y 增大）
        "motion_direction_tolerance": 20,      # 逆向容差(px)
        # --- 跳变检测 ---
        "enable_jump_detection": True,         # 启用跳变检测
        "jump_max_distance": 30,              # 最大帧间位移(px)（超过视为跳变拒绝）
        "jump_max_area_change": 3.0,           # 最大面积变化倍数
        # --- 预测约束 ---
        "constrain_predictions": True,              # 预测框也受方向/跳变/尺寸限制
        "prediction_max_center_jump": 50,
        "prediction_max_y_reverse": 5,
        "prediction_max_area_change_ratio": 1.3,
        "prediction_max_width_change_ratio": 1.3,
        "prediction_max_height_change_ratio": 1.3,
        "prediction_out_of_bounds_policy": "reject",
        "prediction_reject_adds_lost": True,
    },

    # =========================================================================
    # 场景4: 地面光学站跟踪无人机（2448×2048, 6个模板, 3个尺度）
    # 特点: 大画幅(20MP), 无人机目标小(~98x44px), 需 resize_scale 加速
    # =========================================================================
    "scene4_drone": {
        "name": "地面光学站跟踪无人机",                     # 场景显示名称
        "video": "document/地面光学站跟踪无人机.avi",        # 视频文件路径（AVI 格式, 大文件）
        "templates": [                                    # 模板列表（多模板）
            "document/scene4_drone_template_1.png",
            "document/scene4_drone_template_2.png",
            "document/scene4_drone_template_3.png",
            "document/scene4_drone_template_4.png",
            "document/scene4_drone_template_5.png",
            "document/scene4_drone_template_6.png",

        ],
        "output_prefix": "scene4_drone",                    # 输出文件前缀
        # --- 核心匹配 ---
        "threshold": 0.50,            # NCC 匹配阈值（场景4 分数约 0.52~0.53）
        "search_radius": 180,         # 局部搜索窗口半径(px)（大画幅中无人机移动快, 需大范围）
        "max_lost": 25,               # 连续丢失容忍帧数（无人机可能短暂飞出视野）
        "multi_scale": [ 0.9, 1.0, 1.1,],  # 多尺度（无人机距离变化时尺度改变明显）
        "start_frame": 0,             # 起始帧号
        "use_integral_ncc": True,     # 积分图加速 NCC
        "ncc_step": 3,                # 滑窗步长（模板中等~98px, 适中步长平衡速度与精度）
        "resize_scale": 0.5,          # 帧缩放（2448×2048→1224×1024, 像素量降至 1/4, 大幅提速）
        # --- 可视化 ---
        "show_predicted_bbox": False,          # 隐藏预测黄框（大画幅中卡尔曼预测漂移明显）
        "draw_predicted_trajectory": False,     # 预测点不参与轨迹
        # --- 初始化连续确认 ---
        "enable_init_confirmation": True,     # 启用连续确认
        "init_confirm_frames": 3,
        "init_confirm_min_score": 0.5,
        "init_confirm_max_distance": 30,
        # --- 运动方向约束 ---
        "enable_motion_direction": False,      # 不启用（无人机移动方向不固定）
        "motion_direction_y": 0,
        "motion_direction_tolerance": 10,
        # --- 跳变检测 ---
        "enable_jump_detection": False,        # 不启用（无人机帧间位移可能较大）
        "jump_max_distance": 150,
        "jump_max_area_change": 3.0,
        # --- 预测约束 ---
        "constrain_predictions": False,              # 预测框是否受限制（无人机场景不启用）
        "prediction_max_center_jump": 60,
        "prediction_max_y_reverse": 5,
        "prediction_max_area_change_ratio": 1.3,
        "prediction_max_width_change_ratio": 1.3,
        "prediction_max_height_change_ratio": 1.3,
        "prediction_out_of_bounds_policy": "reject",
        "prediction_reject_adds_lost": True,
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
