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
#  max_motion_distance: 允许匹配点离卡尔曼预测点的最大距离(px)。
#                        调大=容忍更远匹配, 调小=限制必须靠近预测。
#                        不设置则回退到 max(50, search_radius*0.75)。
#  ncc_step        : NCC 滑窗步长（像素）。1 = 逐像素（最精确最慢）。
#                    3~4 = 跳着搜（快很多，局部搜索会精调，精度损失很小）。
#  init_search_roi : 初始化搜索 ROI [x, y, w, h]（原始帧坐标）。
#                    限制初始化阶段的 NCC 全图搜索范围。不设置则全图搜索。
#                    设置后可排除画面外干扰区域（如树丛/建筑），防止误初始化。
#  resize_scale    : 帧缩放比例。1.0 = 原始分辨率（最精确，最慢）。
#                    0.5 = 宽高各缩一半（面积 1/4，约 4 倍速）。
#                    0.4 = 缩到 40%（面积约 1/6，约 6 倍速）。
#                    输出坐标始终自动映射回原始帧，不影响 CSV 和画框。
#  show_predicted_bbox     : 是否在输出视频中绘制预测黄框。
#                            False = 预测帧不画黄色预测框（卡尔曼仍正常运行）。
#  draw_predicted_trajectory: 是否将预测中心点加入轨迹线。
#                            False = 轨迹只用 detected 帧的中心点，避免预测不准时污染轨迹。
#  tracking_stop_frame: 停止跟踪帧号。>此帧号后不再调用 NCC 搜索和卡尔曼预测，
#                        不画框，不新增轨迹点，仅输出原视频画面。
#                        不设置或为 None 则跟踪到视频结束。
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
#  --- y 前进速度限制（防止检测框冲到目标前方）---
#  enable_y_forward_speed_limit       : True = 限制候选不能沿正确方向冲太快。
#  max_y_decrease_per_frame           : 候选 y 相对上一帧最多减少多少像素。
#  max_candidate_ahead_of_prediction_y: 候选 y 相对 Kalman 预测最多超前多少像素。
#  max_candidate_behind_prediction_y  : 候选 y 相对 Kalman 预测最多落后多少像素。
#                                       0=不限制。>0 时拒绝落后太多的静止误匹配。
#
#  --- Top-K 候选筛选（不只选最高分，逐个检查前 K 名候选）---
#  enable_topk_candidate_selection    : True = 启用 Top-K 候选逐个检查。
#  topk_candidates                    : 保留前几名空间候选并逐个验证。
#  topk_min_score                     : 候选最低接受分数（低于此分直接跳过）。
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
           
        ],
        "output_prefix": "scene1_animation",           # 输出文件前缀（视频/CSV/截图均以此为名）
        # --- 核心匹配 ---
        "threshold": 0.26,            # NCC 匹配阈值（调高=更严/误检少, 调低=易锁定/可能误匹配）
        "search_radius": 200,         # 局部搜索窗口半径(px)（丢失时自动按 lost_count×0.5 扩大）
        "max_lost": 10,               # 连续丢失多少帧后触发全图重搜索（调大=靠卡尔曼死撑, 调小=快找回）
        "multi_scale": [0.7,0.9,1.0,1.1],         # 模板多尺度因子（每增一个尺度, 搜索时间翻倍；此处目标大小稳定）
        "start_frame": 0,             # 起始帧号（0=第一帧开始；>0 则跳过前 N 帧再初始化）
        "use_integral_ncc": True,     # True=float64积分图加速NCC（大范围快）, False=stride-trick（小范围精调用）
        "max_motion_distance": 9999,     # 允许离预测点的最大距离(px), 0=回退到自动计算
        "ncc_step": 4,                # NCC滑窗步长(px)（1=逐像素最精确, 3~4=跳着搜快很多）
        "resize_scale": 0.4,          # 帧缩放比例（1080×1920→432×768, 像素量降至1/6, 大幅提速）
        # --- 可视化 ---
        "show_predicted_bbox": False,          # 是否在输出视频中绘制黄色预测框
        "draw_predicted_trajectory": False,    # 预测中心点是否加入轨迹线（False=只用 detected 点）
        # --- 初始化连续确认 ---
        "enable_init_confirmation": False,     # 是否启用连续确认（动画表情切换频繁, 不适合）
        "init_confirm_frames": 3,              # 需连续确认的帧数
        "init_confirm_min_score": 0.3,        # 确认期间最低 NCC 分数
        "init_confirm_max_distance": 30,       # 确认期间相邻帧中心最大距离(px)
        # --- 运动方向约束 ---
        "enable_motion_direction": False,      # 是否限制目标运动方向（动画目标移动方向不固定, 不启用）
        "motion_direction_y": 0,               # 0=不限, -1=只能向上(y减小), 1=只能向下(y增大)
        "motion_direction_tolerance": 10,      # 允许的逆向运动容差(px)
        # --- 跳变检测 ---
        "enable_jump_detection": False,        # 是否检测帧间跳变（防止检测框突然跳到远处）
        "jump_max_distance": 150,              # 最大帧间中心位移(px)（超过视为跳变拒绝）
        "jump_max_area_change": 3.0,           # 最大 bbox 面积变化倍数（超过视为跳变拒绝）
        # --- y 前进速度限制 ---
        "enable_y_forward_speed_limit": False,       # 是否限制 y 方向前进速度（动画不启用）
        "max_y_decrease_per_frame": 18,
        "max_candidate_ahead_of_prediction_y": 18,
        "max_candidate_behind_prediction_y": 0,      # 0=不限制
        # --- Top-K 候选筛选 ---
        "enable_topk_candidate_selection": False,    # 是否启用 Top-K 候选筛选（动画不启用）
        "topk_candidates": 8,
        "topk_min_score": 0.38,
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
        "templates": [                                    # 模板列表
            "document/scene2_car_template_1.png",
            # 补充模板: python -m src.template_crop_tool --scene scene2_car --frame 18 --name scene2_car_template_2.png
            "document/scene2_car_template_2.png",
            "document/scene2_car_template_3.png",
        ],
        "output_prefix": "scene2_car",                     # 输出文件前缀
        # --- 核心匹配 ---
        "threshold": 0.36,            # 降低阈值，让候选能先进来再筛选
        "search_radius": 70,          # 扩大搜索范围（车辆在帧20附近位置变化大）
        "max_lost": 8,                # 恢复策略：丢8帧就全图重搜索
        "multi_scale": [0.55,0.65,0.75,0.85,0.95,1.0],  # 更宽尺度范围适应车辆快速变小
        "start_frame": 0,
        "use_integral_ncc": True,
        "max_motion_distance": 90,    # 放宽距离容忍
        "ncc_step": 2,                # 步长2平衡速度精度
        "resize_scale": 1.0,
        # --- 可视化 ---
        "show_predicted_bbox": False,
        "draw_predicted_trajectory": False,
        # --- 初始化连续确认 ---
        "enable_init_confirmation": True,
        "init_confirm_frames": 2,
        "init_confirm_min_score": 0.55,
        "init_confirm_max_distance": 40,
        # --- 运动方向约束 ---
        "enable_motion_direction": True,
        "motion_direction_y": -1,              # 车向上移动(y减小)
        "motion_direction_tolerance": 8,
        # --- 跳变检测 ---
        "enable_jump_detection": True,
        "jump_max_distance": 55,               # 放宽跳变限制
        "jump_max_area_change": 0,
        # --- y 前进速度限制 ---
        "enable_y_forward_speed_limit": False,  # 关闭，防止误拒绝真实车辆候选
        "max_y_decrease_per_frame": 18,
        "max_candidate_ahead_of_prediction_y": 18,
        "max_candidate_behind_prediction_y": 20,
        # --- Top-K 候选筛选 ---
        "enable_topk_candidate_selection": True,
        "topk_candidates": 12,
        "topk_min_score": 0.28,                 # 降低候选门槛
        # --- 预测约束 ---
        "constrain_predictions": True,
        "prediction_max_center_jump": 45,
        "prediction_max_y_reverse": 5,
        "prediction_max_area_change_ratio": 1.5,
        "prediction_max_width_change_ratio": 1.5,
        "prediction_max_height_change_ratio": 1.5,
        "prediction_out_of_bounds_policy": "reject",
        "prediction_reject_adds_lost": True,
        # --- 车辆运动先验 ---
        "scene2_use_vehicle_prior": True,
        "scene2_candidate_topk": 20,
        "scene2_max_lateral_shift": 25,
        "scene2_direction_penalty_weight": 0.15,
        "scene2_distance_penalty_weight": 0.20,
        "scene2_scale_penalty_weight": 0.10,
        "scene2_forward_speed_penalty_weight": 0.20,
        "scene2_ncc_weight": 0.50,
        "scene2_scale_should_decrease": True,
        # 前向/横向硬门控（延迟启用，只在稳定跟踪 N 帧后才激活）
        "scene2_forward_gate_min_stable_frames": 25,    # 连续 detected 25帧后才启用
        "scene2_max_forward_step": 8,                   # 每帧最多向前(y减小)8px
        "scene2_backward_tolerance": 4,                 # 每帧最多后退(y增大)4px
        "scene2_max_lateral_step": 12,                  # 每帧最多横向12px
        "scene2_relaxed_forward_step_when_lost": 15,    # 丢失后放宽到15px
        # 恢复阶段门控（防止 lost 后锁到前方错误目标）
        "scene2_recovery_gate_enabled": True,             # lost 后恢复时启用空间门控
        "scene2_recovery_max_pred_y_error": 8,            # 候选与预测 y 最大偏差(px)
        "scene2_recovery_max_pred_x_error": 12,           # 候选与预测 x 最大偏差(px)
        "scene2_recovery_max_total_distance": 18,         # 候选与预测最大总距离(px)
    },

    # =========================================================================
    # 场景3: 大疆无人机航拍-骑车人目标跟踪（640×512, 3个模板, 5个尺度）
    # 特点: 骑车人目标小(~27x36px), 移动快, 与路面背景对比度低
    # =========================================================================
    "scene3_bicycle": {
        "name": "大疆无人机航拍视频-骑车人目标跟踪",       # 场景显示名称
        "video": "document/大疆无人机航拍骑车人.mp4",      # 视频文件路径
        "templates": [                                    # 模板列表
            "document/scene3_bicycle_template_1.png",
            "document/scene3_bicycle_template_2.png",
            "document/scene3_bicycle_template_3.png",
        ],
        "output_prefix": "scene3_bicycle",                 # 输出文件前缀
        # --- 核心匹配 ---
        "threshold": 0.76,             # NCC 匹配阈值（提高以保证小目标的匹配置信度）
        "search_radius": 80,         # 局部搜索窗口半径(px)（骑车人移动快, 需较大搜索范围）
        "max_lost": 20,               # 连续丢失容忍帧数
        "multi_scale": [0.85, 0.95, 1.0, 1.05, 1.15],  # 多尺度（骑车人尺度变化范围大）
        "start_frame": 83,             # 起始帧号
        "use_integral_ncc": True,     # 积分图加速 NCC
        "max_motion_distance": 0,     # 允许离预测点的最大距离(px), 0=回退到自动计算
        "ncc_step": 2,                # 滑窗步长（模板很小~27px, 必须用细步长）
        "resize_scale": 1.0,          # 帧缩放（640×512 已够小）
        # --- 可视化 ---
        "show_predicted_bbox": False,          # 隐藏预测黄框（避免错误预测框干扰观察）
        "draw_predicted_trajectory": False,     # 预测点不参与轨迹
        # --- 初始化连续确认 ---
        "enable_init_confirmation": False,      # 启用连续确认
        "init_confirm_frames": 3,              # 需连续 3 帧确认
        "init_confirm_min_score": 0.76,         # 确认期间最低 NCC 分数（高阈值保证初始化质量）
        "init_confirm_max_distance": 35,       # 确认期间最大帧间位移(px)
        # --- 运动方向约束 ---
        "enable_motion_direction": True,       # 启用方向约束
        "motion_direction_y": 1,              # 1=只能向下移动（y 增大）
        "motion_direction_tolerance": 18,      # 逆向容差(px)
        # --- 跳变检测 ---
        "enable_jump_detection": True,         # 启用跳变检测
        "jump_max_distance": 35,              # 最大帧间位移(px)（超过视为跳变拒绝）
        "jump_max_area_change": 2.5,           # 最大面积变化倍数
        # --- y 前进速度限制 ---
        "enable_y_forward_speed_limit": False,       # 骑车人场景不启用
        "max_y_decrease_per_frame": 18,
        "max_candidate_ahead_of_prediction_y": 18,
        "max_candidate_behind_prediction_y": 0,      # 0=不限制
        # --- Top-K 候选筛选 ---
        "enable_topk_candidate_selection": False,    # 骑车人场景不启用
        "topk_candidates": 8,
        "topk_min_score": 0.38,
        # --- 预测约束 ---
        "constrain_predictions": True,              # 预测框也受方向/跳变/尺寸限制
        "prediction_max_center_jump": 55,
        "prediction_max_y_reverse": 6,
        "prediction_max_area_change_ratio": 1.35,
        "prediction_max_width_change_ratio": 1.35,
        "prediction_max_height_change_ratio": 1.35,
        "prediction_out_of_bounds_policy": "reject",
        "prediction_reject_adds_lost": True,
        # --- 梯度增强NCC ---
        "scene3_use_gradient_ncc": True,              # 启用Sobel梯度NCC（增强小目标纹理）
        "scene3_gray_weight": 0.7,                    # 灰度NCC权重
        "scene3_grad_weight": 0.3,                    # 梯度NCC权重
        # --- 骑车人退出判断 ---
        "scene3_exit_y": 500,                         # 目标接近此y值时判定可能离开画面
        "scene3_stop_after_exit": True,               # 目标离开画面后停止跟踪
        "scene3_min_visible_score": 0.70,             # 判定目标可见的最低分数
        "scene3_exit_lost_frames": 5,                 # 连续低分帧数后判定退出
    },

    # =========================================================================
    # 场景4: 地面光学站跟踪无人机（2448×2048, 6个模板, 3个尺度）
    # 特点: 大画幅(20MP), 无人机目标小(~98x44px), 需 resize_scale 加速
    # =========================================================================
    "scene4_drone": {
        "name": "地面光学站跟踪无人机",                     # 场景显示名称
        "video": "document/地面光学站跟踪无人机.avi",        # 视频文件路径（AVI 格式, 大文件）
        "templates": [                                    # 模板列表（单模板验证初始化）
            "document/scene4_drone_template_1.png",
            "document/scene4_drone_template_2.png",
            "document/scene4_drone_template_3.png",
            "document/scene4_drone_template_4.png",
            "document/scene4_drone_template_5.png",

            "document/scene4_drone_template_6.png",
            "document/scene4_drone_template_7.png",
            "document/scene4_drone_template_8.png",
            "document/scene4_drone_template_9.png",
            "document/scene4_drone_template_10.png",
            "document/scene4_drone_template_11.png",
            "document/scene4_drone_template_12.png",

            "document/scene4_drone_template_13.png",
            "document/scene4_drone_template_14.png",
            "document/scene4_drone_template_15.png",
            "document/scene4_drone_template_16.png",
        ],
        "output_prefix": "scene4_drone",                    # 输出文件前缀
        # --- 核心匹配 ---
        "threshold": 0.32,            # NCC 匹配阈值
        "init_search_roi": [1450, 980, 500, 260],  # 初始化搜索ROI[原x,原y,原w,原h]—排除右侧树丛误锁区
        "search_radius": 100,         # 局部搜索窗口半径(px)
        "max_lost": 999,              # 极大值=几乎不触发全图重搜索(仅测试初始化)
        "multi_scale": [0.9, 1.0, 1.1],  # 多尺度
        "start_frame": 0,             # 起始帧号
        "use_integral_ncc": True,     # 积分图加速 NCC
        "max_motion_distance": 300,     # 0=回退到自动公式
        "ncc_step": 3,                # 滑窗步长
        "resize_scale": 0.5,          # 帧缩放（2448×2048→1224×1024, 像素量降至 1/4）
        "tracking_stop_frame": 230,    # 超过此帧号后隐藏轨迹(避免树林误框), None=不隐藏
        # --- 可视化 ---
        "show_predicted_bbox": False,
        "draw_predicted_trajectory": False,
        # --- 初始化连续确认 ---
        "enable_init_confirmation": False,     # 关闭(单模板验证阶段)
        "init_confirm_frames": 3,
        "init_confirm_min_score": 0.50,
        "init_confirm_max_distance": 60,
        # --- 运动方向约束 ---
        "enable_motion_direction":False,
        "motion_direction_y": -1,
        "motion_direction_tolerance": 6,
        # --- 跳变检测 ---
        "enable_jump_detection": False,         # 启用
        "jump_max_distance": 90,              # 放宽到 160(无人机帧间大位移)
        "jump_max_area_change": 2.0,
        # --- y 前进速度限制 ---
        "enable_y_forward_speed_limit": False,
        "max_y_decrease_per_frame": 18,
        "max_candidate_ahead_of_prediction_y": 25,
        "max_candidate_behind_prediction_y": 35,
        # --- Top-K 候选筛选 ---
        "enable_topk_candidate_selection": True,
        "topk_candidates": 5,
        "topk_min_score": 0.25,
        # --- 预测约束 ---
        "constrain_predictions": False,
        "prediction_max_center_jump": 60,
        "prediction_max_y_reverse": 5,
        "prediction_max_area_change_ratio": 1.3,
        "prediction_max_width_change_ratio": 1.3,
        "prediction_max_height_change_ratio": 1.3,
        "prediction_out_of_bounds_policy": "reject",
        "prediction_reject_adds_lost": True,
        # --- 可见性判断 ---
        "scene4_use_visibility_gate": True,           # 启用可见性判断（不只看固定帧号）
        "scene4_min_score_for_visible": 0.32,         # 可见最低NCC分数
        "scene4_low_score_patience": 5,               # 连续低分容忍帧数
        "scene4_min_local_contrast": 8.0,             # 目标区域最低局部对比度
        "scene4_stop_when_occluded": True,            # 遮挡时停止画框保留轨迹
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
