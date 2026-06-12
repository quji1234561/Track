# Track — 传统图像处理目标跟踪系统

> 数字图像处理课程综合实践项目  
> 仓库：[github.com/quji1234561/Track](https://github.com/quji1234561/Track)

---

## 一、项目简介

本项目实现了一个**完全不依赖人工智能/深度学习/机器学习的传统图像处理目标跟踪系统**，用于四个实验场景的动态目标跟踪。

**核心方法**：

| 组件 | 实现方式 | 位置 |
|------|---------|------|
| NCC 归一化互相关 | 纯 NumPy 自写，含 stride-trick 矢量化版和积分图加速版 | `src/ncc.py` |
| 积分图 | `np.cumsum` 自建，float64 精度防数值抵消 | `src/ncc.py` |
| 多模板匹配 | 遍历所有模板候选，取最高 NCC 分 | `src/ncc.py` |
| 多尺度模板 | `cv2.resize` 生成多尺度候选，逐尺度参与 NCC 搜索 | `src/preprocess.py` |
| 局部搜索窗口 | 以卡尔曼预测为中心裁剪搜索区域，丢失时自动扩大 | `src/tracker.py` |
| 卡尔曼滤波 | 恒速模型 `[x, y, vx, vy]`，纯 NumPy 自写 | `src/kalman.py` |
| 初始化连续确认 | 连续 N 帧高分检测才正式初始化，防止误锁定 | `src/tracker.py` |
| 运动方向约束 | 限制目标只能沿指定方向移动，拒绝逆向匹配 | `src/tracker.py` |
| 跳变检测 | 帧间位移/面积突变时拒绝匹配，防止检测框乱跳 | `src/tracker.py` |
| 预测约束 | 卡尔曼预测框也受方向/跳变/尺寸限制 | `src/tracker.py` |
| 轨迹绘制 | 绿色检测框 + 红色中心点 + 蓝色轨迹线 | `src/visualize.py` |
| 指标统计 | 检测率、预测率、平均 NCC 分数、FPS | `src/metrics.py` |

---

## 二、作业约束与合规说明

### 禁止使用的技术

本项目**不使用**以下任何方法：

- `cv2.matchTemplate()` — 未使用 OpenCV 封装模板匹配
- `cv2.TrackerXXX_create()` — 未使用 OpenCV Tracker（KCF、CSRT、MOSSE、MIL 等）
- YOLO / CNN / Transformer / OpenCV DNN — 未使用任何深度学习模型
- `torch` / `tensorflow` / `keras` / `sklearn` / `ultralytics` — 未使用任何机器学习框架
- 第三方封装目标跟踪库 — 未使用

运行以下命令可自动验证：

```bash
pixi run check-compliance
```

### 允许使用 OpenCV 的范围

OpenCV **仅**用于基础工程操作，不涉及任何目标识别/匹配/跟踪算法：

| 允许 | 禁止 |
|------|------|
| `cv2.VideoCapture` 读取视频 | `cv2.matchTemplate` |
| `cv2.VideoWriter` 保存视频 | `cv2.TrackerXXX_create` |
| `cv2.cvtColor` 灰度转换 | `cv2.dnn` |
| `cv2.resize` 图像缩放 | `cv2.integral`（自用 `np.cumsum` 替代） |
| `cv2.rectangle/circle/line/putText` 绘图 | `cv2.findHomography` 等高级接口 |
| `cv2.imwrite` 保存图像 | |
| `cv2.selectROI` / 鼠标回调 手动框选 | |

**核心匹配算法由 `src/ncc.py` 中的自写 NCC 完成，跟踪逻辑由 `src/tracker.py` 中的多模板搜索、局部搜索和卡尔曼预测完成。**

---

## 三、功能特性

### 3.1 运行模式

| 功能 | 命令 |
|------|------|
| 运行全部场景 | `pixi run run-all` |
| 运行单个场景 | `pixi run scene1` / `scene2` / `scene3` / `scene4` |
| 调试模式（前100帧+逐帧输出） | `pixi run debug1` / `debug2` / `debug3` / `debug4` |
| 保存关键帧截图 | `pixi run all-with-frames` |
| 限制处理帧数 | `python -m src.main --scene scene2_car --max-frames 300` |
| 人工标注真值 | `pixi run annotate --scene scene2_car --interval 20` |
| 合规检查 | `pixi run check-compliance` |
| 诊断工具（NCC自检+得分分布+模板可视化） | `python -m src.debug_tools --scene all --num-frames 20` |

### 3.2 核心跟踪功能

- 自写 NCC 匹配（`ncc_score` + stride-trick `ncc_search` + 积分图 `ncc_search_integral`）
- 积分图加速（`np.cumsum` 自建，保持 float64 精度防止数值抵消）
- 多模板匹配（遍历所有模板，取最高分；结果包含 `scale`、`template_id`）
- 多尺度模板（每个原始模板按 `multi_scale` 生成多个 `cv2.resize` 候选）
- 卡尔曼滤波预测（恒速模型 `[x, y, vx, vy]`，纯 NumPy 实现）
- 局部搜索窗口（以卡尔曼预测为中心裁剪，丢失时按 `lost_count` 自动扩大）
- 初始化连续确认（需连续 N 帧在附近检测到高分目标才正式初始化）
- 运动方向约束（限制目标只能向上/向下移动，拒绝逆向匹配）
- 跳变检测（帧间位移或面积突变时拒绝匹配）
- 预测约束（卡尔曼预测框也通过方向/跳变/尺寸检查，通不过则保持上一帧位置）
- 预测框显示控制（`show_predicted_bbox` 和 `draw_predicted_trajectory` 独立开关）
- 全图重搜索（连续丢失超过 `max_lost` 帧时触发）

### 3.3 调试与诊断功能

- `score_debug.csv`：每帧一行，含 `best_score`、`scale`、`template_id`、`reject_reason`、`lost_count` 等 30+ 字段
- `scale_scores_debug.csv`：每帧每个候选模板一行，可对比不同尺度的得分
- `debug_tools.py`：NCC 自匹配验证、模板加载检查、前 N 帧全图得分扫描、多尺度模板图片保存
- 运行时打印 `processed_frames` vs `debug_score_rows` 行数验证
- 运行时打印 `Scale usage statistics`（每个尺度被选中的次数）

### 3.4 工具

| 工具 | 用途 |
|------|------|
| `template_crop_tool.py` | 从原始视频帧直接框选裁剪模板（避免播放器截图导致的尺寸偏差） |
| `annotation_tool.py` | 人工点击标注目标中心点，生成真值 CSV（用于后续计算像素误差） |
| `debug_tools.py` | NCC 自检、模板验证、得分分布扫描、多尺度模板可视化 |

---

## 四、项目目录结构

```
Track/
├── src/                              # 完整源代码
│   ├── __init__.py                   # 包初始化
│   ├── config.py                     # 四个场景的全部参数配置
│   ├── preprocess.py                 # 图像预处理（灰度、归一化、多尺度模板生成）
│   ├── ncc.py                        # 自写 NCC（基础 + stride-trick + 积分图加速）+ 多模板搜索
│   ├── kalman.py                     # 自写二维卡尔曼滤波器（恒速模型）
│   ├── tracker.py                    # 跟踪主逻辑（初始化确认/局部搜索/方向约束/跳变检测/预测约束）
│   ├── visualize.py                  # 可视化绘制（检测框/预测框/中心点/轨迹/信息文字）
│   ├── metrics.py                    # 轨迹 CSV 输出 + 基础指标统计 + 汇总表
│   ├── main.py                       # 命令行主入口 + debug CSV 收集
│   ├── annotation_tool.py            # 人工真值标注工具
│   ├── template_crop_tool.py         # 从原视频帧裁剪模板的工具
│   └── debug_tools.py                # 诊断工具（NCC自检/模板检查/得分扫描/多尺度可视化）
│
├── document/                         # 原始实验数据（视频 + 模板 + 作业要求，不建议修改）
├── outputs/                          # 运行生成的输出文件
│   ├── videos/                       # 跟踪结果视频（{scene}_tracking.mp4）
│   ├── frames/                       # 关键帧截图（{scene}_frame_*.png）
│   ├── trajectories/                 # 轨迹 CSV（{scene}_trajectory.csv）
│   ├── metrics/                      # 指标 CSV（{scene}_metrics.csv + summary_metrics.csv）
│   └── debug/                        # debug CSV + 多尺度模板图片
│       └── templates/                # 多尺度模板可视化图片
│
├── report/                           # 报告素材目录（figures/ + tables/）
├── pixi.toml                         # pixi 环境配置 + 全部 task 定义
├── requirements.txt                  # pip 依赖列表
├── PROJECT_PLAN.md                   # 原始项目实现计划
└── README.md                         # 本文件
```

---

## 五、实验数据说明

### 5.1 四个实验场景

| 场景 key | 名称 | 视频 | 模板数 | 多尺度 | 分辨率 | 帧率 |
|----------|------|------|--------|--------|--------|------|
| `scene1_animation` | 动画表情视频跟踪 | `动画表情视频.mp4` | 5 | `[1.0]` | 1080×1920 | 30 fps |
| `scene2_car` | 大疆无人机航拍 - 车辆跟踪 | `大疆无人机航拍视频.mp4` | 2 | `[0.7,0.85,0.95,1.0,1.1]` | 640×512 | 7.5 fps |
| `scene3_bicycle` | 大疆无人机航拍 - 骑车人跟踪 | `大疆无人机航拍骑车人.mp4` | 3 | `[0.8,0.9,1.0,1.1,1.2]` | 640×512 | — |
| `scene4_drone` | 地面光学站跟踪无人机 | `地面光学站跟踪无人机.avi` | 6 | `[0.9,1.0,1.1]` | 2448×2048 | 30 fps |

### 5.2 数据使用说明

- `document/` 是原始数据目录，**不建议修改原始视频和模板**。
- 新增模板图片可以放入 `document/`，然后在 `src/config.py` 对应场景的 `templates` 列表中添加路径。
- 建议使用 `src/template_crop_tool.py` 从原始视频帧直接裁剪模板（避免播放器截图导致的尺寸偏差）。

---

## 六、环境安装

### 方式一：pixi（推荐）

项目已配置 `pixi.toml`，通过 conda-forge 管理全部依赖：

```bash
# 安装依赖（首次）
pixi install

# 进入 pixi 环境
pixi shell
```

### 方式二：pip + venv

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 依赖项

| 包 | 版本 | 用途 |
|----|------|------|
| `numpy` | ≥1.24 | 矩阵运算、自写 NCC、积分图、卡尔曼滤波 |
| `opencv` / `opencv-python` | ≥4.8 | 视频I/O、灰度转换、resize、绘图、ROI 框选 |
| `pandas` | ≥2.0 | CSV 读写、指标汇总 |
| `matplotlib` | ≥3.7 | 可选，报告图表 |
| `python-docx` | ≥0.8 | 可选，读取作业模板 |

---

## 七、运行方法

### 7.1 pixi tasks

```bash
# 运行全部场景
pixi run run-all

# 运行单个场景
pixi run scene1          # 动画表情视频跟踪
pixi run scene2          # 车辆目标跟踪
pixi run scene3          # 骑车人目标跟踪
pixi run scene4          # 地面光学站跟踪无人机

# 调试模式（前 100 帧 + 逐帧打印 NCC 分数）
pixi run debug1
pixi run debug2
pixi run debug3
pixi run debug4

# 保存关键帧截图 + 全部场景
pixi run all-with-frames

# 人工标注工具
pixi run annotate --scene scene2_car --interval 20

# 模板截取工具
pixi run crop-scene2    # 场景2帧30截取模板
pixi run crop-scene3    # 场景3帧50截取模板
pixi run crop-scene4    # 场景4帧100截取模板

# 合规检查
pixi run check-compliance
```

### 7.2 原生 Python 命令

```bash
# 全部场景
python -m src.main --scene all

# 单个场景
python -m src.main --scene scene1_animation
python -m src.main --scene scene2_car --max-frames 300
python -m src.main --scene scene3_bicycle --debug
python -m src.main --scene scene4_drone --save-frames

# 诊断工具
python -m src.debug_tools --scene all --num-frames 20
python -m src.debug_tools --scene scene2_car --num-frames 50

# 模板截取
python -m src.template_crop_tool --scene scene2_car --frame 30 --name my_template.png

# 人工标注
python -m src.annotation_tool --scene scene1_animation --interval 20
```

### 7.3 命令行参数

| 参数 | 说明 |
|------|------|
| `--scene` | 场景 key（`scene1_animation` / `scene2_car` / `scene3_bicycle` / `scene4_drone` / `all`） |
| `--max-frames` | 最多处理帧数（0 = 全部帧） |
| `--debug` | 终端逐帧打印 NCC 分数、中心点、检测状态、reject_reason |
| `--save-frames` | 按间隔保存关键帧截图到 `outputs/frames/` |

---

## 八、配置参数说明

所有配置项在 `src/config.py` 中按场景独立设置，共约 40 个参数。完整参数及中文注释见文件内。

### 8.1 核心匹配参数

| 参数 | 类型 | 说明 |
|------|------|------|
| `threshold` | float | NCC 匹配阈值 (0~1)。调高=更严格，调低=易锁定 |
| `search_radius` | int | 局部搜索窗口半径 (px)。丢失时自动扩大 |
| `max_lost` | int | 连续丢失多少帧后触发全图重搜索 |
| `multi_scale` | list[float] | 模板多尺度因子列表 |
| `start_frame` | int | 起始帧号（目标在后面才出现时设 >0） |
| `use_integral_ncc` | bool | True=积分图加速, False=stride-trick |
| `ncc_step` | int | NCC 滑窗步长 (px) |
| `resize_scale` | float | 帧缩放比例（输出坐标自动映射回原始分辨率） |

### 8.2 可视化参数

| 参数 | 类型 | 说明 |
|------|------|------|
| `show_predicted_bbox` | bool | 是否绘制黄色预测框 |
| `draw_predicted_trajectory` | bool | 预测中心点是否加入轨迹线 |

### 8.3 初始化连续确认

| 参数 | 类型 | 说明 |
|------|------|------|
| `enable_init_confirmation` | bool | 是否启用 |
| `init_confirm_frames` | int | 需连续确认的帧数 |
| `init_confirm_min_score` | float | 确认期间最低 NCC 分数 |
| `init_confirm_max_distance` | int | 确认期间最大帧间位移 (px) |

### 8.4 运动方向约束

| 参数 | 类型 | 说明 |
|------|------|------|
| `enable_motion_direction` | bool | 是否启用 |
| `motion_direction_y` | int | 0=不限, -1=只能向上, 1=只能向下 |
| `motion_direction_tolerance` | int | 逆向运动容差 (px) |

### 8.5 跳变检测

| 参数 | 类型 | 说明 |
|------|------|------|
| `enable_jump_detection` | bool | 是否启用 |
| `jump_max_distance` | int | 最大帧间位移 (px)，超过拒绝 |
| `jump_max_area_change` | float | 最大面积变化倍数，超过拒绝 |

### 8.6 预测约束

| 参数 | 类型 | 说明 |
|------|------|------|
| `constrain_predictions` | bool | 预测框是否也受方向/跳变/尺寸限制 |
| `prediction_max_center_jump` | int | 预测框最大帧间位移 (px) |
| `prediction_max_y_reverse` | int | 预测框最大逆向位移 (px) |
| `prediction_max_area_change_ratio` | float | 预测框面积变化上限 |
| `prediction_max_width_change_ratio` | float | 预测框宽度变化上限 |
| `prediction_max_height_change_ratio` | float | 预测框高度变化上限 |
| `prediction_out_of_bounds_policy` | str | 出界策略：`"reject"` 或 `"clamp"` |
| `prediction_reject_adds_lost` | bool | 预测被拒时是否计入 lost_count |

---

## 九、输出文件说明

运行后自动生成以下文件：

### 9.1 跟踪视频

`outputs/videos/{scene}_tracking.mp4`

含目标框（绿色=检测 / 黄色=预测）、红色中心点、蓝色轨迹线、场景名、帧号、NCC 分数、状态标签。

### 9.2 轨迹 CSV

`outputs/trajectories/{scene}_trajectory.csv`

| 字段 | 说明 |
|------|------|
| `frame_id` | 帧号 |
| `x, y, w, h` | bbox（原始帧坐标） |
| `center_x, center_y` | 中心点 |
| `score` | NCC 分数 |
| `detected` | 0/1 是否为 NCC 检测 |
| `predicted` | 0/1 是否为卡尔曼预测 |
| `used_for_trajectory` | 0/1 该帧中心点是否参与轨迹绘制 |
| `template_id` | 最佳模板编号 |

### 9.3 指标 CSV

`outputs/metrics/{scene}_metrics.csv`

| 字段 | 说明 |
|------|------|
| `scene` | 场景名称 |
| `total_frames` | 总帧数 |
| `detected_frames` | 检测成功帧数 |
| `predicted_frames` | 预测补偿帧数 |
| `lost_frames` | 完全丢失帧数 |
| `detection_rate` | 检测率 |
| `prediction_rate` | 预测率 |
| `average_score` | 平均 NCC 分数 |
| `average_fps` | 平均处理速度 |

`outputs/metrics/summary_metrics.csv` 汇总全部四个场景的指标。

### 9.4 Debug 输出

| 文件 | 说明 |
|------|------|
| `outputs/debug/{scene}_score_debug.csv` | 每帧一行，30+ 字段（score, scale, template_id, search window, match pos, reject_reason, lost_count 等） |
| `outputs/debug/{scene}_scale_scores_debug.csv` | 每帧每个候选模板一行，可对比不同尺度/模板的得分 |
| `outputs/debug/templates/{scene}/` | 多尺度模板可视化图片 |

### 9.5 关键帧截图

`outputs/frames/{scene}_frame_*.png`（需运行时加 `--save-frames`）

---

## 十、多尺度模板匹配说明

1. `multi_scale` 参数控制模板按不同比例缩放（`cv2.resize`），生成多个候选。
2. `multi_template_search` 遍历所有候选模板，对每个执行自写 NCC 搜索，取最高分。
3. 获胜模板的 `scale` 和 `template_id` 会记录在 `score_debug.csv` 和 `scale_scores_debug.csv` 中。
4. 运行结束时终端打印 `Scale usage statistics`，显示每个尺度被选中的次数。
5. `resize_scale`（视频帧缩放）和 `multi_scale`（模板尺度变化）是两个独立参数，程序自动处理坐标映射。
6. 运行 `python -m src.debug_tools --scene scene2_car` 会自动保存多尺度模板图片。

---

## 十一、积分图加速说明

1. `src/ncc.py` 中的 `compute_integral_images_numpy()` 使用 `np.cumsum` 自建积分图和平方积分图。
2. 积分图使任意矩形的 `sum(I)` 和 `sum(I²)` 可在 O(1) 时间内完成，用于快速计算 NCC 分母（窗口方差）。
3. 积分图保持 **float64** 精度，防止高均值低方差区域（如浅色背景）发生灾难性数值抵消。
4. NCC 分子 `sum(patch × template_zero)` 仍由自写代码逐窗口计算。
5. 可通过 `use_integral_ncc` 参数在积分图版和 stride-trick 版之间切换。

---

## 十二、预测框与轨迹显示配置

| 参数 | 默认 | 说明 |
|------|------|------|
| `show_predicted_bbox` | True（场景1）/ False（场景2/3/4） | 是否绘制黄色预测框 |
| `draw_predicted_trajectory` | False | 预测中心点是否加入轨迹线 |

- `detected=True` 的帧永远绘制绿色框并加入轨迹（不受这两个参数影响）。
- 关闭预测框不影响卡尔曼内部运行，只是不显示。
- 关闭预测轨迹可避免卡尔曼漂移时污染运动轨迹。

---

## 十三、如何正确截取模板

**不要用播放器全屏截图作为模板**——播放器截图包含 UI 边框、窗口缩放、字幕等，与原视频帧尺寸不一致。

使用项目自带的 `template_crop_tool.py`：

```bash
# 从场景2的第30帧截取模板
python -m src.template_crop_tool --scene scene2_car --frame 30 --name my_template.png

# 或使用 pixi task
pixi run crop-scene2
```

操作流程：
1. 运行命令 → 弹出视频帧窗口
2. 鼠标拖拽框选目标区域（包含目标 + 少量背景）
3. Enter 确认 / C 取消
4. 模板保存到 `document/{name}`，预览图保存到 `outputs/debug/templates/`

裁完后在 `src/config.py` 对应场景的 `templates` 列表中添加路径即可。

---

## 十四、调参建议

### 14.1 通用流程

1. 先运行诊断工具查看实际 NCC 分数分布：
   ```bash
   python -m src.debug_tools --scene scene2_car --num-frames 20
   ```
2. 根据输出的 `Score statistics` 设定 `threshold`（建议 `mean - 1×std`）
3. 用 debug 模式试跑少量帧确认效果：
   ```bash
   python -m src.main --scene scene2_car --max-frames 50 --debug
   ```
4. 查看 `outputs/debug/{scene}_score_debug.csv` 中的 `reject_reason` 列，了解每帧被拒原因

### 14.2 常见问题速查

| 现象 | 可能原因 | 调参方向 |
|------|---------|---------|
| 初始化锁到错误目标 | 目标未出现、阈值太低 | 提高 `threshold`、设置 `start_frame`、启用 `enable_init_confirmation` |
| 跳到其他目标 | 全图重搜索太频繁 | 提高 `max_lost`、启用 `enable_jump_detection`、缩小 `search_radius` |
| 远距离丢失 | 模板尺度不匹配、阈值太高 | 增加 `multi_scale` 范围、降低 `threshold` |
| 预测框漂移 | 卡尔曼速度累积 | 启用 `constrain_predictions`、调小 `prediction_max_center_jump` |
| 目标反向移动被拒 | `motion_direction_y` 设反 | 检查目标实际移动方向，调整 ±1 |
| 速度太慢 | 模板/尺度太多 | 增大 `ncc_step`、减小 `search_radius`、设 `resize_scale < 1.0` |

---

## 十五、未使用 AI / 封装匹配接口的声明

1. 本项目未使用任何深度学习、机器学习或 AI 模型（无 YOLO、CNN、Transformer、DNN）。
2. 未使用 `cv2.matchTemplate` 或任何 OpenCV 封装模板匹配接口。
3. 未使用 OpenCV Tracker（KCF、CSRT、MOSSE、MIL 等）或任何第三方跟踪库。
4. NCC 归一化互相关算法由 `src/ncc.py` 中的纯 NumPy 代码自主实现。
5. 积分图由 `np.cumsum` 自建，不使用 `cv2.integral`。
6. 卡尔曼滤波器由 `src/kalman.py` 中的纯 NumPy 代码自主实现。
7. 多模板匹配、多尺度搜索、局部搜索窗口、初始化确认、运动约束、跳变检测、预测约束等策略均在 `src/tracker.py` 中自主实现。

---

## 十六、后续报告引用

- 跟踪视频截图可直接用于报告中的实验结果展示。
- `outputs/trajectories/` 中的 CSV 可在 Excel/Python 中绘制轨迹图。
- `outputs/metrics/summary_metrics.csv` 可直接作为报告中的指标汇总表。
- `outputs/debug/{scene}_score_debug.csv` 可用于分析每帧的匹配细节（分数、尺度、拒绝原因）。
- 如需计算像素误差，先用 `src/annotation_tool.py` 标注真值，再用 `src/metrics.py` 的 `compute_metrics_with_ground_truth` 函数。
- `outputs/debug/templates/` 中的多尺度模板图片可放入报告展示多尺度匹配原理。
- `report/` 目录用于存放报告相关图表源文件。
