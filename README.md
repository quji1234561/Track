# Track — 基于传统图像处理的目标跟踪系统

> 数字图像处理课程综合实践项目  
> 纯传统图像处理方法，不使用任何 AI / 机器学习 / 深度学习

---

## 1. 项目简介

本项目实现了一个**完全不依赖人工智能、深度学习或机器学习的传统图像处理目标跟踪系统**，用于四个实验场景的动态目标跟踪。

核心方法包括：自写 NCC 归一化互相关、多模板匹配、多尺度搜索、积分图加速、局部搜索窗口、卡尔曼运动预测、Top-K 候选筛选、初始化 ROI、运动方向约束、跳变检测、预测约束。跟踪结果输出可视化视频、轨迹 CSV 和性能指标 CSV。

---

## 2. 合规声明

本项目严格遵守课程作业要求：

**未使用以下任何技术**：
- ❌ 深度学习、机器学习、神经网络
- ❌ YOLO、CNN、DNN、Transformer
- ❌ `cv2.matchTemplate` — 自写 NCC 替代
- ❌ OpenCV Tracker（KCF、CSRT、MOSSE、MIL 等）
- ❌ torch、tensorflow、keras、sklearn、ultralytics
- ❌ 任何第三方封装目标跟踪库

**OpenCV 仅用于**：
- 视频读写（`VideoCapture` / `VideoWriter`）
- 灰度转换（`cvtColor`）
- 图像缩放（`resize`）
- 绘图标注（`rectangle` / `circle` / `line` / `putText`）
- 手动 ROI 裁剪（`selectROI`，仅 `template_crop_tool.py`）

**自实现核心算法**：
- NCC 归一化互相关：`src/ncc.py`，纯 NumPy
- 积分图：`np.cumsum` 构建，非 `cv2.integral`
- 卡尔曼滤波：`src/kalman.py`，纯 NumPy 矩阵运算，非 `cv2.KalmanFilter`

运行以下命令可自动验证合规性：

```bash
pixi run check-compliance
```

---

## 3. 核心算法

### 3.1 归一化互相关 NCC

```
NCC = Σ((I - μ_I) × (T - μ_T)) / (σ_I × σ_T)
```

减均值消除亮度差异，除标准差消除对比度差异，使 NCC 在 [-1, 1] 范围内衡量两个图像块的相似度。1 表示完全相同，-1 表示完全相反。

### 3.2 多模板匹配 + 多尺度搜索

每个原始模板按 `multi_scale` 列表生成多个缩放版本（如 [0.9, 1.0, 1.1]），所有候选逐一参与 NCC 搜索，取最高分。适应目标在视频中的尺度变化。

### 3.3 积分图加速

`np.cumsum` 自建积分图，使任意矩形区域的像素和、平方和可在 O(1) 时间内计算。用于快速求窗口均值与方差（NCC 分母），避免重复遍历。保持 float64 精度防止高均值低方差区域的数值抵消。

### 3.4 局部搜索窗口

每帧以卡尔曼预测位置为中心，`search_radius` 为半径裁剪局部区域做 NCC，大幅减少搜索空间。丢失时自动按 `lost_count × 0.5` 扩大搜索半径。

### 3.5 卡尔曼运动预测

恒速模型 `[x, y, vx, vy]`，预测目标下一帧位置。当 NCC 匹配失败时由卡尔曼预测填充，保持轨迹连续。纯 NumPy 矩阵运算实现。

### 3.6 Top-K 候选筛选

不只取 NCC 最高分，而是取前 K 个空间候选，按分数从高到低逐个通过约束检查（阈值/距离/方向/跳变/y-speed），接受第一个合法候选。降低最高分是误匹配时锁错的风险。

### 3.7 初始化 ROI

`init_search_roi = [x, y, w, h]` 限制初始化阶段的 NCC 搜索在指定区域内（原始帧坐标），排除画面外的干扰区域。例如场景 4 用 `[1450, 980, 500, 260]` 排除右侧树丛背景，防止误初始化。

### 3.8 目标丢失处理

连续丢失超过 `max_lost` 帧后触发全图重搜索。丢失期间卡尔曼预测维持位置估计，可受 `constrain_predictions` 约束限制漂移。

### 3.9 停止跟踪策略（场景 4）

`tracking_stop_frame` 用于目标进入树林模糊区域后停止 NCC 搜索和检测框绘制，但保留历史轨迹线继续显示。避免在不可靠区域误框树叶/建筑。

### 3.10 轨迹绘制

绿色框 = NCC 检测成功，黄色框 = 卡尔曼预测。中心点及轨迹线仅使用 detected 帧或配置允许的预测帧。`draw_predicted_trajectory=False` 可避免预测不准时污染轨迹。

---

## 4. 项目结构

```
Track/
├── src/
│   ├── config.py                 # 四个场景的全部配置参数
│   ├── ncc.py                    # 自写 NCC + 积分图 + 多模板匹配 + Top-K
│   ├── tracker.py                # 跟踪主逻辑（初始终/局部搜索/约束/预测）
│   ├── kalman.py                 # 自写二维卡尔曼滤波器
│   ├── preprocess.py             # 灰度化、归一化、多尺度模板生成
│   ├── visualize.py              # 检测框/预测框/中心点/轨迹线/文字绘制
│   ├── main.py                   # CLI 入口 + debug CSV 收集
│   ├── metrics.py                # 轨迹 CSV + 指标统计 + 汇总表
│   ├── template_crop_tool.py     # 从原视频帧手动框选裁剪模板
│   ├── annotation_tool.py        # 人工标注目标真值
│   └── debug_tools.py            # 诊断工具（NCC 自检/得分扫描/模板可视化）
├── document/                     # 原始视频 + 模板图片（建议只读）
├── outputs/                      # 运行输出
│   ├── videos/                   # 跟踪可视化视频
│   ├── trajectories/             # 每帧轨迹 CSV
│   ├── metrics/                  # 指标 CSV + 汇总表
│   └── debug/                    # debug CSV + 多尺度模板图片
├── archive_docs/                 # 归档的临时文档和调参记录
├── pixi.toml                     # pixi 环境 + 全部 task
├── requirements.txt              # pip 依赖
└── README.md                     # 本文件
```

---

## 5. 四个实验场景

| 场景 | 名称 | 视频分辨率 | 帧率 | 关键配置 |
|------|------|-----------|------|---------|
| scene1_animation | 动画表情视频跟踪 | 1080×1920 | 30fps | 单尺度, resize_scale=0.4 |
| scene2_car | 大疆无人机航拍—车辆跟踪 | 640×512 | 7.5fps | 方向约束+跳变+Top-K, 车从下方向上 |
| scene3_bicycle | 大疆无人机航拍—骑车人跟踪 | 640×512 | — | start_frame=83 起始, 方向约束+跳变, 目标极小 |
| scene4_drone | 地面光学站无人机跟踪 | 2448×2048 | 30fps | resize_scale=0.5, init_search_roi 防误锁, 多模板接力, tracking_stop_frame=230 停止画框但保留轨迹 |

### 场景 3 说明

骑车人目标极小（~27×36px），第 0 帧目标不可见。`start_frame=83` 跳过前 83 帧直接初始化；关闭初始化确认（目标第 83 帧已清晰出现，直接初始化更稳定）；结合方向约束和跳变检测防止跳到其他相似小目标。

### 场景 4 说明

大画幅（2448×2048），使用 `resize_scale=0.5` 加速。`init_search_roi=[1450,980,500,260]` 限制初始化搜索范围在无人机出现区域，排除右侧树丛/建筑背景。多模板接力适配无人机不同帧的外观和尺度变化。前 230 帧稳定跟踪，`tracking_stop_frame=230` 后目标进入树林模糊区，停止画检测框但保留前 230 帧历史轨迹，避免误框树叶。

---

## 6. 环境安装

### 方式一：pixi（推荐）

```bash
pixi install
pixi shell
```

### 方式二：pip

```bash
python -m venv venv
# Windows: venv\Scripts\activate
# macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
```

### 依赖

| 包 | 用途 |
|----|------|
| numpy | 矩阵运算（NCC、积分图、卡尔曼） |
| opencv | 视频I/O、颜色转换、resize、绘图 |
| pandas | CSV 数据处理 |
| matplotlib | 可选，报告图表 |
| python-docx | 可选，读取作业模板 |

---

## 7. 运行方式

```bash
# 单个场景
python -m src.main --scene scene1_animation
python -m src.main --scene scene2_car
python -m src.main --scene scene3_bicycle
python -m src.main --scene scene4_drone

# 全部场景
python -m src.main --scene all

# 调试模式（限制帧数 + 逐帧输出）
python -m src.main --scene scene4_drone --max-frames 600 --debug

# pixi shortcuts
pixi run scene1
pixi run debug2
pixi run run-all
pixi run check-compliance
```

---

## 8. 模板裁剪

使用项目自带工具从原始视频帧直接裁剪模板（避免播放器截图导致的尺寸偏差）：

```bash
python -m src.template_crop_tool --scene scene4_drone --frame 209 --name my_template.png
```

操作：运行后弹出视频帧 → 鼠标拖框选目标 → Enter 确认 → 模板保存到 `document/`。

**裁剪原则**：
- 只包含目标本体 + 少量周围背景
- 不要截太大（背景过多会主导 NCC 匹配）
- 多模板用于适配目标不同帧的姿态、尺度、模糊和背景变化

---

## 9. 输出结果

```
outputs/
├── videos/{scene}_tracking.mp4        # 跟踪可视化视频
├── trajectories/{scene}_trajectory.csv # 每帧位置 + 状态
├── metrics/{scene}_metrics.csv        # 单场景指标
├── metrics/summary_metrics.csv         # 四场景汇总
└── debug/{scene}_score_debug.csv      # 每帧 40+ 字段详细 debug
```

轨迹 CSV 字段：`frame_id, x, y, w, h, center_x, center_y, score, detected, predicted, used_for_trajectory, template_id`

指标 CSV 字段：`scene, total_frames, detected_frames, predicted_frames, lost_frames, detection_rate, prediction_rate, average_score, average_fps`

---

## 10. 调参建议

| 现象 | 可能原因 | 建议 |
|------|---------|------|
| 初始化锁错 | 模板不匹配 / 搜索区域过大 | 更换模板、提高 threshold、使用 `init_search_roi` |
| 中途丢失 | 模板不足 / 搜索半径太小 | 补模板、增大 `search_radius` |
| 跳到背景 | 错误高分区域 | 缩小 `search_radius`、启用 Top-K + 提高 `topk_min_score`、补更干净模板 |
| 速度太慢 | 模板/尺度太多 | 增大 `ncc_step`、减少模板、调整 `resize_scale` |
| 目标进入遮挡/模糊区 | NCC 不可靠 | 使用 `tracking_stop_frame` 停止画框，保留历史轨迹 |
| 预测框乱飘 | 卡尔曼无约束 | 启用 `constrain_predictions`、限制 `prediction_max_center_jump` |

---

## 11. 实验结果摘要

- 四个场景均已输出完整的跟踪视频、轨迹 CSV 和性能指标 CSV。
- 场景 1 动画表情：多模板 NCC 跟踪动画表情在画面中的移动。
- 场景 2 车辆：方向约束 + Top-K 候选筛选有效防止误锁到其他车道车辆。
- 场景 3 骑车人：从第 83 帧目标出现开始跟踪，方向约束和跳变检测保持跟踪稳定性，跟踪至目标离开画面。
- 场景 4 无人机：初始化 ROI 排除树丛背景干扰，多模板接力适配外观变化，前 230 帧稳定跟踪，进入树林模糊区后停止画检测框但保留完整轨迹，避免误识别树叶。
- 所有输出文件可用于课程报告的量化和可视化分析。
