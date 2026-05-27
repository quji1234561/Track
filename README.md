# Track — 传统图像处理目标跟踪系统

数字图像处理课程大作业：基于 **NCC 归一化互相关 + 多模板匹配 + 局部搜索 + 卡尔曼滤波预测** 的目标跟踪系统。

---

## 作业约束说明

本项目严格遵守课程要求：

### 禁止使用的技术
- `cv2.matchTemplate()` — 不使用 OpenCV 封装模板匹配
- `cv2.TrackerXXX_create()` — 不使用 OpenCV Tracker（KCF、CSRT、MOSSE 等）
- YOLO / CNN / Transformer / OpenCV DNN — 不使用任何深度学习模型
- 机器学习分类器
- 第三方封装目标跟踪库

### 技术栈限制
| 组件 | 使用方式 |
|------|----------|
| OpenCV | 仅用于视频读写、灰度转换、画框/画线/写字、图像保存 |
| NumPy | 矩阵运算，自写 NCC 算法 |
| NCC 归一化互相关 | 自主实现在 `src/ncc.py` |
| 多模板匹配 | 自主实现在 `src/ncc.py` 的 `multi_template_search` |
| 卡尔曼滤波 | 自主实现在 `src/kalman.py` |
| 局部搜索窗口 | 自主实现在 `src/tracker.py` |

**OpenCV 仅用于视频读写、基础图像转换和结果绘制；核心目标匹配由 `src/ncc.py` 中的自写 NCC 完成；目标跟踪由 `src/tracker.py` 中的多模板搜索、局部搜索和卡尔曼预测完成。**

---

## 环境安装

### 方式一：pixi（推荐）

项目已配置 `pixi.toml`，一键初始化：

```bash
# 安装依赖（首次）
pixi install

# 进入 pixi shell 环境
pixi shell
```

pixi 使用 conda-forge 渠道，所有依赖（包括 OpenCV、NumPy、pandas）均由 conda 管理，无需单独安装系统级包。

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

| 包 | 用途 |
|----|------|
| `numpy` | 矩阵运算，自写 NCC 算法 |
| `opencv` (或 `opencv-python`) | 视频I/O、灰度转换、可视化绘制 |
| `pandas` | CSV 数据处理 |
| `matplotlib` | 可选，报告图表 |
| `python-docx` | 可选，读取作业模板 |

---

## 目录结构

```
Track/
├── document/                     # 原始视频、模板、作业要求（不修改）
├── src/                          # 完整源代码
│   ├── __init__.py
│   ├── config.py                 # 四个场景配置
│   ├── preprocess.py             # 图像预处理（灰度、归一化、多尺度模板）
│   ├── ncc.py                    # 自写 NCC 归一化互相关匹配算法
│   ├── kalman.py                 # 自写二维卡尔曼滤波器
│   ├── tracker.py                # 跟踪主逻辑
│   ├── visualize.py              # 绘制目标框、轨迹、信息文字
│   ├── metrics.py                # 轨迹保存、指标统计
│   ├── annotation_tool.py        # 人工真值标注工具
│   └── main.py                   # 命令行主入口
├── outputs/                      # 输出文件
│   ├── videos/                   # 跟踪视频
│   ├── frames/                   # 关键帧截图
│   ├── trajectories/             # 轨迹 CSV
│   ├── metrics/                  # 指标 CSV
│   └── logs/                     # 日志
├── report/                       # 报告素材
│   ├── figures/
│   └── tables/
├── requirements.txt
├── README.md
└── PROJECT_PLAN.md
```

---

## 运行方法

### 使用 pixi tasks（推荐）

```bash
# 运行全部场景
pixi run run-all

# 运行单个场景
pixi run scene1          # 动画表情视频跟踪
pixi run scene2          # 大疆无人机航拍-车辆目标跟踪
pixi run scene3          # 大疆无人机航拍-骑车人目标跟踪
pixi run scene4          # 地面光学站跟踪无人机

# 调试模式（前100帧，逐帧输出NCC分数）
pixi run debug1
pixi run debug2
pixi run debug3
pixi run debug4

# 保存关键帧截图
pixi run all-with-frames

# 人工标注工具（指定场景+间隔帧数）
pixi run annotate --scene scene1_animation --interval 20

# 验证代码合规性（检查无禁止API调用）
pixi run check-compliance
```

### 使用原生 Python（在 pixi shell 或 venv 内）

```bash
# 运行全部场景
python -m src.main --scene all

# 运行单个场景
python -m src.main --scene scene1_animation
python -m src.main --scene scene2_car
python -m src.main --scene scene3_bicycle
python -m src.main --scene scene4_drone

# 调试模式
python -m src.main --scene scene1_animation --max-frames 100 --debug

# 保存关键帧截图
python -m src.main --scene scene1_animation --save-frames

# 人工标注工具
python -m src.annotation_tool --scene scene1_animation --interval 20
```

---

## 四个实验场景

| 场景 | 名称 | 模板数 | 多尺度 |
|------|------|--------|--------|
| scene1_animation | 动画表情视频跟踪 | 3 | 1.0 |
| scene2_car | 大疆无人机航拍-车辆目标跟踪 | 1 | 0.9, 1.0, 1.1 |
| scene3_bicycle | 大疆无人机航拍-骑车人目标跟踪 | 1 | 0.8~1.2 (5级) |
| scene4_drone | 地面光学站跟踪无人机 | 1 | 0.8~1.2 (5级) |

---

## 输出文件说明

每个场景输出以下文件：

| 类型 | 路径 | 说明 |
|------|------|------|
| 跟踪视频 | `outputs/videos/{scene}_tracking.mp4` | 含目标框、中心点、轨迹、NCC分数 |
| 轨迹CSV | `outputs/trajectories/{scene}_trajectory.csv` | 逐帧位置信息 |
| 指标CSV | `outputs/metrics/{scene}_metrics.csv` | 检测率、平均NCC分数、FPS |
| 关键帧 | `outputs/frames/{scene}_frame_*.png` | 定期间隔截图 |
| 综合表 | `outputs/metrics/summary_metrics.csv` | 四个场景汇总对比 |

轨迹 CSV 字段：`frame_id, x, y, w, h, center_x, center_y, score, detected, predicted, template_id`

指标 CSV 字段：`scene, total_frames, detected_frames, predicted_frames, lost_frames, detection_rate, prediction_rate, average_score, average_fps`

---

## 未使用 AI / 封装匹配接口的声明

1. 本项目未使用任何深度学习、机器学习或 AI 模型（无 YOLO、CNN、Transformer、DNN）。
2. 未使用 `cv2.matchTemplate` 或任何 OpenCV 封装模板匹配接口。
3. 未使用 OpenCV Tracker（KCF、CSRT、MOSSE 等）或任何第三方跟踪库。
4. NCC 归一化互相关算法由 `src/ncc.py` 中的纯 NumPy 代码自主实现。
5. 卡尔曼滤波器由 `src/kalman.py` 中的纯 NumPy 代码自主实现。
6. 多模板匹配、局部搜索窗口、目标丢失容错等策略均在 `src/tracker.py` 中自主实现。

---

## 后续报告引用

- 跟踪视频截图可直接用于报告中的实验结果展示。
- `outputs/trajectories/` 中的 CSV 可在 Excel/Python 中绘制轨迹图。
- `outputs/metrics/summary_metrics.csv` 可直接作为报告中的指标汇总表。
- 如需计算像素误差，先用 `annotation_tool.py` 标注真值，再调用 `compute_metrics_with_ground_truth`。
- `report/` 目录用于存放报告相关图表源文件。
