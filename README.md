# Track — 传统图像处理目标跟踪系统

数字图像处理课程综合实践项目。纯传统图像处理方法，不使用任何 AI / 机器学习 / 深度学习。

---

## 技术约束

**未使用以下任何技术**：
- ❌ 深度学习、机器学习、神经网络（YOLO、CNN、DNN、Transformer）
- ❌ `cv2.matchTemplate` — 自写 NCC 替代
- ❌ OpenCV Tracker（KCF、CSRT、MOSSE、MIL 等）
- ❌ torch、tensorflow、keras、sklearn、ultralytics
- ❌ 任何第三方目标跟踪库

**OpenCV 仅用于**：视频读写、灰度转换、图像缩放、绘图标注、形态学操作。

**自实现核心算法**：NCC 归一化互相关（`src/ncc.py`）、积分图加速、卡尔曼滤波（`src/kalman.py`）。

验证合规：

```bat
pixi run check-compliance
```

---

## 四个场景

| 场景 | 名称 | 分辨率 | 跟踪方法 |
|------|------|--------|---------|
| scene1_animation | 动画表情视频跟踪 | 1080×1920 | NCC 多模板匹配 + 卡尔曼预测 |
| scene2_car | 航拍车辆跟踪 | 640×512 | NCC + 方向约束 + 跳变检测 + GMC 补偿 + 遮挡状态机 |
| scene3_bicycle | 航拍骑车人跟踪 | 640×512 | NCC + 反向补轨 + 合并渲染 |
| scene4_drone | 地面站无人机跟踪 | 2448×2048 | 帧差法 + 轮廓检测 + 可选交互式校正 |

### scene3 说明

骑车人目标极小（~27×36px），第 0 帧不可见。`start_frame=83` 跳过前 83 帧直接初始化。使用独立 `Scene3LegacyTracker` 避免与 scene2 复杂状态机冲突。正向跟踪后运行反向补轨填充前 83 帧轨迹。

### scene4 说明

大画幅（2448×2048），使用帧差法检测运动区域：帧差 → ROI 内最大连通域 → 作为无人机位置。

交互模式（`scene4_interactive=True`）：按空格暂停，鼠标框选重定位无人机，按 Q 退出。跟丢时可手动纠正。

---

## 环境安装

### pixi（推荐）

```bat
pixi install
pixi shell
```

### pip

```bat
python -m venv venv
venv\Scripts\activate
pip install numpy opencv-python pandas matplotlib python-docx
```

---

## 运行命令

```bat
# 编译检查
python -m compileall -q src

# 单个场景
pixi run python -m src.main --scene scene1_animation
pixi run python -m src.main --scene scene2_car
pixi run python -m src.main --scene scene3_bicycle
pixi run python -m src.main --scene scene4_drone

# 调试模式（限制帧数 + 逐帧输出）
pixi run python -m src.main --scene scene2_car --max-frames 120 --debug
pixi run python -m src.main --scene scene3_bicycle --max-frames 120 --debug
pixi run python -m src.main --scene scene4_drone --max-frames 120 --debug

# 全部场景
pixi run python -m src.main --scene all

# pixi 快捷命令
pixi run scene1
pixi run scene2
pixi run scene3
pixi run scene4
pixi run debug2
pixi run debug4
pixi run run-all
```

### scene4 交互式校正

```bat
# 编辑 src/config.py，设置：
#   "scene4_interactive": True,
#   "scene4_detection_mode": "roi_largest_component",
# 然后运行：
pixi run python -m src.main --scene scene4_drone --debug
```

操作：空格暂停 → 鼠标拖拽框选无人机 → 空格恢复。跟丢时暂停重新框选即可恢复跟踪。

---

## 输出文件

```
outputs/
├── videos/{scene}_tracking.mp4           # 跟踪可视化视频
├── videos/scene4_drone_diff_debug.avi    # scene4 帧差 debug 视频
├── trajectories/{scene}_trajectory.csv   # 每帧位置 + 检测状态
├── metrics/{scene}_metrics.csv           # 单场景指标
├── metrics/summary_metrics.csv           # 四场景汇总
└── debug/{scene}_score_debug.csv         # 每帧详细 debug 字段
```

| CSV | 关键字段 |
|-----|---------|
| trajectory | `frame_id, x, y, w, h, center_x, center_y, score, detected, predicted, used_for_trajectory` |
| metrics | `scene, total_frames, detected_frames, detection_rate, average_score, average_fps` |

---

## 项目结构

```
Track/
├── src/
│   ├── main.py                         # CLI 入口、场景调度、CSV/视频输出
│   ├── config.py                       # 四个场景全部配置参数
│   ├── tracker.py                      # scene1/2/3 通用 NCC 跟踪器
│   ├── scene4_frame_diff_tracker.py    # scene4 帧差跟踪 + 交互模式
│   ├── scene3_legacy_tracker.py        # scene3 独立简单 NCC 跟踪器
│   ├── scene3_backward_fill.py         # scene3 反向补轨
│   ├── scene3_render_merged.py         # scene3 合并渲染
│   ├── scene3_motion.py                # scene3 GMC 运动检测（备用）
│   ├── global_motion.py                # GMC 全局运动估计
│   ├── ncc.py                          # 自写 NCC + 积分图 + 多模板匹配
│   ├── kalman.py                       # 自写二维卡尔曼滤波器
│   ├── preprocess.py                   # 帧预处理、模板加载
│   ├── visualize.py                    # 框/轨迹/文字绘制
│   ├── metrics.py                      # CSV 输出、指标计算
│   ├── annotation_tool.py              # 人工标注工具
│   ├── template_crop_tool.py           # 模板裁剪工具
│   └── debug_tools.py                  # NCC 诊断工具
├── document/                           # 视频 + 模板图片（只读）
├── archive_docs/                       # 归档的开发文档和调参记录
├── outputs/                            # 运行输出（不提交 Git）
├── pixi.toml                           # pixi 环境配置
└── README.md
```

---

## 调参建议

| 现象 | 可能原因 | 建议 |
|------|---------|------|
| 初始化锁错 | 模板不匹配 / 搜索区域过大 | 更换模板、提高 `threshold`、使用 `init_search_roi` |
| 中途丢失 | 模板不足 / 搜索半径太小 | 补模板、增大 `search_radius` |
| 跳到背景 | 错误高分区域 | 缩小 `search_radius`、启用 Top-K、补更干净模板 |
| 速度太慢 | 模板/尺度太多 | 增大 `ncc_step`、减少模板、调整 `resize_scale` |
| scene4 检测率低 | ROI 半径/面积阈值不当 | 调整 `scene4_roi_component_search_radius`、`scene4_component_min_area` |
| scene4 框乱跳 | 选了错误连通域 | 降低 `scene4_component_max_jump` 或用交互模式手动纠正 |

---

## 注意事项

- 所有坐标均以原始视频分辨率为准，`resize_scale` 只影响内部搜索速度
- scene2 的遮挡状态机和 GMC 补偿逻辑与场景强耦合，调参需谨慎
- scene4 交互模式依赖 OpenCV GUI 窗口，远程 SSH 环境无法使用
- `outputs/` 目录内容不应提交 Git（已在 `.gitignore` 中）
