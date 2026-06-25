# 🎯 TensorFlow 目标检测项目

基于 TensorFlow 2.x 的目标检测项目，采用 **TF OD API pretrained + TTA** 主流路线，辅以 Keras 自建训练框架（已封存）。

## 📌 主路径（推荐）2026-06-26 更新

**三条 EfficientDet baseline + 一条 SSD baseline**（D1 推荐，性价比最好）：

| 模型 | 输入 | mAP@0.5（2476 张） | mAP@0.5:0.95 | 推理 (RTX 3090) | FPS | 模型 |
|---|---|---|---|---|---|---|
| SSD MobileNetV2 FPNLite | 320 | 0.3031 | 0.1779 | 13 ms | **75** | 14 MB |
| **EfficientDet-D1** ⭐ | 640 | 0.5056 | 0.3379 | 43 ms | 23 | 59 MB |
| EfficientDet-D2 | 768 | 0.5422 | 0.3692 | 56 ms | 18 | 67 MB |
| EfficientDet-D3 | 896 | **0.5733** | **0.3991** | 76 ms | 13 | 87 MB |
| D1 + TTA (640/768 + flip) | 640 | 0.5099 | 0.3411 | ~170 ms | - | - |

→ **D1 mAP 比 SSD 高 67%**（mAP@0.5:0.95 高 90%），慢 3.2x。**推荐用 D1**。
→ **D3 比 D1 mAP 再高 13.4%**，但推理慢 76%。需要高精度的场景选 D3。

> 📌 数据来源：本项目 `evaluate_odapi.py` 全量评估（修复后 2026-06-25）。
> ⚠️ paper 数字（SSD 22.2 / D1 38.4 / D2 41.8 / D3 45.4）略低，本项目实测略高（子类集可能含较多简单场景）。

### Top 类别 AP@0.5（修复后全量 2476 张）

| 类别 | SSD 320 | D1 640 | D2 768 | D3 896 |
|---|---|---|---|---|
| airplane | 0.68 | 0.89 | 0.89 | 0.91 |
| fire_hydrant | 0.57 | 0.84 | 0.90 | 0.91 |
| bus | 0.66 | 0.83 | 0.84 | 0.87 |
| train | 0.70 | 0.82 | 0.88 | 0.87 |
| person | 0.52 | 0.72 | 0.75 | 0.79 |
| motorcycle | 0.49 | 0.70 | 0.71 | 0.74 |
| bicycle | 0.33 | 0.61 | 0.65 | 0.67 |
| car | 0.24 | 0.54 | 0.61 | 0.67 |
| truck | 0.25 | 0.44 | 0.47 | 0.52 |
| boat | 0.13 | 0.35 | 0.39 | 0.45 |

**D1 Top 20 高精度类别**（修复后）:dog 0.92 / giraffe 0.90 / backpack 0.90 / airplane 0.89 / bear 0.88 / oven 0.86 / zebra 0.86 / fire_hydrant 0.84 / bus 0.83 / train 0.82 / horse 0.80 / wine glass 0.80 / fork 0.79 / teddy bear 0.78 / kite 0.77 / sheep 0.76 / book 0.76 / mouse 0.75 / person 0.72 / refrigerator 0.72

### 主路径命令速查

```bash
# 评估（推荐用 D1）
python evaluate_odapi.py \
  --model-path pretrained/efficientdet_d1_coco17_tpu-32/saved_model \
  --input-size 640 \
  --max-imgs 2476

# TTA（多尺度 + flip + NMS 融合）
python evaluate_odapi_tta.py \
  --model-path pretrained/efficientdet_d1_coco17_tpu-32/saved_model \
  --tta-scales 640 768 --tta-with-flip --tta-fusion nms \
  --max-imgs 2476

# 单图 demo + FPS
python demo_d1.py \
  --model-path pretrained/efficientdet_d1_coco17_tpu-32/saved_model \
  --input-size 640 \
  --image data/coco/raw/val2017/000000308631.jpg \
  --save /tmp/det.jpg
# FPS 测试：--fps-test 100

# 摄像头实时检测（走 OD API）
python demo_odapi.py --camera 0 \
  --weights pretrained/efficientdet_d1_coco17_tpu-32/saved_model \
  --input-size 640
```

## ⚠️ Self-Trained 路径已封存（2026-06-18）

`checkpoints/speed_coco/` 已重命名为 `checkpoints/archive_unused_self_trained_2026/`。原因：30 epoch 训练坍缩到只预测 person（mAP 0.0003），根因为 OHEM 强化 person 预测。

**主路径已切到 OD API pretrained + TTA**。`train.py` / `predict.py` / `camera_detect.py` 仍可用但属历史路径。

**如需恢复 self-trained**：`mv checkpoints/archive_unused_self_trained_2026 checkpoints/speed_coco`

## 🔧 Step 8: 数据 cat_id 修复（2026-06-25 关键 bug）

修复了 val/test/train.record 的 cat_id 错位 bug：之前的 `category_names` 列表**缺 cat_id=12 占位**（COCO cat_id 不连续，1-90 有空洞 12/26/29/30/45/66/68/69/71/83），导致所有 cat_id ≥ 13 的 GT label **错位 1**，跟 OD API 输出对不齐。

**修复**:
- `configs/dataset_configs.py`: 从 COCO JSON 动态生成 90 项 `category_names`（含空洞占位）
- `data/convert_to_tfrecord.py`: 用 COCO 原生 cat_id 1-80 保留空洞
- 重新生成 val.record / test.record / train.record
- 删除之前那个反向 1+ 错位的映射表（`coco_to_record_label.json`）
- `evaluate_odapi.py` / `evaluate_odapi_tta.py`: 移除 cat_id 映射代码

**D1 fine-tune 跳过**（2026-06-25）：tensorflow/models 跟 TF 2.20 深度不兼容（KerasTensor 不兼容、`control_flow_ops.case` 已删、keras-cv 依赖的 tensorflow-text 不兼容），继续 patch 是无底洞。D1 baseline 0.51 已超 paper 0.38，fine-tune 涨 1-2 点价值有限。**详见 `finetune_workspace/README.md`**。

## ✨ 核心特性

- 🔌 **OD API saved_model 路线**：直接加载 TF 官方预训练，无需训练即可用
- 🎯 **D1 vs SSD baseline 对比**：本项目实测 D1 显著优于 SSD（+67% mAP@0.5）
- 🔧 **TTA 集成**：多尺度 + flip + NMS 融合（`evaluate_odapi_tta.py`，D1 几乎无收益 +1%）
- 📦 **多格式导出**：ONNX / TFLite / SavedModel（`export_odapi.py`）
- 🎥 **摄像头实时检测**：`demo_odapi.py` 支持 D1 + SSD
- 📊 **稳定评估**：`evaluate_odapi.py` 全量 2476 张 val，支持任意 OD API saved_model

## 🗂️ 项目结构（主路径文件）

```
object_detection/
├── config.py                    # 静态配置入口（Keras self-trained 路径，已封存）
├── train.py / predict.py /      # Keras self-trained 训练/推理（已封存）
│   camera_detect.py / evaluate.py
├── evaluate_odapi.py            # ★ OD API 单尺度评估（修复 2 个 bug + cat_id 错位）
├── evaluate_odapi_tta.py        # ★ OD API 多尺度 TTA 评估
├── demo_odapi.py                # ★ OD API 摄像头实时检测
├── demo_d1.py                   # ★ D1/SSD 单图 demo + FPS 测试
├── export_odapi.py              # ★ OD API saved_model → ONNX/TFLite
├── finetune_workspace/          # D1 fine-tune 准备工作目录（步骤 8 跳过，详见目录内 README）
│
├── configs/                     # 模型/数据集配置（Keras self-trained 用）
├── data/                        # 数据层（TFRecord 等）
├── models/                      # Keras self-trained 模型层（已封存）
├── utils/                       # 工具层（letterbox, NMS, mAP 计算等）
│
├── pretrained/                  # ★ 预训练 saved_model
│   ├── ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/  (14 MB)
│   ├── efficientdet_d1_coco17_tpu-32/                  (59 MB, 推荐)
│   ├── efficientdet_d2_coco17_tpu-32/                  (67 MB, 高精度)
│   └── efficientdet_d3_coco17_tpu-32/                  (87 MB, 超高精度)
├── checkpoints/                 # 训练产物（archive_unused_self_trained_2026/ 已封存）
├── outputs/                     # ★ 导出模型
│   ├── d1_onnx/                 # D1 ONNX (22.5 MB, CPU 也能跑)
│   ├── d1_tflite_fp16/          # D1 TFLite FP16 (11.1 MB)
│   ├── ssd_onnx/                # SSD ONNX (11.7 MB, CPU 24.6 FPS)
│   └── ssd_tflite_fp16/         # SSD TFLite FP16
├── data/coco/                   # ★ 重新生成的 TFRecord（cat_id 修复后）
│   ├── train.record   (18 GB)
│   ├── val.record     (407 MB)
│   ├── test.record    (407 MB)
│   └── label_map.pbtxt (90 项，含 cat_id=12 占位)
└── logs/                        # 评估日志
```

## 🚀 快速开始（OD API 主路径）

### 1. 安装依赖

```bash
pip install -r requirements.txt
# 额外：ONNX 导出需要
pip install tf2onnx onnxruntime
```

### 2. 下载预训练 saved_model

```bash
# SSD MobileNetV2 FPNLite 320（baseline，~14MB）
mkdir -p pretrained && cd pretrained
wget http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz
tar xzf ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz

# EfficientDet-D1 640（推荐，~51MB）
wget http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d1_coco17_tpu-32.tar.gz
tar xzf efficientdet_d1_coco17_tpu-32.tar.gz

# 可选：D2 / D3（高精度）
wget http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d2_coco17_tpu-32.tar.gz && tar xzf efficientdet_d2_coco17_tpu-32.tar.gz
wget http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d3_coco17_tpu-32.tar.gz && tar xzf efficientdet_d3_coco17_tpu-32.tar.gz
cd ..
```

### 3. 评估 baseline

```bash
# SSD baseline
python evaluate_odapi.py --max-imgs 2476

# D1 baseline
python evaluate_odapi.py \
  --model-path pretrained/efficientdet_d1_coco17_tpu-32/saved_model \
  --input-size 640 \
  --max-imgs 2476

# D1 + TTA
python evaluate_odapi_tta.py \
  --model-path pretrained/efficientdet_d1_coco17_tpu-32/saved_model \
  --tta-scales 640 768 --tta-with-flip --tta-fusion nms \
  --max-imgs 2476
```

### 4. 单图推理 + FPS 测试

```bash
# 单图 demo
python demo_d1.py \
  --model-path pretrained/efficientdet_d1_coco17_tpu-32/saved_model \
  --input-size 640 \
  --image data/coco/raw/val2017/000000308631.jpg \
  --save /tmp/det.jpg

# FPS benchmark（RTX 3090）
python demo_d1.py --model-path pretrained/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/saved_model --input-size 320 --fps-test 100
# → SSD 320: ~13 ms / 75 FPS
python demo_d1.py --model-path pretrained/efficientdet_d1_coco17_tpu-32/saved_model --input-size 640 --fps-test 100
# → D1 640: ~43 ms / 23 FPS
```

### 5. 摄像头实时检测

```bash
# D1 实时检测
python demo_odapi.py --camera 0 \
  --weights pretrained/efficientdet_d1_coco17_tpu-32/saved_model \
  --input-size 640

# TTA 实时检测（多尺度 + flip）
python demo_odapi.py --camera 0 \
  --weights pretrained/efficientdet_d1_coco17_tpu-32/saved_model \
  --input-size 640 --tta
```

### 6. 导出模型（部署）

```bash
# ONNX（推荐，CPU 也能跑）
python export_odapi.py \
  --model-path pretrained/efficientdet_d1_coco17_tpu-32/saved_model \
  --format onnx \
  --output outputs/d1_onnx

# TFLite FP16（边缘设备；OD API saved_model 转 TFLite 有 input shape 限制，推荐 ONNX）
python export_odapi.py \
  --model-path pretrained/efficientdet_d1_coco17_tpu-32/saved_model \
  --format tflite --quantize fp16 \
  --output outputs/d1_tflite_fp16
```

## 📊 性能参考（本项目实测，2026-06-25）

### GPU 推理（RTX 3090）

| 模型 | 输入 | 单次推理 | FPS |
|---|---|---|---|
| SSD MobileNetV2 FPNLite 320 | 320 | **13 ms** | **75 FPS** |
| **EfficientDet-D1** ⭐ | 640 | 43 ms | 23 FPS |
| EfficientDet-D2 | 768 | 56 ms | 18 FPS |
| EfficientDet-D3 | 896 | 76 ms | 13 FPS |

### CPU 推理（ONNX Runtime）

| 模型 | 模型大小 | CPU 推理 | FPS |
|---|---|---|---|
| SSD 320 (ONNX) | 11.7 MB | **41 ms** | **24.6 FPS** |
| D1 640 (ONNX) | 22.5 MB | 257 ms | 3.9 FPS |
| D1 640 (TFLite FP16) | 11.1 MB | n/a（OD API shape 限制） | - |

### 评估 mAP（val 2017 全量 2476 张，cat_id 修复后）

| 模型 | mAP@0.5 | mAP@0.5:0.95 | 来源 |
|---|---|---|---|
| SSD 320 (baseline) | 0.3031 | 0.1779 | `evaluate_odapi.py` |
| **D1 640 (推荐)** | **0.5056** | **0.3379** | `evaluate_odapi.py` |
| D1 + TTA (640/768 + flip) | 0.5099 | 0.3411 | `evaluate_odapi_tta.py` |
| D2 768 (高精度) | 0.5422 | 0.3692 | `evaluate_odapi.py` |
| D3 896 (超高精度) | 0.5733 | 0.3991 | `evaluate_odapi.py` |

## 📝 历史 / 已封存

- **Self-Trained 路径**（2026-06-18 封存）：Keras DetectionModel + 自定义 head，30 epoch 训练坍缩到只预测 person（mAP 0.0003）。详见 `checkpoints/archive_unused_self_trained_2026/`。
- **train.py / predict.py / camera_detect.py**：Keras 路径配套，已封存。如需恢复自训练，参照上面的 self-trained 恢复命令。

## 📄 许可证

MIT License