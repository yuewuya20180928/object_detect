# 🎯 TensorFlow 目标检测项目

基于 TensorFlow 2.x 的通用目标检测框架，支持 **三档预训练模型** + **两档数据集** 自由组合，采用 **迁移学习** 训练策略。

## ✨ 核心特性

- 🔧 **配置驱动**：`config.py` 改一行切换模型/数据集
- 🧠 **三档模型**：速度优先 / 均衡 / 精度优先
- 📊 **两档数据集**：COCO 2017 / Objects365
- 🏗️ **A+B 混合架构**：TF 官方 Backbone + 自定义 Keras 检测头
- 📦 **多格式导出**：SavedModel / TFLite (FP16/INT8) / ONNX
- 🎥 **摄像头实时检测**：参考 face_detection 项目成熟实现
- 🔄 **完整隔离**：模型+数据集组合的 checkpoint / 日志 / 数据完全隔离

## 🗂️ 项目结构

```
object_detection/
├── config.py                    # 静态配置入口（模型/数据集切换）
├── train.py                     # 训练主入口
├── predict.py                   # 单图/批量推理
├── camera_detect.py             # 摄像头实时检测
├── evaluate.py                  # 评估（mAP / FPS）
├── export.py                    # 模型导出（SavedModel / TFLite / ONNX）
│
├── configs/                     # 模型与数据集配置
│   ├── model_configs.py         # 三档模型配置
│   └── dataset_configs.py       # 两档数据集配置
│
├── data/                        # 数据层
│   ├── download.py              # 数据集下载
│   ├── convert_to_tfrecord.py   # 标注→TFRecord
│   └── dataset_builder.py       # tf.data pipeline
│
├── models/                      # 模型层
│   ├── detector.py              # 检测器工厂
│   ├── losses.py                # Focal Loss / Smooth L1
│   ├── anchors.py               # Anchor 生成与分配
│   ├── postprocess.py           # NMS / 解码
│   └── backbones/               # 各模型实现
│
├── utils/                       # 工具层
│   ├── logger.py
│   ├── visualization.py
│   └── metrics.py
│
├── pretrained/                  # 预训练权重
├── checkpoints/                 # 训练产物（按 model_dataset 隔离）
├── logs/                        # TensorBoard 日志
└── outputs/                     # 推理输出
```

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 选择模型和数据集

编辑 `config.py`：

```python
MODEL_MODE = "balanced"   # speed / balanced / accuracy
DATASET_MODE = "coco"     # coco / objects365
```

### 3. 下载预训练 backbone 权重

```bash
# 首次运行需联网（Keras 自动下载 ImageNet 权重，~70MB）
# 下载后保存到 pretrained/keras/，下次训练自动离线使用
python scripts/download_backbone.py --model balanced
# 或一次性下三个
python scripts/download_backbone.py --model all
```

### 4. 准备数据集

```bash
python data/download.py --dataset coco
python data/convert_to_tfrecord.py --dataset coco
```

### 5. 训练

```bash
# 默认配置（balanced + coco，batch_size=8，~20 epoch）
python train.py

# 手动指定关键参数
python train.py --epochs 20 --batch-size 4

# 断点续训
python train.py --resume

# 解冻 backbone 微调
python train.py --unfreeze-backbone --fine-tune-lr 4e-5
```

> **批大小建议**（RTX 3090 24G）：
> - `balanced` (EfficientDet-D4, 512²): batch_size=8 甜点，OOM 则降到 4
> - `speed` (SSD MobileNetV2, 320²): batch_size=32
> - `accuracy` (EfficientDet-D7, 1024²): batch_size=2 极限

### 6. 推理

```bash
# 单图推理
python predict.py --image test.jpg

# 摄像头实时检测
python camera_detect.py --device 0
```

### 7. 导出模型

```bash
python export.py --format savedmodel
python export.py --format tflite --quantize float16
python export.py --format onnx
```

## 🎛️ 模型/数据集矩阵

| | COCO 2017 | Objects365 |
|---|-----------|-----------|
| **SSD MobileNetV2** (speed) | ✅ | ✅ |
| **EfficientDet-D4** (balanced) | ✅ | ✅ |
| **EfficientDet-D7** (accuracy) | ✅ | ✅ |

每种组合有独立的 checkpoint / log / data 目录，互不干扰。

## 📊 性能参考

| 模型 | mAP (COCO) | 推理速度 (RTX 3090) | 模型大小 |
|------|-----------|-------------------|---------|
| SSD MobileNetV2 FPNLite 320 | 29.2% | ~5ms | 14 MB |
| EfficientDet-D4 | 43.0% | ~40ms | 65 MB |
| EfficientDet-D7 | 52.2% | ~250ms | 256 MB |

## 📝 开发状态

- [x] Phase 1: 项目框架搭建
- [ ] Phase 2: 数据处理层
- [ ] Phase 3: 模型层
- [ ] Phase 4: 训练流程
- [ ] Phase 5: 推理与部署
- [ ] Phase 6: 调优与文档

## 📄 许可证

MIT License
