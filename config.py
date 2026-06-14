#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
项目静态配置入口

切换模型 / 数据集只需改这两个变量：
    MODEL_MODE   = "speed" | "balanced" | "accuracy"
    DATASET_MODE = "coco"   | "objects365"

其他变量会自动联动：
    - 预训练权重路径
    - 数据集目录
    - 训练产物目录（checkpoint / log）
    - 类别数
    - 训练超参（batch_size / learning_rate / input_size / epochs）
"""

import os
import sys
from pathlib import Path

import tensorflow as tf

# 把项目根目录加入 Python 路径（支持 `python config.py` 直接运行调试）
PROJECT_ROOT = Path(__file__).parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.model_configs import get_model_config       # noqa: E402
from configs.dataset_configs import get_dataset_config   # noqa: E402

# ============================================================================
# ★★★ 核心配置（切换只需改这里） ★★★
# ============================================================================
MODEL_MODE = "speed"
DATASET_MODE = "coco"

# ============================================================================
# GPU 配置
# ============================================================================
# 1 张 RTX 3090 (24G)
GPU_ID = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID

# 是否启用混合精度训练（3090 支持 Tensor Core，启用可加速约 1.5x）
USE_MIXED_PRECISION = True

# 训练时显存按需分配（避免一次性占满）
ENABLE_MEMORY_GROWTH = True

# 随机种子（保证可复现）
RANDOM_SEED = 42

# ============================================================================
# 自动联动配置（不建议手动改）
# ============================================================================
_MODEL_CFG = get_model_config(MODEL_MODE)
_DATASET_CFG = get_dataset_config(DATASET_MODE)

# 模型
MODEL_NAME = _MODEL_CFG["name"]
INPUT_SIZE = _MODEL_CFG["input_size"]
BATCH_SIZE = _MODEL_CFG["batch_size"]
LEARNING_RATE = _MODEL_CFG["learning_rate"]
FINE_TUNE_LR = _MODEL_CFG["fine_tune_lr"]
WARMUP_STEPS = _MODEL_CFG["warmup_steps"]
TOTAL_EPOCHS = _MODEL_CFG["total_epochs"]
INFERENCE_SCORE_THRESH = _MODEL_CFG["inference_score_thresh"]
NMS_IOU_THRESH = _MODEL_CFG["nms_iou_thresh"]

# 数据集
NUM_CLASSES = _DATASET_CFG["num_classes"]
DATASET_NAME = _DATASET_CFG["name"]
MAX_BOXES = 100   # 每张图最大目标数（用于 padding）

# ============================================================================
# 目录配置（按 model+dataset 组合自动隔离）
# ============================================================================
# 共享目录
PRETRAINED_DIR = PROJECT_ROOT / "pretrained"
DATA_ROOT      = PROJECT_ROOT / "data"

# 隔离目录（核心：组合命名）
_EXPERIMENT_NAME = f"{MODEL_MODE}_{DATASET_MODE}"
CHECKPOINT_DIR  = PROJECT_ROOT / "checkpoints" / _EXPERIMENT_NAME
LOG_DIR         = PROJECT_ROOT / "logs"        / _EXPERIMENT_NAME
OUTPUT_DIR      = PROJECT_ROOT / "outputs"     / _EXPERIMENT_NAME

# 数据集专属目录
DATA_DIR        = DATA_ROOT   / DATASET_MODE
TRAIN_RECORD    = DATA_DIR    / "train.record"
VAL_RECORD      = DATA_DIR    / "val.record"
TEST_RECORD     = DATA_DIR    / "test.record"
LABEL_MAP_PATH  = DATA_DIR    / "label_map.pbtxt"

# 预训练模型目录
PRETRAINED_MODEL_DIR = PRETRAINED_DIR / _MODEL_CFG["pretrained_dir"]
PRETRAINED_CKPT = PRETRAINED_MODEL_DIR / "checkpoint" / "ckpt-0"

# 预训练权重自动检测（优先级）
# 1. PRETRAINED_DIR/keras/efficientnet{b0-b7}_notop.h5 （手动下载）
# 2. PRETRAINED_DIR/efficientnet{b0-b7}_notop.h5
# 3. ~/.keras/models/（keras 默认下载路径）
# 4. None → keras 联网下载
_BACKBONE_WEIGHTS_CANDIDATES = {
    # backbone 名 (B0~B7) -> 备选文件名列表
    "B0": ["efficientnetb0_notop.h5", "efficientnet_B0_imagenet.h5"],
    "B1": ["efficientnetb1_notop.h5", "efficientnet_B1_imagenet.h5"],
    "B2": ["efficientnetb2_notop.h5", "efficientnet_B2_imagenet.h5"],
    "B3": ["efficientnetb3_notop.h5", "efficientnet_B3_imagenet.h5"],
    "B4": ["efficientnetb4_notop.h5", "efficientnet_B4_imagenet.h5"],
    "B5": ["efficientnetb5_notop.h5", "efficientnet_B5_imagenet.h5"],
    "B6": ["efficientnetb6_notop.h5", "efficientnet_B6_imagenet.h5"],
    "B7": ["efficientnetb7_notop.h5", "efficientnet_B7_imagenet.h5"],
    "MobileNetV2": [
        "mobilenet_v2_weights.h5",
        "mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5",
    ],
}


def find_backbone_weights(backbone_name: str) -> "Path | None":
    """
    查找预训练 backbone 权重本地路径

    Args:
        backbone_name: "B0"~"B7" | "MobileNetV2"

    Returns:
        找到则返回 Path，否则 None（走 keras 在线下载）
    """
    candidates = _BACKBONE_WEIGHTS_CANDIDATES.get(backbone_name, [])
    if not candidates:
        return None

    # 搜索路径优先级
    import os
    search_dirs = [
        PRETRAINED_DIR / "keras",
        PRETRAINED_DIR,
        Path(os.path.expanduser("~/.keras/models")),
    ]
    for d in search_dirs:
        if not d.exists():
            continue
        for name in candidates:
            p = d / name
            if p.exists():
                return p
    return None


def get_model_backbone_name(model_mode: str = None) -> str:
    """根据 model_mode 返回 backbone 名（用于查找本地权重）"""
    mode = model_mode or MODEL_MODE
    if mode == "speed":
        return "MobileNetV2"
    elif mode == "balanced":
        return "B4"
    elif mode == "accuracy":
        return "B7"
    raise ValueError(f"未知 model_mode: {mode}")


# 自动检测本地权重路径（None 表示走 keras 在线下载）
BACKBONE_WEIGHTS_PATH = find_backbone_weights(get_model_backbone_name())

# ============================================================================
# 训练流程配置
# ============================================================================
# 早停：监控 val_loss，连续 N 个 epoch 不降则停
EARLY_STOP_PATIENCE = 5
EARLY_STOP_MIN_DELTA = 1e-4

# Checkpoint
SAVE_BEST_ONLY = True         # 只保存 val_loss 最优的 best.weights.h5
KEEP_LATEST = True            # 同时保留 latest.weights.h5 便于断点续训

# 数据增强
AUGMENT_TRAIN = True
AUGMENT_VAL = False           # 验证集不增强

# 数据切分（三分：train / val / test = 80/10/10）
SPLIT_TRAIN = 0.8
SPLIT_VAL = 0.1
SPLIT_TEST = 0.1

# num_workers：tf.data 加载线程数
# 设为 tf.data.AUTOTUNE 让 TF 根据 CPU 负载自动调优（8+ 核机器推荐）
NUM_PARALLEL_CALLS = tf.data.AUTOTUNE

# Prefetch buffer
# 增大 prefetch 让 GPU 不会等待数据加载
PREFETCH_BUFFER = 4           # 预取 batch 数（3090 推荐 4）

# ============================================================================
# 导出配置
# ============================================================================
EXPORT_DIR = OUTPUT_DIR / "exported"
TFLITE_DIR  = OUTPUT_DIR / "exported_tflite"
ONNX_DIR    = OUTPUT_DIR / "exported_onnx"

# ============================================================================
# 调试 & 日志
# ============================================================================
VERBOSE = True                # 是否打印详细日志
LOG_TO_FILE = True            # 是否记录到日志文件
TF_CPP_MIN_LOG_LEVEL = "2"    # TF C++ 日志级别: 0=all, 1=info, 2=warning, 3=error

# TensorBoard 配置
TENSORBOARD_HISTOGRAM_FREQ = 1
TENSORBOARD_WRITE_GRAPH = True
TENSORBOARD_UPDATE_FREQ = "epoch"

# ============================================================================
# 打印配置概览
# ============================================================================
def print_config():
    """打印当前配置概览"""
    print("=" * 70)
    print(f"🎯 TensorFlow 目标检测 - 实验配置")
    print("=" * 70)
    print(f"  实验名称:      {_EXPERIMENT_NAME}")
    print(f"  模型档位:      {MODEL_MODE:10s}  ({MODEL_NAME})")
    print(f"  数据集:        {DATASET_MODE:10s}  ({DATASET_NAME}, {NUM_CLASSES} 类)")
    print(f"  GPU:           {GPU_ID} (单卡 RTX 3090)")
    print(f"  混合精度:      {USE_MIXED_PRECISION}")
    print(f"  随机种子:      {RANDOM_SEED}")
    print("-" * 70)
    print(f"  输入尺寸:      {INPUT_SIZE}x{INPUT_SIZE}")
    print(f"  Batch Size:    {BATCH_SIZE}")
    print(f"  初始学习率:    {LEARNING_RATE}")
    print(f"  微调学习率:    {FINE_TUNE_LR}")
    print(f"  Warmup Steps:  {WARMUP_STEPS}")
    print(f"  总 Epochs:     {TOTAL_EPOCHS}")
    print(f"  早停 Patience: {EARLY_STOP_PATIENCE}")
    print("-" * 70)
    print(f"  数据目录:      {DATA_DIR}")
    print(f"  预训练权重:    {PRETRAINED_MODEL_DIR}")
    print(f"  Checkpoint:    {CHECKPOINT_DIR}")
    print(f"  TensorBoard:   {LOG_DIR}")
    print(f"  导出目录:      {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    print_config()
