#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tf.data 数据管道

提供：
  - parse_tfexample(): 解析单条 TFRecord
  - build_train_dataset(): 训练集 pipeline（带增强、打乱）
  - build_val_dataset(): 验证集 pipeline
  - build_test_dataset(): 测试集 pipeline

输入：TFRecord 文件
输出：(images, labels_dict)，可直接喂给 Keras 模型
"""

import os
import sys
from pathlib import Path
from typing import Tuple

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

import tensorflow as tf
import config  # noqa


# ============================================================================
# TFRecord 特征描述
# ============================================================================
_FEATURE_DESCRIPTION = {
    "image/height":             tf.io.FixedLenFeature([], tf.int64),
    "image/width":              tf.io.FixedLenFeature([], tf.int64),
    "image/filename":           tf.io.FixedLenFeature([], tf.string),
    "image/source_id":          tf.io.FixedLenFeature([], tf.string),
    "image/key/sha256":         tf.io.FixedLenFeature([], tf.string),
    "image/encoded":            tf.io.FixedLenFeature([], tf.string),
    "image/format":             tf.io.FixedLenFeature([], tf.string),
    "image/object/bbox/xmin":   tf.io.VarLenFeature(tf.float32),
    "image/object/bbox/xmax":   tf.io.VarLenFeature(tf.float32),
    "image/object/bbox/ymin":   tf.io.VarLenFeature(tf.float32),
    "image/object/bbox/ymax":   tf.io.VarLenFeature(tf.float32),
    "image/object/class/text":  tf.io.VarLenFeature(tf.string),
    "image/object/class/label": tf.io.VarLenFeature(tf.int64),
}


def parse_tfexample(example_proto):
    """
    解析单条 TFRecord Example

    Returns:
        (image, labels_dict)
        image:        tf.float32 [H, W, 3] (归一化到 [0, 1])
        labels_dict:  {
            "boxes":   tf.float32 [N, 4] (cx, cy, w, h 归一化),
            "classes": tf.int32   [N]
        }
    """
    parsed = tf.io.parse_single_example(example_proto, _FEATURE_DESCRIPTION)

    # 原始图片尺寸（供后处理恢复坐标）
    height = parsed["image/height"]
    width = parsed["image/width"]

    # 解码图片
    image = tf.io.decode_jpeg(parsed["image/encoded"], channels=3)
    image = tf.cast(image, tf.float32) / 255.0

    # 解析 bbox（[xmin, ymin, xmax, ymax] 归一化 → [cx, cy, w, h] 归一化）
    xmin = tf.sparse.to_dense(parsed["image/object/bbox/xmin"])
    ymin = tf.sparse.to_dense(parsed["image/object/bbox/ymin"])
    xmax = tf.sparse.to_dense(parsed["image/object/bbox/xmax"])
    ymax = tf.sparse.to_dense(parsed["image/object/bbox/ymax"])

    cx = (xmin + xmax) / 2.0
    cy = (ymin + ymax) / 2.0
    w  = xmax - xmin
    h  = ymax - ymin
    boxes = tf.stack([cx, cy, w, h], axis=-1)   # [N, 4]

    classes = tf.sparse.to_dense(parsed["image/object/class/label"])
    classes = tf.cast(classes, tf.int32)

    # Padding 到固定长度（用于 batch）
    boxes = tf.pad(boxes, [[0, config.MAX_BOXES - tf.shape(boxes)[0]], [0, 0]])
    classes = tf.pad(classes, [[0, config.MAX_BOXES - tf.shape(classes)[0]]])

    labels = {
        "boxes":   boxes,   # (MAX_BOXES, 4)
        "classes": classes,  # (MAX_BOXES,)
        "original_shape": tf.stack([height, width], axis=0),  # (H, W) 原始尺寸，供后处理使用
    }
    return image, labels


# ============================================================================
# 数据增强
# ============================================================================
def random_flip_horizontal(image, labels):
    """50% 概率水平翻转（图像 + bbox 同步）"""
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
        boxes = labels["boxes"]
        # cx → 1 - cx
        cx = boxes[..., 0]
        cy = boxes[..., 1]
        w = boxes[..., 2]
        h = boxes[..., 3]
        boxes = tf.stack([1.0 - cx, cy, w, h], axis=-1)
        labels["boxes"] = boxes
    return image, labels


def random_color_jitter(image, labels, brightness=0.1, contrast=0.1, saturation=0.1):
    """随机色彩抖动（不影响 bbox）"""
    image = tf.image.random_brightness(image, max_delta=brightness)
    image = tf.image.random_contrast(image, lower=1-contrast, upper=1+contrast)
    image = tf.image.random_saturation(image, lower=1-saturation, upper=1+saturation)
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image, labels


def resize_with_padding(image, labels, target_size):
    """保持宽高比缩放后 padding 到 target_size×target_size（letterbox）"""
    h = tf.shape(image)[0]
    w = tf.shape(image)[1]
    scale = tf.cast(target_size, tf.float32) / tf.cast(tf.maximum(h, w), tf.float32)
    new_h = tf.cast(tf.cast(h, tf.float32) * scale, tf.int32)
    new_w = tf.cast(tf.cast(w, tf.float32) * scale, tf.int32)

    image = tf.image.resize(image, [new_h, new_w], method=tf.image.ResizeMethod.BILINEAR)
    image = tf.image.pad_to_bounding_box(
        image,
        (target_size - new_h) // 2,
        (target_size - new_w) // 2,
        target_size, target_size
    )
    labels.pop("image_shape", None)  # 移除 resize_with_padding 写入的错误的 image_shape
    return image, labels


# ============================================================================
# 合并 Pipeline：parse + augment + resize 单次 map，避免多次序列化
# ============================================================================
def _process_sample(example_proto, target_size, augment, augment_color):
    """
    合并后的单图处理函数（替代原 3 个 map）

    Args:
        example_proto: TFRecord 原始字节
        target_size:   模型输入尺寸
        augment:       是否启用增强
        augment_color: 是否启用颜色抖动（augment 为 True 时生效）
    """
    # 1) 解析 TFRecord（拆开原 parse_tfexample 以合并后处理）
    parsed = tf.io.parse_single_example(example_proto, _FEATURE_DESCRIPTION)
    height = parsed["image/height"]
    width = parsed["image/width"]
    image = tf.io.decode_jpeg(parsed["image/encoded"], channels=3)
    image = tf.cast(image, tf.float32) / 255.0

    xmin = tf.sparse.to_dense(parsed["image/object/bbox/xmin"])
    ymin = tf.sparse.to_dense(parsed["image/object/bbox/ymin"])
    xmax = tf.sparse.to_dense(parsed["image/object/bbox/xmax"])
    ymax = tf.sparse.to_dense(parsed["image/object/bbox/ymax"])
    cx = (xmin + xmax) / 2.0
    cy = (ymin + ymax) / 2.0
    w  = xmax - xmin
    h  = ymax - ymin
    boxes = tf.stack([cx, cy, w, h], axis=-1)
    classes = tf.cast(tf.sparse.to_dense(parsed["image/object/class/label"]), tf.int32)
    boxes = tf.pad(boxes, [[0, config.MAX_BOXES - tf.shape(boxes)[0]], [0, 0]])
    classes = tf.pad(classes, [[0, config.MAX_BOXES - tf.shape(classes)[0]]])

    labels = {
        "boxes":   boxes,
        "classes": classes,
        "original_shape": tf.stack([height, width], axis=0),
    }

    # 2) 增强
    if augment:
        image, labels = random_flip_horizontal(image, labels)
        if augment_color:
            image, labels = random_color_jitter(image, labels)

    # 3) 缩放
    image, labels = resize_with_padding(image, labels, target_size)
    return image, labels


# ============================================================================
# Pipeline 构造
# ============================================================================
def build_train_dataset(
    tfrecord_path: Path = None,
    batch_size: int = None,
    input_size: int = None,
    shuffle: bool = True,
    buffer_size: int = 10000,           # 调大默认值 1000 → 10000
    augment: bool = True,
    augment_color: bool = False,         # 默认关闭颜色抖动（开发期）
) -> tf.data.Dataset:
    """
    构造训练集 pipeline

    Args:
        tfrecord_path: TFRecord 文件路径
        batch_size:    batch size
        input_size:    输入图像尺寸
        shuffle:       是否打乱
        buffer_size:   打乱 buffer（默认 10000，更接近全局打乱）
        augment:       是否启用数据增强
        augment_color: 是否启用颜色抖动（augment=True 时生效）
    """
    if tfrecord_path is None:
        tfrecord_path = config.TRAIN_RECORD
    if batch_size is None:
        batch_size = config.BATCH_SIZE
    if input_size is None:
        input_size = config.INPUT_SIZE

    files = [str(tfrecord_path)]
    dataset = tf.data.TFRecordDataset(files, num_parallel_reads=config.NUM_PARALLEL_CALLS)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=buffer_size, reshuffle_each_iteration=True)

    # 合并后的单次 map（parse + augment + resize 一步完成）
    dataset = dataset.map(
        lambda raw: _process_sample(raw, input_size, augment, augment_color),
        num_parallel_calls=config.NUM_PARALLEL_CALLS
    )

    # 批量化
    dataset = dataset.batch(batch_size, drop_remainder=True)

    # repeat() 让多 epoch 不报 “input ran out of data”
    # det.fit 会按 steps_per_epoch × epochs 计算总 step
    dataset = dataset.repeat()

    dataset = dataset.prefetch(buffer_size=config.PREFETCH_BUFFER)

    return dataset


def build_eval_dataset(
    tfrecord_path: Path = None,
    batch_size: int = None,
    input_size: int = None,
    shuffle: bool = False,
) -> tf.data.Dataset:
    """构造验证/测试集 pipeline（不增强、不打乱）"""
    if tfrecord_path is None:
        tfrecord_path = config.VAL_RECORD
    if batch_size is None:
        batch_size = config.BATCH_SIZE
    if input_size is None:
        input_size = config.INPUT_SIZE

    files = [str(tfrecord_path)]
    dataset = tf.data.TFRecordDataset(files, num_parallel_reads=config.NUM_PARALLEL_CALLS)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=10000)
    # 验证集也走合并后的 _process_sample（不增强）
    dataset = dataset.map(
        lambda raw: _process_sample(raw, input_size, False, False),
        num_parallel_calls=config.NUM_PARALLEL_CALLS
    )
    dataset = dataset.batch(batch_size, drop_remainder=False)
    dataset = dataset.prefetch(buffer_size=config.PREFETCH_BUFFER)
    return dataset


# ============================================================================
# 调试入口
# ============================================================================
def _main():
    """打印一个 batch 验证 pipeline 正确性"""
    print("构建训练集...")
    ds = build_train_dataset()
    for i, (images, labels) in enumerate(ds.take(1)):
        print(f"Batch {i}:")
        print(f"  images: {images.shape}, dtype={images.dtype}")
        print(f"  boxes:  {labels['boxes'].shape}, dtype={labels['boxes'].dtype}")
        print(f"  classes: {labels['classes'].shape}, dtype={labels['classes'].dtype}")
        print(f"  image min/max: {images.numpy().min():.3f} / {images.numpy().max():.3f}")
    print("\n构建验证集...")
    ds = build_eval_dataset()
    for i, (images, labels) in enumerate(ds.take(1)):
        print(f"Batch {i}: images {images.shape}, boxes {labels['boxes'].shape}")


if __name__ == "__main__":
    _main()
