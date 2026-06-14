#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Anchor 生成与分配

提供：
  - AnchorGenerator: 根据特征图层级生成 anchor
  - assign_anchors_to_gt: anchor 与真值匹配
  - encode_boxes / decode_boxes: 编码/解码 bbox 偏移
"""

from __future__ import annotations
import math
from typing import List, Tuple
import tensorflow as tf
import numpy as np


# ============================================================================
# Anchor 生成
# ============================================================================
class AnchorGenerator:
    """
    FPN 多尺度 Anchor 生成器

    用法：
        gen = AnchorGenerator(
            feature_sizes={'P3': 64, 'P4': 32, 'P5': 16},
            image_size=512,
            base_sizes={'P3': 0.05, 'P4': 0.1, 'P5': 0.2},
            ratios=[0.5, 1.0, 2.0],
        )
        anchors = gen.generate_all()  # (total_N, 4) [cx, cy, w, h] 归一化
    """

    def __init__(
        self,
        feature_sizes: dict,
        image_size: int,
        base_sizes: dict,
        ratios: List[float] = (0.5, 1.0, 2.0),
        scales_per_octave: int = 3,
    ):
        self.feature_sizes = feature_sizes
        self.image_size = image_size
        self.base_sizes = base_sizes
        self.ratios = list(ratios)
        # 缩放系数（每个 octave）
        # 1.0 1.26 1.59 (2^(1/3)) 之类的，实际工程中常用固定值
        self.scales = [2 ** (i / scales_per_octave) for i in range(scales_per_octave)]

    def generate_level(self, level: str) -> np.ndarray:
        """
        生成单层 anchor

        Returns:
            (N, 4) [cx, cy, w, h] 归一化
        """
        fsize = self.feature_sizes[level]
        cell = self.image_size / fsize
        base = self.base_sizes[level]

        anchors = []
        for gy in range(fsize):
            for gx in range(fsize):
                cx = (gx + 0.5) * cell / self.image_size
                cy = (gy + 0.5) * cell / self.image_size
                for ratio in self.ratios:
                    for scale in self.scales:
                        w = base * scale * math.sqrt(ratio)
                        h = base * scale / math.sqrt(ratio)
                        # 截断到 [0, 1]
                        w = min(w, 1.0)
                        h = min(h, 1.0)
                        anchors.append([cx, cy, w, h])
        return np.array(anchors, dtype=np.float32)

    def generate_all(self) -> np.ndarray:
        """生成所有层的 anchors"""
        all_anchors = []
        for level in self.feature_sizes:
            level_anchors = self.generate_level(level)
            all_anchors.append(level_anchors)
        return np.concatenate(all_anchors, axis=0)

    @property
    def num_anchors_per_cell(self) -> int:
        return len(self.ratios) * len(self.scales)


# ============================================================================
# Bbox 编解码
# ============================================================================
def encode_boxes(anchors: tf.Tensor, gt_boxes: tf.Tensor) -> tf.Tensor:
    """
    将 GT 编码为相对 anchor 的偏移

    Args:
        anchors: (N, 4) [cx, cy, w, h] 归一化
        gt_boxes: (N, 4) [cx, cy, w, h] 归一化

    Returns:
        (N, 4) [dx, dy, dw, dh]
    """
    # 中心点偏移（除以 anchor 宽高）
    dx = (gt_boxes[..., 0] - anchors[..., 0]) / anchors[..., 2]
    dy = (gt_boxes[..., 1] - anchors[..., 1]) / anchors[..., 3]
    # 宽高对数偏移
    dw = tf.math.log(gt_boxes[..., 2] / anchors[..., 2])
    dh = tf.math.log(gt_boxes[..., 3] / anchors[..., 3])
    return tf.stack([dx, dy, dw, dh], axis=-1)


def decode_boxes(anchors: tf.Tensor, deltas: tf.Tensor) -> tf.Tensor:
    """
    将网络预测的偏移解码为绝对 bbox

    Args:
        anchors: (N, 4) [cx, cy, w, h] 归一化
        deltas:  (N, 4) [dx, dy, dw, dh]

    Returns:
        (N, 4) [cx, cy, w, h] 归一化
    """
    # 中心点
    cx = deltas[..., 0] * anchors[..., 2] + anchors[..., 0]
    cy = deltas[..., 1] * anchors[..., 3] + anchors[..., 1]
    # 宽高
    w = anchors[..., 2] * tf.exp(deltas[..., 2])
    h = anchors[..., 3] * tf.exp(deltas[..., 3])
    return tf.stack([cx, cy, w, h], axis=-1)


def xywh_to_xyxy(boxes: tf.Tensor) -> tf.Tensor:
    """[cx, cy, w, h] -> [x1, y1, x2, y2]"""
    cx, cy, w, h = tf.split(boxes, 4, axis=-1)
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return tf.concat([x1, y1, x2, y2], axis=-1)


def xyxy_to_xywh(boxes: tf.Tensor) -> tf.Tensor:
    """[x1, y1, x2, y2] -> [cx, cy, w, h]"""
    x1, y1, x2, y2 = tf.split(boxes, 4, axis=-1)
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return tf.concat([cx, cy, w, h], axis=-1)


# ============================================================================
# IoU
# ============================================================================
def box_iou(boxes_a: tf.Tensor, boxes_b: tf.Tensor) -> tf.Tensor:
    """
    计算两组框之间的 IoU 矩阵

    Args:
        boxes_a: (N, 4) [cx, cy, w, h] 归一化
        boxes_b: (M, 4) [cx, cy, w, h] 归一化

    Returns:
        (N, M) IoU 矩阵
    """
    # 转为 xyxy
    a_xyxy = xywh_to_xyxy(boxes_a)
    b_xyxy = xywh_to_xyxy(boxes_b)

    # 交集
    x1 = tf.maximum(a_xyxy[..., 0:1], b_xyxy[..., 0:1, tf.newaxis])
    y1 = tf.maximum(a_xyxy[..., 1:2], b_xyxy[..., 1:2, tf.newaxis])
    x2 = tf.minimum(a_xyxy[..., 2:3], b_xyxy[..., 2:3, tf.newaxis])
    y2 = tf.minimum(a_xyxy[..., 3:4], b_xyxy[..., 3:4, tf.newaxis])

    inter = tf.maximum(x2 - x1, 0) * tf.maximum(y2 - y1, 0)
    area_a = (a_xyxy[..., 2:3] - a_xyxy[..., 0:1]) * (a_xyxy[..., 3:4] - a_xyxy[..., 1:2])
    area_b = (b_xyxy[..., 2:3, tf.newaxis] - b_xyxy[..., 0:1, tf.newaxis]) * \
             (b_xyxy[..., 3:4, tf.newaxis] - b_xyxy[..., 1:2, tf.newaxis])
    union = area_a + area_b - inter + 1e-6
    return tf.squeeze(inter / union, axis=-1) if inter.shape[-1] == 1 else inter / union


# ============================================================================
# Anchor 分配（使用独立的 @tf.function 模块避免 XLA 兼容问题）
# ============================================================================
from models.anchors_assign import assign_anchors_to_gt
