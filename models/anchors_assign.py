# -*- coding: utf-8 -*-
"""Anchor 分配逻辑 - 独立文件，单独 tf.function 避免 XLA 兼容问题"""

from __future__ import annotations
import tensorflow as tf
from typing import Tuple


def box_iou(boxes_a: tf.Tensor, boxes_b: tf.Tensor) -> tf.Tensor:
    """计算两组框之间的 IoU 矩阵 (N,4)×(M,4) → (N,M)
    用显式 reshape 而非 tf.newaxis，确保 XLA 兼容性
    """
    # xywh → xyxy
    def xywh_to_xyxy(boxes):
        cx, cy, w, h = tf.split(boxes, 4, axis=-1)
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        return tf.concat([x1, y1, x2, y2], axis=-1)

    a_xyxy = xywh_to_xyxy(boxes_a)  # (N, 4)
    b_xyxy = xywh_to_xyxy(boxes_b)  # (M, 4)

    # 显式 reshape 为 (N,1,4) 和 (1,M,4) 以便广播
    a = tf.reshape(a_xyxy, [-1, 1, 4])  # (N, 1, 4)
    b = tf.reshape(b_xyxy, [1, -1, 4])  # (1, M, 4)

    # 交集
    x1 = tf.maximum(a[..., 0], b[..., 0])  # (N, M)
    y1 = tf.maximum(a[..., 1], b[..., 1])
    x2 = tf.minimum(a[..., 2], b[..., 2])
    y2 = tf.minimum(a[..., 3], b[..., 3])
    inter = tf.maximum(x2 - x1, 0) * tf.maximum(y2 - y1, 0)

    # 面积
    area_a = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])  # (N, M)
    area_b = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])  # (N, M)
    union = area_a + area_b - inter

    iou = tf.where(union > 0, inter / (union + 1e-10), tf.zeros_like(inter))
    return iou  # (N, M)


def encode_boxes(anchors: tf.Tensor, gt_boxes: tf.Tensor) -> tf.Tensor:
    """[cx,cy,w,h]→[dx,dy,dw,dh]"""
    dx = (gt_boxes[..., 0] - anchors[..., 0]) / anchors[..., 2]
    dy = (gt_boxes[..., 1] - anchors[..., 1]) / anchors[..., 3]
    dw = tf.math.log(gt_boxes[..., 2] / anchors[..., 2])
    dh = tf.math.log(gt_boxes[..., 3] / anchors[..., 3])
    return tf.stack([dx, dy, dw, dh], axis=-1)


# 独立函数 + 独立 tf.function（experimental_compile=False 避免 XLA）
@tf.function(experimental_compile=False)
def assign_anchors_to_gt(
    anchors: tf.Tensor,
    gt_boxes: tf.Tensor,
    gt_labels: tf.Tensor,
    pos_iou_threshold: float = 0.5,
    neg_iou_threshold: float = 0.4,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    将每个 anchor 分配为正/负/忽略（纯 tf.where，无 tf.cond）
    """
    N = tf.shape(anchors)[0]
    M = tf.shape(gt_boxes)[0]

    # 初始化（全忽略）
    cls_targets = tf.fill([N], tf.constant(-1, dtype=tf.int32))
    box_targets = tf.zeros([N, 4], dtype=tf.float32)

    # gt_boxes 为空时，IoU 矩阵 (N,0)，后续 gather 返回空张量，安全
    iou = box_iou(anchors, gt_boxes)  # (N, M) 或 (N, 0)
    max_iou = tf.reduce_max(iou, axis=1)  # (N,)
    best_gt = tf.argmax(iou, axis=1, output_type=tf.int32)  # (N,)

    # 取 GT 类别（gather 空张量返回空张量，安全）
    matched_labels = tf.gather(gt_labels, best_gt)  # (N,)

    # 正样本 / 负样本 mask
    is_pos = max_iou >= pos_iou_threshold  # (N,)
    is_neg = max_iou < neg_iou_threshold   # (N,)

    # cls_targets: 全用 tf.where，无条件分支
    cls_targets = tf.where(is_pos, tf.ones_like(cls_targets), cls_targets)
    cls_targets = tf.where(is_neg, tf.zeros_like(cls_targets), cls_targets)
    cls_targets = tf.where(cls_targets == 1, matched_labels, cls_targets)

    # 编码 box
    matched_gt_boxes = tf.gather(gt_boxes, best_gt)  # (N, 4)
    box_targets = encode_boxes(anchors, matched_gt_boxes)  # (N, 4)
    # 忽略样本 box 置 0
    box_targets = tf.where(
        (cls_targets >= 0)[:, tf.newaxis],
        box_targets,
        tf.zeros_like(box_targets)
    )
    return cls_targets, box_targets