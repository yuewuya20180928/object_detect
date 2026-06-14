#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
后处理：NMS、bbox 转换

提供：
  - nms(): 非极大值抑制
  - batched_nms(): 批处理 NMS（跨类别）
  - decode_predictions(): 解码网络输出为最终检测结果
"""

import tensorflow as tf
import numpy as np


def nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float = 0.5,
    score_threshold: float = 0.0,
    max_output_size: int = 100,
) -> tuple:
    """
    非极大值抑制

    Args:
        boxes: (N, 4) [x1, y1, x2, y2] 像素坐标
        scores: (N,)
        iou_threshold: IoU 阈值
        score_threshold: 分数阈值
        max_output_size: 最多保留框数

    Returns:
        (selected_indices, selected_scores)
    """
    if len(boxes) == 0:
        return np.array([], dtype=np.int32), np.array([], dtype=np.float32)

    # 用 TF 的 NMS（GPU 加速）
    boxes_tf = tf.constant(boxes, dtype=tf.float32)
    scores_tf = tf.constant(scores, dtype=tf.float32)
    selected = tf.image.non_max_suppression(
        boxes_tf, scores_tf,
        max_output_size=max_output_size,
        iou_threshold=iou_threshold,
        score_threshold=score_threshold,
    )
    selected_np = selected.numpy()
    return selected_np, scores[selected_np]


def batched_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    iou_threshold: float = 0.5,
    score_threshold: float = 0.0,
    max_output_size: int = 100,
) -> tuple:
    """
    跨类别 NMS（同一目标多个类预测时，取最高分类）

    Returns:
        (selected_boxes, selected_scores, selected_classes)
    """
    if len(boxes) == 0:
        return np.array([], dtype=np.float32).reshape(0, 4), np.array([]), np.array([], dtype=np.int32)

    # 加 offset 防止跨类合并
    max_coord = boxes.max() + 1
    offsets = class_ids.astype(np.float32) * max_coord
    boxes_offset = boxes + offsets[:, None]

    selected = tf.image.non_max_suppression(
        tf.constant(boxes_offset, dtype=tf.float32),
        tf.constant(scores, dtype=tf.float32),
        max_output_size=max_output_size,
        iou_threshold=iou_threshold,
        score_threshold=score_threshold,
    )
    idx = selected.numpy()
    return boxes[idx], scores[idx], class_ids[idx]


def decode_predictions(
    raw_boxes: np.ndarray,
    raw_scores: np.ndarray,
    anchors: np.ndarray,
    image_shape: tuple,
    input_size: int,
    score_thresh: float = 0.5,
    nms_iou_thresh: float = 0.5,
    max_detections: int = 100,
    num_classes: int = 80,
) -> dict:
    """
    解码网络原始输出

    Args:
        raw_boxes:   (N, 4) 网络预测的 box deltas
        raw_scores:  (N, num_classes+1) 各类别 logits
        anchors:     (N, 4) [cx, cy, w, h] 归一化
        image_shape: (H, W) 原始图像尺寸
        input_size:  模型输入尺寸
        score_thresh: 分数阈值
        nms_iou_thresh: NMS IoU 阈值
        max_detections: 最多检测数
        num_classes:   类别数

    Returns:
        {
            "boxes":   (M, 4) 像素坐标 [x1, y1, x2, y2]
            "scores":  (M,)
            "class_ids": (M,)
        }
    """
    if len(raw_boxes) == 0:
        return {
            "boxes": np.zeros((0, 4), dtype=np.float32),
            "scores": np.array([], dtype=np.float32),
            "class_ids": np.array([], dtype=np.int32),
        }

    # 1. 解码 box
    boxes_xywh = _decode_deltas(anchors, raw_boxes)
    # 转为 xyxy 并截断到 [0, 1]
    cx, cy, w, h = boxes_xywh[..., 0], boxes_xywh[..., 1], boxes_xywh[..., 2], boxes_xywh[..., 3]
    x1 = np.clip(cx - w / 2, 0, 1)
    y1 = np.clip(cy - h / 2, 0, 1)
    x2 = np.clip(cx + w / 2, 0, 1)
    y2 = np.clip(cy + h / 2, 0, 1)
    boxes_norm = np.stack([x1, y1, x2, y2], axis=-1)  # (N, 4) 归一化 xyxy

    # 2. softmax → scores
    exp = np.exp(raw_scores - raw_scores.max(axis=-1, keepdims=True))
    probs = exp / exp.sum(axis=-1, keepdims=True)
    class_probs = probs[:, 1:]  # 去掉背景
    class_ids = class_probs.argmax(axis=-1)
    class_scores = class_probs.max(axis=-1)

    # 3. 过滤低分
    keep = class_scores >= score_thresh
    boxes_norm = boxes_norm[keep]
    class_scores = class_scores[keep]
    class_ids = class_ids[keep]

    if len(boxes_norm) == 0:
        return {
            "boxes": np.zeros((0, 4), dtype=np.float32),
            "scores": np.array([], dtype=np.float32),
            "class_ids": np.array([], dtype=np.int32),
        }

    # 4. NMS（跨类别）
    boxes_px = boxes_norm * [image_shape[1], image_shape[0], image_shape[1], image_shape[0]]
    final_boxes, final_scores, final_classes = batched_nms(
        boxes_px, class_scores, class_ids,
        iou_threshold=nms_iou_thresh,
        max_output_size=max_detections,
    )

    return {
        "boxes": final_boxes,
        "scores": final_scores,
        "class_ids": final_classes,
    }


def _decode_deltas(anchors: np.ndarray, deltas: np.ndarray) -> np.ndarray:
    """解码 anchor 偏移"""
    cx = deltas[:, 0] * anchors[:, 2] + anchors[:, 0]
    cy = deltas[:, 1] * anchors[:, 3] + anchors[:, 1]
    w = anchors[:, 2] * np.exp(np.clip(deltas[:, 2], -4, 4))
    h = anchors[:, 3] * np.exp(np.clip(deltas[:, 3], -4, 4))
    return np.stack([cx, cy, w, h], axis=-1)


if __name__ == "__main__":
    # 自测
    boxes = np.array([
        [10, 10, 50, 50],
        [15, 15, 55, 55],   # 高度重叠
        [100, 100, 150, 150],
    ], dtype=np.float32)
    scores = np.array([0.9, 0.7, 0.8])
    sel, _ = nms(boxes, scores, iou_threshold=0.3)
    print(f"NMS 选择: {sel}")  # 应为 [0, 2]
