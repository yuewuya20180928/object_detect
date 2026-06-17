#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
推理后处理工具

提供：
  - soft_nms(): Gaussian soft-NMS（不直接砍掉 IoU 高的低分 det，而是衰减分数）
  - top_k_filter(): 每张图只保留 top-K 个检测
  - load_per_class_thresh(): 加载 tune_thresh.py 调出的最优阈值
  - apply_per_class_thresh(): 应用 per-class 阈值过滤
"""
import json
from pathlib import Path
import numpy as np
import tensorflow as tf


# ============================================================================
# Soft-NMS (Gaussian)
# ============================================================================
def soft_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    sigma: float = 0.5,
    score_thresh: float = 0.001,
    max_det: int = 100,
) -> tuple:
    """
    Gaussian soft-NMS（参考 https://arxiv.org/abs/1704.04503）

    与 hard NMS 不同：IoU 高的低分 det 不会被砍掉，而是分数按高斯衰减。
    这样能保留被 hard NMS 误砍的、实际是另一个目标的检测。

    Args:
        boxes:        (N, 4) xyxy
        scores:       (N,) 置信度
        sigma:        高斯衰减系数，越大衰减越慢
        score_thresh: 最终保留的最低分数（衰减后低于此值丢弃）
        max_det:      最多保留几个

    Returns:
        (kept_boxes, kept_scores, kept_indices)
    """
    if len(boxes) == 0:
        return boxes, scores, np.array([], dtype=np.int32)

    boxes = boxes.astype(np.float32, copy=True)
    scores = scores.astype(np.float32, copy=True).copy()
    N = len(boxes)
    indices = np.arange(N)

    for i in range(N - 1):
        # 找到 [i, N) 中分数最高的
        max_idx = i + int(np.argmax(scores[i:]))
        if max_idx != i:
            # 交换
            indices[i], indices[max_idx] = indices[max_idx], indices[i]
            boxes[[i, max_idx]] = boxes[[max_idx, i]]
            scores[[i, max_idx]] = scores[[max_idx, i]]

        # 当前最大分 box 是 indices[i]，衰减后续所有 box 的分数
        xx1 = np.maximum(boxes[i, 0], boxes[i+1:, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[i+1:, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[i+1:, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[i+1:, 3])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_j = (boxes[i+1:, 2] - boxes[i+1:, 0]) * (boxes[i+1:, 3] - boxes[i+1:, 1])
        union = area_i + area_j - inter + 1e-6
        iou = inter / union

        # Gaussian 衰减: score = score * exp(-iou^2 / sigma)
        weight = np.exp(-(iou * iou) / sigma)
        scores[i+1:] *= weight

    # 按 score_thresh 过滤
    keep = scores >= score_thresh
    kept_indices = indices[keep]
    kept_scores = scores[keep]
    kept_boxes = boxes[keep]

    # 取 top-K (按分数排序)
    if len(kept_scores) > max_det:
        top_k = np.argsort(-kept_scores)[:max_det]
        kept_indices = kept_indices[top_k]
        kept_scores = kept_scores[top_k]
        kept_boxes = kept_boxes[top_k]

    return kept_boxes, kept_scores, kept_indices


# ============================================================================
# Top-K 过滤
# ============================================================================
def top_k_filter(boxes, scores, class_ids, k: int = 20):
    """每张图只保留 top-K 个检测（按 score 降序）"""
    if len(scores) <= k:
        return boxes, scores, class_ids
    top_k = np.argsort(-scores)[:k]
    return boxes[top_k], scores[top_k], class_ids[top_k]


# ============================================================================
# Per-class 阈值
# ============================================================================
def load_per_class_thresh(json_path: str = None) -> dict:
    """加载 tune_thresh.py 调出的最优阈值 dict，key 是 0-based class_id"""
    if json_path is None:
        json_path = Path(__file__).parent.parent / "best_per_class_thresh.json"
    if not Path(json_path).exists():
        return None
    with open(json_path) as f:
        data = json.load(f)
    return {int(k): float(v) for k, v in data.items()}


def apply_per_class_thresh(boxes, scores, class_ids, per_class_thresh: dict, default_thresh: float = 0.3):
    """
    应用 per-class score 阈值

    Args:
        boxes, scores, class_ids: 模型的输出
        per_class_thresh: {class_id_0based: thresh} 字典
        default_thresh: 字典里没有的类用这个值

    Returns:
        (filtered_boxes, filtered_scores, filtered_class_ids)
    """
    if per_class_thresh is None or len(scores) == 0:
        return boxes, scores, class_ids
    keep = np.array([
        scores[i] >= per_class_thresh.get(int(class_ids[i]), default_thresh)
        for i in range(len(scores))
    ], dtype=bool)
    return boxes[keep], scores[keep], class_ids[keep]


# ============================================================================
# Batched NMS per class（TTA concat 用）
# ============================================================================
def batched_nms_per_class(boxes, scores, class_ids, iou_threshold: float = 0.5, max_output_size: int = 200):
    """
    Class-aware NMS：对每个 class 独立做 NMS，避免跨类合并

    Args:
        boxes:     (N, 4) xyxy
        scores:    (N,)
        class_ids: (N,) int
        iou_threshold: 同类内 IoU 阈值
        max_output_size: 每个类最多保留数

    Returns:
        (kept_boxes, kept_scores, kept_class_ids)
    """
    if len(boxes) == 0:
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.int32),
        )
    boxes = boxes.astype(np.float32)
    scores = scores.astype(np.float32)
    class_ids = class_ids.astype(np.int32)

    out_b, out_s, out_c = [], [], []
    for cls in np.unique(class_ids):
        mask = class_ids == cls
        cls_boxes = boxes[mask]
        cls_scores = scores[mask]
        if len(cls_boxes) == 0:
            continue
        # 用 TF NMS
        sel = tf.image.non_max_suppression(
            tf.constant(cls_boxes),
            tf.constant(cls_scores),
            max_output_size=max_output_size,
            iou_threshold=iou_threshold,
        ).numpy()
        out_b.append(cls_boxes[sel])
        out_s.append(cls_scores[sel])
        out_c.append(np.full(len(sel), cls, dtype=np.int32))

    if not out_b:
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.int32),
        )
    return (
        np.concatenate(out_b, axis=0),
        np.concatenate(out_s, axis=0),
        np.concatenate(out_c, axis=0),
    )
