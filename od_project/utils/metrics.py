#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
评估指标工具

提供：
  - compute_iou_matrix(): 计算两组框之间的 IoU 矩阵
  - compute_ap(): 单类别 AP (Average Precision)
  - compute_map(): mAP (mean Average Precision)
  - timer(): 推理计时上下文管理器
"""

import time
import contextlib
from typing import List, Tuple
import numpy as np


# ============================================================================
# IoU 计算
# ============================================================================
def compute_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """
    计算一个框与多个框的 IoU

    Args:
        box:   (4,) [x1, y1, x2, y2]
        boxes: (N, 4) [x1, y1, x2, y2]

    Returns:
        (N,) IoU 值
    """
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area1 = (box[2] - box[0]) * (box[3] - box[1])
    area2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = area1 + area2 - inter + 1e-6
    return inter / union


def compute_iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """
    计算两组框之间的 IoU 矩阵

    Args:
        boxes_a: (N, 4)
        boxes_b: (M, 4)

    Returns:
        (N, M) IoU 矩阵
    """
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)

    boxes_a = np.asarray(boxes_a)
    boxes_b = np.asarray(boxes_b)

    # 计算交集
    x1 = np.maximum(boxes_a[:, None, 0], boxes_b[None, :, 0])
    y1 = np.maximum(boxes_a[:, None, 1], boxes_b[None, :, 1])
    x2 = np.minimum(boxes_a[:, None, 2], boxes_b[None, :, 2])
    y2 = np.minimum(boxes_a[:, None, 3], boxes_b[None, :, 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter + 1e-6
    return inter / union


# ============================================================================
# AP / mAP 计算
# ============================================================================
def compute_ap(
    recall: np.ndarray,
    precision: np.ndarray,
    use_11_point: bool = False,
) -> float:
    """
    计算 AP (Average Precision)

    Args:
        recall:    (N,) 召回率（递增）
        precision: (N,) 精确率
        use_11_point: True=COCO 旧标准 11 点插值；False=面积法（COCO 新标准）

    Returns:
        AP 值
    """
    # 在两端补点
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # 让 precision 单调递减（包络）
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])

    if use_11_point:
        # 11 点插值（旧 COCO 标准）
        ap = 0.0
        for t in np.linspace(0, 1, 11):
            mask = mrec >= t
            p = mpre[mask].max() if mask.any() else 0
            ap += p / 11
    else:
        # 面积法（PR 曲线下面积）
        i = np.where(mrec[1:] != mrec[:-1])[0]
        ap = float(np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1]))
    return ap


def compute_map(
    all_detections: List[dict],
    all_groundtruths: List[dict],
    num_classes: int,
    iou_thresh: float = 0.5,
) -> Tuple[float, List[float]]:
    """
    计算 mAP@iou_thresh

    Args:
        all_detections:    每张图的检测结果 [{boxes, scores, class_ids}, ...]
        all_groundtruths:  每张图的真值      [{boxes, class_ids}, ...]
        num_classes:       类别数
        iou_thresh:        IoU 阈值（0.5 表示 mAP@0.5）

    Returns:
        (mAP, [ap_per_class])
    """
    aps = []
    for cls_id in range(num_classes):
        # 收集该类所有检测和真值
        y_scores = []
        y_true = []
        n_gt = 0

        for dets, gts in zip(all_detections, all_groundtruths):
            gt_mask = gts["class_ids"] == cls_id
            gt_boxes = gts["boxes"][gt_mask]
            n_gt += len(gt_boxes)

            det_mask = dets["class_ids"] == cls_id
            det_boxes = dets["boxes"][det_mask]
            det_scores = dets["scores"][det_mask]

            if len(det_boxes) == 0:
                continue

            # 该检测框与该类所有 GT 的 IoU
            if len(gt_boxes) > 0:
                ious = compute_iou_matrix(det_boxes, gt_boxes)
            else:
                ious = np.zeros((len(det_boxes), 0))

            matched = np.zeros(len(gt_boxes), dtype=bool)
            for i, (box, score) in enumerate(zip(det_boxes, det_scores)):
                if ious.shape[1] > 0:
                    best_iou = ious[i].max()
                    best_gt = ious[i].argmax()
                else:
                    best_iou = 0
                    best_gt = -1
                if best_iou >= iou_thresh and not matched[best_gt]:
                    y_true.append(1)
                    matched[best_gt] = True
                else:
                    y_true.append(0)
                y_scores.append(score)

        if n_gt == 0:
            # 该类没有 GT，AP 记为 0
            aps.append(0.0)
            continue
        if not y_scores:
            # 有 GT 但没检测到
            aps.append(0.0)
            continue

        # 按分数排序
        y_scores = np.array(y_scores)
        y_true = np.array(y_true, dtype=np.int32)
        order = np.argsort(-y_scores)
        y_true = y_true[order]

        tp = np.cumsum(y_true)
        fp = np.cumsum(1 - y_true)
        recall = tp / max(n_gt, 1)
        precision = tp / np.maximum(tp + fp, 1)
        ap = compute_ap(recall, precision)
        aps.append(ap)

    return float(np.mean(aps)), aps


# ============================================================================
# 计时
# ============================================================================
@contextlib.contextmanager
def timer(name: str = "操作", logger=None):
    """
    计时上下文管理器

    用法：
        with timer("模型推理"):
            outputs = model(inputs)
    """
    t0 = time.perf_counter()
    yield
    elapsed = time.perf_counter() - t0
    msg = f"[{name}] 耗时 {elapsed*1000:.1f} ms"
    if logger:
        logger.info(msg)
    else:
        print(msg)


def benchmark(model_fn, input_shape, n_warmup: int = 5, n_runs: int = 50, logger=None):
    """
    基准测试：测量模型推理速度

    Args:
        model_fn:    接受 input_shape 的 callable，返回输出
        input_shape: 输入 shape
        n_warmup:    预热次数
        n_runs:      正式测试次数
    """
    import tensorflow as tf
    # 预热
    for _ in range(n_warmup):
        _ = model_fn(tf.random.normal(input_shape))
    # 测试
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        _ = model_fn(tf.random.normal(input_shape))
        times.append(time.perf_counter() - t0)
    times = np.array(times) * 1000  # 转为 ms
    msg = (
        f"[Benchmark] input={input_shape}  "
        f"mean={times.mean():.2f}ms  "
        f"median={np.median(times):.2f}ms  "
        f"p95={np.percentile(times, 95):.2f}ms  "
        f"FPS={1000/times.mean():.1f}"
    )
    if logger:
        logger.info(msg)
    else:
        print(msg)
    return times


if __name__ == "__main__":
    # 自测
    print("=== IoU 矩阵 ===")
    a = np.array([[0, 0, 10, 10], [5, 5, 15, 15]])
    b = np.array([[0, 0, 10, 10], [20, 20, 30, 30]])
    print(compute_iou_matrix(a, b))

    print("\n=== 计时 ===")
    with timer("测试"):
        time.sleep(0.05)
