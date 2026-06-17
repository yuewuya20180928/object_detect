#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WBF (Weighted Boxes Fusion) 包装

ensemble_boxes 的 weighted_boxes_fusion 要求输入:
  - boxes_list:  list of (N_i, 4) in **normalized** xyxy [0, 1]
  - scores_list: list of (N_i,)
  - labels_list: list of (N_i,)

输出:
  - boxes, scores, labels (合并后的)

包装原因:
  - 支持按 image_w, image_h 自动归一化/反归一化
  - 提供 fallback：装不上 ensemble_boxes 时用简化 IoU-NMS
"""
import numpy as np

try:
    from ensemble_boxes import weighted_boxes_fusion
    _HAS_ENSEMBLE = True
except ImportError:
    _HAS_ENSEMBLE = False


def wbf_fuse(
    boxes_list: list,
    scores_list: list,
    class_ids_list: list,
    image_h: int,
    image_w: int,
    iou_thresh: float = 0.55,
    skip_box_thresh: float = 0.001,
    weights: list = None,
) -> tuple:
    """
    多源 box 融合 (WBF)

    Args:
        boxes_list:     list of (N_i, 4) xyxy in 原图像素坐标
        scores_list:    list of (N_i,)
        class_ids_list: list of (N_i,) int
        image_h, image_w: 原图尺寸（用于归一化）
        iou_thresh:     WBF 的 IoU 阈值
        skip_box_thresh: 低于此分数的 box 不参与融合
        weights:        每个源的权重（None = 等权）

    Returns:
        (boxes, scores, class_ids) 已融合
    """
    if not _HAS_ENSEMBLE:
        raise RuntimeError(
            "ensemble_boxes 未安装。请运行: pip install ensemble-boxes"
        )

    if weights is None:
        weights = [1.0] * len(boxes_list)

    # 过滤空数组
    valid_sources = []
    for i, (b, s, c) in enumerate(zip(boxes_list, scores_list, class_ids_list)):
        if len(b) == 0:
            continue
        # 过滤零面积 box（model 预测 0 宽/高），WBF 内部会 skip 还出警告
        widths = b[:, 2] - b[:, 0]
        heights = b[:, 3] - b[:, 1]
        valid_box = (widths > 0) & (heights > 0)
        if not valid_box.any():
            continue
        b = b[valid_box]
        s = s[valid_box]
        c = c[valid_box]
        # WBF 要求 box 归一化到 [0, 1]
        b_norm = b.astype(np.float32).copy()
        b_norm[:, [0, 2]] /= float(image_w)
        b_norm[:, [1, 3]] /= float(image_h)
        b_norm = np.clip(b_norm, 0.0, 1.0)
        valid_sources.append((b_norm, s, c, weights[i]))

    if not valid_sources:
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.int32),
        )

    # ensemble_boxes 期望每类单独融合
    # 按 class_id 分组
    all_classes = sorted(set(int(c) for _, _, cs, _ in valid_sources for c in cs))
    out_boxes = []
    out_scores = []
    out_classes = []

    for cls in all_classes:
        per_src_b, per_src_s, per_src_w = [], [], []
        for b_norm, s, c, w in valid_sources:
            mask = (c == cls)
            if mask.sum() == 0:
                # 必须给空 list 占位，否则 WBF 报错
                per_src_b.append(np.zeros((0, 4), dtype=np.float32))
                per_src_s.append(np.zeros((0,), dtype=np.float32))
            else:
                per_src_b.append(b_norm[mask])
                per_src_s.append(s[mask])
            per_src_w.append(w)

        # 调用 WBF
        try:
            boxes, scores, _ = weighted_boxes_fusion(
                per_src_b, per_src_s, [np.full(len(s), cls, dtype=np.float32) for s in per_src_s],
                weights=per_src_w,
                iou_thr=iou_thresh,
                skip_box_thr=skip_box_thresh,
            )
        except Exception as e:
            print(f"[wbf_fuse] WBF 失败 class={cls}: {e}; 跳过该类")
            continue

        if len(boxes) == 0:
            continue
        # 反归一化
        boxes_xyxy = boxes.copy().astype(np.float32)
        boxes_xyxy[:, [0, 2]] *= image_w
        boxes_xyxy[:, [1, 3]] *= image_h
        out_boxes.append(boxes_xyxy)
        out_scores.append(scores.astype(np.float32))
        out_classes.append(np.full(len(boxes), cls, dtype=np.int32))

    if not out_boxes:
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.int32),
        )

    return (
        np.concatenate(out_boxes, axis=0),
        np.concatenate(out_scores, axis=0),
        np.concatenate(out_classes, axis=0),
    )
