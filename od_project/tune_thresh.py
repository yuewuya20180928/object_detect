#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Per-class score 阈值自动调参

对每类在 val 上 grid search 最优 score 阈值，使得该类 AP 最大。
最优阈值保存到 best_per_class_thresh.json，给 evaluate/demo 用。

预期收益: mAP +1-3%
"""
import os
import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT.parent))

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import numpy as np
import tensorflow as tf
import config
from data.dataset_builder import build_eval_dataset
from utils.metrics import compute_map, compute_iou_matrix, compute_ap
from utils.logger import get_logger

COCO_NAMES = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck",
    "boat","traffic light","fire hydrant","stop sign","parking meter","bench",
    "bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe",
    "backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard",
    "sports ball","kite","baseball bat","baseball glove","skateboard","surfboard",
    "tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl",
    "banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza",
    "donut","cake","chair","couch","potted plant","bed","dining table","toilet",
    "tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven",
    "toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear",
    "hair drier","toothbrush",
]

# 阈值搜索范围（不要 < 0.05 防止噪声）
THRESH_CANDIDATES = [0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40,
                     0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
# 0.01 是关键: baseline 等效阈值, tune 要把它包进去才能合理比较


def collect_predictions(val_ds, sig, max_imgs: int = 1000):
    """跑模型，收集所有 dets + gts"""
    all_dets, all_gts = [], []
    n_imgs = 0
    for images, labels in val_ds:
        if n_imgs >= max_imgs:
            break
        B = images.shape[0]
        for i in range(B):
            if n_imgs >= max_imgs:
                break
            n_imgs += 1
            inp = tf.cast(images[i:i+1] * 255.0, tf.uint8)
            out = sig(inp)
            boxes = out["detection_boxes"][0].numpy()
            sc = out["detection_scores"][0].numpy()
            cls = out["detection_classes"][0].numpy().astype(np.int32)
            num = int(out["num_detections"][0].numpy())
            xyxy = np.zeros((100, 4), dtype=np.float32)
            xyxy[:, 0] = boxes[:, 1] * config.INPUT_SIZE
            xyxy[:, 1] = boxes[:, 0] * config.INPUT_SIZE
            xyxy[:, 2] = boxes[:, 3] * config.INPUT_SIZE
            xyxy[:, 3] = boxes[:, 2] * config.INPUT_SIZE
            cids = cls[:num] - 1
            valid = (cids >= 0) & (cids < 80)
            all_dets.append({
                "boxes": xyxy[:num][valid],
                "scores": sc[:num][valid],
                "class_ids": cids[valid],
            })
            gt_boxes_raw = labels["boxes"][i].numpy()
            gt_classes = labels["classes"][i].numpy()
            ok = (gt_boxes_raw[..., 2] > 0) & (gt_boxes_raw[..., 3] > 0) & (gt_classes > 0)
            gt_boxes_raw = gt_boxes_raw[ok]
            gt_c = gt_classes[ok]
            orig_h, orig_w = labels["original_shape"][i].numpy()
            scale = config.INPUT_SIZE / max(orig_h, orig_w)
            new_w = int(orig_w * scale)
            new_h = int(orig_h * scale)
            pad_x = (config.INPUT_SIZE - new_w) // 2
            pad_y = (config.INPUT_SIZE - new_h) // 2
            cx, cy, w, h = gt_boxes_raw[:, 0], gt_boxes_raw[:, 1], gt_boxes_raw[:, 2], gt_boxes_raw[:, 3]
            gt_x1 = cx * new_w + pad_x - (w * new_w) / 2
            gt_y1 = cy * new_h + pad_y - (h * new_h) / 2
            gt_x2 = cx * new_w + pad_x + (w * new_w) / 2
            gt_y2 = cy * new_h + pad_y + (h * new_h) / 2
            all_gts.append({
                "boxes": np.stack([gt_x1, gt_y1, gt_x2, gt_y2], -1).astype(np.float32),
                "class_ids": gt_c - 1,
            })
    return all_dets, all_gts, n_imgs


def compute_ap_for_class(all_dets, all_gts, cls_id, score_thresh=0.0, iou_thresh=0.5):
    """只算单类的 AP, 比 compute_map 快 80x (因为不用算其他 79 类)"""
    y_scores = []
    y_true = []
    n_gt = 0
    for dets, gts in zip(all_dets, all_gts):
        gt_mask = gts["class_ids"] == cls_id
        gt_boxes = gts["boxes"][gt_mask]
        n_gt += len(gt_boxes)
        det_mask = (dets["class_ids"] == cls_id) & (dets["scores"] >= score_thresh)
        det_boxes = dets["boxes"][det_mask]
        det_scores = dets["scores"][det_mask]
        if len(det_boxes) == 0:
            continue
        if len(gt_boxes) > 0:
            ious = compute_iou_matrix(det_boxes, gt_boxes)
        else:
            ious = np.zeros((len(det_boxes), 0))
        matched = np.zeros(len(gt_boxes), dtype=bool)
        for i in range(len(det_boxes)):
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
            y_scores.append(det_scores[i])
    if n_gt == 0 or not y_scores:
        return 0.0
    y_scores = np.array(y_scores)
    y_true = np.array(y_true, dtype=np.int32)
    tp = np.cumsum(y_true)
    fp = np.cumsum(1 - y_true)
    recall = tp / max(n_gt, 1)
    precision = tp / np.maximum(tp + fp, 1)
    return compute_ap(recall, precision)


def tune_per_class_thresh(all_dets, all_gts, logger):
    """对每类 grid search 最优 score 阈值
    改进: 包含 0.01 作为候选, 避免抬高 thresh 损失低分 TPs
    """
    best_thresh = {}
    best_ap = {}
    for cls_id in range(80):
        best_t = 0.01
        best_a = 0.0
        for t in THRESH_CANDIDATES:
            a = compute_ap_for_class(all_dets, all_gts, cls_id, score_thresh=t)
            if a > best_a:
                best_a = a
                best_t = t
        best_thresh[cls_id] = best_t
        best_ap[cls_id] = best_a
        if (cls_id + 1) % 10 == 0:
            logger.info(f"  tuned {cls_id + 1}/80  (current: {COCO_NAMES[cls_id]:18s} thresh={best_t:.2f} AP={best_a:.4f})")

    return best_thresh, best_ap


def main():
    logger = get_logger("tune_thresh", PROJECT_ROOT / "logs" / "tune_thresh")
    logger.info("=" * 70)
    logger.info("Per-class score 阈值自动调参 (TF OD API SSD MobileNetV2 FPNLite 320x320)")
    logger.info("=" * 70)

    logger.info(f"加载模型: {PROJECT_ROOT}/pretrained/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/saved_model")
    detect_fn = tf.saved_model.load(str(PROJECT_ROOT / "pretrained" / "ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8" / "saved_model"))
    sig = detect_fn.signatures["serving_default"]

    val_ds = build_eval_dataset(
        tfrecord_path=config.VAL_RECORD,
        batch_size=4,
        input_size=config.INPUT_SIZE,
    )

    logger.info("收集 1000 张 val 的所有 predictions...")
    all_dets, all_gts, n_imgs = collect_predictions(val_ds, sig, max_imgs=1000)
    logger.info(f"  收集到 {n_imgs} 张图的 predictions")

    logger.info("Grid search 80 类最优阈值 (16 个候选)...")
    best_thresh, best_ap = tune_per_class_thresh(all_dets, all_gts, logger)

    # 保存
    out_path = PROJECT_ROOT / "best_per_class_thresh.json"
    with open(out_path, "w") as f:
        json.dump({str(k): float(v) for k, v in best_thresh.items()}, f, indent=2)
    logger.info(f"已保存最优阈值到: {out_path}")

    # 打印结果
    logger.info("\n=== 80 类最优阈值 (按阈值升序) ===")
    sorted_cls = sorted(range(80), key=lambda c: best_thresh[c])
    for c in sorted_cls:
        logger.info(f"  {COCO_NAMES[c]:18s} (1-based {c+1:3d}): thresh={best_thresh[c]:.2f}  AP={best_ap[c]:.4f}")

    # 验证: 调参后 mAP
    logger.info("\n=== 验证: 应用 per-class thresh 后全局 mAP ===")
    N = len(all_dets)
    filtered_dets = []
    for k in range(N):
        det = all_dets[k]
        thresh_arr = np.array([best_thresh.get(int(c), 0.01) for c in det["class_ids"]])
        keep = det["scores"] >= thresh_arr
        filtered_dets.append({
            "boxes": det["boxes"][keep],
            "scores": det["scores"][keep],
            "class_ids": det["class_ids"][keep],
        })
    mAP_50, _ = compute_map(filtered_dets, all_gts, num_classes=80, iou_thresh=0.5)
    iou_thrs = np.arange(0.5, 1.0, 0.05)
    aps_all = [compute_map(filtered_dets, all_gts, num_classes=80, iou_thresh=t)[0] for t in iou_thrs]
    mAP_50_95 = float(np.mean(aps_all))
    logger.info(f"  mAP@0.5:     {mAP_50:.4f}")
    logger.info(f"  mAP@0.5:0.95: {mAP_50_95:.4f}")
    logger.info(f"  baseline mAP@0.5 (无 thresh): 0.0628")


if __name__ == "__main__":
    main()
