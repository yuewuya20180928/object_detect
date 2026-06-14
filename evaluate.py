#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型评估脚本

计算：
  - mAP@0.5
  - mAP@0.5:0.95
  - 各类别 AP
  - 推理 FPS

用法：
    python evaluate.py
    python evaluate.py --weights checkpoints/balanced_coco/best.weights.h5
    python evaluate.py --benchmark              # 只跑性能测试
"""

import os
import sys
import argparse
import time
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import numpy as np
import tensorflow as tf
import config
from utils.logger import get_logger
from utils.metrics import compute_map, benchmark
from models.detector import DetectionModel
from models.postprocess import decode_predictions
from data.dataset_builder import build_eval_dataset


# ============================================================================
# 加载类别名
# ============================================================================
def load_class_names(label_map_path: Path) -> list:
    names = []
    if not label_map_path.exists():
        return names
    with open(label_map_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("name:"):
                name = line.split("'")[1] if "'" in line else line.split('"')[1]
                names.append(name)
    return names


# ============================================================================
# mAP 评估
# ============================================================================
def evaluate_map(det, test_ds, class_names, score_thresh, nms_iou_thresh, logger, max_samples=500):
    """计算 mAP"""
    logger.info("=" * 50)
    logger.info("开始 mAP 评估")
    logger.info("=" * 50)

    all_dets = []
    all_gts = []

    sample_count = 0
    for images, labels in test_ds:
        if sample_count >= max_samples:
            break

        # 推理
        outputs = det.predict(images, verbose=0)

        # 收集该 batch 的预测
        for i in range(images.shape[0]):
            raw_boxes_list, raw_scores_list = [], []
            for level in sorted(outputs.keys()):
                if not level.startswith("cls_"):
                    continue
                box_key = level.replace("cls_", "box_")
                cls = outputs[level][i].numpy()
                box = outputs[box_key][i].numpy()
                H, W = cls.shape[:2]
                cls = cls.reshape(H, W, -1, config.NUM_CLASSES + 1).reshape(-1, config.NUM_CLASSES + 1)
                box = box.reshape(H, W, -1, 4).reshape(-1, 4)
                raw_boxes_list.append(box)
                raw_scores_list.append(cls)

            raw_boxes = np.concatenate(raw_boxes_list, axis=0)
            raw_scores = np.concatenate(raw_scores_list, axis=0)

            # 解码
            result = decode_predictions(
                raw_boxes, raw_scores, det.anchors,
                image_shape=(images.shape[1], images.shape[2]),
                input_size=config.INPUT_SIZE,
                score_thresh=score_thresh,
                nms_iou_thresh=nms_iou_thresh,
                num_classes=config.NUM_CLASSES,
            )
            all_dets.append({
                "boxes": result["boxes"],
                "scores": result["scores"],
                "class_ids": result["class_ids"],
            })

            # 真值
            gt_boxes = labels["boxes"][i].numpy()
            gt_classes = labels["classes"][i].numpy()
            # 过滤空框
            valid = (gt_boxes[..., 2] > 0) & (gt_boxes[..., 3] > 0)
            gt_boxes = gt_boxes[valid]
            gt_classes = gt_classes[valid]
            # 转为 xyxy
            cx, cy, w, h = gt_boxes[..., 0], gt_boxes[..., 1], gt_boxes[..., 2], gt_boxes[..., 3]
            gt_boxes_xyxy = np.stack([
                cx - w/2, cy - h/2, cx + w/2, cy + h/2
            ], axis=-1)
            all_gts.append({
                "boxes": gt_boxes_xyxy,
                "class_ids": gt_classes,
            })

        sample_count += images.shape[0]
        if sample_count % 50 == 0:
            logger.info(f"  已评估: {sample_count}/{max_samples}")

    logger.info(f"共评估 {len(all_dets)} 张图")

    # mAP@0.5
    mAP_50, aps_50 = compute_map(all_dets, all_gts, num_classes=len(class_names), iou_thresh=0.5)
    logger.info(f"mAP@0.5:     {mAP_50:.4f}")

    # mAP@0.5:0.95
    iou_thresholds = np.arange(0.5, 1.0, 0.05)
    aps_all = []
    for iou_thr in iou_thresholds:
        mAP, _ = compute_map(all_dets, all_gts, num_classes=len(class_names), iou_thresh=iou_thr)
        aps_all.append(mAP)
    mAP_50_95 = np.mean(aps_all)
    logger.info(f"mAP@0.5:0.95: {mAP_50_95:.4f}")

    # 各类别 AP (mAP@0.5)
    logger.info("\n各类别 AP@0.5:")
    for i, (name, ap) in enumerate(zip(class_names, aps_50)):
        if i < 10 or ap > 0.3:  # 只打印前 10 个 + AP 较高的
            logger.info(f"  {i+1:3d} {name:20s}: {ap:.4f}")

    return mAP_50, mAP_50_95


# ============================================================================
# 性能基准测试
# ============================================================================
def evaluate_fps(det, input_size, n_warmup=10, n_runs=50, logger=None):
    """测 FPS"""
    logger.info("=" * 50)
    logger.info("性能基准测试")
    logger.info("=" * 50)
    times = benchmark(
        det.model,
        input_shape=(1, input_size, input_size, 3),
        n_warmup=n_warmup,
        n_runs=n_runs,
        logger=logger,
    )
    fps = 1000 / times.mean()
    logger.info(f"  FPS: {fps:.1f}")
    return fps


# ============================================================================
# 主函数
# ============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default=None,
                        help="权重路径（默认 checkpoints/{experiment}/best.weights.h5）")
    parser.add_argument("--score-thresh", type=float, default=config.INFERENCE_SCORE_THRESH)
    parser.add_argument("--nms-iou-thresh", type=float, default=config.NMS_IOU_THRESH)
    parser.add_argument("--max-samples", type=int, default=500,
                        help="最大评估样本数（控制耗时）")
    parser.add_argument("--benchmark-only", action="store_true", help="只跑性能测试")
    parser.add_argument("--map-only", action="store_true", help="只跑 mAP 评估")
    args = parser.parse_args()

    if args.weights is None:
        args.weights = config.CHECKPOINT_DIR / "best.weights.h5"
    weights = Path(args.weights)
    if not weights.exists():
        print(f"❌ 权重不存在: {weights}")
        sys.exit(1)

    logger = get_logger("evaluate", config.LOG_DIR)
    logger.info(f"加载权重: {weights}")

    # 构建检测器
    det = DetectionModel()
    det.load_weights(str(weights))
    logger.info("模型加载完成")

    class_names = load_class_names(config.LABEL_MAP_PATH)

    # 性能基准
    if not args.map_only:
        evaluate_fps(det, config.INPUT_SIZE, logger=logger)

    # mAP 评估
    if not args.benchmark_only:
        test_ds = build_eval_dataset(
            tfrecord_path=config.TEST_RECORD,
            batch_size=config.BATCH_SIZE,
            input_size=config.INPUT_SIZE,
        )
        evaluate_map(
            det, test_ds, class_names,
            args.score_thresh, args.nms_iou_thresh,
            logger, max_samples=args.max_samples,
        )

    logger.info("评估完成")


if __name__ == "__main__":
    main()
