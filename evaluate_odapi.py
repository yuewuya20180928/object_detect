#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
评估 TF 官方 SSD MobileNetV2 FPNLite 320x320 预训练权重
用我们自己的 val TFRecord + mAP 计算逻辑, 验证官方模型在我们数据集上 mAP

预期: mAP@0.5 ≈ 0.28-0.32 (paper 是 0.292, 我们的 val 应该一致)
"""
import os
import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import numpy as np
import tensorflow as tf

import config
from data.dataset_builder import build_eval_dataset
from utils.metrics import compute_map
from utils.logger import get_logger

ODAPI_MODEL_DEFAULT = PROJECT_ROOT / "pretrained/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/saved_model"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", type=str, default=str(ODAPI_MODEL_DEFAULT),
                   help="OD API saved_model 路径（默认 SSD MobileNetV2 FPNLite 320）")
    p.add_argument("--input-size", type=int, default=None,
                   help="输入尺寸（None 则从 ODAPI 模型 pipeline.config 读；默认 None）")
    p.add_argument("--max-imgs", type=int, default=2476,
                   help="最多评估张数（val.record 共 2476 张，默认全量）")
    p.add_argument("--warmup-imgs", type=int, default=10,
                   help="前 N 张只计时，不计入 mAP")
    p.add_argument("--score-thresh", type=float, default=0.01,
                   help="score 过滤阈值（默认 0.01；OD API 默认 0.5 会砍掉 98% 低分预测）")
    p.add_argument("--no-pad-filter", dest="pad_filter", action="store_false", default=True,
                   help="关闭 padding 类（1-based id 81-90）过滤")
    return p.parse_args()


def main():
    args = parse_args()
    logger = get_logger("eval_odapi", PROJECT_ROOT / "logs" / "eval_odapi")
    logger.info("=" * 70)
    logger.info(f"评估 TF 官方 OD API 模型: {args.model_path}")
    logger.info(f"max_imgs={args.max_imgs}, warmup_imgs={args.warmup_imgs}, "
                f"score_thresh={args.score_thresh}, pad_filter={args.pad_filter}, "
                f"input_size={args.input_size}")
    logger.info("=" * 70)

    # 加载 TF OD API SavedModel
    logger.info(f"加载 OD API SavedModel: {args.model_path}")
    detect_fn = tf.saved_model.load(str(args.model_path))
    signature = detect_fn.signatures["serving_default"]
    logger.info(f"签名输入: {list(signature.structured_input_signature[1].keys())}")
    logger.info(f"签名输出: {list(signature.structured_outputs.keys())}")

    # 数据
    val_ds = build_eval_dataset(
        tfrecord_path=config.VAL_RECORD,
        batch_size=4,
        input_size=args.input_size or config.INPUT_SIZE,
    )
    INPUT_SIZE = args.input_size or config.INPUT_SIZE

    # 推理 + 收集
    all_dets = []
    all_gts = []
    n_imgs = 0
    for images, labels in val_ds:
        if n_imgs >= args.max_imgs:
            break
        B = images.shape[0]
        for i in range(B):
            if n_imgs >= args.max_imgs:
                break
            n_imgs += 1

            # OD API 输入需要 uint8, shape (1, H, W, 3)
            input_tensor = tf.cast(images[i:i+1] * 255.0, tf.uint8)
            detections = signature(input_tensor)
            # detections 包含:
            #   detection_boxes: (1, 100, 4) ymin, xmin, ymax, xmax (归一化 0-1)
            #   detection_scores: (1, 100)
            #   detection_classes: (1, 100) (1-based COCO id, 1-90)
            #   num_detections: (1,)
            boxes = detections["detection_boxes"][0].numpy()  # (100, 4) ymin xmin ymax xmax
            scores = detections["detection_scores"][0].numpy()  # (100,)
            classes = detections["detection_classes"][0].numpy().astype(np.int32)  # (100,) COCO 原生 1-80 (含 padding 81-90)
            num = int(detections["num_detections"][0].numpy())

            # 转成 xyxy in (input_size, input_size) 像素空间
            # OD API 返回的是归一化的 0-1 坐标
            xyxy = np.zeros((100, 4), dtype=np.float32)
            xyxy[:, 0] = boxes[:, 1] * INPUT_SIZE  # x1 = xmin
            xyxy[:, 1] = boxes[:, 0] * INPUT_SIZE  # y1 = ymin
            xyxy[:, 2] = boxes[:, 3] * INPUT_SIZE  # x2 = xmax
            xyxy[:, 3] = boxes[:, 2] * INPUT_SIZE  # y2 = ymax

            # ★ 三层过滤：
            # 1) score >= score_thresh (默认 0.01，OD API 默认 0.5 会砍掉 98% 低分预测)
            # 2) 过滤 padding 类 (1-based id 81-90)：compute_map 内部 ignore，不影响 mAP
            #    但 7.8% 预测 + 9% top-1 是 padding → 占名额 + demo 画乱标签
            score_mask = scores[:num] >= args.score_thresh
            if args.pad_filter:
                score_mask &= (classes[:num] >= 1) & (classes[:num] <= 80)
            all_dets.append({
                "boxes": xyxy[:num][score_mask],
                "scores": scores[:num][score_mask],
                "class_ids": classes[:num][score_mask] - 1,  # 1-based → 0-based (0-79)
            })

            # GT: 转 letterbox 320 空间
            gt_boxes_raw = labels["boxes"][i].numpy()
            gt_classes = labels["classes"][i].numpy()
            valid = (gt_boxes_raw[..., 2] > 0) & (gt_boxes_raw[..., 3] > 0) & (gt_classes > 0)
            gt_boxes_raw = gt_boxes_raw[valid]
            gt_classes = gt_classes[valid]
            gt_classes = gt_classes - 1
            orig_h, orig_w = labels["original_shape"][i].numpy()
            scale = INPUT_SIZE / max(orig_h, orig_w)
            # ★ 跟 dataset_builder.resize_with_padding 对齐：new_w/new_h 用 int 截断, pad 用 // 2
            # 原代码用 float 算 pad_x/pad_y，跟 image 实际像素位置偏差 0~0.75 px，拉低 mAP
            new_w = int(orig_w * scale)
            new_h = int(orig_h * scale)
            pad_x = (INPUT_SIZE - new_w) // 2
            pad_y = (INPUT_SIZE - new_h) // 2
            cx, cy, w, h = gt_boxes_raw[:, 0], gt_boxes_raw[:, 1], gt_boxes_raw[:, 2], gt_boxes_raw[:, 3]
            gt_x1 = cx * orig_w * scale + pad_x - (w * orig_w * scale) / 2
            gt_y1 = cy * orig_h * scale + pad_y - (h * orig_h * scale) / 2
            gt_x2 = cx * orig_w * scale + pad_x + (w * orig_w * scale) / 2
            gt_y2 = cy * orig_h * scale + pad_y + (h * orig_h * scale) / 2
            gt_boxes_xyxy = np.stack([gt_x1, gt_y1, gt_x2, gt_y2], axis=-1).astype(np.float32)
            all_gts.append({
                "boxes": gt_boxes_xyxy,
                "class_ids": gt_classes,
            })

    logger.info(f"共评估 {n_imgs} 张图")
    mAP_50, aps_50 = compute_map(all_dets, all_gts, num_classes=80, iou_thresh=0.5)
    logger.info(f"mAP@0.5:     {mAP_50:.4f}")
    # mAP@0.5:0.95
    iou_thresholds = np.arange(0.5, 1.0, 0.05)
    aps_all = []
    for iou_thr in iou_thresholds:
        mAP, _ = compute_map(all_dets, all_gts, num_classes=80, iou_thresh=iou_thr)
        aps_all.append(mAP)
    mAP_50_95 = float(np.mean(aps_all))
    logger.info(f"mAP@0.5:0.95: {mAP_50_95:.4f}")

    # Top 20 类别 AP
    aps_50_arr = np.array(aps_50)
    sorted_idx = np.argsort(-aps_50_arr)[:20]
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
    logger.info("\nTop 20 类别 AP@0.5:")
    for i in sorted_idx:
        logger.info(f"  {i+1:3d} {COCO_NAMES[i]:20s}: {aps_50_arr[i]:.4f}")

    logger.info("\n[对比] Paper 报告 (mAP@0.5:0.95): 0.224 / mAP@0.5: 0.292")
    logger.info(f"我们的评估 ({n_imgs} 张): mAP@0.5: {mAP_50:.4f} / mAP@0.5:0.95: {mAP_50_95:.4f}")


if __name__ == "__main__":
    main()
