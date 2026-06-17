#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TTA 评估: 多尺度 + 水平 flip + WBF 合并 (评估我们 fine-tune 的 Keras 模型)

加载: checkpoints/speed_coco/best.weights.h5 (新版训练产物)
对比: evaluate.py baseline (单尺度, 不带 TTA)

用法:
    # 默认: 加载 config.CHECKPOINT_DIR/best.weights.h5
    python evaluate_keras_tta.py

    # 自定义权重
    python evaluate_keras_tta.py --weights checkpoints/speed_coco/best.weights.h5

    # 自定义 TTA 尺度
    python evaluate_keras_tta.py --tta-scales 320 384 512 --tta-with-flip
"""
import os
import sys
import argparse
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import numpy as np
import tensorflow as tf
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy("mixed_float16")

import config
from models.detector import DetectionModel, build_anchors_for_detector
from models.postprocess import decode_predictions
from utils.letterbox import letterbox_image, unletterbox_boxes, flip_h_image, flip_h_boxes
from utils.wbf import wbf_fuse
from utils.metrics import compute_map
from utils.inference import soft_nms, top_k_filter, load_per_class_thresh, apply_per_class_thresh
from utils.logger import get_logger


def detect_at_scale(det: DetectionModel, image_float01: np.ndarray,
                    scale_size: int, flip: bool,
                    nms_iou_thresh: float, score_thresh: float,
                    max_detections: int) -> tuple:
    """在指定尺度 + flip 上做单次推理，返回原图坐标系的 (boxes, scores, class_ids)

    多尺度修复: det.anchors 是基于训练时 input_size 生成的 (19206 个)。
    TTA 在 scale_size 上推理时输出的 box 数量是 scale_size 对应的 anchor 数。
    需根据该尺度实际 feature map 形状重新生成 anchors。
    """
    img = image_float01
    if flip:
        img = flip_h_image(img)

    letterboxed, scale, pad_x, pad_y = letterbox_image(img, scale_size)
    batch = letterboxed[None]
    outputs = det.predict(batch, verbose=0)

    raw_boxes_list, raw_scores_list, actual_feature_sizes = [], [], {}
    for level in det.feature_spec.keys():
        cls = np.asarray(outputs[f"cls_{level}"][0])
        box = np.asarray(outputs[f"box_{level}"][0])
        H, W = cls.shape[:2]
        actual_feature_sizes[level] = (H, W)
        cls = cls.reshape(H, W, -1, config.NUM_CLASSES + 1).reshape(-1, config.NUM_CLASSES + 1)
        box = box.reshape(H, W, -1, 4).reshape(-1, 4)
        raw_boxes_list.append(box)
        raw_scores_list.append(cls)
    raw_boxes = np.concatenate(raw_boxes_list, axis=0)
    raw_scores = np.concatenate(raw_scores_list, axis=0)

    # 为该尺度重新生成 anchors
    if scale_size == det.input_size:
        anchors = det.anchors
    else:
        gen = build_anchors_for_detector(actual_feature_sizes, scale_size)
        anchors = gen.generate_all()
    assert len(anchors) == len(raw_boxes), \
        f"anchor count mismatch: anchors={len(anchors)}, raw_boxes={len(raw_boxes)} (scale={scale_size})"

    result = decode_predictions(
        raw_boxes, raw_scores, anchors,
        image_shape=(scale_size, scale_size),
        input_size=scale_size,
        score_thresh=score_thresh,
        nms_iou_thresh=nms_iou_thresh,
        max_detections=max_detections,
        num_classes=config.NUM_CLASSES,
    )
    xyxy_lb = result["boxes"]
    sc = result["scores"]
    cids = result["class_ids"]

    xyxy = unletterbox_boxes(xyxy_lb, scale, pad_x, pad_y)
    if flip:
        orig_w = image_float01.shape[1]
        xyxy = flip_h_boxes(xyxy, orig_w)

    return xyxy, sc, cids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default=None,
                        help="Keras 权重路径（默认 config.CHECKPOINT_DIR/best.weights.h5）")
    parser.add_argument("--tta-scales", type=int, nargs="+", default=[320, 384, 448])
    parser.add_argument("--tta-with-flip", action="store_true", default=True)
    parser.add_argument("--no-tta-flip", dest="tta_with_flip", action="store_false")
    parser.add_argument("--wbf-iou-thr", type=float, default=0.55)
    parser.add_argument("--nms-iou-thresh", type=float, default=0.5)
    parser.add_argument("--score-thresh", type=float, default=0.01)
    parser.add_argument("--use-per-class-thresh", action="store_true")
    parser.add_argument("--use-soft-nms", action="store_true")
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--max-imgs", type=int, default=1000)
    parser.add_argument("--warmup-imgs", type=int, default=10)
    args = parser.parse_args()

    if args.weights is None:
        args.weights = config.CHECKPOINT_DIR / "best.weights.h5"
    weights = Path(args.weights)
    if not weights.exists():
        print(f"❌ 权重不存在: {weights}")
        sys.exit(1)

    logger = get_logger("eval_keras_tta", config.LOG_DIR / "eval_keras_tta")
    logger.info("=" * 70)
    logger.info("TTA 评估: 多尺度 + flip + WBF (我们 fine-tune 的 Keras 模型)")
    logger.info("=" * 70)
    logger.info(f"权重: {weights}")
    logger.info(f"TTA 配置: scales={args.tta_scales}, flip={args.tta_with_flip}, WBF iou_thr={args.wbf_iou_thr}")
    logger.info(f"后处理: nms_iou={args.nms_iou_thresh}, score_thresh={args.score_thresh}, "
                f"per_class_thresh={args.use_per_class_thresh}, "
                f"soft_nms={args.use_soft_nms}, top_k={args.top_k}")

    logger.info("构建 DetectionModel...")
    det = DetectionModel()
    det.load_weights(str(weights))
    logger.info("✅ 模型 + 权重加载完成")

    val_tfrecord = config.VAL_RECORD
    logger.info(f"读 TFRecord: {val_tfrecord}")
    raw_ds = tf.data.TFRecordDataset([str(val_tfrecord)])
    FEATURE_DESCRIPTION = {
        "image/height": tf.io.FixedLenFeature([], tf.int64),
        "image/width":  tf.io.FixedLenFeature([], tf.int64),
        "image/encoded": tf.io.FixedLenFeature([], tf.string),
        "image/object/bbox/xmin": tf.io.VarLenFeature(tf.float32),
        "image/object/bbox/xmax": tf.io.VarLenFeature(tf.float32),
        "image/object/bbox/ymin": tf.io.VarLenFeature(tf.float32),
        "image/object/bbox/ymax": tf.io.VarLenFeature(tf.float32),
        "image/object/class/label": tf.io.VarLenFeature(tf.int64),
    }

    per_class_thresh = load_per_class_thresh() if args.use_per_class_thresh else None
    if per_class_thresh:
        logger.info("已加载 per-class 阈值")

    all_dets = []
    all_gts = []
    n_imgs = 0
    n_tta = 0
    t_warmup_end = time.time()

    for raw in raw_ds:
        if n_imgs >= args.max_imgs:
            break
        parsed = tf.io.parse_single_example(raw, FEATURE_DESCRIPTION)
        height = int(parsed["image/height"].numpy())
        width = int(parsed["image/width"].numpy())
        image_bytes = parsed["image/encoded"].numpy()
        img = tf.io.decode_jpeg(image_bytes, channels=3)
        image = tf.cast(img, tf.float32).numpy() / 255.0

        # GT: TFRecord 0-1 归一化 → 原图像素
        xmin = tf.sparse.to_dense(parsed["image/object/bbox/xmin"]).numpy()
        xmax = tf.sparse.to_dense(parsed["image/object/bbox/xmax"]).numpy()
        ymin = tf.sparse.to_dense(parsed["image/object/bbox/ymin"]).numpy()
        ymax = tf.sparse.to_dense(parsed["image/object/bbox/ymax"]).numpy()
        gt_classes_raw = tf.sparse.to_dense(parsed["image/object/class/label"]).numpy().astype(np.int32)
        gt_w_norm = xmax - xmin
        gt_h_norm = ymax - ymin
        valid = (gt_w_norm > 0) & (gt_h_norm > 0) & (gt_classes_raw > 0)
        gt_boxes_xyxy = np.stack([
            xmin[valid] * width,
            ymin[valid] * height,
            xmax[valid] * width,
            ymax[valid] * height,
        ], axis=-1).astype(np.float32)
        gt_cids = gt_classes_raw[valid] - 1

        # TTA 推理
        boxes_list, scores_list, cids_list = [], [], []
        for scale_size in args.tta_scales:
            xyxy, sc, cids = detect_at_scale(det, image, scale_size, flip=False,
                                              nms_iou_thresh=args.nms_iou_thresh,
                                              score_thresh=args.score_thresh,
                                              max_detections=args.top_k)
            boxes_list.append(xyxy)
            scores_list.append(sc)
            cids_list.append(cids)
            n_tta += 1
            if args.tta_with_flip:
                xyxy, sc, cids = detect_at_scale(det, image, scale_size, flip=True,
                                                  nms_iou_thresh=args.nms_iou_thresh,
                                                  score_thresh=args.score_thresh,
                                                  max_detections=args.top_k)
                boxes_list.append(xyxy)
                scores_list.append(sc)
                cids_list.append(cids)
                n_tta += 1

        fused_boxes, fused_scores, fused_cids = wbf_fuse(
            boxes_list, scores_list, cids_list,
            image_h=height, image_w=width,
            iou_thresh=args.wbf_iou_thr,
            skip_box_thresh=args.score_thresh,
        )

        valid_mask = (fused_cids >= 0) & (fused_cids < config.NUM_CLASSES)
        fused_boxes = fused_boxes[valid_mask]
        fused_scores = fused_scores[valid_mask]
        fused_cids = fused_cids[valid_mask]

        if per_class_thresh is not None and len(fused_scores) > 0:
            fused_boxes, fused_scores, fused_cids = apply_per_class_thresh(
                fused_boxes, fused_scores, fused_cids, per_class_thresh,
                default_thresh=args.score_thresh
            )
        if args.use_soft_nms and len(fused_scores) > 0:
            fused_boxes, fused_scores, _keep = soft_nms(
                fused_boxes, fused_scores, sigma=0.5, score_thresh=args.score_thresh
            )
            fused_cids = fused_cids[_keep] if len(_keep) > 0 else fused_cids[:0]
        if args.top_k < 100 and len(fused_scores) > args.top_k:
            fused_boxes, fused_scores, fused_cids = top_k_filter(
                fused_boxes, fused_scores, fused_cids, k=args.top_k
            )

        all_dets.append({
            "boxes": fused_boxes,
            "scores": fused_scores,
            "class_ids": fused_cids,
        })
        all_gts.append({
            "boxes": gt_boxes_xyxy,
            "class_ids": gt_cids,
        })

        n_imgs += 1
        if n_imgs == args.warmup_imgs:
            t_warmup_end = time.time()
            logger.info(f"[Warmup 完成] {n_imgs} 张, {n_tta} 次推理")
        elif n_imgs % 100 == 0:
            logger.info(f"已处理 {n_imgs} 张图, {n_tta} 次推理")

    elapsed = time.time() - t_warmup_end
    n_eval = n_imgs - args.warmup_imgs
    logger.info("=" * 50)
    logger.info(f"共评估 {n_imgs} 张图, {n_tta} 次推理")
    logger.info(f"实际评估 {n_eval} 张, 耗时 {elapsed:.1f}s, FPS={n_eval/elapsed:.2f}" if n_eval > 0
                else f"全部 warmup: {n_imgs} 张, {elapsed:.1f}s")

    mAP_50, aps_50 = compute_map(all_dets, all_gts, num_classes=config.NUM_CLASSES, iou_thresh=0.5)
    logger.info(f"mAP@0.5:     {mAP_50:.4f}")
    iou_thresholds = np.arange(0.5, 1.0, 0.05)
    aps_all = []
    for iou_thr in iou_thresholds:
        mAP, _ = compute_map(all_dets, all_gts, num_classes=config.NUM_CLASSES, iou_thresh=iou_thr)
        aps_all.append(mAP)
    mAP_50_95 = float(np.mean(aps_all))
    logger.info(f"mAP@0.5:0.95: {mAP_50_95:.4f}")

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

    logger.info("\n[对比] Paper SSD MobileNetV2 320 (mAP@0.5:0.95=0.224 / mAP@0.5=0.292)")
    logger.info(f"Keras TTA ({n_imgs} 张): mAP@0.5: {mAP_50:.4f} / mAP@0.5:0.95: {mAP_50_95:.4f}")


if __name__ == "__main__":
    main()
