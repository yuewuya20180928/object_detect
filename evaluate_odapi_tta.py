#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TTA 评估: 多尺度 + 水平 flip + WBF 合并

用法:
    # 默认 3 个尺度 [320, 384, 448] + flip = 6 次推理
    python evaluate_odapi_tta.py

    # 自定义尺度
    python evaluate_odapi_tta.py --tta-scales 320 384 512 --tta-with-flip

    # 单尺度 flip TTA（最快）
    python evaluate_odapi_tta.py --tta-scales 320 --tta-with-flip

    # 跑评估 + per-class thresh + soft-NMS + Top-K
    python evaluate_odapi_tta.py \
        --use-per-class-thresh --use-soft-nms --top-k 30 \
        --tta-scales 320 384 448 --tta-with-flip
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

import config
from utils.letterbox import letterbox_image, unletterbox_boxes, flip_h_image, flip_h_boxes
from utils.wbf import wbf_fuse, _HAS_ENSEMBLE
from utils.metrics import compute_map
from utils.inference import soft_nms, top_k_filter, load_per_class_thresh, apply_per_class_thresh, batched_nms_per_class
from utils.logger import get_logger

ODAPI_MODEL_DEFAULT = PROJECT_ROOT / "pretrained/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/saved_model"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", type=str, default=str(ODAPI_MODEL_DEFAULT),
                   help="OD API saved_model 路径")
    p.add_argument("--tta-scales", type=int, nargs="+", default=[320, 384, 448],
                   help="TTA 多尺度推理的尺寸列表（默认 [320, 384, 448]）")
    p.add_argument("--tta-with-flip", action="store_true", default=True,
                   help="在每个尺度上额外做水平 flip（默认 True）")
    p.add_argument("--no-tta-flip", dest="tta_with_flip", action="store_false")
    p.add_argument("--tta-fusion", type=str, default="wbf", choices=["wbf", "nms", "none"],
                   help="TTA 融合方式: wbf (Weighted Boxes Fusion) | nms (concat + class-aware NMS) | none (单尺度, baseline 对照)")
    p.add_argument("--wbf-iou-thr", type=float, default=0.55, help="WBF 融合 IoU 阈值")
    p.add_argument("--nms-iou-thr", type=float, default=0.5, help="NMS 融合 IoU 阈值")
    p.add_argument("--use-per-class-thresh", action="store_true")
    p.add_argument("--use-soft-nms", action="store_true")
    p.add_argument("--top-k", type=int, default=100)
    p.add_argument("--max-imgs", type=int, default=1000)
    p.add_argument("--score-thresh", type=float, default=0.01)
    p.add_argument("--warmup-imgs", type=int, default=10, help="前 N 张只计时，不计入 mAP")
    return p.parse_args()


def detect_at_scale(detect_fn, image_float01, scale_size, flip: bool):
    """
    在指定尺度 + flip 上做单次推理，返回原图坐标系的 (boxes, scores, class_ids)

    Args:
        detect_fn:     OD API saved_model 的 serving_default signature
        image_float01: (H, W, 3) float32 [0, 1] 原图
        scale_size:    letterbox 目标尺寸
        flip:          是否先水平翻转

    Returns:
        boxes (N, 4) 原图像素坐标 xyxy
        scores (N,)
        class_ids (N,) 0-based
    """
    img = image_float01
    if flip:
        img = flip_h_image(img)

    # letterbox
    letterboxed, scale, pad_x, pad_y = letterbox_image(img, scale_size)

    # 推理 (OD API 需要 uint8 (1, H, W, 3))
    input_tensor = tf.cast(letterboxed[None] * 255.0, tf.uint8)
    detections = detect_fn(input_tensor)

    boxes = detections["detection_boxes"][0].numpy()  # (100, 4) ymin xmin ymax xmax (归一化)
    scores = detections["detection_scores"][0].numpy()
    classes = detections["detection_classes"][0].numpy().astype(np.int32)  # 1-based COCO
    num = int(detections["num_detections"][0].numpy())

    # 转 xyxy in letterbox 像素空间
    xyxy_letter = np.zeros((num, 4), dtype=np.float32)
    xyxy_letter[:, 0] = boxes[:num, 1] * scale_size  # x1
    xyxy_letter[:, 1] = boxes[:num, 0] * scale_size  # y1
    xyxy_letter[:, 2] = boxes[:num, 3] * scale_size  # x2
    xyxy_letter[:, 3] = boxes[:num, 2] * scale_size  # y2

    # 反 letterbox 到 原图坐标
    xyxy = unletterbox_boxes(xyxy_letter, scale, pad_x, pad_y)

    # 如果是 flip 推理，把 box 翻回去（flip 后的图像宽 = 原图宽）
    if flip:
        orig_w = image_float01.shape[1]
        xyxy = flip_h_boxes(xyxy, orig_w)

    class_ids = classes[:num] - 1
    return xyxy, scores[:num], class_ids


def main():
    args = parse_args()
    logger = get_logger("eval_odapi_tta", PROJECT_ROOT / "logs" / "eval_odapi_tta")
    logger.info("=" * 70)
    logger.info("TTA 评估: 多尺度 + flip + WBF")
    logger.info("=" * 70)
    if not _HAS_ENSEMBLE:
        logger.error("ensemble_boxes 未安装！pip install ensemble-boxes")
        sys.exit(1)
    logger.info(f"TTA 配置: scales={args.tta_scales}, flip={args.tta_with_flip}, "
                f"fusion={args.tta_fusion}, WBF iou_thr={args.wbf_iou_thr}")

    # 加载 OD API
    logger.info(f"加载 OD API SavedModel: {args.model_path}")
    detect_fn = tf.saved_model.load(str(args.model_path))
    signature = detect_fn.signatures["serving_default"]
    logger.info(f"签名输入: {list(signature.structured_input_signature[1].keys())}")

    # 读 TFRecord 原始字节（不走 build_eval_dataset，因为我们要原图）
    logger.info(f"读 TFRecord: {config.VAL_RECORD}")
    raw_ds = tf.data.TFRecordDataset([str(config.VAL_RECORD)])
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
        logger.info(f"已加载 per-class 阈值")

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

        # 解码成 float32 [0, 1]
        img = tf.io.decode_jpeg(image_bytes, channels=3)
        image = tf.cast(img, tf.float32).numpy() / 255.0  # (H, W, 3)

        # GT (TFRecord 存的是**归一化** 0-1 坐标，必须 × (orig_w, orig_h) 转像素)
        # 旧版 bug: 直接当像素坐标用 → GT 退化为 (0,0,0,0) 或 (1,1,1,1)，跟 100+ 像素级预测完全错位，mAP=0
        xmin = tf.sparse.to_dense(parsed["image/object/bbox/xmin"]).numpy()
        xmax = tf.sparse.to_dense(parsed["image/object/bbox/xmax"]).numpy()
        ymin = tf.sparse.to_dense(parsed["image/object/bbox/ymin"]).numpy()
        ymax = tf.sparse.to_dense(parsed["image/object/bbox/ymax"]).numpy()
        gt_classes_raw = tf.sparse.to_dense(parsed["image/object/class/label"]).numpy().astype(np.int32)
        # valid
        gt_w_norm = xmax - xmin
        gt_h_norm = ymax - ymin
        valid = (gt_w_norm > 0) & (gt_h_norm > 0) & (gt_classes_raw > 0)
        # 归一化 → 原图像素坐标
        gt_boxes_xyxy = np.stack([
            xmin[valid] * width,
            ymin[valid] * height,
            xmax[valid] * width,
            ymax[valid] * height,
        ], axis=-1).astype(np.float32)
        gt_cids = gt_classes_raw[valid] - 1  # 1-based → 0-based

        # TTA: 每个尺度 + 可选 flip → 推理
        boxes_list, scores_list, cids_list = [], [], []
        for scale_size in args.tta_scales:
            xyxy, sc, cids = detect_at_scale(detect_fn, image, scale_size, flip=False)
            boxes_list.append(xyxy)
            scores_list.append(sc)
            cids_list.append(cids)
            n_tta += 1
            if args.tta_with_flip:
                xyxy, sc, cids = detect_at_scale(detect_fn, image, scale_size, flip=True)
                boxes_list.append(xyxy)
                scores_list.append(sc)
                cids_list.append(cids)
                n_tta += 1

        # 融合多源 box
        if args.tta_fusion == "wbf":
            fused_boxes, fused_scores, fused_cids = wbf_fuse(
                boxes_list, scores_list, cids_list,
                image_h=height, image_w=width,
                iou_thresh=args.wbf_iou_thr,
                skip_box_thresh=args.score_thresh,
            )
        elif args.tta_fusion == "nms":
            # NMS-concat: 拼接所有 box + class-aware NMS
            all_b = np.concatenate(boxes_list, axis=0) if boxes_list else np.zeros((0, 4), dtype=np.float32)
            all_s = np.concatenate(scores_list, axis=0) if scores_list else np.zeros((0,), dtype=np.float32)
            all_c = np.concatenate(cids_list, axis=0) if cids_list else np.zeros((0,), dtype=np.int32)
            # 预过滤低分
            keep = all_s >= args.score_thresh
            all_b, all_s, all_c = all_b[keep], all_s[keep], all_c[keep]
            if len(all_s) > 0:
                fused_boxes, fused_scores, fused_cids = batched_nms_per_class(
                    all_b, all_s, all_c,
                    iou_threshold=args.nms_iou_thr,
                    max_output_size=200,
                )
            else:
                fused_boxes, fused_scores, fused_cids = np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.int32)
        else:  # none: 只用第 1 个 source（baseline 对照）
            fused_boxes = boxes_list[0] if boxes_list else np.zeros((0, 4), dtype=np.float32)
            fused_scores = scores_list[0] if scores_list else np.zeros((0,), dtype=np.float32)
            fused_cids = cids_list[0] if cids_list else np.zeros((0,), dtype=np.int32)

        # 过滤 padding 类
        valid_mask = (fused_cids >= 0) & (fused_cids < 80)
        fused_boxes = fused_boxes[valid_mask]
        fused_scores = fused_scores[valid_mask]
        fused_cids = fused_cids[valid_mask]

        # 后处理
        if per_class_thresh is not None and len(fused_scores) > 0:
            fused_boxes, fused_scores, fused_cids = apply_per_class_thresh(
                fused_boxes, fused_scores, fused_cids, per_class_thresh, default_thresh=args.score_thresh
            )
        if args.use_soft_nms and len(fused_scores) > 0:
            fused_boxes, fused_scores, _keep = soft_nms(
                fused_boxes, fused_scores, sigma=0.5, score_thresh=args.score_thresh
            )
            # _keep 是绝对索引（原始 N 长度数组的子集），直接用它过滤 cids
            # 旧逻辑用 if len(_keep) == len(fused_cids) 判断，永远 false，导致 cids 不被同步
            # → dets["boxes"] 25 个 vs cids 48 个，compute_map 报 IndexError
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
    logger.info(f"=" * 50)
    logger.info(f"共评估 {n_imgs} 张图, {n_tta} 次推理")
    logger.info(f"实际评估 {n_eval} 张, 耗时 {elapsed:.1f}s, FPS={n_eval/elapsed:.2f}" if n_eval > 0 else f"全部 warmup: {n_imgs} 张, {elapsed:.1f}s")

    mAP_50, aps_50 = compute_map(all_dets, all_gts, num_classes=80, iou_thresh=0.5)
    logger.info(f"mAP@0.5:     {mAP_50:.4f}")
    iou_thresholds = np.arange(0.5, 1.0, 0.05)
    aps_all = []
    for iou_thr in iou_thresholds:
        mAP, _ = compute_map(all_dets, all_gts, num_classes=80, iou_thresh=iou_thr)
        aps_all.append(mAP)
    mAP_50_95 = float(np.mean(aps_all))
    logger.info(f"mAP@0.5:0.95: {mAP_50_95:.4f}")

    # Top 20 AP
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
    logger.info(f"我们的 TTA 评估 ({n_imgs} 张): mAP@0.5: {mAP_50:.4f} / mAP@0.5:0.95: {mAP_50_95:.4f}")


if __name__ == "__main__":
    main()
