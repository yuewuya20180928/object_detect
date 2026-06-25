#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对照实验：用 COCO pycocotools 标准评估工具算 mAP
目的：诊断 SSD/D1 在本项目数据上的 mAP 偏低是 letterbox 问题还是 mAP 算法问题

对比：
  A. dataset_builder letterbox (int padding, 0 填充) → 用 evaluate_odapi.py 我们的 compute_map
  B. cv2 letterbox (gray padding 114) → 用本脚本 + pycocotools
  C. tf.image.resize_with_pad (OD API 训练时的标准 letterbox) → 用本脚本 + pycocotools

如果 B 或 C 接近 paper 数字(SSD 0.222 / D1 0.384)→ 是 letterbox 问题
如果 B/C 也都是 0.06 → 是 mAP 计算代码问题
如果三者接近 → letterbox + mAP 都没问题,可能是 OD API saved_model 加载问题
"""
import os
import sys
import json
import argparse
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import numpy as np
import tensorflow as tf
import cv2
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

ODAPI_MODEL_DEFAULT = PROJECT_ROOT / "pretrained/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/saved_model"
COCO_ANN = PROJECT_ROOT / "data/coco/raw/annotations/instances_val2017.json"
COCO_VAL_DIR = PROJECT_ROOT / "data/coco/raw/val2017"


def letterbox_cv2(img, target_size):
    """OpenCV-style letterbox（gray padding 114，跟 TF OD API 训练一致）"""
    h, w = img.shape[:2]
    scale = target_size / max(h, w)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    pad_x = (target_size - new_w) // 2
    pad_y = (target_size - new_h) // 2
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
    canvas[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized
    return canvas, scale, pad_x, pad_y


def letterbox_tf(image_uint8, target_size):
    """tf.image.resize_with_pad — TF OD API 训练时用的 letterbox"""
    img_tf = tf.constant(image_uint8)
    padded = tf.image.resize_with_pad(img_tf, target_size, target_size, method="bilinear")
    padded = tf.cast(padded, tf.uint8).numpy()
    # 计算 scale 和 pad（用跟 dataset_builder 一致的公式）
    h, w = image_uint8.shape[:2]
    scale = target_size / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    pad_x = (target_size - new_w) // 2
    pad_y = (target_size - new_h) // 2
    return padded, scale, pad_x, pad_y


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", type=str, default=str(ODAPI_MODEL_DEFAULT))
    p.add_argument("--input-size", type=int, default=320)
    p.add_argument("--max-imgs", type=int, default=200)
    p.add_argument("--letterbox", type=str, default="cv2", choices=["cv2", "tf", "zero"],
                   help="cv2=gray padding 114, tf=tf.image.resize_with_pad, zero=0 padding (dataset_builder 风格)")
    args = p.parse_args()

    print(f"模型: {args.model_path}")
    print(f"输入: {args.input_size}")
    print(f"Letterbox: {args.letterbox}")
    print(f"张数: {args.max_imgs}")

    # 加载 COCO annotations
    print(f"加载 annotations: {COCO_ANN}")
    coco_gt = COCO(str(COCO_ANN))
    img_ids = coco_gt.getImgIds()[:args.max_imgs]
    img_id_to_ann = {}
    for img_id in img_ids:
        ann_ids = coco_gt.getAnnIds(imgIds=[img_id], iscrowd=False)
        img_id_to_ann[img_id] = coco_gt.loadAnns(ann_ids)
    img_infos = coco_gt.loadImgs(img_ids)
    print(f"GT loaded: {len(img_ids)} 张图")

    # 加载 OD API
    print(f"加载 OD API SavedModel: {args.model_path}")
    detect_fn = tf.saved_model.load(str(args.model_path))
    signature = detect_fn.signatures["serving_default"]

    # 推理 + 收集 det（COCO 格式）
    detections = []
    t0 = time.time()
    for i, img_info in enumerate(img_infos):
        img_path = COCO_VAL_DIR / img_info["file_name"]
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[WARN] 读图失败: {img_path}")
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if args.letterbox == "cv2":
            letterboxed, scale, pad_x, pad_y = letterbox_cv2(img_rgb, args.input_size)
        elif args.letterbox == "tf":
            letterboxed, scale, pad_x, pad_y = letterbox_tf(img_rgb, args.input_size)
        else:  # zero padding (dataset_builder 风格)
            h, w = img_rgb.shape[:2]
            scale = args.input_size / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            pad_x = (args.input_size - new_w) // 2
            pad_y = (args.input_size - new_h) // 2
            resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            letterboxed = np.zeros((args.input_size, args.input_size, 3), dtype=np.uint8)
            letterboxed[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized

        input_tensor = tf.constant(letterboxed[None], dtype=tf.uint8)
        out = signature(input_tensor)
        boxes = out["detection_boxes"][0].numpy()  # (100, 4) ymin xmin ymax xmax 归一化
        scores = out["detection_scores"][0].numpy()
        classes = out["detection_classes"][0].numpy().astype(np.int32)
        num = int(out["num_detections"][0].numpy())

        # 过滤 padding 类
        valid = (classes[:num] >= 1) & (classes[:num] <= 80)
        boxes = boxes[:num][valid]
        scores = scores[:num][valid]
        classes = classes[:num][valid] - 1  # 1-based → 0-based (COCO catIds 1-90)
        num = len(scores)

        # box 转原图 xyxy 像素
        orig_h, orig_w = img_info["height"], img_info["width"]
        xyxy = np.zeros((num, 4), dtype=np.float32)
        xyxy[:, 0] = boxes[:, 1] * args.input_size  # x1 = xmin
        xyxy[:, 1] = boxes[:, 0] * args.input_size  # y1 = ymin
        xyxy[:, 2] = boxes[:, 3] * args.input_size  # x2 = xmax
        xyxy[:, 3] = boxes[:, 2] * args.input_size  # y2 = ymax
        # letterbox 像素 → 原图像素
        xyxy[:, [0, 2]] -= pad_x
        xyxy[:, [1, 3]] -= pad_y
        xyxy /= scale
        xyxy[:, [0, 2]] = np.clip(xyxy[:, [0, 2]], 0, orig_w)
        xyxy[:, [1, 3]] = np.clip(xyxy[:, [1, 3]], 0, orig_h)

        for j in range(num):
            if scores[j] < 0.001:
                continue
            x1, y1, x2, y2 = xyxy[j].tolist()
            detections.append({
                "image_id": img_info["id"],
                "category_id": int(classes[j] + 1),  # 0-based → 1-based COCO cat_id
                "bbox": [x1, y1, x2 - x1, y2 - y1],  # xywh
                "score": float(scores[j]),
            })

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"  已处理 {i+1}/{len(img_infos)} 张, 耗时 {elapsed:.1f}s")

    print(f"\n共收集 {len(detections)} 个 det")

    # pycocotools COCOeval
    print("\n[pycocotools COCOeval]")
    if len(detections) == 0:
        print("没有 det,无法评估")
        return
    with open("/tmp/detections.json", "w") as f:
        json.dump(detections, f)
    coco_dt = coco_gt.loadRes("/tmp/detections.json")
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.params.imgIds = img_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # 提取关键数字
    stats = coco_eval.stats
    print(f"\n关键指标:")
    print(f"  mAP@0.5:0.95: {stats[0]:.4f}")
    print(f"  mAP@0.5:     {stats[1]:.4f}")
    print(f"  mAP@0.75:    {stats[2]:.4f}")
    print(f"  mAP small:   {stats[3]:.4f}")
    print(f"  mAP medium:  {stats[4]:.4f}")
    print(f"  mAP large:   {stats[5]:.4f}")


if __name__ == "__main__":
    main()