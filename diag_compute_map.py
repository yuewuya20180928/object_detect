#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
诊断：build_eval_dataset 数据源 vs cv2.imread 数据源 对 mAP 影响
两套都跑 pycocotools 算 mAP,排除 compute_map 算法 bug

A. 数据源 = build_eval_dataset (TFRecord decode + dataset_builder letterbox)
   用 evaluate_odapi.py 流程,但 det 收集成 COCO 格式,pycocotools 算 mAP
B. 数据源 = cv2.imread + cv2 letterbox
   evaluate_pycocotools.py 已经跑过,作为对照
"""
import os
import sys
import json
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
import config
from data.dataset_builder import build_eval_dataset

ODAPI_MODEL = PROJECT_ROOT / "pretrained/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/saved_model"
COCO_ANN = PROJECT_ROOT / "data/coco/raw/annotations/instances_val2017.json"
COCO_VAL_DIR = PROJECT_ROOT / "data/coco/raw/val2017"


def main():
    max_imgs = 200
    input_size = config.INPUT_SIZE  # 320

    # 加载 COCO GT
    coco_gt = COCO(str(COCO_ANN))
    # val.record 的 img_id 跟原图的 img_id 是 COCO 原生 id
    # 但 TFRecord 里没存 image_id,只有文件名(filename / key sha256)
    # 先拿前 N 张原图,然后用文件名匹配
    img_infos = coco_gt.loadImgs(coco_gt.getImgIds()[:max_imgs])

    # 加载 OD API
    print(f"加载 OD API: {ODAPI_MODEL}")
    detect_fn = tf.saved_model.load(str(ODAPI_MODEL))
    signature = detect_fn.signatures["serving_default"]

    # 加载 val.record (build_eval_dataset 走 dataset_builder 流程)
    print("加载 val.record (走 dataset_builder pipeline)")
    val_ds = build_eval_dataset(
        tfrecord_path=config.VAL_RECORD,
        batch_size=1,
        input_size=input_size,
    )

    # 用 filename 匹配 TFRecord 跟 COCO 原图
    coco_by_filename = {info["file_name"]: info for info in img_infos}

    detections = []
    n_imgs = 0
    t0 = time.time()
    for images, labels in val_ds:
        if n_imgs >= max_imgs:
            break
        # 拿到 image 后,送 OD API
        input_tensor = tf.cast(images * 255.0, tf.uint8)
        out = signature(input_tensor)
        boxes = out["detection_boxes"][0].numpy()
        scores = out["detection_scores"][0].numpy()
        classes = out["detection_classes"][0].numpy().astype(np.int32)
        num = int(out["num_detections"][0].numpy())

        # 过滤 padding 类
        valid = (classes[:num] >= 1) & (classes[:num] <= 80)
        boxes = boxes[:num][valid]
        scores = scores[:num][valid]
        classes = classes[:num][valid] - 1
        num = len(scores)

        # GT 转 letterbox 像素（用 dataset_builder 一致的 int padding）
        gt_boxes_raw = labels["boxes"][0].numpy()
        gt_classes = labels["classes"][0].numpy()
        valid_gt = (gt_boxes_raw[..., 2] > 0) & (gt_boxes_raw[..., 3] > 0) & (gt_classes > 0)
        gt_boxes_raw = gt_boxes_raw[valid_gt]
        gt_classes = gt_classes[valid_gt] - 1  # 1-based → 0-based
        orig_h, orig_w = labels["original_shape"][0].numpy()
        scale = input_size / max(orig_h, orig_w)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        pad_x = (input_size - new_w) // 2
        pad_y = (input_size - new_h) // 2

        # box 转原图 xyxy 像素
        xyxy = np.zeros((num, 4), dtype=np.float32)
        xyxy[:, 0] = boxes[:, 1] * input_size
        xyxy[:, 1] = boxes[:, 0] * input_size
        xyxy[:, 2] = boxes[:, 3] * input_size
        xyxy[:, 3] = boxes[:, 2] * input_size
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
                # 用 COCO 原生 image_id（这里 dataset 没直接给,先用 0 占位）
                "image_id": n_imgs,  # 占位,后面用 idx 映射
                "category_id": int(classes[j] + 1),
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "score": float(scores[j]),
            })

        n_imgs += 1
        if n_imgs % 50 == 0:
            print(f"  已处理 {n_imgs}/{max_imgs}, 耗时 {time.time()-t0:.1f}s")

    print(f"\n收集 {len(detections)} 个 det ({n_imgs} 张图)")
    print("\n注意:build_eval_dataset 走 TFRecord,跟 COCO 原图不是 1-1 对应（顺序可能不同）")
    print("这个实验不能精确映射到 COCO img_id,只能定性看 mAP 量级")


if __name__ == "__main__":
    main()