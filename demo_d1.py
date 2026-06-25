#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
D1 / SSD 单图推理 + FPS 测试

用法:
    # 默认 SSD 320
    python3 demo_d1.py --image data/coco/raw/val2017/000000308631.jpg

    # 切 D1 640
    python3 demo_d1.py \
        --model-path pretrained/efficientdet_d1_coco17_tpu-32/saved_model \
        --input-size 640 \
        --image data/coco/raw/val2017/000000308631.jpg

    # 跑 100 次测 FPS
    python3 demo_d1.py \
        --model-path pretrained/efficientdet_d1_coco17_tpu-32/saved_model \
        --input-size 640 \
        --fps-test 100
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
import cv2

# COCO 80 类（1-based id 1-80；0 是背景）
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

SSD_MODEL = PROJECT_ROOT / "pretrained/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/saved_model"
D1_MODEL = PROJECT_ROOT / "pretrained/efficientdet_d1_coco17_tpu-32/saved_model"


def letterbox(img, target_size):
    """保持宽高比 letterbox 到 target_size x target_size，0 填充"""
    h, w = img.shape[:2]
    scale = target_size / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    pad_x = (target_size - new_w) // 2
    pad_y = (target_size - new_h) // 2
    img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    canvas[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = img_resized
    return canvas, scale, pad_x, pad_y


def unletterbox_boxes(xyxy, scale, pad_x, pad_y, orig_w, orig_h):
    """letterbox 像素坐标 → 原图像素坐标"""
    out = xyxy.copy().astype(np.float32)
    out[:, [0, 2]] -= pad_x
    out[:, [1, 3]] -= pad_y
    out /= scale
    out[:, [0, 2]] = np.clip(out[:, [0, 2]], 0, orig_w)
    out[:, [1, 3]] = np.clip(out[:, [1, 3]], 0, orig_h)
    return out


def detect(detect_fn, image_bgr, input_size, score_thresh=0.5):
    """单图推理，返回 xyxy in 原图坐标"""
    orig_h, orig_w = image_bgr.shape[:2]
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    letterboxed, scale, pad_x, pad_y = letterbox(img_rgb, input_size)
    input_tensor = tf.constant(letterboxed[None], dtype=tf.uint8)
    detections = detect_fn(input_tensor)
    boxes = detections["detection_boxes"][0].numpy()
    scores = detections["detection_scores"][0].numpy()
    classes = detections["detection_classes"][0].numpy().astype(np.int32)
    num = int(detections["num_detections"][0].numpy())
    # letterbox 像素 → 原图像素
    xyxy = np.zeros((num, 4), dtype=np.float32)
    xyxy[:, 0] = boxes[:num, 1] * input_size
    xyxy[:, 1] = boxes[:num, 0] * input_size
    xyxy[:, 2] = boxes[:num, 3] * input_size
    xyxy[:, 3] = boxes[:num, 2] * input_size
    xyxy = unletterbox_boxes(xyxy, scale, pad_x, pad_y, orig_w, orig_h)
    # score 过滤
    keep = scores[:num] >= score_thresh
    xyxy = xyxy[keep]
    scores = scores[:num][keep]
    classes = classes[:num][keep]
    return xyxy, scores, classes


def draw_dets(image_bgr, xyxy, scores, classes):
    img = image_bgr.copy()
    for (x1, y1, x2, y2), s, c in zip(xyxy, scores, classes):
        name = COCO_NAMES[c-1] if 1 <= c <= 80 else f"cls{c}"
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        label = f"{name} {s:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (int(x1), int(y1) - th - 4), (int(x1) + tw, int(y1)), (0, 255, 0), -1)
        cv2.putText(img, label, (int(x1), int(y1) - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return img


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", type=str, default=str(SSD_MODEL),
                   help=f"OD API saved_model 路径（默认 SSD，可换 D1: {D1_MODEL}）")
    p.add_argument("--input-size", type=int, default=320, help="输入尺寸")
    p.add_argument("--image", type=str, default=None, help="单张图推理并保存可视化")
    p.add_argument("--score-thresh", type=float, default=0.5)
    p.add_argument("--fps-test", type=int, default=0, help="跑 N 次推理测 FPS（不保存可视化）")
    p.add_argument("--fps-warmup", type=int, default=10, help="FPS 测试前的 warmup 次数")
    p.add_argument("--save", type=str, default=None, help="可视化结果保存路径")
    args = p.parse_args()

    print(f"[INFO] 模型: {args.model_path}")
    print(f"[INFO] 输入尺寸: {args.input_size}")
    print(f"[INFO] score_thresh: {args.score_thresh}")

    detect_fn = tf.saved_model.load(str(args.model_path))
    signature = detect_fn.signatures["serving_default"]
    print(f"[INFO] SavedModel 加载完成 | 输入: input_tensor (uint8, {args.input_size}x{args.input_size}x3)")

    # FPS 测试
    if args.fps_test > 0:
        dummy = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        # warmup
        for _ in range(args.fps_warmup):
            _ = detect(detect_fn, dummy, args.input_size, args.score_thresh)
        # 正式测
        t0 = time.time()
        for _ in range(args.fps_test):
            _ = detect(detect_fn, dummy, args.input_size, args.score_thresh)
        elapsed = time.time() - t0
        fps = args.fps_test / elapsed
        ms_per = elapsed / args.fps_test * 1000
        print(f"\n[FPS] 模型={Path(args.model_path).name}, input={args.input_size}")
        print(f"[FPS] {args.fps_test} 次推理, 总耗时 {elapsed:.2f}s")
        print(f"[FPS] 单次推理 {ms_per:.1f} ms, FPS = {fps:.1f}")
        return

    # 单图 demo
    if args.image:
        img = cv2.imread(args.image)
        if img is None:
            print(f"[ERROR] 读图失败: {args.image}")
            sys.exit(1)
        print(f"[INFO] 输入图: {args.image} ({img.shape[1]}x{img.shape[0]})")
        xyxy, scores, classes = detect(detect_fn, img, args.input_size, args.score_thresh)
        print(f"\n[DETECT] 共检出 {len(scores)} 个目标:")
        for (x1, y1, x2, y2), s, c in zip(xyxy[:10], scores[:10], classes[:10]):
            name = COCO_NAMES[c-1] if 1 <= c <= 80 else f"cls{c}"
            print(f"  {name:20s} score={s:.3f} box=[{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}]")
        if len(scores) > 10:
            print(f"  ... 还有 {len(scores)-10} 个")
        if args.save:
            vis = draw_dets(img, xyxy, scores, classes)
            cv2.imwrite(args.save, vis)
            print(f"\n[SAVE] 可视化已保存: {args.save}")
        return

    print("[ERROR] 至少指定 --image 或 --fps-test")
    sys.exit(1)


if __name__ == "__main__":
    main()