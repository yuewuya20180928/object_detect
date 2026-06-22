#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
USB 摄像头实时目标检测 - Headless 版本

服务器没 X display？用这个。
- 不弹窗口，把带检测框的画面写到 MP4 视频文件
- 实时打印到 stdout（webchat 也能看到）
- 定期抓截图保存到 PNG（方便快速预览）
- 支持 RTSP/HLS 推流（可选）

用法:
  python demo_usb_camera_headless.py --score-thresh 0.1
  python demo_usb_camera_headless.py --output /tmp/detect.mp4 --duration 30
  python demo_usb_camera_headless.py --rtsp 8554  # 推 RTSP 流
"""

import os
import sys
import time
import argparse
import signal
import threading
from pathlib import Path
from collections import deque, Counter

PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
# 关键：headless 模式禁用 Qt
os.environ.setdefault("OPENCV_VIDEOIO_PRIORITY_MSMF", "0")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import mixed_precision

mixed_precision.set_global_policy("mixed_float16")

import config  # noqa: E402
from models.detector import DetectionModel  # noqa: E402
from models.postprocess import decode_predictions  # noqa: E402


# COCO 80 类名
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

# HSV 离散色彩（黄金角）
np.random.seed(42)
CLASS_COLORS = np.zeros((80, 3), dtype=np.uint8)
for i in range(80):
    h = (i * 137) % 180
    s = 200 + (i % 3) * 18
    v = 220 + (i % 2) * 35
    CLASS_COLORS[i] = cv2.cvtColor(np.array([[[h, s, v]]], dtype=np.uint8), cv2.COLOR_HSV2BGR)[0, 0]


def letterbox(image, input_size):
    h, w = image.shape[:2]
    scale = input_size / max(h, w)
    new_h = int(round(h * scale))
    new_w = int(round(w * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_y = (input_size - new_h) // 2
    pad_x = (input_size - new_w) // 2
    padded = np.full((input_size, input_size, 3), 0, dtype=np.uint8)
    padded[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    return padded, scale, pad_x, pad_y


def unletterbox_boxes(boxes_320, scale, pad_x, pad_y):
    if len(boxes_320) == 0:
        return boxes_320
    out = boxes_320.astype(np.float32).copy()
    out[:, 0] = (boxes_320[:, 0] - pad_x) / scale
    out[:, 1] = (boxes_320[:, 1] - pad_y) / scale
    out[:, 2] = (boxes_320[:, 2] - pad_x) / scale
    out[:, 3] = (boxes_320[:, 3] - pad_y) / scale
    return out


def draw_boxes(image, boxes, scores, class_ids, fps=None, n_det=None, threshold=None):
    """在 BGR 图像上画检测框 + 状态条

    2026-06-22 patch (Patch C)：标签背景 clamp 到画面内
      - 横向贴边往内缩（避免 x1≈0 时贴死左边缘）
      - 纵向顶部放不下挪到框内（避免 y1<th+6 时画到画面外）"""
    H, W = image.shape[:2]
    for box, score, cid in zip(boxes, scores, class_ids):
        x1, y1, x2, y2 = box.astype(int)
        x1 = max(0, min(W - 1, x1))
        y1 = max(0, min(H - 1, y1))
        x2 = max(0, min(W - 1, x2))
        y2 = max(0, min(H - 1, y2))
        if x2 <= x1 or y2 <= y1:
            continue
        color = tuple(int(c) for c in CLASS_COLORS[cid % 80])
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        name = COCO_NAMES[cid] if cid < len(COCO_NAMES) else f"cls{cid}"
        label = f"{name} {score:.0%}"
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        # 标签背景 clamp
        lbl_x0 = max(2, min(x1, W - tw - 6))
        lbl_x1 = lbl_x0 + tw + 4
        lbl_y1 = y1
        lbl_y0 = lbl_y1 - th - 6
        if lbl_y0 < 2:
            lbl_y0 = y1 + 2
            lbl_y1 = lbl_y0 + th + 6
        cv2.rectangle(image, (lbl_x0, lbl_y0), (lbl_x1, lbl_y1), color, -1)
        cv2.putText(image, label, (lbl_x0 + 2, lbl_y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    # 状态条
    if fps is not None:
        info = f"FPS: {fps:.1f}  Det: {n_det or 0}  Thr: {threshold or 0:.2f}"
        cv2.putText(image, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    return image


def decode_outputs(outputs, anchors, input_size, score_thresh, nms_iou, num_classes):
    raw_boxes_list, raw_scores_list = [], []
    for level in sorted(outputs.keys()):
        if not level.startswith("cls_"):
            continue
        box_key = level.replace("cls_", "box_")
        cls = np.asarray(outputs[level][0])
        box = np.asarray(outputs[box_key][0])
        H, W = cls.shape[:2]
        cls = cls.reshape(H, W, -1, num_classes + 1).reshape(-1, num_classes + 1)
        box = box.reshape(H, W, -1, 4).reshape(-1, 4)
        raw_boxes_list.append(box)
        raw_scores_list.append(cls)
    raw_boxes = np.concatenate(raw_boxes_list, axis=0)
    raw_scores = np.concatenate(raw_scores_list, axis=0)
    result = decode_predictions(
        raw_boxes, raw_scores, anchors,
        image_shape=(input_size, input_size),
        input_size=input_size,
        score_thresh=score_thresh,
        nms_iou_thresh=nms_iou,
        num_classes=num_classes,
    )
    return result["boxes"], result["scores"], result["class_ids"]


def main():
    parser = argparse.ArgumentParser(description="USB 摄像头实时检测 headless 版")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--score-thresh", type=float, default=0.2)
    parser.add_argument("--nms-iou", type=float, default=0.5)
    parser.add_argument("--skip-frames", type=int, default=0)
    parser.add_argument("--input-size", type=int, default=config.INPUT_SIZE)
    parser.add_argument("--weights", type=str, default=None)
    parser.add_argument("--output", type=str, default=None,
                        help="输出 MP4 路径（默认 ./detections/run_YYYYMMDD_HHMMSS.mp4）")
    parser.add_argument("--duration", type=int, default=None,
                        help="运行时长（秒），默认无限")
    parser.add_argument("--screenshot-every", type=int, default=0,
                        help="每隔 N 帧保存 PNG 截图（0=不存）")
    args = parser.parse_args()

    weights = args.weights or str(config.CHECKPOINT_DIR / "best.weights.h5")
    if not Path(weights).exists():
        print(f"[ERROR] 权重不存在: {weights}")
        sys.exit(1)

    # 输出路径
    out_dir = Path("./detections")
    out_dir.mkdir(exist_ok=True)
    if args.output:
        video_path = Path(args.output)
    else:
        ts = time.strftime("%Y%m%d_%H%M%S")
        video_path = out_dir / f"run_{ts}.mp4"

    # 探测摄像头
    available = []
    for i in range(4):
        cap_test = cv2.VideoCapture(i)
        if cap_test.isOpened() and cap_test.get(cv2.CAP_PROP_FRAME_WIDTH) > 0:
            available.append(i)
            cap_test.release()
    if not available:
        print("[ERROR] 没有可用的摄像头")
        sys.exit(1)
    if args.camera not in available:
        print(f"[WARN] /dev/video{args.camera} 不可用，自动选 /dev/video{available[0]}")
        args.camera = available[0]

    # 加载模型
    print(f"[INFO] 加载模型: {weights}")
    det = DetectionModel(input_size=args.input_size)
    det.load_weights(weights)
    print(f"[INFO] anchors={det.num_anchors}, input={args.input_size}")

    # 打开摄像头
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_cap = cap.get(cv2.CAP_PROP_FPS) or 30
    print(f"[INFO] 摄像头 {args.camera}: {actual_w}x{actual_h} @ {fps_cap}fps")

    # 准备视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(video_path), fourcc, fps_cap, (actual_w, actual_h))
    if not writer.isOpened():
        print(f"[WARN] mp4v 失败，试试 XVID")
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_path = video_path.with_suffix('.avi')
        writer = cv2.VideoWriter(str(video_path), fourcc, fps_cap, (actual_w, actual_h))
    print(f"[INFO] 视频输出: {video_path.resolve()}")

    # 模型 warmup
    print("[INFO] 模型 warmup (XLA compile)...")
    t0 = time.perf_counter()
    dummy = np.zeros((1, args.input_size, args.input_size, 3), dtype=np.float32)
    _ = det.predict(dummy, verbose=0)
    print(f"[INFO] warmup 完成: {(time.perf_counter() - t0) * 1000:.0f}ms")

    # Ctrl+C 优雅退出
    running = True
    def handle_sigint(sig, frame):
        nonlocal running
        running = False
    signal.signal(signal.SIGINT, handle_sigint)
    signal.signal(signal.SIGTERM, handle_sigint)

    print()
    print("=" * 60)
    print(f"运行中...  score_thresh={args.score_thresh}")
    print("实时统计每 30 帧打印一次 (Ctrl+C 退出)")
    print("=" * 60)
    print()

    fps_history = deque(maxlen=30)
    last_result = None
    class_counter = Counter()  # 累计检测的类别
    total_det = 0
    frame_idx = 0
    t_start = time.perf_counter()
    t_last_log = t_start

    try:
        while running:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.05)
                continue

            t0 = time.perf_counter()

            # 跑检测
            if frame_idx % (args.skip_frames + 1) == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                padded, scale, pad_x, pad_y = letterbox(rgb, args.input_size)
                input_tensor = padded.astype(np.float32) / 255.0
                input_tensor = np.expand_dims(input_tensor, axis=0)
                outputs = det.predict(input_tensor, verbose=0)
                boxes_320, scores, class_ids = decode_outputs(
                    outputs, det.anchors, args.input_size,
                    args.score_thresh, args.nms_iou, config.NUM_CLASSES
                )
                boxes_orig = unletterbox_boxes(boxes_320, scale, pad_x, pad_y)
                last_result = (boxes_orig, scores, class_ids)
                # 累计统计
                for cid, sc in zip(class_ids, scores):
                    if sc >= args.score_thresh:
                        class_counter[cid] += 1
                        total_det += 1

            # 画框
            display_frame = frame
            if last_result is not None:
                boxes, scores, class_ids = last_result
                # 状态条用最近 30 帧平均 FPS
                avg_fps = sum(fps_history) / len(fps_history) if fps_history else 0
                display_frame = draw_boxes(
                    frame.copy(), boxes, scores, class_ids,
                    fps=avg_fps, n_det=len(boxes), threshold=args.score_thresh
                )

            # 写视频
            writer.write(display_frame)

            # 截图
            if args.screenshot_every > 0 and frame_idx % args.screenshot_every == 0:
                ss_path = out_dir / f"frame_{frame_idx:07d}.jpg"
                cv2.imwrite(str(ss_path), display_frame)

            t1 = time.perf_counter()
            fps_history.append(1.0 / max(t1 - t0, 1e-6))

            # 30 帧 / 5 秒打印一次
            if frame_idx % 30 == 0:
                t_now = time.perf_counter()
                if t_now - t_last_log > 1.0:
                    avg_fps = sum(fps_history) / len(fps_history)
                    elapsed = t_now - t_start
                    top5 = class_counter.most_common(5)
                    top5_str = ", ".join(
                        f"{COCO_NAMES[c]}({n})" for c, n in top5
                    ) if top5 else "(none)"
                    print(
                        f"[F{frame_idx:06d} t={elapsed:6.1f}s] "
                        f"FPS={avg_fps:5.1f}  Det={len(last_result[0]) if last_result else 0:2d}  "
                        f"TotalDet={total_det:5d}  Top5: {top5_str}"
                    )
                    t_last_log = t_now

            frame_idx += 1
            if args.duration and (time.perf_counter() - t_start) >= args.duration:
                print(f"\n[INFO] 已达时长 {args.duration}s，自动停止")
                break

    finally:
        cap.release()
        writer.release()
        elapsed = time.perf_counter() - t_start
        avg_fps = frame_idx / elapsed if elapsed > 0 else 0
        print()
        print("=" * 60)
        print(f"运行结束 | {frame_idx} 帧 | {elapsed:.1f}s | 平均 {avg_fps:.1f} fps")
        print(f"视频保存: {video_path.resolve()}")
        print(f"总检测数: {total_det}")
        print("Top 10 类别:")
        for cid, n in class_counter.most_common(10):
            print(f"  {COCO_NAMES[cid]:20s} {n:5d}")
        print("=" * 60)


if __name__ == "__main__":
    main()
