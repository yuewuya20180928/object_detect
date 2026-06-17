#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
USB 摄像头实时目标检测

模型: SSD MobileNetV2 FPNLite 320x320 (COCO 80 类)
输入: USB 摄像头（默认 /dev/video0，可用 --camera 切换）
显示: OpenCV 窗口实时显示带检测框的画面

按键:
  q       - 退出
  s       - 保存当前帧到 ./detections/
  +/-     - 调整 score 阈值（步进 0.05）
  d       - 切换是否绘制检测框
  f       - 冻结/解冻当前画面（方便观察）

性能优化:
  - XLA 编译（首次推理会触发 ~3min compile，之后稳态快 20-30%）
  - 混合精度 FP16（跟训练时一致）
  - 帧跳过（--skip-frames，每 N+1 帧跑一次检测）
  - cv2.VideoCapture 用 MJPG 格式 + BUFFERSIZE=1（降低延迟）
  - 反推 box 复用同一份（skip-frames > 0 时）
"""

import os
import sys
import time
import argparse
from pathlib import Path
from collections import deque

PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import mixed_precision

# 跟训练时一致的混合精度策略（权重是 FP16 存盘）
mixed_precision.set_global_policy("mixed_float16")

import config  # noqa: E402
from models.detector import DetectionModel  # noqa: E402
from models.postprocess import decode_predictions  # noqa: E402


# ============================================================================
# COCO 80 类名
# ============================================================================
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

# 每个类固定颜色（HSV 离散，颜色分散易区分）
np.random.seed(42)
CLASS_COLORS = np.zeros((80, 3), dtype=np.uint8)
for i in range(80):
    h = (i * 137) % 180  # 黄金角分布，色相分散
    s = 200 + (i % 3) * 18
    v = 220 + (i % 2) * 35
    hsv = np.array([[[h, s, v]]], dtype=np.uint8)
    CLASS_COLORS[i] = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]


# ============================================================================
# 工具函数
# ============================================================================
def letterbox(image: np.ndarray, input_size: int) -> tuple:
    """保持宽高比缩放到 input_size×input_size（跟 dataset_builder.py 一致用整数除法 pad）

    Returns:
        (padded_image, scale, pad_x, pad_y)
    """
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


def unletterbox_boxes(boxes_320: np.ndarray, scale: float, pad_x: int, pad_y: int) -> np.ndarray:
    """把 320 空间 box 映射回原图坐标空间"""
    if len(boxes_320) == 0:
        return boxes_320
    out = boxes_320.astype(np.float32).copy()
    out[:, 0] = (boxes_320[:, 0] - pad_x) / scale
    out[:, 1] = (boxes_320[:, 1] - pad_y) / scale
    out[:, 2] = (boxes_320[:, 2] - pad_x) / scale
    out[:, 3] = (boxes_320[:, 3] - pad_y) / scale
    return out


def draw_boxes(image: np.ndarray, boxes: np.ndarray, scores: np.ndarray, class_ids: np.ndarray,
               thickness: int = 2) -> np.ndarray:
    """画 bbox + 标签（类名 + 置信度），class 颜色 + 白色文字 + LINE_AA"""
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
        # 框
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        # 标签
        name = COCO_NAMES[cid] if cid < len(COCO_NAMES) else f"cls{cid}"
        label = f"{name} {score:.0%}"
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        # 标签背景（filled，颜色和框一致）
        cv2.rectangle(image, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        # 标签文字（白色抗锯齿）
        cv2.putText(image, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return image


def decode_outputs(outputs: dict, anchors: np.ndarray, input_size: int,
                  score_thresh: float, nms_iou: float, num_classes: int) -> tuple:
    """把模型 raw output → (boxes, scores, class_ids) in 320 像素空间"""
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


def list_cameras(max_check: int = 5) -> list:
    """探测可用的摄像头设备"""
    available = []
    for i in range(max_check):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            available.append((i, w, h))
            cap.release()
    return available


# ============================================================================
# 主函数
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="USB 摄像头实时目标检测（SSD MobileNetV2 FPNLite）")
    parser.add_argument("--camera", type=int, default=0, help="摄像头设备索引（默认 0）")
    parser.add_argument("--list-cameras", action="store_true", help="列出可用摄像头并退出")
    parser.add_argument("--width", type=int, default=1280, help="采集宽度")
    parser.add_argument("--height", type=int, default=720, help="采集高度")
    parser.add_argument("--score-thresh", type=float, default=0.2,
                        help="score 阈值（训练不充分时建议先用 0.1-0.2，按 +/- 调）")
    parser.add_argument("--nms-iou", type=float, default=0.5, help="NMS IoU 阈值")
    parser.add_argument("--skip-frames", type=int, default=0,
                        help="每 N+1 帧跑一次检测（0=每帧；2 = 每 3 帧一次）")
    parser.add_argument("--input-size", type=int, default=config.INPUT_SIZE,
                        help="模型输入尺寸（默认 320，改了需要 retrain 或重新生成 anchors）")
    parser.add_argument("--weights", type=str, default=None,
                        help="权重路径（默认 checkpoints/speed_coco/best.weights.h5）")
    parser.add_argument("--save-dir", type=str, default="./detections",
                        help="按 's' 保存的目录")
    args = parser.parse_args()

    if args.list_cameras:
        print("可用摄像头:")
        for idx, w, h in list_cameras():
            print(f"  /dev/video{idx}  →  {w}x{h}")
        return

    weights = args.weights or str(config.CHECKPOINT_DIR / "best.weights.h5")
    if not Path(weights).exists():
        print(f"[ERROR] 权重不存在: {weights}")
        sys.exit(1)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ===== 加载模型 =====
    print(f"[INFO] 加载模型: {weights}")
    det = DetectionModel(input_size=args.input_size)
    det.load_weights(weights)
    print(f"[INFO] 模型加载完成 | anchors={det.num_anchors} | input={args.input_size}x{args.input_size}")

    # ===== 打开摄像头 =====
    # 先探测所有可用设备，自动选个能用的
    available = []
    for i in range(4):
        cap_test = cv2.VideoCapture(i)
        if cap_test.isOpened() and cap_test.get(cv2.CAP_PROP_FRAME_WIDTH) > 0:
            available.append(i)
            cap_test.release()
    if not available:
        print("[ERROR] 没有可用的摄像头设备")
        print("提示: 用 --list-cameras 看可用设备")
        sys.exit(1)
    if args.camera not in available:
        print(f"[WARN] /dev/video{args.camera} 不可用，自动选 /dev/video{available[0]}")
        args.camera = available[0]

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"[ERROR] 无法打开摄像头 {args.camera}")
        sys.exit(1)

    # 用 MJPG 格式提帧率，buffer=1 降低延迟
    # 注意：有些摄像头不支持改 FOURCC，必须 try/except 避免一错就退出
    try:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    except Exception as e:
        print(f"[WARN] 设 FOURCC 失败，用默认: {e}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # 验证真的打开了
    if not cap.isOpened() or cap.get(cv2.CAP_PROP_FRAME_WIDTH) == 0:
        print(f"[ERROR] 摄像头 {args.camera} 设参后不可用，尝试其他索引")
        cap.release()
        for alt in [0, 1, 2]:
            if alt == args.camera:
                continue
            cap2 = cv2.VideoCapture(alt)
            if cap2.isOpened() and cap2.get(cv2.CAP_PROP_FRAME_WIDTH) > 0:
                print(f"[INFO] 自动切到 /dev/video{alt}")
                cap = cap2
                args.camera = alt
                break
            cap2.release()
        if not cap.isOpened():
            print(f"[ERROR] 所有摄像头都不可用")
            sys.exit(1)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] 摄像头 {args.camera}: {actual_w}x{actual_h}")
    print(f"[INFO] 保存目录: {save_dir.resolve()}")
    print("[INFO] 首次推理会触发 XLA compile (约 2-3 分钟)，之后稳态 20-30ms/帧")
    print()
    print("=== 控制键 ===")
    print("  q  退出        |  s  保存当前帧")
    print("  +/- score阈值 |  d  切换绘制")
    print("  f  冻结画面")
    print()

    # ===== 模型 warmup (触发 XLA compile) =====
    print("[INFO] 模型 warmup 中...")
    t0 = time.perf_counter()
    dummy = np.zeros((1, args.input_size, args.input_size, 3), dtype=np.float32)
    _ = det.predict(dummy, verbose=0)
    print(f"[INFO] Warmup 完成: {(time.perf_counter() - t0) * 1000:.0f}ms")

    # ===== 主循环 =====
    fps_history = deque(maxlen=30)
    last_result = None  # 用于 skip-frames 模式：每 N 帧复用上一次结果
    draw_enabled = True
    score_thresh = args.score_thresh
    frame_idx = 0
    saved_count = 0
    frozen_frame = None  # 冻结时的原始帧

    try:
        while True:
            if frozen_frame is not None:
                frame = frozen_frame.copy()
            else:
                ret, frame = cap.read()
                if not ret:
                    print("[WARN] 读帧失败，重试中...")
                    time.sleep(0.05)
                    continue

            t0 = time.perf_counter()

            # 跑检测（按 skip-frames 决定频率）
            if frame_idx % (args.skip_frames + 1) == 0 and frozen_frame is None:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                padded, scale, pad_x, pad_y = letterbox(rgb, args.input_size)
                input_tensor = padded.astype(np.float32) / 255.0
                input_tensor = np.expand_dims(input_tensor, axis=0)

                outputs = det.predict(input_tensor, verbose=0)
                boxes_320, scores, class_ids = decode_outputs(
                    outputs, det.anchors, args.input_size,
                    score_thresh, args.nms_iou, config.NUM_CLASSES
                )
                boxes_orig = unletterbox_boxes(boxes_320, scale, pad_x, pad_y)
                last_result = (boxes_orig, scores, class_ids)
            elif frame_idx % (args.skip_frames + 1) == 0 and frozen_frame is not None:
                # 冻结模式：用当前冻结帧重新跑一次检测（用最新阈值）
                rgb = cv2.cvtColor(frozen_frame, cv2.COLOR_BGR2RGB)
                padded, scale, pad_x, pad_y = letterbox(rgb, args.input_size)
                input_tensor = padded.astype(np.float32) / 255.0
                input_tensor = np.expand_dims(input_tensor, axis=0)
                outputs = det.predict(input_tensor, verbose=0)
                boxes_320, scores, class_ids = decode_outputs(
                    outputs, det.anchors, args.input_size,
                    score_thresh, args.nms_iou, config.NUM_CLASSES
                )
                boxes_orig = unletterbox_boxes(boxes_320, scale, pad_x, pad_y)
                last_result = (boxes_orig, scores, class_ids)

            # 画框
            display_frame = frame
            if draw_enabled and last_result is not None:
                boxes, scores, class_ids = last_result
                display_frame = draw_boxes(frame.copy(), boxes, scores, class_ids)

            # FPS
            t1 = time.perf_counter()
            fps = 1.0 / max(t1 - t0, 1e-6)
            fps_history.append(fps)
            avg_fps = sum(fps_history) / len(fps_history)

            # 状态条
            n_det = len(last_result[0]) if last_result else 0
            status_lines = [
                f"FPS: {avg_fps:.1f}  |  Det: {n_det}  |  Thr: {score_thresh:.2f}  |  "
                f"Draw: {'ON' if draw_enabled else 'OFF'}  |  {'FROZEN' if frozen_frame is not None else 'LIVE'}",
            ]
            for i, line in enumerate(status_lines):
                cv2.putText(display_frame, line, (10, 30 + i * 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

            cv2.imshow("USB Camera - Object Detection  (q:quit  s:save  +/-:thr  d:draw  f:freeze)",
                       display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                fname = save_dir / f"det_{int(time.time())}_{saved_count:03d}.jpg"
                cv2.imwrite(str(fname), display_frame)
                print(f"[SAVE] {fname}")
                saved_count += 1
            elif key in (ord('+'), ord('=')):
                score_thresh = min(0.95, score_thresh + 0.05)
                print(f"[INFO] score_thresh = {score_thresh:.2f}")
                # 阈值变了立即重跑（用冻结帧或当前帧）
                if frozen_frame is not None:
                    pass  # 下一轮会自动重跑（见上面 if frozen_frame is not None 分支）
            elif key == ord('-'):
                score_thresh = max(0.05, score_thresh - 0.05)
                print(f"[INFO] score_thresh = {score_thresh:.2f}")
            elif key == ord('d'):
                draw_enabled = not draw_enabled
                print(f"[INFO] draw = {draw_enabled}")
            elif key == ord('f'):
                if frozen_frame is None:
                    frozen_frame = frame.copy()
                    print("[INFO] 画面已冻结（再按 f 解冻）")
                else:
                    frozen_frame = None
                    print("[INFO] 画面解冻")

            frame_idx += 1

    except KeyboardInterrupt:
        print("\n[INFO] Ctrl+C 中断")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"[INFO] 完成，共保存 {saved_count} 张到 {save_dir.resolve()}")


if __name__ == "__main__":
    main()
