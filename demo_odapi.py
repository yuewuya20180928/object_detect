#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
USB 摄像头实时目标检测 (TF OD API 官方预训练权重版)

模型: TF 官方 SSD MobileNetV2 FPNLite 320x320 (COCO 80 类, mAP=0.292)
输入: USB 摄像头（默认 /dev/video0，可用 --camera 切换）
显示: OpenCV 窗口实时显示带检测框的画面

按你:
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

TTA (Test-Time Augmentation) 性能：
  - 多尺度 + 水平 flip + NMS-concat 融合：val mAP 0.05→0.053 (+8%)
  - 默认 3 尺度 [320, 384, 448] + flip = 6 次推理/帧, 实时变慢
  - 生产推荐加 --skip-frames 5 + --tta-scales 320 384 减少延迟
  - 验证脚本: od_project/evaluate_odapi_tta.py --tta-fusion nms
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

# 使用 TF OD API 官方 SavedModel (跳我们自己的 DetectionModel)
ODAPI_MODEL_PATH = PROJECT_ROOT / "pretrained" / "ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8" / "saved_model"
from models.postprocess import decode_predictions  # noqa: E402
sys.path.insert(0, str(PROJECT_ROOT / "od_project"))
from utils.inference import batched_nms_per_class  # noqa: E402


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


def detect_tta(
    signature,
    rgb: np.ndarray,
    scales: list,
    with_flip: bool,
    fusion: str,
    nms_iou: float = 0.5,
) -> tuple:
    """
    多尺度 + flip TTA 推理（在原图坐标系返回 boxes）

    Args:
        signature: OD API saved_model serving_default
        rgb:       (H, W, 3) uint8 原图
        scales:    多个推理尺寸，如 [320, 384, 448]
        with_flip: 每个尺度是否额外做水平 flip
        fusion:    "nms" (推荐，+8% mAP) | "single" (仅第 1 尺度)
        nms_iou:   NMS IoU 阈值（仅 fusion="nms"）

    Returns:
        (boxes, scores, class_ids) in 原图像素坐标
    """
    H, W = rgb.shape[:2]
    boxes_all, scores_all, cids_all = [], [], []
    for scale_size in scales:
        # 原图
        padded, scale, pad_x, pad_y = letterbox(rgb, scale_size)
        tensor = tf.cast(tf.expand_dims(padded, 0), tf.uint8)
        outputs = signature(tensor)
        n = int(outputs["num_detections"][0].numpy())
        b = outputs["detection_boxes"][0].numpy()[:n]
        s = outputs["detection_scores"][0].numpy()[:n]
        c = outputs["detection_classes"][0].numpy()[:n].astype(np.int32) - 1
        xy = np.zeros((n, 4), dtype=np.float32)
        xy[:, 0] = b[:, 1] * scale_size
        xy[:, 1] = b[:, 0] * scale_size
        xy[:, 2] = b[:, 3] * scale_size
        xy[:, 3] = b[:, 2] * scale_size
        boxes_all.append(unletterbox_boxes(xy, scale, pad_x, pad_y))
        scores_all.append(s)
        cids_all.append(c)
        # flip
        if with_flip:
            rgb_flip = rgb[:, ::-1, :]
            padded_f, scale_f, pad_x_f, pad_y_f = letterbox(rgb_flip, scale_size)
            tensor_f = tf.cast(tf.expand_dims(padded_f, 0), tf.uint8)
            outputs_f = signature(tensor_f)
            nf = int(outputs_f["num_detections"][0].numpy())
            bf = outputs_f["detection_boxes"][0].numpy()[:nf]
            sf = outputs_f["detection_scores"][0].numpy()[:nf]
            cf = outputs_f["detection_classes"][0].numpy()[:nf].astype(np.int32) - 1
            xyf = np.zeros((nf, 4), dtype=np.float32)
            xyf[:, 0] = bf[:, 1] * scale_size
            xyf[:, 1] = bf[:, 0] * scale_size
            xyf[:, 2] = bf[:, 3] * scale_size
            xyf[:, 3] = bf[:, 2] * scale_size
            xyf_orig = unletterbox_boxes(xyf, scale_f, pad_x_f, pad_y_f)
            # unflip: 水平翻转 box
            xyf_orig[:, 0] = W - xyf_orig[:, 2]
            xyf_orig[:, 2] = W - xyf_orig[:, 0]
            boxes_all.append(xyf_orig)
            scores_all.append(sf)
            cids_all.append(cf)

    if fusion == "single" or len(boxes_all) == 1:
        return boxes_all[0], scores_all[0], cids_all[0]

    # NMS-concat: 拼所有源，class-aware NMS 去重
    all_b = np.concatenate(boxes_all, axis=0)
    all_s = np.concatenate(scores_all, axis=0)
    all_c = np.concatenate(cids_all, axis=0)
    keep = all_s >= 0.01
    all_b, all_s, all_c = all_b[keep], all_s[keep], all_c[keep]
    if len(all_s) == 0:
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.int32),
        )
    return batched_nms_per_class(
        all_b, all_s, all_c, iou_threshold=nms_iou, max_output_size=200,
    )


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
    parser.add_argument("--score-thresh", type=float, default=0.5,
                        help="score 阈值（官方预训练权重，0.5 是合理默认）")
    parser.add_argument("--nms-iou", type=float, default=0.5, help="NMS IoU 阈值")
    parser.add_argument("--skip-frames", type=int, default=0,
                        help="每 N+1 帧跑一次检测（0=每帧；2 = 每 3 帧一次）")
    parser.add_argument("--input-size", type=int, default=config.INPUT_SIZE,
                        help="模型输入尺寸（默认 320，改了需要 retrain 或重新生成 anchors）")
    parser.add_argument("--weights", type=str, default=None,
                        help="OD API SavedModel 路径（默认 pretrained/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/saved_model）")
    parser.add_argument("--save-dir", type=str, default="./detections",
                        help="按 's' 保存的目录")
    # TTA 参数
    parser.add_argument("--tta", action="store_true", help="启用 Test-Time Augmentation（多尺度 + flip）")
    parser.add_argument("--tta-fusion", type=str, default="nms", choices=["nms", "single"],
                       help="TTA 融合方式：nms (concat+class-NMS，推荐) | single (仅首尺度)")
    parser.add_argument("--tta-scales", type=int, nargs="+", default=[320, 384, 448],
                       help="TTA 多尺度列表（默认 [320, 384, 448]）")
    parser.add_argument("--no-tta-flip", dest="tta_with_flip", action="store_false", default=True,
                       help="TTA 不做水平 flip")
    parser.add_argument("--tta-nms-iou", type=float, default=0.5, help="TTA NMS IoU 阈值")
    args = parser.parse_args()

    if args.list_cameras:
        print("可用摄像头:")
        for idx, w, h in list_cameras():
            print(f"  /dev/video{idx}  →  {w}x{h}")
        return

    weights = args.weights or str(ODAPI_MODEL_PATH)
    if not Path(weights).exists():
        print(f"[ERROR] SavedModel 不存在: {weights}")
        sys.exit(1)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ===== 加载 TF OD API SavedModel =====
    print(f"[INFO] 加载 TF OD API SavedModel: {weights}")
    detect_fn = tf.saved_model.load(str(weights))
    signature = detect_fn.signatures["serving_default"]
    print(f"[INFO] SavedModel 加载完成 | 输入: input_tensor (uint8, H×W×3)")

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
    dummy = tf.zeros((1, args.input_size, args.input_size, 3), dtype=tf.uint8)
    _ = signature(dummy)
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
                if args.tta:
                    boxes_orig, sc, class_ids = detect_tta(
                        signature, rgb, args.tta_scales, args.tta_with_flip,
                        fusion=args.tta_fusion, nms_iou=args.tta_nms_iou,
                    )
                else:
                    padded, scale, pad_x, pad_y = letterbox(rgb, args.input_size)
                    input_tensor = tf.cast(tf.expand_dims(padded, 0), tf.uint8)
                    outputs = signature(input_tensor)
                    boxes = outputs["detection_boxes"][0].numpy()
                    sc = outputs["detection_scores"][0].numpy()
                    cls = outputs["detection_classes"][0].numpy().astype(np.int32)
                    num = int(outputs["num_detections"][0].numpy())
                    keep = sc[:num] >= score_thresh
                    boxes_320 = np.zeros((num, 4), dtype=np.float32)
                    boxes_320[:, 0] = boxes[:num, 1] * args.input_size
                    boxes_320[:, 1] = boxes[:num, 0] * args.input_size
                    boxes_320[:, 2] = boxes[:num, 3] * args.input_size
                    boxes_320[:, 3] = boxes[:num, 2] * args.input_size
                    class_ids = cls - 1
                    boxes_320 = boxes_320[keep]
                    sc = sc[:num][keep]
                    class_ids = class_ids[:num][keep]
                    boxes_orig = unletterbox_boxes(boxes_320, scale, pad_x, pad_y)
                # 统一应用 score_thresh
                keep_final = sc >= score_thresh
                last_result = (boxes_orig[keep_final], sc[keep_final], class_ids[keep_final])
            elif frame_idx % (args.skip_frames + 1) == 0 and frozen_frame is not None:
                rgb = cv2.cvtColor(frozen_frame, cv2.COLOR_BGR2RGB)
                if args.tta:
                    boxes_orig, sc, class_ids = detect_tta(
                        signature, rgb, args.tta_scales, args.tta_with_flip,
                        fusion=args.tta_fusion, nms_iou=args.tta_nms_iou,
                    )
                else:
                    padded, scale, pad_x, pad_y = letterbox(rgb, args.input_size)
                    input_tensor = tf.cast(tf.expand_dims(padded, 0), tf.uint8)
                    outputs = signature(input_tensor)
                    boxes = outputs["detection_boxes"][0].numpy()
                    sc = outputs["detection_scores"][0].numpy()
                    cls = outputs["detection_classes"][0].numpy().astype(np.int32)
                    num = int(outputs["num_detections"][0].numpy())
                    keep = sc[:num] >= score_thresh
                    boxes_320 = np.zeros((num, 4), dtype=np.float32)
                    boxes_320[:, 0] = boxes[:num, 1] * args.input_size
                    boxes_320[:, 1] = boxes[:num, 0] * args.input_size
                    boxes_320[:, 2] = boxes[:num, 3] * args.input_size
                    boxes_320[:, 3] = boxes[:num, 2] * args.input_size
                    class_ids = cls - 1
                    boxes_320 = boxes_320[keep]
                    sc = sc[:num][keep]
                    class_ids = class_ids[:num][keep]
                    boxes_orig = unletterbox_boxes(boxes_320, scale, pad_x, pad_y)
                keep_final = sc >= score_thresh
                last_result = (boxes_orig[keep_final], sc[keep_final], class_ids[keep_final])

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
