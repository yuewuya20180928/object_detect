#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
摄像头实时目标检测

用法：
    # 默认摄像头
    python camera_detect.py

    # 指定摄像头
    python camera_detect.py --device 1

    # 调整阈值
    python camera_detect.py --conf 0.5 --nms 0.4

按 q 退出。
"""

import os
import sys
import argparse
import queue
import threading
import time
from pathlib import Path
from collections import deque

PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import cv2
import numpy as np
import tensorflow as tf
import config
from utils.logger import get_logger
from utils.visualization import draw_boxes
from models.detector import DetectionModel
from models.postprocess import decode_predictions


# ============================================================================
# 预处理 / 后处理
# ============================================================================
def letterbox(image: np.ndarray, size: int) -> tuple:
    """保持宽高比的 letterbox"""
    h, w = image.shape[:2]
    scale = size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(image, (new_w, new_h))
    pad = np.full((size, size, 3), 114, dtype=np.uint8)
    pad_top = (size - new_h) // 2
    pad_left = (size - new_w) // 2
    pad[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized
    return pad, scale, pad_top, pad_left


def _box_iou(a, b):
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    iw = max(ix2 - ix1, 0); ih = max(iy2 - iy1, 0)
    inter = iw * ih
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-6)


# ============================================================================
# 时序平滑（过滤单帧假阳性）
# ============================================================================
class TemporalFilter:
    def __init__(self, min_hits: int = 2, iou_thresh: float = 0.4):
        self.history = deque(maxlen=3)
        self.min_hits = min_hits
        self.iou_thresh = iou_thresh

    def update(self, dets):
        self.history.append(dets)
        if len(self.history) < self.min_hits:
            return []
        confirmed = []
        for det in self.history[-1]:
            hits = 1
            for prev in list(self.history)[:-1]:
                if any(_box_iou(det, p) > self.iou_thresh for p in prev):
                    hits += 1
            if hits >= self.min_hits:
                confirmed.append(det)
        return confirmed


# ============================================================================
# 推理线程
# ============================================================================
def infer_worker(model, det, infer_q, result_q, conf_thresh, nms_thresh, anchors, num_classes, input_size, class_names):
    tf_filter = TemporalFilter(min_hits=2, iou_thresh=0.4)
    while True:
        frame = infer_q.get()
        if frame is None:
            break
        try:
            h, w = frame.shape[:2]
            # 预处理
            padded, scale, pad_top, pad_left = letterbox(frame, input_size)
            inp = (padded.astype(np.float32) / 255.0)[None, ...]

            # 推理
            outputs = model(inp, training=False)

            # 收集
            raw_boxes, raw_scores = [], []
            for level in sorted(outputs.keys()):
                if not level.startswith("cls_"):
                    continue
                box_key = level.replace("cls_", "box_")
                cls = outputs[level][0].numpy()
                box = outputs[box_key][0].numpy()
                H, W = cls.shape[:2]
                cls = cls.reshape(H, W, -1, num_classes + 1).reshape(-1, num_classes + 1)
                box = box.reshape(H, W, -1, 4).reshape(-1, 4)
                raw_boxes.append(box)
                raw_scores.append(cls)
            raw_boxes = np.concatenate(raw_boxes, axis=0)
            raw_scores = np.concatenate(raw_scores, axis=0)

            # 解码 + NMS
            result = decode_predictions(
                raw_boxes, raw_scores, anchors,
                image_shape=(h, w), input_size=input_size,
                score_thresh=conf_thresh, nms_iou_thresh=nms_thresh,
                num_classes=num_classes,
            )

            # Letterbox 逆变换
            dets = []
            for box, score, cls_id in zip(result["boxes"], result["scores"], result["class_ids"]):
                x1, y1, x2, y2 = box
                rx1 = max(0, int((x1 - pad_left) / scale))
                ry1 = max(0, int((y1 - pad_top) / scale))
                rx2 = min(w, int((x2 - pad_left) / scale))
                ry2 = min(h, int((y2 - pad_top) / scale))
                if rx2 > rx1 and ry2 > ry1:
                    dets.append((rx1, ry1, rx2, ry2, float(score), int(cls_id)))

            confirmed = tf_filter.update(dets)

            # 替换最新结果
            while not result_q.empty():
                try:
                    result_q.get_nowait()
                except queue.Empty:
                    break
            result_q.put(confirmed)
        except Exception as e:
            print(f"[infer worker error] {e}")


# ============================================================================
# 类别名加载
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
# 主循环
# ============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0, help="摄像头设备索引")
    parser.add_argument("--conf", type=float, default=config.INFERENCE_SCORE_THRESH)
    parser.add_argument("--nms", type=float, default=config.NMS_IOU_THRESH)
    parser.add_argument("--weights", type=str, default=None,
                        help="模型权重路径（默认 checkpoints/{experiment}/best.weights.h5）")
    parser.add_argument("--mirror", action="store_true", help="水平翻转画面（镜像模式）")
    parser.add_argument("--width", type=int, default=1280, help="摄像头分辨率宽")
    parser.add_argument("--height", type=int, default=720, help="摄像头分辨率高")
    args = parser.parse_args()

    if args.weights is None:
        weights = config.CHECKPOINT_DIR / "best.weights.h5"
    else:
        weights = Path(args.weights)
    if not weights.exists():
        print(f"❌ 权重不存在: {weights}")
        sys.exit(1)

    logger = get_logger("camera", None, log_to_file=False)
    logger.info(f"加载权重: {weights}")

    det = DetectionModel()
    det.load_weights(str(weights))
    logger.info("模型加载完成")

    class_names = load_class_names(config.LABEL_MAP_PATH)

    cap = cv2.VideoCapture(args.device)
    if not cap.isOpened():
        logger.error(f"无法打开摄像头 {args.device}")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    logger.info(f"摄像头: device={args.device}, 分辨率={args.width}x{args.height}")
    logger.info(f"参数: conf={args.conf}, nms={args.nms}")
    logger.info("按 q 退出")

    # 启动推理线程
    infer_q = queue.Queue(maxsize=1)
    result_q = queue.Queue(maxsize=1)
    worker = threading.Thread(
        target=infer_worker,
        args=(
            det.model, det, infer_q, result_q,
            args.conf, args.nms, det.anchors,
            config.NUM_CLASSES, config.INPUT_SIZE, class_names,
        ),
        daemon=True,
    )
    worker.start()

    last_dets = []
    frame_id = 0
    fps_counter = deque(maxlen=30)
    fps_t0 = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            logger.warning("读取帧失败")
            continue

        frame_id += 1
        fps_counter.append(time.time())
        if len(fps_counter) > 5:
            fps = (len(fps_counter) - 1) / (fps_counter[-1] - fps_counter[0])

        # 每隔 2 帧送一次推理
        if frame_id % 2 == 0:
            if infer_q.empty():
                try:
                    infer_q.put_nowait(frame.copy())
                except queue.Full:
                    pass

        # 非阻塞取最新结果
        try:
            last_dets = result_q.get_nowait()
        except queue.Empty:
            pass

        # 绘制
        display = frame.copy()
        if last_dets:
            boxes = np.array([[d[0], d[1], d[2], d[3]] for d in last_dets], dtype=np.float32)
            scores = np.array([d[4] for d in last_dets], dtype=np.float32)
            class_ids = np.array([d[5] for d in last_dets], dtype=np.int32)
            display = draw_boxes(display, boxes, scores, class_ids, class_names)

        # 绘制 FPS
        if len(fps_counter) > 5:
            cv2.putText(display, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if args.mirror:
            display = cv2.flip(display, 1)

        cv2.imshow("Object Detection  (q to quit)", display)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    try:
        infer_q.put_nowait(None)
    except queue.Full:
        pass
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
