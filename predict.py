#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单图/批量推理

用法：
    # 单图
    python predict.py --image test.jpg

    # 批量（目录下所有图片）
    python predict.py --image-dir ./test_images/

    # 指定权重
    python predict.py --image test.jpg --weights checkpoints/balanced_coco/best.weights.h5
"""

import os
import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import cv2
import numpy as np
import tensorflow as tf
import config
from utils.logger import get_logger
from utils.visualization import draw_boxes, save_detection_image
from models.detector import DetectionModel
from models.postprocess import decode_predictions
from models.anchors import xywh_to_xyxy


# ============================================================================
# 图像预处理
# ============================================================================
def preprocess_image(image: np.ndarray, input_size: int) -> tuple:
    """
    Letterbox 预处理（保持宽高比）

    Args:
        image: BGR 图片
        input_size: 模型输入尺寸

    Returns:
        (input_tensor, scale, pad_top, pad_left, orig_h, orig_w)
    """
    orig_h, orig_w = image.shape[:2]
    scale = input_size / max(orig_h, orig_w)
    new_h, new_w = int(orig_h * scale), int(orig_w * scale)
    resized = cv2.resize(image, (new_w, new_h))

    pad = np.full((input_size, input_size, 3), 114, dtype=np.uint8)  # 灰色 padding
    pad_top = (input_size - new_h) // 2
    pad_left = (input_size - new_w) // 2
    pad[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized

    # 归一化 + batch 维度
    inp = pad.astype(np.float32) / 255.0
    inp = np.expand_dims(inp, axis=0)  # (1, H, W, 3)
    return inp, scale, pad_top, pad_left, orig_h, orig_w


# ============================================================================
# 加载类别名
# ============================================================================
def load_class_names(label_map_path: Path) -> list:
    """从 label_map.pbtxt 加载类别名"""
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
# 推理单图
# ============================================================================
def infer_single(det, image_path: Path, save_dir: Path, class_names: list,
                 score_thresh: float, nms_iou_thresh: float, logger):
    """单图推理"""
    image = cv2.imread(str(image_path))
    if image is None:
        logger.error(f"无法读取图片: {image_path}")
        return

    # 预处理
    inp, scale, pad_top, pad_left, h, w = preprocess_image(image, config.INPUT_SIZE)

    # 推理
    outputs = det.predict(inp, verbose=0)

    # 收集所有层的预测
    # ★ 用 det.feature_spec 顺序而非 sorted：保证和训练时 anchor 拼接顺序一致
    raw_boxes = []
    raw_scores = []
    for level in det.feature_spec.keys():
        box_key = f"box_{level}"
        # 形状: (1, H, W, num_anchors * (C+1)) / (1, H, W, num_anchors * 4)
        cls = outputs[f"cls_{level}"][0]  # (H, W, num_anchors * (C+1))
        box = outputs[box_key][0]         # (H, W, num_anchors * 4)
        H, W = cls.shape[:2]
        cls = cls.reshape(H, W, -1, config.NUM_CLASSES + 1)
        box = box.reshape(H, W, -1, 4)
        # 移到 (N, ...)
        cls = cls.reshape(-1, config.NUM_CLASSES + 1)
        box = box.reshape(-1, 4)
        raw_boxes.append(box)
        raw_scores.append(cls)

    raw_boxes = np.concatenate(raw_boxes, axis=0)
    raw_scores = np.concatenate(raw_scores, axis=0)

    # 解码
    result = decode_predictions(
        raw_boxes, raw_scores, det.anchors,
        image_shape=(h, w),
        input_size=config.INPUT_SIZE,
        score_thresh=score_thresh,
        nms_iou_thresh=nms_iou_thresh,
        num_classes=config.NUM_CLASSES,
    )

    # Letterbox 坐标逆变换
    boxes_out = []
    for box in result["boxes"]:
        x1, y1, x2, y2 = box
        # 减 padding，除以 scale
        rx1 = max(0, int((x1 - pad_left) / scale))
        ry1 = max(0, int((y1 - pad_top) / scale))
        rx2 = min(w, int((x2 - pad_left) / scale))
        ry2 = min(h, int((y2 - pad_top) / scale))
        boxes_out.append([rx1, ry1, rx2, ry2])
    boxes_out = np.array(boxes_out, dtype=np.float32)

    # 画框
    img_out = draw_boxes(
        image, boxes_out, result["scores"], result["class_ids"],
        class_names, score_thresh=score_thresh,
    )

    # 保存
    save_path = save_dir / f"pred_{image_path.stem}.png"
    save_detection_image(img_out, [], save_path=save_path)

    # 打印结果
    logger.info(f"  {image_path.name}: 检测到 {len(boxes_out)} 个目标")
    for i, (box, score, cls_id) in enumerate(zip(boxes_out, result["scores"], result["class_ids"])):
        cls_name = class_names[int(cls_id)] if int(cls_id) < len(class_names) else f"id={cls_id}"
        logger.info(f"    [{i+1}] {cls_name} {score:.2f} @ {[int(v) for v in box]}")


# ============================================================================
# 主函数
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="单图/批量推理")
    parser.add_argument("--image", type=str, help="单张图片路径")
    parser.add_argument("--image-dir", type=str, help="图片目录")
    parser.add_argument("--weights", type=str, default=None,
                        help="模型权重路径（默认 checkpoints/{experiment}/best.weights.h5）")
    parser.add_argument("--score-thresh", type=float, default=config.INFERENCE_SCORE_THRESH)
    parser.add_argument("--nms-iou-thresh", type=float, default=config.NMS_IOU_THRESH)
    parser.add_argument("--save-dir", type=str, default=None,
                        help="结果保存目录（默认 outputs/{experiment}/predictions）")
    args = parser.parse_args()

    if not args.image and not args.image_dir:
        parser.error("必须指定 --image 或 --image-dir")

    # 默认权重
    if args.weights is None:
        args.weights = config.CHECKPOINT_DIR / "best.weights.h5"
    weights = Path(args.weights)
    if not weights.exists():
        print(f"❌ 权重不存在: {weights}")
        sys.exit(1)

    # 默认保存目录
    if args.save_dir is None:
        save_dir = config.OUTPUT_DIR / "predictions"
    else:
        save_dir = Path(args.save_dir)

    save_dir.mkdir(parents=True, exist_ok=True)
    logger = get_logger("predict", save_dir)
    logger.info(f"加载权重: {weights}")
    logger.info(f"保存目录: {save_dir}")

    # 构建检测器
    det = DetectionModel()
    det.load_weights(str(weights))
    logger.info("模型加载完成")

    # 类别名
    class_names = load_class_names(config.LABEL_MAP_PATH)
    logger.info(f"类别数: {len(class_names)}")

    # 收集图片
    images = []
    if args.image:
        images.append(Path(args.image))
    if args.image_dir:
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]:
            images.extend(Path(args.image_dir).glob(ext))

    logger.info(f"待推理图片: {len(images)} 张")

    for img_path in images:
        infer_single(det, img_path, save_dir, class_names,
                     args.score_thresh, args.nms_iou_thresh, logger)

    logger.info("推理完成")


if __name__ == "__main__":
    main()
