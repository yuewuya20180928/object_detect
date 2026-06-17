#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Letterbox resize 工具（TTA 必需）

保持宽高比缩放后 padding 到 target_size×target_size（灰边）
配套提供 inverse 函数，把推理 box 从 letterbox 空间映射回原图空间

约定:
  - 图像 float32 [0, 1]（训练 / 推理都是这个格式）
  - box 格式 xyxy，像素坐标
"""
import numpy as np
import tensorflow as tf


def letterbox_params(orig_h: int, orig_w: int, target_size: int):
    """
    计算 letterbox 参数 (与 dataset_builder.resize_with_padding 一致)

    Args:
        orig_h, orig_w: 原图尺寸
        target_size:    目标方形尺寸

    Returns:
        scale:  缩放比例 (new / orig)
        new_h, new_w: 缩放后尺寸（int 截断）
        pad_x, pad_y: padding 偏移（int）
    """
    scale = target_size / float(max(orig_h, orig_w))
    new_h = int(orig_h * scale)
    new_w = int(orig_w * scale)
    pad_x = (target_size - new_w) // 2
    pad_y = (target_size - new_h) // 2
    return scale, new_h, new_w, pad_x, pad_y


def letterbox_image(image: np.ndarray, target_size: int) -> tuple:
    """
    对单张图像做 letterbox resize

    Args:
        image: (H, W, 3) float32 [0, 1] 或 uint8 [0, 255]
        target_size: 目标方形尺寸

    Returns:
        (letterboxed, scale, pad_x, pad_y)
    """
    if image.ndim == 3:
        h, w = image.shape[:2]
    else:
        raise ValueError(f"image must be (H,W,3), got {image.shape}")

    scale, new_h, new_w, pad_x, pad_y = letterbox_params(h, w, target_size)

    # 缩放（用 tf.image.resize 跟训练时一致）
    if image.dtype == np.float32:
        img_tf = tf.constant(image)
    else:
        img_tf = tf.constant(image.astype(np.float32))
    resized = tf.image.resize(img_tf, [new_h, new_w], method=tf.image.ResizeMethod.BILINEAR).numpy()

    # padding（灰色，dataset_builder 默认 0）
    if target_size - new_h > 0 or target_size - new_w > 0:
        padded = np.zeros((target_size, target_size, resized.shape[2]), dtype=resized.dtype)
        padded[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized
        return padded, scale, pad_x, pad_y
    return resized, scale, pad_x, pad_y


def unletterbox_boxes(xyxy: np.ndarray, scale: float, pad_x: int, pad_y: int) -> np.ndarray:
    """
    把 letterbox 空间 (target_size×target_size) 的 xyxy box 映射回原图空间

    公式:
      x_orig = (x_letter - pad_x) / scale
      y_orig = (y_letter - pad_y) / scale
    """
    if len(xyxy) == 0:
        return xyxy
    out = xyxy.copy().astype(np.float32)
    out[:, 0] = (xyxy[:, 0] - pad_x) / scale
    out[:, 1] = (xyxy[:, 1] - pad_y) / scale
    out[:, 2] = (xyxy[:, 2] - pad_x) / scale
    out[:, 3] = (xyxy[:, 3] - pad_y) / scale
    return out


def flip_h_boxes(xyxy: np.ndarray, image_w: int) -> np.ndarray:
    """水平翻转 xyxy box（在图像坐标系）"""
    if len(xyxy) == 0:
        return xyxy
    out = xyxy.copy()
    x1, x2 = xyxy[:, 0].copy(), xyxy[:, 2].copy()
    out[:, 0] = image_w - x2
    out[:, 2] = image_w - x1
    return out


def flip_h_image(image: np.ndarray) -> np.ndarray:
    """水平翻转图像 (H, W, 3)"""
    return image[:, ::-1, :].copy()
