#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型导出工具

支持三种格式：
  - savedmodel : TF SavedModel（通用，PC/服务器部署）
  - tflite     : TFLite（移动端/嵌入式）
  - onnx       : ONNX（跨平台，Intel/ARM/GPU/NPU）

用法：
    # 导出 SavedModel
    python export.py --format savedmodel

    # 导出 TFLite (FP16)
    python export.py --format tflite --quantize float16

    # 导出 TFLite (INT8)
    python export.py --format tflite --quantize int8 --calib-data 100

    # 导出 ONNX
    python export.py --format onnx

    # 全部导出
    python export.py --format all
"""

import os
import sys
import argparse
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import numpy as np
import tensorflow as tf
import config
from utils.logger import get_logger
from models.detector import DetectionModel


# ============================================================================
# 包装模型（导出用）
# ============================================================================
class ExportModel(tf.Module):
    """
    导出用的包装模型
    接收原始图像 tensor (B, H, W, 3) 归一化到 [0, 1]
    返回 dict {boxes, scores, classes}
    """

    def __init__(self, detection_model: DetectionModel, score_thresh: float = 0.3, nms_iou_thresh: float = 0.5):
        super().__init__()
        self.det = detection_model
        self.model = detection_model.model
        self.anchors = tf.constant(detection_model.anchors, dtype=tf.float32)
        self.num_classes = detection_model.num_classes
        self.input_size = detection_model.input_size
        self.score_thresh = score_thresh
        self.nms_iou_thresh = nms_iou_thresh

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, None, None, 3], dtype=tf.float32, name="input_image"),
    ])
    def __call__(self, input_image):
        """
        Args:
            input_image: (B, H, W, 3) float32, 归一化到 [0, 1]

        Returns:
            {
                "boxes":   (B, N, 4) float32 [x1, y1, x2, y2] 归一化
                "scores":  (B, N)    float32
                "classes": (B, N)    int32
            }
        """
        # 1. Letterbox 预处理
        B = tf.shape(input_image)[0]
        H = tf.shape(input_image)[1]
        W = tf.shape(input_image)[2]
        size = self.input_size
        scale = tf.cast(size, tf.float32) / tf.cast(tf.maximum(H, W), tf.float32)
        new_h = tf.cast(tf.cast(H, tf.float32) * scale, tf.int32)
        new_w = tf.cast(tf.cast(W, tf.float32) * scale, tf.int32)

        # 缩放
        resized = tf.image.resize(input_image, [new_h, new_w])
        # Pad
        padded = tf.image.pad_to_bounding_box(
            resized,
            (size - new_h) // 2, (size - new_w) // 2,
            size, size,
        )

        # 2. 模型推理
        outputs = self.model(padded, training=False)

        # 3. 收集各层预测
        raw_boxes_list = []
        raw_scores_list = []
        for level in sorted(outputs.keys()):
            if not level.startswith("cls_"):
                continue
            box_key = level.replace("cls_", "box_")
            cls = outputs[level]   # (B, H, W, num_anchors * (C+1))
            box = outputs[box_key] # (B, H, W, num_anchors * 4)
            B_, H_, W_, C_ = cls.shape
            cls = tf.reshape(cls, [B_, H_ * W_, -1, self.num_classes + 1])
            cls = tf.reshape(cls, [B_, H_ * W_ * tf.shape(cls)[2], self.num_classes + 1])
            box = tf.reshape(box, [B_, H_ * W_, -1, 4])
            box = tf.reshape(box, [B_, H_ * W_ * tf.shape(box)[2], 4])
            raw_boxes_list.append(box)
            raw_scores_list.append(cls)

        raw_boxes = tf.concat(raw_boxes_list, axis=1)  # (B, N, 4)
        raw_scores = tf.concat(raw_scores_list, axis=1)  # (B, N, C+1)

        # 4. 解码 + NMS
        # 这里用 TF 原生 NMS，批处理
        # 为简化，返回前 100 个最高分（实际工程可做更精细的 NMS）
        # 取前景概率
        probs = tf.nn.softmax(raw_scores, axis=-1)
        fg_probs = probs[..., 1:]  # (B, N, C)
        fg_scores = tf.reduce_max(fg_probs, axis=-1)  # (B, N)
        fg_classes = tf.argmax(fg_probs, axis=-1)    # (B, N)

        # 解码 box
        anchors_batch = tf.expand_dims(self.anchors, 0)  # (1, N, 4)
        cx = raw_boxes[..., 0] * anchors_batch[..., 2] + anchors_batch[..., 0]
        cy = raw_boxes[..., 1] * anchors_batch[..., 3] + anchors_batch[..., 1]
        w = anchors_batch[..., 2] * tf.exp(tf.clip_by_value(raw_boxes[..., 2], -4, 4))
        h = anchors_batch[..., 3] * tf.exp(tf.clip_by_value(raw_boxes[..., 3], -4, 4))
        x1 = tf.clip_by_value(cx - w / 2, 0, 1)
        y1 = tf.clip_by_value(cy - h / 2, 0, 1)
        x2 = tf.clip_by_value(cx + w / 2, 0, 1)
        y2 = tf.clip_by_value(cy + h / 2, 0, 1)
        decoded_boxes = tf.stack([y1, x1, y2, x2], axis=-1)  # (B, N, 4) 注意 NMS 用 yxyx

        # 过滤低分
        mask = fg_scores >= self.score_thresh

        # 这里为简化，假设 batch=1 的典型场景
        # 实际工程建议用 tf.image.combined_non_max_suppression
        # 用 combined_nms 处理多 batch
        selected_boxes, selected_scores, selected_classes, _ = tf.image.combined_non_max_suppression(
            boxes=tf.expand_dims(decoded_boxes[0], 0),  # (1, N, 4) yxyx
            scores=tf.expand_dims(fg_probs[0], 0),      # (1, N, C)
            max_output_size_per_class=100,
            max_total_size=100,
            iou_threshold=self.nms_iou_thresh,
            score_threshold=self.score_thresh,
        )
        # 转为 xyxy
        out_boxes_yx = selected_boxes  # (1, max_total, 4) yxyx
        out_boxes = tf.stack([
            out_boxes_yx[..., 1], out_boxes_yx[..., 0],
            out_boxes_yx[..., 3], out_boxes_yx[..., 2],
        ], axis=-1)  # (1, max_total, 4) xyxy

        # 扩展到 batch
        out_boxes = tf.tile(out_boxes, [B, 1, 1])
        out_scores = tf.tile(selected_scores, [B, 1])
        out_classes = tf.tile(selected_classes, [B, 1])

        return {
            "boxes":   out_boxes,
            "scores":  out_scores,
            "classes": out_classes,
        }


# ============================================================================
# 导出函数
# ============================================================================
def export_savedmodel(det, save_dir: Path, logger):
    """导出 SavedModel"""
    save_dir.mkdir(parents=True, exist_ok=True)
    export_model = ExportModel(det)
    signatures = {
        "serving_default": export_model.__call__.get_concrete_function(
            tf.TensorSpec(shape=[None, None, None, 3], dtype=tf.float32, name="input_image")
        )
    }
    tf.saved_model.save(export_model, str(save_dir), signatures=signatures)
    logger.info(f"✅ SavedModel: {save_dir}")


def export_tflite(det, save_dir: Path, quantize: str, logger, calib_data=None):
    """导出 TFLite"""
    save_dir.mkdir(parents=True, exist_ok=True)

    # 用原始 detector model 导出（不含后处理，便于跨平台）
    converter = tf.lite.TFLiteConverter.from_keras_model(det.model)

    if quantize == "float16":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    elif quantize == "int8":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8
        ]
        # INT8 需要校准数据
        if calib_data is None:
            logger.warning("INT8 量化需要校准数据，使用默认（精度可能下降）")
        else:
            def representative_dataset():
                for batch in calib_data.take(50):
                    # batch 是 (image, label) 形式
                    images = batch[0] if isinstance(batch, tuple) else batch
                    yield [tf.cast(images, tf.float32)]
            converter.representative_dataset = representative_dataset

    tflite_model = converter.convert()
    save_path = save_dir / f"model_{quantize}.tflite"
    save_path.write_bytes(tflite_model)

    size_mb = len(tflite_model) / (1024 * 1024)
    logger.info(f"✅ TFLite ({quantize}): {save_path}  ({size_mb:.1f} MB)")


def export_onnx(det, save_dir: Path, logger):
    """导出 ONNX"""
    try:
        import tf2onnx
    except ImportError:
        logger.error("未安装 tf2onnx，无法导出 ONNX。pip install tf2onnx")
        return

    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / "model.onnx"

    # 转 ONNX
    input_signature = [
        tf.TensorSpec(
            [1, config.INPUT_SIZE, config.INPUT_SIZE, 3],
            tf.float32, name="input_image"
        )
    ]
    onnx_model, _ = tf2onnx.convert.from_keras(
        det.model,
        input_signature=input_signature,
        opset=13,
    )
    onnx_model.save(str(save_path))
    size_mb = save_path.stat().st_size / (1024 * 1024)
    logger.info(f"✅ ONNX: {save_path}  ({size_mb:.1f} MB)")


# ============================================================================
# 主函数
# ============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--format", choices=["savedmodel", "tflite", "onnx", "all"],
                        default="all")
    parser.add_argument("--quantize", choices=["none", "float16", "int8"],
                        default="float16", help="TFLite 量化类型")
    parser.add_argument("--weights", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    if args.weights is None:
        args.weights = config.CHECKPOINT_DIR / "best.weights.h5"
    weights = Path(args.weights)
    if not weights.exists():
        print(f"❌ 权重不存在: {weights}")
        sys.exit(1)

    if args.output_dir is None:
        output_dir = config.EXPORT_DIR
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = get_logger("export", output_dir / "logs")
    logger.info(f"加载权重: {weights}")

    # 构建并加载
    det = DetectionModel()
    det.load_weights(str(weights))
    logger.info("模型加载完成")

    formats = ["savedmodel", "tflite", "onnx"] if args.format == "all" else [args.format]

    for fmt in formats:
        logger.info(f"\n{'='*50}\n导出 {fmt} ...\n{'='*50}")
        try:
            if fmt == "savedmodel":
                export_savedmodel(det, output_dir, logger)
            elif fmt == "tflite":
                export_tflite(
                    det,
                    output_dir,
                    args.quantize if args.quantize != "none" else "float16",
                    logger,
                )
            elif fmt == "onnx":
                export_onnx(det, output_dir, logger)
        except Exception as e:
            logger.error(f"导出 {fmt} 失败: {e}")
            import traceback
            traceback.print_exc()

    logger.info("\n所有导出完成")


if __name__ == "__main__":
    main()
