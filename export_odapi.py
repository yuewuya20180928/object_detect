#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
导出 OD API SavedModel → TFLite / ONNX

支持:
  - SSD MobileNetV2 FPNLite 320
  - EfficientDet-D1 640
  - 其他 TF OD API 模型

用法:
    # 导出 ONNX
    python3 export_odapi.py --model-path pretrained/efficientdet_d1_coco17_tpu-32/saved_model --format onnx --output outputs/d1_onnx

    # 导出 TFLite（FP16，可能需要 allow_custom_ops）
    python3 export_odapi.py --model-path pretrained/efficientdet_d1_coco17_tpu-32/saved_model --format tflite --quantize float16

注意:
  - TFLite 转换 EfficientDet 通常需要 allow_custom_ops=True，BiFPN 有自定义算子
  - ONNX 转换相对稳定，tf2onnx 会处理大多数算子
"""
import os
import sys
import argparse
import shutil
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import numpy as np
import tensorflow as tf
from utils.logger import get_logger


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", type=str, required=True,
                   help="OD API SavedModel 路径")
    p.add_argument("--format", type=str, default="onnx",
                   choices=["onnx", "tflite", "savedmodel"],
                   help="导出格式")
    p.add_argument("--output", type=str, default=None,
                   help="输出目录（默认 outputs/<model_name>_<format>）")
    p.add_argument("--quantize", type=str, default="fp32",
                   choices=["fp32", "fp16", "int8"],
                   help="TFLite 量化方式")
    p.add_argument("--opset", type=int, default=13, help="ONNX opset")
    return p.parse_args()


def export_onnx(model_path: Path, output_dir: Path, opset: int, logger):
    """导出 ONNX（走 tf2onnx CLI 避开 _WrapperFunction 兼容性问题）"""
    import subprocess
    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = output_dir / "model.onnx"

    # tf2onnx 命令行模式
    tf2onnx_bin = sys.executable + " -m tf2onnx.convert"
    cmd = [
        sys.executable, "-m", "tf2onnx.convert",
        "--saved-model", str(model_path),
        "--output", str(onnx_path),
        "--opset", str(opset),
    ]
    logger.info(f"转 ONNX (opset={opset})...")
    logger.info(f"  cmd: {' '.join(cmd)}")
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - t0

    if proc.returncode != 0:
        logger.error(f"ONNX 转换失败:\n{proc.stderr[-2000:]}")
        return None

    size_mb = onnx_path.stat().st_size / (1024 * 1024)
    logger.info(f"✅ ONNX 导出完成: {onnx_path} ({size_mb:.1f} MB, {elapsed:.1f}s)")
    return onnx_path


def export_tflite(model_path: Path, output_dir: Path, quantize: str, logger):
    """导出 TFLite"""
    logger.info(f"加载 SavedModel: {model_path}")
    model = tf.saved_model.load(str(model_path))
    concrete_func = model.signatures["serving_default"]

    output_dir.mkdir(parents=True, exist_ok=True)
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])

    # 从 saved_model 读 input shape，传给 TFLite（避免默认固定为 1x1x1x3）
    input_spec = concrete_func.structured_input_signature[1]
    for name, spec in input_spec.items():
        shape_list = spec.shape.as_list()
        # batch 固定为 1，其他维度从 signature 读（None 填 640）
        fixed_shape = [1] + [640 if s is None else s for s in shape_list[1:]]
        converter.input_shapes = {name: fixed_shape}
        logger.info(f"  TFLite input shape [{name}]: {fixed_shape}")

    if quantize == "fp16":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    elif quantize == "int8":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        # INT8 需要 representative_dataset
        fixed_h = fixed_shape[1] if fixed_shape[1] else 320
        fixed_w = fixed_shape[2] if fixed_shape[2] else 320
        def representative_dataset():
            for _ in range(100):
                yield [np.random.randint(0, 255, (1, fixed_h, fixed_w, 3), dtype=np.uint8)]
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.float32

    converter.allow_custom_ops = True
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]

    logger.info(f"转 TFLite ({quantize})...")
    t0 = time.time()
    try:
        tflite_model = converter.convert()
    except Exception as e:
        logger.error(f"TFLite 转换失败: {e}")
        logger.error("EfficientDet 的 BiFPN 自定义算子可能不支持，需要 TF Model Garden 的 export_tflite_graph 工具")
        return None
    elapsed = time.time() - t0

    suffix = f"_{quantize}" if quantize != "fp32" else ""
    tflite_path = output_dir / f"model{suffix}.tflite"
    tflite_path.write_bytes(tflite_model)
    size_mb = len(tflite_model) / (1024 * 1024)
    logger.info(f"✅ TFLite 导出完成: {tflite_path} ({size_mb:.1f} MB, {elapsed:.1f}s)")

    # 同时复制 saved_model 到 output dir（给 benchmark 读 shape 用）
    import shutil
    sm_dst = output_dir / "saved_model"
    if not sm_dst.exists():
        shutil.copytree(model_path, sm_dst)
        logger.info(f"  复制 saved_model 到: {sm_dst}")

    return tflite_path


def benchmark_tflite(tflite_path: Path, input_shape, logger):
    """TFLite 推理测时"""
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # 动态读 input shape（避免硬编码错误）
    # TFLite 的 input_details shape 可能都是 1 (因为转换器对动态维度处理)
    # 总是 batch=1, H/W 从 tflite 文件名/saved_model 读
    sm_dir = tflite_path.parent / "saved_model"
    if not sm_dir.exists():
        # 导出 script 把 saved_model 复制到了同目录
        for cand in tflite_path.parent.rglob("saved_model.pb"):
            sm_dir = cand.parent
            break
    try:
        sm = tf.saved_model.load(str(sm_dir))
        sig = sm.signatures["serving_default"]
        input_spec = sig.structured_input_signature[1]
        sm_shape = list(input_spec.values())[0].shape.as_list()
        # batch=1, H/W 从 sm, channel=3 (如果是的话)
        real_shape = [1]
        for s in sm_shape[1:]:
            real_shape.append(s if s is not None else 640)
        real_shape = tuple(real_shape)
    except Exception as e:
        logger.warning(f"读 saved_model shape 失败: {e}, 用 TFLite details")
        actual_shape = list(input_details[0]["shape"])
        real_shape = tuple(int(s) if s > 1 else (640 if i > 0 else 1) for i, s in enumerate(actual_shape))
    logger.info(f"  TFLite input shape: {real_shape}")

    input_dtype = input_details[0]["dtype"]
    if input_dtype == np.uint8:
        dummy = np.random.randint(0, 255, real_shape, dtype=np.uint8)
    else:
        dummy = np.random.rand(*real_shape).astype(np.float32)

    try:
        interpreter.set_tensor(input_details[0]["index"], dummy)
        interpreter.invoke()
        for _ in range(10):
            interpreter.set_tensor(input_details[0]["index"], dummy)
            interpreter.invoke()
        t0 = time.time()
        n_runs = 50
        for _ in range(n_runs):
            interpreter.set_tensor(input_details[0]["index"], dummy)
            interpreter.invoke()
        elapsed = time.time() - t0
        ms = elapsed / n_runs * 1000
        fps = n_runs / elapsed
        logger.info(f"  TFLite CPU {real_shape}: {ms:.1f} ms/run, {fps:.1f} FPS")
    except Exception as e:
        logger.warning(f"  TFLite benchmark 跳过 (input shape 被 fix 为 {list(input_details[0]['shape'])}): {e}")
        logger.info(f"  ⚠️ OD API SavedModel 转 TFLite 有 known input shape 限制。")
        logger.info(f"  建议用 TF Model Garden 的 export_tflite_graph，或走 ONNX 路径。")


def benchmark_onnx(onnx_path: Path, input_shape, logger):
    """ONNX Runtime 推理测时"""
    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    input_type = sess.get_inputs()[0].type
    # 根据模型期望的 dtype 构造 dummy
    if "uint8" in input_type:
        dummy = np.random.randint(0, 255, input_shape, dtype=np.uint8)
    elif "float" in input_type:
        dummy = np.random.rand(*input_shape).astype(np.float32)
    else:
        dummy = np.random.randint(0, 255, input_shape, dtype=np.uint8)
    for _ in range(10):
        sess.run(None, {input_name: dummy})
    t0 = time.time()
    n_runs = 50
    for _ in range(n_runs):
        sess.run(None, {input_name: dummy})
    elapsed = time.time() - t0
    ms = elapsed / n_runs * 1000
    fps = n_runs / elapsed
    logger.info(f"  ONNX CPU {input_shape}: {ms:.1f} ms/run, {fps:.1f} FPS")


def main():
    args = parse_args()
    model_path = Path(args.model_path)
    if not (model_path / "saved_model.pb").exists():
        print(f"❌ 不是 SavedModel: {model_path}")
        sys.exit(1)
    if args.output is None:
        args.output = f"outputs/{model_path.name}_{args.format}"
    output_dir = Path(args.output)
    log_path = PROJECT_ROOT / "logs" / f"export_odapi_{model_path.name}"
    logger = get_logger("export_odapi", log_path)
    logger.info(f"模型: {model_path}")
    logger.info(f"格式: {args.format}, 量化: {args.quantize}")
    logger.info(f"输出: {output_dir}")

    if args.format == "onnx":
        onnx_path = export_onnx(model_path, output_dir, args.opset, logger)
        logger.info("\n[BENCH] ONNX CPU 推理:")
        try:
            # 从 saved_model 读 input shape
            sm = tf.saved_model.load(str(model_path))
            sig = sm.signatures["serving_default"]
            input_spec = sig.structured_input_signature[1]
            input_shape = list(input_spec.values())[0].shape.as_list()
            # 替换 None 为 640
            input_shape = [640 if s is None else s for s in input_shape]
            input_shape = tuple(input_shape)
            benchmark_onnx(onnx_path, input_shape, logger)
        except Exception as e:
            logger.warning(f"ONNX benchmark 失败: {e}")
    elif args.format == "tflite":
        tflite_path = export_tflite(model_path, output_dir, args.quantize, logger)
        if tflite_path:
            logger.info("\n[BENCH] TFLite CPU 推理:")
            try:
                benchmark_tflite(tflite_path, None, logger)
            except Exception as e:
                logger.warning(f"TFLite benchmark 失败: {e}")
    elif args.format == "savedmodel":
        # 已经是 SavedModel，复制到 output
        output_dir.mkdir(parents=True, exist_ok=True)
        saved_model_dir = output_dir / "saved_model"
        if saved_model_dir.exists():
            shutil.rmtree(saved_model_dir)
        shutil.copytree(model_path / "saved_model", saved_model_dir)
        size_mb = sum(f.stat().st_size for f in saved_model_dir.rglob("*")) / (1024 * 1024)
        logger.info(f"✅ SavedModel 复制完成: {saved_model_dir} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()