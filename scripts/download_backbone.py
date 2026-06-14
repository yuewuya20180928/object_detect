#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
预训练 backbone 权重下载工具

用法：
    # 下载 balanced (EfficientNet-B4) 默认权重
    python scripts/download_backbone.py

    # 下载其他模型
    python scripts/download_backbone.py --model speed
    python scripts/download_backbone.py --model accuracy

下载路径：
    pretrained/keras/efficientnet{b0-b7}_notop.h5
    pretrained/keras/mobilenet_v2_weights.h5
"""

import os
import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

import config
from utils.logger import get_logger


def download_efficientnet(model_name: str, dest: Path, logger):
    """
    从 keras applications 触发下载并保存到指定路径

    Keras 不会直接暴露 .h5 路径，但可以通过 model.save_weights() 把 imagenet 权重保存为本地文件。
    """
    import tensorflow as tf
    from tensorflow.keras.applications import (
        EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3,
        EfficientNetB4, EfficientNetB5, EfficientNetB6, EfficientNetB7,
    )
    efnet_map = {
        "B0": EfficientNetB0, "B1": EfficientNetB1, "B2": EfficientNetB2,
        "B3": EfficientNetB3, "B4": EfficientNetB4, "B5": EfficientNetB5,
        "B6": EfficientNetB6, "B7": EfficientNetB7,
    }
    if model_name not in efnet_map:
        raise ValueError(f"不支持的 EfficientNet: {model_name}")

    logger.info(f"下载 EfficientNet-{model_name[1]} ImageNet 权重（首次需联网）...")
    model = efnet_map[model_name](
        input_shape=(None, None, 3),
        include_top=False,
        weights="imagenet",
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    model.save_weights(str(dest))
    logger.info(f"已保存: {dest} ({dest.stat().st_size / 1024 / 1024:.1f} MB)")


def download_mobilenetv2(dest: Path, logger):
    """下载 MobileNetV2 ImageNet 权重"""
    import tensorflow as tf
    from tensorflow.keras.applications import MobileNetV2
    logger.info("下载 MobileNetV2 ImageNet 权重（首次需联网）...")
    model = MobileNetV2(
        input_shape=(None, None, 3),
        include_top=False,
        weights="imagenet",
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    model.save_weights(str(dest))
    logger.info(f"已保存: {dest} ({dest.stat().st_size / 1024 / 1024:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description="下载 backbone ImageNet 权重")
    parser.add_argument(
        "--model",
        choices=["speed", "balanced", "accuracy", "all"],
        default="balanced",
        help="下载哪个模型的权重（all=全部下载）",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=config.PRETRAINED_DIR / "keras",
        help="保存目录",
    )
    args = parser.parse_args()

    logger = get_logger("download_backbone", args.out_dir.parent / "logs", verbose=False)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.model == "all":
        targets = ["speed", "balanced", "accuracy"]
    else:
        targets = [args.model]

    for mode in targets:
        if mode == "speed":
            dest = args.out_dir / "mobilenet_v2_weights.h5"
            download_mobilenetv2(dest, logger)
        elif mode == "balanced":
            dest = args.out_dir / "efficientnetb4_notop.h5"
            download_efficientnet("B4", dest, logger)
        elif mode == "accuracy":
            dest = args.out_dir / "efficientnetb7_notop.h5"
            download_efficientnet("B7", dest, logger)

    logger.info("=" * 50)
    logger.info("下载完成。下次训练将自动使用本地权重（无需联网）。")
    logger.info(f"权重目录: {args.out_dir}")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
