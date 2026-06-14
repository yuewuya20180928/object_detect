#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SSD MobileNetV2 风格 Backbone

输出 6 个尺度特征图（对应 SSD 的 extra layers）
P3~P7 + 一个 extra layer
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV2


def build_ssd_mobilenet_backbone(
    input_size: int = 320,
    weights: str = "imagenet",
    weights_path: str = None,
):
    """
    构建 SSD MobileNetV2 Backbone

    Args:
        input_size:   输入尺寸
        weights:      "imagenet" | None
                      - "imagenet"：自动从 keras 下载 ImageNet 权重（需联网）
                      - None：不加载预训练权重
        weights_path: 本地预训练权重文件路径（.h5），优先于 weights

    Returns:
        (base_model, fmap_outputs, feature_layers)
        base_model:    tf.keras.Model（MobileNetV2 本身）
        fmap_outputs:  dict {level: KerasTensor} 符号张量，可直接接 FPN
        feature_layers: dict {level: layer_name_str}
    """
    if weights_path is not None:
        from pathlib import Path
        weights_path = Path(weights_path).expanduser()
        if not weights_path.exists():
            raise FileNotFoundError(
                f"未找到本地权重文件: {weights_path}\n"
                f"请下载 mobilenet_v2_weights.h5 放到该路径，"
                f"或设置 weights='imagenet' 联网下载"
            )
        print(f"[SSD MobileNet] 加载本地权重: {weights_path}")
        base = MobileNetV2(
            input_shape=(input_size, input_size, 3),
            include_top=False,
            weights=None,
        )
        base.load_weights(str(weights_path))
    else:
        # 联网下载 ImageNet 预训练 MobileNetV2
        base = MobileNetV2(
            input_shape=(input_size, input_size, 3),
            include_top=False,
            weights=weights,
        )
    # 全部冻结（迁移学习第一阶段）
    base.trainable = False

    # SSD 标准特征层
    feature_layers = {
        "P3": "block_6_expand_relu",   # 20x20 (input 320)
        "P4": "block_13_expand_relu",  # 10x10
        "P5": "out_relu",              # 5x5  (顶层)
    }

    # 抽取多尺度特征（返回 KerasTensor）
    fmap_outputs = {lvl: base.get_layer(name).output for lvl, name in feature_layers.items()}

    # 额外的 SSD 层（继续下采样）— 这些都接受 KerasTensor 输入
    x = fmap_outputs["P5"]
    # P6: 3x3
    x = layers.Conv2D(256, 3, strides=2, padding="same", name="P6_conv1")(x)
    x = layers.BatchNormalization(name="P6_bn1")(x)
    x = layers.ReLU(6.0, name="P6_relu1")(x)
    fmap_outputs["P6"] = x
    # P7: 3x3
    x = layers.Conv2D(128, 3, strides=2, padding="same", name="P7_conv1")(x)
    x = layers.BatchNormalization(name="P7_bn1")(x)
    x = layers.ReLU(6.0, name="P7_relu1")(x)
    fmap_outputs["P7"] = x

    return base, fmap_outputs, feature_layers


if __name__ == "__main__":
    bb, layers_ = build_ssd_mobilenet_backbone(320)
    print("Feature layers:", layers_)
    for lvl, name in layers_.items():
        try:
            shape = bb.get_layer(name).output_shape
            print(f"  {lvl}: {shape}")
        except Exception:
            pass
