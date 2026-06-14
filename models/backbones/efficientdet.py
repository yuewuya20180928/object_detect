#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EfficientDet 风格 Backbone（基于 EfficientNet V1）

EfficientDet-D0/D4/D7 分别对应 EfficientNet-B0/B4/B7
输出 5 个尺度特征图 P3~P7
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import (
    EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3,
    EfficientNetB4, EfficientNetB5, EfficientNetB6, EfficientNetB7,
)


_EFNET_MAP = {
    "B0": EfficientNetB0, "B1": EfficientNetB1, "B2": EfficientNetB2,
    "B3": EfficientNetB3, "B4": EfficientNetB4, "B5": EfficientNetB5,
    "B6": EfficientNetB6, "B7": EfficientNetB7,
}

# EfficientNet 各 stage 输出对应的特征层名
#
# 关键：相邻 stage 必须是 2x 关系（让 FPN 上采样正常工作）
# 实际尺寸（input=512，B4 为例）：
#   block2a = 256, block3a = 128, block4a = 64, block5a = 32, block6a = 16
# 全部 2x 关系，5 个 stage 选 4 个，剩下 P7 由 FPN 下采样生成
#
# 不要选 ["block2a", "block3a", "block5a", "block7a"]：
#   这样 P3=256, P4=128, P5=32, P6=16
#   P4→P5 是 4x 跳变，FPN 上采样一次不够
_STAGE_NAMES = {
    "B0": ["block2a_expand_activation", "block3a_expand_activation",
           "block4a_expand_activation", "block5a_expand_activation"],
    "B1": ["block2a_expand_activation", "block3a_expand_activation",
           "block4a_expand_activation", "block5a_expand_activation"],
    "B2": ["block2a_expand_activation", "block3a_expand_activation",
           "block4a_expand_activation", "block5a_expand_activation"],
    "B3": ["block2a_expand_activation", "block3a_expand_activation",
           "block4a_expand_activation", "block5a_expand_activation"],
    "B4": ["block2a_expand_activation", "block3a_expand_activation",
           "block4a_expand_activation", "block5a_expand_activation"],
    "B5": ["block2a_expand_activation", "block3a_expand_activation",
           "block4a_expand_activation", "block5a_expand_activation"],
    "B6": ["block2a_expand_activation", "block3a_expand_activation",
           "block4a_expand_activation", "block5a_expand_activation"],
    "B7": ["block2a_expand_activation", "block3a_expand_activation",
           "block4a_expand_activation", "block5a_expand_activation"],
}


def build_efficientdet_backbone(
    model_name: str = "B4",
    input_size: int = 512,
    weights: str = "imagenet",
    weights_path: str = None,
):
    """
    构建 EfficientDet Backbone

    Args:
        model_name:    "B0"~"B7"（D0~D7 配置）
        input_size:    输入尺寸
        weights:       "imagenet" | None
                       - "imagenet"：自动从 keras 下载 ImageNet 权重（需联网）
                       - None：不加载预训练权重（随机初始化）
        weights_path:  本地预训练权重文件路径（.h5）
                       - 优先级高于 weights
                       - None：使用 weights 参数

    Returns:
        (base_model, fmap_outputs, feature_layer_names)
        base_model:       tf.keras.Model（EfficientNet 本身）
        fmap_outputs:     dict {level: KerasTensor} 符号张量，可直接接 FPN
        feature_layer_names: dict {level: layer_name_str}
    """
    if model_name not in _EFNET_MAP:
        raise ValueError(f"不支持的 EfficientNet: {model_name}, 可选: {list(_EFNET_MAP.keys())}")

    # 加载预训练 EfficientNet
    if weights_path is not None:
        # 优先用本地权重（离线环境）
        from pathlib import Path
        weights_path = Path(weights_path).expanduser()
        if not weights_path.exists():
            raise FileNotFoundError(
                f"未找到本地权重文件: {weights_path}\n"
                f"请下载 EfficientNetB{model_name[1]}_weights.h5 放到该路径，"
                f"或设置 weights='imagenet' 联网下载"
            )
        print(f"[EfficientDet] 加载本地权重: {weights_path}")
        base = _EFNET_MAP[model_name](
            input_shape=(input_size, input_size, 3),
            include_top=False,
            weights=None,  # 不传 imagenet
        )
        base.load_weights(str(weights_path))
    else:
        # 走 keras 默认逻辑（联网下载 imagenet 权重）
        base = _EFNET_MAP[model_name](
            input_shape=(input_size, input_size, 3),
            include_top=False,
            weights=weights,
        )
    base.trainable = False  # 迁移学习第一阶段全冻结

    # 抽取多尺度特征（返回 KerasTensor，不调用模型）
    stage_names = _STAGE_NAMES[model_name]
    fmap_outputs = {f"P{i+3}": base.get_layer(name).output
                    for i, name in enumerate(stage_names)}

    return base, fmap_outputs, {f"P{i+3}": name for i, name in enumerate(stage_names)}


def get_recommended_input_size(model_name: str) -> int:
    """获取各 EfficientDet 配置的推荐输入尺寸"""
    return {
        "B0": 512, "B1": 640, "B2": 768, "B3": 896,
        "B4": 1024, "B5": 1280, "B6": 1280, "B7": 1536,
    }.get(model_name, 512)


if __name__ == "__main__":
    for name in ["B0", "B4"]:
        print(f"\n=== EfficientNet-{name} ===")
        size = get_recommended_input_size(name)
        bb, names = build_efficientdet_backbone(name, size)
        print(f"Input: {size}x{size}")
        print(f"Feature layers: {names}")
