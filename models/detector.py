#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检测器工厂与统一接口

提供：
  - build_detection_head(): 构建 FPN + 分类/回归头
  - build_detector():         根据 MODEL_MODE 构建完整检测器
  - DetectionModel:           统一封装（训练/推理接口）
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List
import tensorflow as tf
from tensorflow.keras import layers, Model

import sys
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

import config
from models.backbones import (
    build_ssd_mobilenet_backbone,
    build_efficientdet_backbone,
    get_recommended_input_size,
)
from models.anchors import AnchorGenerator
from models.losses import DetectionLoss


# ============================================================================
# FPN（Feature Pyramid Network）
# ============================================================================
def _fpn_block(x, out_channels: int, name: str):
    """单层 FPN：1x1 conv 调整通道 + 3x3 conv 平滑"""
    x = layers.Conv2D(out_channels, 1, padding="same", name=f"{name}_lat")(x)
    x = layers.BatchNormalization(name=f"{name}_lat_bn")(x)
    x = layers.Activation("relu", name=f"{name}_lat_relu")(x)
    x = layers.Conv2D(out_channels, 3, padding="same", name=f"{name}_smooth")(x)
    x = layers.BatchNormalization(name=f"{name}_smooth_bn")(x)
    x = layers.Activation("relu", name=f"{name}_smooth_relu")(x)
    return x


def build_fpn(feature_maps: Dict[str, tf.Tensor], out_channels: int = 256, num_levels: int = 5) -> Dict[str, tf.Tensor]:
    """
    自顶向下 FPN

    Args:
        feature_maps: dict {level: tensor}  来自 backbone（通道数可能不一致）
        out_channels: FPN 输出通道数
        num_levels:   FPN 层数（默认 5 = P3~P7）
    """
    levels = sorted(feature_maps.keys())  # ['P3', 'P4', 'P5', ...]

    # 1. 补 P6/P7（如果 backbone 输出层级不够）
    if num_levels != len(levels):
        x = feature_maps[levels[-1]]
        while len(levels) < num_levels:
            new_level = f"P{int(levels[-1][1:]) + 1}"
            x = layers.Conv2D(out_channels, 3, strides=2, padding="same",
                              name=f"{new_level}_downsample")(x)
            x = layers.BatchNormalization(name=f"{new_level}_bn")(x)
            x = layers.Activation("relu", name=f"{new_level}_relu")(x)
            feature_maps[new_level] = x
            levels.append(new_level)
            if len(levels) >= num_levels:
                break

    # 2. 横向连接：1x1 conv 把所有 backbone 特征通道数对齐到 out_channels
    #    （这是修复 Bug 的关键步骤！）
    lateral_features = {}
    for level in levels:
        lat = layers.Conv2D(
            out_channels, 1, padding="same",
            name=f"fpn_lat_{level}"
        )(feature_maps[level])
        lat = layers.BatchNormalization(name=f"fpn_lat_{level}_bn")(lat)
        lat = layers.Activation("relu", name=f"fpn_lat_{level}_relu")(lat)
        lateral_features[level] = lat

    # 3. 自顶向下融合
    fpn_outputs = {}
    # P6/P7 是向下采样产生的终端层，不参与 top-down 合并
    merge_levels = [l for l in levels if int(l[1:]) <= 5]
    top = merge_levels[-1]  # 顶层 = merge_levels 中最深层（如 P5）
    # 顶层直接用 lateral 特征，再 3x3 平滑
    fpn_outputs[top] = layers.Conv2D(
        out_channels, 3, padding="same",
        name=f"fpn_{top}_smooth"
    )(lateral_features[top])
    fpn_outputs[top] = layers.BatchNormalization(
        name=f"fpn_{top}_smooth_bn"
    )(fpn_outputs[top])
    fpn_outputs[top] = layers.Activation(
        "relu", name=f"fpn_{top}_smooth_relu"
    )(fpn_outputs[top])

    # 4. 其它层：上采样 + 1x1 conv 调整通道 + 横向相加 + 3x3 平滑
    for i in range(len(merge_levels) - 2, -1, -1):
        cur_level = merge_levels[i]
        next_level = merge_levels[i + 1]
        # 上采样
        upsampled = layers.UpSampling2D(
            size=2, name=f"fpn_up_{cur_level}"
        )(fpn_outputs[next_level])
        # 1x1 conv 把上采样特征的通道对齐到当前层 lateral 的通道
        lat_ch = lateral_features[cur_level].shape[-1]
        aligned = layers.Conv2D(
            lat_ch, 1, padding="same",
            name=f"fpn_align_{cur_level}"
        )(upsampled)
        aligned = layers.BatchNormalization(name=f"fpn_align_{cur_level}_bn")(aligned)
        aligned = layers.Activation("relu", name=f"fpn_align_{cur_level}_relu")(aligned)
        # 横向相加
        merged = layers.Add(name=f"fpn_merge_{cur_level}")(
            [aligned, lateral_features[cur_level]]
        )
        # 3x3 平滑
        smoothed = layers.Conv2D(
            out_channels, 3, padding="same",
            name=f"fpn_{cur_level}_smooth"
        )(merged)
        smoothed = layers.BatchNormalization(
            name=f"fpn_{cur_level}_smooth_bn"
        )(smoothed)
        smoothed = layers.Activation(
            "relu", name=f"fpn_{cur_level}_smooth_relu"
        )(smoothed)
        fpn_outputs[cur_level] = smoothed

    # 5. P6/P7：终端层，不合并，只做 3x3 平滑后直接输出
    for level in levels:
        if int(level[1:]) >= 6 and level not in fpn_outputs:
            smoothed = layers.Conv2D(
                out_channels, 3, padding="same",
                name=f"fpn_{level}_smooth"
            )(lateral_features[level])
            smoothed = layers.BatchNormalization(name=f"fpn_{level}_smooth_bn")(smoothed)
            smoothed = layers.Activation("relu", name=f"fpn_{level}_smooth_relu")(smoothed)
            fpn_outputs[level] = smoothed

    # ★ 关键修复：按 P3→P7 字母序排序返回 dict，确保训练和推理时拼接顺序一致
    # （修复前是插入序 P5, P4, P3, P6, P7，与 sorted 顺序 P3, P4, P5, P6, P7 错位）
    fpn_outputs = {k: fpn_outputs[k] for k in sorted(fpn_outputs.keys())}
    return fpn_outputs


# ============================================================================
# 检测头
# ============================================================================
def build_detection_head(
    feature_maps: Dict[str, tf.Tensor],
    num_anchors: int,
    num_classes: int,
    name_prefix: str = "head",
):
    """
    构建分类 + 回归检测头

    Args:
        feature_maps: FPN 输出 {level: tensor}
        num_anchors:  每格 anchor 数
        num_classes:  类别数（不含背景）

    Returns:
        {
            "cls_logits": dict {level: (B, H, W, num_anchors * (num_classes+1))}
            "box_deltas": dict {level: (B, H, W, num_anchors * 4)}
        }
    """
    cls_outputs = {}
    box_outputs = {}
    for level, fmap in feature_maps.items():
        # 共享特征（增强：3 层卷积 + BN）
        x = layers.Conv2D(256, 3, padding="same",
                          name=f"{name_prefix}_{level}_shared1")(fmap)
        x = layers.BatchNormalization(name=f"{name_prefix}_{level}_shared1_bn")(x)
        x = layers.Activation("relu", name=f"{name_prefix}_{level}_shared1_relu")(x)

        x = layers.Conv2D(256, 3, padding="same",
                          name=f"{name_prefix}_{level}_shared2")(x)
        x = layers.BatchNormalization(name=f"{name_prefix}_{level}_shared2_bn")(x)
        x = layers.Activation("relu", name=f"{name_prefix}_{level}_shared2_relu")(x)

        x = layers.Conv2D(256, 3, padding="same",
                          name=f"{name_prefix}_{level}_shared3")(x)
        x = layers.BatchNormalization(name=f"{name_prefix}_{level}_shared3_bn")(x)
        x = layers.Activation("relu", name=f"{name_prefix}_{level}_shared3_relu")(x)

        # 分类分支（2 层：中间层 + 输出层）
        cls_x = layers.Conv2D(256, 3, padding="same",
                              name=f"{name_prefix}_{level}_cls_hidden")(x)
        cls_x = layers.BatchNormalization(name=f"{name_prefix}_{level}_cls_hidden_bn")(cls_x)
        cls_x = layers.Activation("relu", name=f"{name_prefix}_{level}_cls_hidden_relu")(cls_x)
        # 分类 (num_classes+1 含背景)
        cls = layers.Conv2D(num_anchors * (num_classes + 1), 1, padding="same",
                            name=f"{name_prefix}_{level}_cls")(cls_x)

        # 回归分支（2 层：中间层 + 输出层）
        box_x = layers.Conv2D(256, 3, padding="same",
                               name=f"{name_prefix}_{level}_box_hidden")(x)
        box_x = layers.BatchNormalization(name=f"{name_prefix}_{level}_box_hidden_bn")(box_x)
        box_x = layers.Activation("relu", name=f"{name_prefix}_{level}_box_hidden_relu")(box_x)
        # 回归
        box = layers.Conv2D(num_anchors * 4, 1, padding="same",
                            name=f"{name_prefix}_{level}_box")(box_x)
        cls_outputs[level] = cls
        box_outputs[level] = box
    return {"cls_logits": cls_outputs, "box_deltas": box_outputs}


# ============================================================================
# Backbone + FPN + Head 拼接
# ============================================================================
def build_detector(
    model_mode: str = None,
    num_classes: int = None,
    input_size: int = None,
    fpn_channels: int = 256,
    num_fpn_levels: int = 5,
    weights: str = "imagenet",
    weights_path: str = None,
) -> tuple:
    """
    工厂函数：构建完整检测器

    Args:
        model_mode:   "speed" / "balanced" / "accuracy"
        num_classes:  类别数（不含背景）
        input_size:   输入尺寸（None 则从 config 取）
        fpn_channels: FPN 输出通道数
        num_fpn_levels: FPN 层数
        weights:      "imagenet" | None（传给 backbone）
        weights_path: 本地预训练权重文件路径（离线环境用）

    Returns:
        (detector_model, feature_spec)
        detector_model: tf.keras.Model，输入 image, 输出 {cls_logits, box_deltas}
        feature_spec:   dict {level: (feature_h, feature_w)}
    """
    if model_mode is None:
        model_mode = config.MODEL_MODE
    if num_classes is None:
        num_classes = config.NUM_CLASSES
    if input_size is None:
        input_size = config.INPUT_SIZE

    # 1. Backbone
    if model_mode == "speed":
        base, fmap_outputs, layer_names = build_ssd_mobilenet_backbone(
            input_size, weights=weights, weights_path=weights_path
        )
    elif model_mode in ("balanced", "accuracy"):
        # balanced = B4, accuracy = B7
        bn = "B4" if model_mode == "balanced" else "B7"
        base, fmap_outputs, layer_names = build_efficientdet_backbone(
            bn, input_size, weights=weights, weights_path=weights_path
        )
    else:
        raise ValueError(f"未知 model_mode: {model_mode}")

    # 2. FPN（直接接 fmap_outputs 符号张量，不需要调用 backbone）
    # === DEBUG: 打印各 backbone 特征图实际尺寸 ===
    print(f"[DEBUG] Backbone output shapes (input={input_size}x{input_size}):")
    for level, tensor in fmap_outputs.items():
        print(f"  {level}: {tensor.shape}")
    # === DEBUG END ===
    fpn_outputs = build_fpn(fmap_outputs, out_channels=fpn_channels, num_levels=num_fpn_levels)

    # 3. 检测头
    # 暂定每格 9 个 anchor（3 ratios × 3 scales，SSD 默认）
    num_anchors = 9
    head_outputs = build_detection_head(
        fpn_outputs, num_anchors=num_anchors, num_classes=num_classes
    )

    # 4. 构建完整 Model
    image_input = base.input
    cls_logits = head_outputs["cls_logits"]
    box_deltas = head_outputs["box_deltas"]

    # 输出结构化为 dict（每层单独键）
    outputs = {}
    for level in cls_logits:
        outputs[f"cls_{level}"] = cls_logits[level]
        outputs[f"box_{level}"] = box_deltas[level]

    detector = Model(
        inputs=image_input,
        outputs=outputs,
        name=f"detector_{model_mode}",
    )

    # 特征图规格
    feature_spec = {
        level: tuple(fmap.shape[1:3])
        for level, fmap in fpn_outputs.items()
    }

    return detector, feature_spec


# ============================================================================
# Anchor 生成（与 backbone 同步）
# ============================================================================
def build_anchors_for_detector(
    feature_spec: Dict[str, tuple],
    input_size: int,
    base_sizes: Dict[str, float] = None,
    ratios: List[float] = (0.5, 1.0, 2.0),
) -> "AnchorGenerator":
    """
    为检测器构建 anchor 生成器

    Args:
        feature_spec: {level: (H, W)}  来自 build_detector
        input_size:   输入尺寸
        base_sizes:   每层基础 anchor 大小（归一化）
        ratios:       宽高比

    Returns:
        AnchorGenerator 实例
    """
    if base_sizes is None:
        # 默认按特征图大小线性递减
        levels = sorted(feature_spec.keys())
        max_size = max(feature_spec[levels[0]])
        base_sizes = {}
        for lvl in levels:
            h, w = feature_spec[lvl]
            # 越小特征图 → 越大 anchor
            base_sizes[lvl] = 0.1 * (max_size / h)
            base_sizes[lvl] = min(base_sizes[lvl], 0.9)  # 截断

    gen = AnchorGenerator(
        feature_sizes={lvl: spec[0] for lvl, spec in feature_spec.items()},
        image_size=input_size,
        base_sizes=base_sizes,
        ratios=list(ratios),
    )
    return gen


# ============================================================================
# 统一检测器封装（训练/推理接口）
# ============================================================================
class DetectionModel(tf.keras.Model):
    """
    检测器统一封装（继承 tf.keras.Model）

    用法：
        # 训练
        det = DetectionModel()
        det.compile(optimizer=tf.keras.optimizers.Adam(1e-4))
        det.fit(train_ds, validation_data=val_ds, epochs=20)

        # 推理
        outputs = det(images)  # 或 det.predict(image_batch)
    """

    def __init__(
        self,
        model_mode: str = None,
        num_classes: int = None,
        input_size: int = None,
        weights: str = "imagenet",
        weights_path: str = None,
    ):
        # 先调父类 __init__，暂不传 inputs（Functional 构造稍后在 call 里拼装）
        super().__init__(name="detection_model")

        self.model_mode = model_mode or config.MODEL_MODE
        self.num_classes = num_classes or config.NUM_CLASSES
        # 如果调用方传了 input_size，则覆盖 config 默认值
        self.input_size = input_size if input_size is not None else config.INPUT_SIZE

        # 构建网络（detector 是一个 Functional 子模型，输出是 dict）
        self.model, self.feature_spec = build_detector(
            model_mode=self.model_mode,
            num_classes=self.num_classes,
            input_size=self.input_size,
            weights=weights,
            weights_path=weights_path,
        )

        # 构建 anchor 生成器
        self.anchor_gen = build_anchors_for_detector(self.feature_spec, self.input_size)
        self.anchors = self.anchor_gen.generate_all()  # (N, 4) numpy
        self.num_anchors = len(self.anchors)

        # 损失函数
        self.detection_loss = DetectionLoss(num_classes=self.num_classes)

        # Anchor 转为 tensor（用于 train_step 里的赋值计算）
        self.anchors_tf = tf.constant(self.anchors, dtype=tf.float32)

        print(f"[DetectionModel] mode={self.model_mode}, "
              f"num_classes={self.num_classes}, "
              f"anchors={self.num_anchors}, "
              f"input={self.input_size}x{self.input_size}")

    # ------------------------------------------------------------------
    # Keras 接口
    # ------------------------------------------------------------------
    def call(self, inputs, training=False):
        """前向传播：返回 dict {cls_P3, box_P3, cls_P4, box_P4, ...}"""
        return self.model(inputs, training=training)

    def compile(self, optimizer=None, **kwargs):
        """覆盖以不传 loss（损失在 train_step 里手动算）"""
        if optimizer is None:
            optimizer = tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE)
        # 不传 loss / metrics（我们在 train_step 里手算损失并返回）
        super().compile(optimizer=optimizer, **kwargs)

    def summary(self, **kwargs):
        return self.model.summary(**kwargs)

    def save_weights(self, path: str, **kwargs):
        """保存权重（保存为子模型 self.model 的格式，与预测脚本兼容）"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.model.save_weights(path, **kwargs)

    def load_weights(self, path: str):
        self.model.load_weights(path)

    def predict(self, image_batch, verbose=0):
        """推理：image_batch (B, H, W, 3) 归一化"""
        return self.model.predict(image_batch, verbose=verbose)

    # ------------------------------------------------------------------
    # 自定义训练循环（处理 anchor assignment + 多输出 loss）
    # ------------------------------------------------------------------
    def _assign_anchors_batch(self, gt_boxes, gt_classes):
        """
        对 batch 中每张图进行 anchor assignment

        Args:
            gt_boxes:   (B, MAX_BOXES, 4) 归一化 [cx, cy, w, h]，零填充表示无效
            gt_classes: (B, MAX_BOXES)    类别 id，0 表示无效
        Returns:
            cls_targets: (B, N_anchors)   -1=忽略, 0=背景, 1..C=前景
            box_targets: (B, N_anchors, 4) 编码后的 deltas
            num_pos:     (B,)             每张图的正样本数
        """
        from models.anchors import assign_anchors_to_gt

        B = tf.shape(gt_boxes)[0]

        # 走 tf.map_fn 避开 unstack 在动态 batch 上的限制
        def _per_image(args):
            boxes, classes = args
            valid_mask = boxes[:, 2] > 1e-6
            valid_boxes = tf.boolean_mask(boxes, valid_mask)
            valid_classes = tf.boolean_mask(classes, valid_mask)
            cls_t, box_t = assign_anchors_to_gt(
                self.anchors_tf, valid_boxes, valid_classes
            )
            num_pos = tf.reduce_sum(tf.cast(cls_t > 0, tf.int32))
            return cls_t, box_t, num_pos

        cls_targets, box_targets, num_pos = tf.map_fn(
            _per_image,
            elems=(gt_boxes, gt_classes),
            fn_output_signature=(tf.int32, tf.float32, tf.int32),
            parallel_iterations=8,
        )
        return cls_targets, box_targets, num_pos

    def _compute_loss(self, predictions, labels):
        """
        损失计算：anchor assignment + 多输出 focal + smooth L1
        """
        gt_boxes = labels["boxes"]    # (B, MAX, 4)
        gt_classes = labels["classes"]  # (B, MAX)

        # 1. 给每个 anchor 分配目标
        cls_targets, box_targets, num_pos = self._assign_anchors_batch(gt_boxes, gt_classes)

        # 2. 把 predictions 重组为 {cls_logits, box_deltas} 字典
        cls_logits_dict = {}
        box_deltas_dict = {}
        for level in self.feature_spec:
            cls_logits_dict[level] = predictions[f"cls_{level}"]
            box_deltas_dict[level] = predictions[f"box_{level}"]

        # 3. 每层 reshape: (B, H, W, A*C) -> (B, H*W*A, C)，再拼
        # 直接从 tensor shape 提取 H, W, 避免 feature_spec 与实际不符
        cls_flat, box_flat = [], []
        for level in cls_logits_dict:
            t_cls = cls_logits_dict[level]       # (B, H, W, A*(C+1))
            t_box = box_deltas_dict[level]        # (B, H, W, A*4)
            B = tf.shape(t_cls)[0]
            H = tf.shape(t_cls)[1]
            W = tf.shape(t_cls)[2]
            cls_c = tf.shape(t_cls)[3]            # A*(C+1)
            box_c = tf.shape(t_box)[3]            # A*4
            n_anchors = box_c // 4               # 每格 anchor 数

            # cls: (B,H,W,A,C) 再 flatten 到 (B, H*W*A, C)
            cls_l = tf.reshape(t_cls, [B, H, W, n_anchors, -1])
            cls_l = tf.reshape(cls_l, [B, H * W * n_anchors, -1])
            # box: (B,H,W,A,4) flatten 到 (B, H*W*A, 4)
            box_l = tf.reshape(t_box, [B, H, W, n_anchors, 4])
            box_l = tf.reshape(box_l, [B, H * W * n_anchors, 4])
            cls_flat.append(cls_l)
            box_flat.append(box_l)

        cls_logits = tf.concat(cls_flat, axis=1)   # (B, N_total, C+1)
        box_deltas = tf.concat(box_flat, axis=1)  # (B, N_total, 4)

        # 4. 调损失函数
        loss_dict = self.detection_loss(
            y_true={
                "cls_targets":   cls_targets,
                "box_targets":   box_targets,
                "num_positives": num_pos,
            },
            y_pred={
                "cls_logits": cls_logits,
                "box_deltas": box_deltas,
            },
        )
        return loss_dict

    def train_step(self, data):
        """重写：手动算 loss 和梯度"""
        images, labels = data

        with tf.GradientTape() as tape:
            predictions = self(images, training=True)
            loss_dict = self._compute_loss(predictions, labels)
            total_loss = loss_dict["total"]

        grads = tape.gradient(total_loss, self.model.trainable_variables)
        # 跳过 None 梯度（冻结的变量）
        grads_and_vars = [
            (g, v) for g, v in zip(grads, self.model.trainable_variables) if g is not None
        ]
        self.optimizer.apply_gradients(grads_and_vars)

        return {
            "loss":    loss_dict["total"],
            "cls_loss": loss_dict["cls"],
            "box_loss": loss_dict["box"],
        }

    def test_step(self, data):
        """重写：验证 step（不需梯度）"""
        images, labels = data
        predictions = self(images, training=False)
        loss_dict = self._compute_loss(predictions, labels)
        return {
            "loss":    loss_dict["total"],
            "cls_loss": loss_dict["cls"],
            "box_loss": loss_dict["box"],
        }


if __name__ == "__main__":
    print("=== 构建检测器（仅模型定义，不实际加载 ImageNet 权重）===")
    print(f"模型档位: {config.MODEL_MODE}")
    print(f"输入尺寸: {config.INPUT_SIZE}")
    print(f"类别数:   {config.NUM_CLASSES}")
    print("\nBackbone + FPN + Head 结构:")
    det, spec = build_detector()
    print(f"Feature spec: {spec}")
    print(f"Output keys: {[k for k in det.output.keys()]}")
