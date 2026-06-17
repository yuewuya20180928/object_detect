#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检测损失函数

提供：
  - FocalLoss:    分类损失（处理类别不平衡）
  - SmoothL1Loss: 回归损失（bbox 偏移）
  - DetectionLoss: 总损失 = α * Focal + β * SmoothL1
"""

import tensorflow as tf
from tensorflow.keras import losses as keras_losses


# ============================================================================
# Focal Loss（分类）
# ============================================================================
class FocalLoss(keras_losses.Loss):
    """
    Focal Loss for Dense Object Detection
    论文: https://arxiv.org/abs/1708.02002

    用于解决正负样本严重不平衡问题（背景 anchor 远多于前景）
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, name: str = "focal_loss"):
        super().__init__(name=name, reduction=tf.keras.losses.Reduction.NONE)
        self.alpha = alpha
        self.gamma = gamma

    def call(self, y_true, y_pred_logits):
        """
        Args:
            y_true:        (N,) 0/1 或 (N, C) one-hot multi-class
            y_pred_logits: (N,) sigmoid logit 或 (N, C) softmax logits
        """
        y_true = tf.cast(y_true, tf.float32)
        rank = len(y_true.shape)

        if rank == 1:
            # Binary sigmoid focal（原有逻辑）
            p_fg = tf.sigmoid(y_pred_logits)
            p_bg = 1.0 - p_fg
            pt = y_true * p_fg + (1.0 - y_true) * p_bg
            at = y_true * self.alpha + (1.0 - y_true) * (1.0 - self.alpha)
            focal_weight = tf.pow(1.0 - pt, self.gamma)
            ce = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=y_true, logits=y_pred_logits
            )
            loss = at * focal_weight * ce
        else:
            # Multi-class softmax focal
            # y_true: (N, C) one-hot, y_pred_logits: (N, C) logits
            # softmax cross entropy per sample: (N,)
            ce_per_sample = tf.nn.softmax_cross_entropy_with_logits(
                labels=y_true, logits=y_pred_logits
            )  # (N,)
            # pt = 前景类的 softmax 概率
            p = tf.nn.softmax(y_pred_logits, axis=-1)          # (N, C)
            pt = tf.reduce_sum(p * y_true, axis=-1)             # (N,)
            # focal weight
            focal_weight = tf.pow(1.0 - pt, self.gamma)
            # alpha mask（只对前景加权）
            fg_label = tf.reduce_sum(y_true, axis=-1)           # (N,) 1=fg, 0=bg
            at = fg_label * self.alpha + (1.0 - fg_label) * (1.0 - self.alpha)
            loss = at * focal_weight * ce_per_sample           # (N,)

        return tf.reduce_mean(loss)

    def get_config(self):
        config = super().get_config()
        config.update({"alpha": self.alpha, "gamma": self.gamma})
        return config


# ============================================================================
# Smooth L1 Loss（回归）
# ============================================================================
class SmoothL1Loss(keras_losses.Loss):
    """
    Smooth L1 / Huber Loss
    比 L2 鲁棒，比 L1 在零点平滑
    """

    def __init__(self, sigma: float = 1.0, name: str = "smooth_l1"):
        super().__init__(name=name, reduction=tf.keras.losses.Reduction.NONE)
        self.sigma = sigma

    def call(self, y_true, y_pred):
        """
        Args:
            y_true: (N, 4) 目标 (cx, cy, w, h) 归一化
            y_pred: (N, 4) 预测 (cx, cy, w, h) 归一化
        """
        diff = y_pred - y_true
        abs_diff = tf.abs(diff)
        # Smooth L1 公式
        sigma_sq = self.sigma ** 2
        smooth_l1 = tf.where(
            abs_diff < (1.0 / sigma_sq),
            0.5 * sigma_sq * tf.square(diff),
            abs_diff - 0.5 / sigma_sq
        )
        return tf.reduce_mean(smooth_l1)

    def get_config(self):
        config = super().get_config()
        config.update({"sigma": self.sigma})
        return config


# ============================================================================
# 总损失
# ============================================================================
class DetectionLoss:
    """
    检测器总损失

    用法：
        loss_fn = DetectionLoss(num_classes=80)
        loss_dict = loss_fn(y_true_dict, y_pred_dict)
        total_loss = loss_dict["total"]
    """

    def __init__(
        self,
        num_classes: int,
        cls_weight: float = 1.0,
        box_weight: float = 1.0,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,         # 06-17 22:56 改回 2.0（retrain v4 用 1.5 卡 cls_loss 0.73；早 30 epoch 训练用 2.0 健康）
        neg_pos_ratio: float = 3.0,        # 06-17 22:56 改回 3.0（retrain v4 用 1.0 破坏 OHEM）
        min_neg_per_image: int = 16,       # 每图最少保留负样本数（防 0 正样本图全空）
    ):
        self.num_classes = num_classes
        self.cls_weight = cls_weight
        self.box_weight = box_weight
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.neg_pos_ratio = neg_pos_ratio
        self.min_neg_per_image = min_neg_per_image
        # 保留 focal_loss 实例（供外部兼容/单元测试）
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.box_loss = SmoothL1Loss()

    def __call__(self, y_true, y_pred):
        """
        Args:
            y_true: {
                "cls_targets": (B, N) int class id, -1=ignore
                "box_targets": (B, N, 4)
                "num_positives": (B,) 正样本数
            }
            y_pred: {
                "cls_logits": (B, N, num_classes+1)
                "box_deltas": (B, N, 4)
            }

        Returns:
            {"total": scalar, "cls": scalar, "box": scalar}
        """
        cls_targets = y_true["cls_targets"]   # (B, N) int class id, -1=ignore
        box_targets = y_true["box_targets"]   # (B, N, 4)
        num_positives = y_true["num_positives"]  # (B,)

        cls_logits = y_pred["cls_logits"]     # (B, N, C+1)
        box_deltas = y_pred["box_deltas"]     # (B, N, 4)

        # 统一到 float32（避免混精度 dtype 不匹配），最后 loss scalar 再转回模型 dtype
        target_dtype = tf.float32
        cls_logits = tf.cast(cls_logits, target_dtype)
        box_deltas = tf.cast(box_deltas, target_dtype)

        # ----- 分类损失（focal + OHEM）-----
        num_classes = self.num_classes
        C = num_classes + 1

        # 掩码
        ignore_mask = tf.cast(cls_targets == -1, tf.float32)   # (B, N)
        fg_mask = tf.cast(cls_targets > 0, tf.float32)         # (B, N)
        bg_mask = tf.cast(cls_targets == 0, tf.float32)         # (B, N)

        # one-hot（ignore 落 background 槽，后续 mask 会剔除）
        cls_targets_clamped = tf.maximum(cls_targets, 0)
        cls_targets_onehot = tf.one_hot(
            cls_targets_clamped, depth=C, dtype=tf.float32
        )  # (B, N, C)

        # ===== 计算 per-anchor focal loss（不 reduce）=====
        # softmax cross entropy per anchor: (B, N)
        ce_per_anchor = tf.nn.softmax_cross_entropy_with_logits(
            labels=cls_targets_onehot, logits=cls_logits
        )
        # pt: 真实类的 softmax 概率
        p = tf.nn.softmax(cls_logits, axis=-1)
        pt = tf.reduce_sum(p * cls_targets_onehot, axis=-1)         # (B, N)
        # focal weight
        focal_weight = tf.pow(1.0 - pt, self.focal_gamma)           # (B, N)
        # alpha mask
        fg_label = tf.reduce_sum(cls_targets_onehot, axis=-1)       # (B, N) 1=fg, 0=bg
        at = fg_label * self.focal_alpha + (1.0 - fg_label) * (1.0 - self.focal_alpha)
        cls_loss_per_anchor = at * focal_weight * ce_per_anchor     # (B, N)

        # ===== OHEM：对负样本按 loss 排序，取 top-K hard negative =====
        # K = max(num_pos * neg_pos_ratio, min_neg_per_image)
        neg_ratio = tf.cast(self.neg_pos_ratio, num_positives.dtype)
        num_neg_to_keep = tf.maximum(
            num_positives * neg_ratio,
            tf.cast(self.min_neg_per_image, num_positives.dtype)
        )  # (B,)
        num_neg_to_keep = tf.cast(num_neg_to_keep, tf.int32)

        def _ohem_per_image(args):
            """对单张图做 OHEM：保留 top-K 损失最大的负样本"""
            losses, num_neg_keep, fg_m, bg_m = args
            # losses: (N,) per-anchor cls loss
            # 仅对 bg 排序；fg / ignore 不参与排序
            # 把 fg/ignore 的 loss 设为 -1，确保它们不会被 top-K 选中
            sort_keys = tf.where(bg_m > 0, losses, tf.ones_like(losses) * -1.0)
            # 取 top num_neg_keep（多的填 0 loss，不影响结果）
            k = tf.minimum(num_neg_keep, tf.cast(tf.reduce_sum(bg_m), tf.int32))
            topk_vals, topk_idx = tf.math.top_k(sort_keys, k=tf.shape(losses)[0])
            # 前 k 个标记为 1，其余为 0
            range_idx = tf.range(tf.shape(losses)[0])
            bg_keep = tf.cast(range_idx < k, tf.float32)
            # scatter 回原位置
            bg_keep_scattered = tf.scatter_nd(
                topk_idx[:, tf.newaxis], bg_keep, tf.shape(losses)
            )
            return bg_keep_scattered

        bg_keep_mask = tf.map_fn(
            _ohem_per_image,
            elems=(cls_loss_per_anchor, num_neg_to_keep, fg_mask, bg_mask),
            fn_output_signature=tf.float32,
        )  # (B, N)

        # 最终 cls loss mask：所有 fg + 选中的 hard bg
        cls_keep_mask = fg_mask + bg_keep_mask   # (B, N)
        num_cls_keep = tf.maximum(tf.reduce_sum(cls_keep_mask), 1.0)
        cls_loss = tf.reduce_sum(cls_loss_per_anchor * cls_keep_mask) / num_cls_keep

        # ===== 回归损失（不变，仅对正样本）=====
        box_loss_per = tf.abs(box_deltas - box_targets)
        box_loss_per = tf.where(
            box_loss_per < 1.0,
            0.5 * tf.square(box_loss_per),
            box_loss_per - 0.5
        )
        box_loss_per = tf.reduce_sum(box_loss_per, axis=-1)  # (B, N)
        box_loss_per = box_loss_per * fg_mask
        num_pos = tf.maximum(tf.reduce_sum(fg_mask, axis=-1), 1.0)  # (B,)
        box_loss = tf.reduce_mean(tf.reduce_sum(box_loss_per, axis=-1) / num_pos)

        # ----- 总损失 -----
        total = self.cls_weight * cls_loss + self.box_weight * box_loss

        return {
            "total": total,
            "cls":   cls_loss,
            "box":   box_loss,
        }