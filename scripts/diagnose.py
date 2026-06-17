#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练后诊断脚本

检查项目：
  1. 数据 pipeline 是否正确（解析 TFRecord，bbox 范围，类别分布）
  2. 模型前向输出是否合理（logit 范围，权重初始化）
  3. 训练后模型的预测分布（bg/fg 比例，box delta 范围）
  4. Anchor 顺序在训练和推理时是否一致
  5. Loss 数值是否合理（trained vs random init 对比）

用法：
    python scripts/diagnose.py
    python scripts/diagnose.py --ckpt checkpoints/speed_coco/best.weights.h5
    python scripts/diagnose.py --ckpt checkpoints/speed_coco/latest.weights.h5
"""

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import tensorflow as tf

import config
from data.dataset_builder import build_train_dataset, build_eval_dataset, parse_tfexample
from models.detector import DetectionModel
from models.postprocess import decode_predictions


# ============================================================================
# 1. 数据诊断
# ============================================================================
def diagnose_data():
    print("=" * 70)
    print("1. 数据诊断（解析 train.record 单条样本）")
    print("=" * 70)
    ds = tf.data.TFRecordDataset([str(config.TRAIN_RECORD)])
    for raw in ds.take(1):
        img, lbl = parse_tfexample(raw)
        print(f"  image shape:    {img.shape}, dtype={img.dtype}")
        print(f"  image min/max:  {tf.reduce_min(img).numpy():.3f} / {tf.reduce_max(img).numpy():.3f}")
        print(f"  boxes shape:    {lbl['boxes'].shape}")
        print(f"  classes shape:  {lbl['classes'].shape}")
        print(f"  orig shape:     {lbl['original_shape'].numpy()}")
        valid_mask = lbl['boxes'][:, 2] > 1e-6
        n_valid = tf.reduce_sum(tf.cast(valid_mask, tf.int32)).numpy()
        print(f"  valid boxes:    {n_valid}")
        if n_valid > 0:
            valid_boxes = lbl['boxes'][valid_mask].numpy()
            valid_classes = lbl['classes'][valid_mask].numpy()
            print(f"  sample boxes:   {valid_boxes[:3]}")
            print(f"  sample classes: {valid_classes[:3]} (range {valid_classes.min()}..{valid_classes.max()})")
            in_range = np.all((valid_boxes >= 0) & (valid_boxes <= 1))
            print(f"  all boxes in [0, 1]? {in_range}")
            if valid_classes.max() > config.NUM_CLASSES:
                print(f"  ⚠️ 警告：类别 ID 超过 NUM_CLASSES ({config.NUM_CLASSES})")

    # 80 张图的统计
    train_ds = build_train_dataset(batch_size=4, input_size=320, augment=False)
    class_counts = {}
    n_with_objects = 0
    n_empty = 0
    pos_anchor_counts = []
    for i, (imgs, lbls) in enumerate(train_ds.take(20)):
        for j in range(imgs.shape[0]):
            valid_mask = lbls['boxes'][j, :, 2] > 1e-6
            n_valid = int(tf.reduce_sum(tf.cast(valid_mask, tf.int32)).numpy())
            if n_valid == 0:
                n_empty += 1
            else:
                n_with_objects += 1
                pos_anchor_counts.append(n_valid)
                for cls_id in lbls['classes'][j][valid_mask].numpy():
                    class_counts[int(cls_id)] = class_counts.get(int(cls_id), 0) + 1

    print(f"\n  80 张图: 有目标={n_with_objects}, 无目标={n_empty}")
    if pos_anchor_counts:
        print(f"  每图目标数: min={min(pos_anchor_counts)}, max={max(pos_anchor_counts)}, "
              f"avg={np.mean(pos_anchor_counts):.1f}")
    print(f"  类别分布 top 10:")
    for cls_id, count in sorted(class_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"    class {cls_id:3d}: {count} 次")


# ============================================================================
# 2. 模型结构诊断
# ============================================================================
def diagnose_model(det):
    print()
    print("=" * 70)
    print("2. 模型结构诊断")
    print("=" * 70)
    print(f"  anchors shape: {det.anchors.shape}")
    print(f"  anchors cx in [0, 1]? {((det.anchors[:, 0] >= 0) & (det.anchors[:, 0] <= 1)).all()}")
    print(f"  anchors w range: [{det.anchors[:, 2].min():.4f}, {det.anchors[:, 2].max():.4f}]")
    print(f"  anchors h range: [{det.anchors[:, 3].min():.4f}, {det.anchors[:, 3].max():.4f}]")

    print(f"\n  feature_spec (训练时拼接顺序): {list(det.feature_spec.keys())}")
    print(f"  sorted(feature_spec) (推理时拼接顺序): {sorted(det.feature_spec.keys())}")
    if list(det.feature_spec.keys()) != sorted(det.feature_spec.keys()):
        print("  ⚠️ 警告：训练/推理 anchor 顺序不一致（错位 Bug）")

    print(f"\n  feature_sizes: {dict(det.anchor_gen.feature_sizes)}")
    print(f"  base_sizes: {dict(det.anchor_gen.base_sizes)}")
    print(f"  num anchors per cell: {det.anchor_gen.num_anchors_per_cell}")

    # 前向输出
    inp = np.random.uniform(0, 1, (1, 320, 320, 3)).astype(np.float32)
    preds = det.predict(inp, verbose=0)
    print(f"\n  前向输出（随机初始化权重）:")
    for k, v in preds.items():
        if hasattr(v, 'shape'):
            print(f"    {k:15s}: {v.shape}  range=[{v.min():.3f}, {v.max():.3f}]")


# ============================================================================
# 3. 训练后预测分布诊断
# ============================================================================
def diagnose_predictions(det, n_batches=5):
    print()
    print("=" * 70)
    print("3. 训练后预测分布（真实数据）")
    print("=" * 70)
    val_ds = build_eval_dataset(batch_size=4, input_size=320)

    bg_probs = []
    fg_probs = []
    bg_logits = []
    fg_logits = []
    pos_anchor_counts = []
    total_pos = 0
    total_neg = 0
    total_ignore = 0
    total_loss_sum = 0.0
    total_cls_sum = 0.0
    total_box_sum = 0.0
    n_iter = 0

    for imgs, lbls in val_ds.take(n_batches):
        outputs = det(imgs, training=False)
        loss_dict = det._compute_loss(outputs, lbls)
        total_loss_sum += float(loss_dict['total'])
        total_cls_sum += float(loss_dict['cls'])
        total_box_sum += float(loss_dict['box'])
        n_iter += 1

        # anchor assignment
        cls_targets, _, num_pos = det._assign_anchors_batch(lbls['boxes'], lbls['classes'])
        n_pos = int(tf.reduce_sum(tf.cast(cls_targets > 0, tf.int32)).numpy())
        n_neg = int(tf.reduce_sum(tf.cast(cls_targets == 0, tf.int32)).numpy())
        n_ignore = int(tf.reduce_sum(tf.cast(cls_targets == -1, tf.int32)).numpy())
        total_pos += n_pos
        total_neg += n_neg
        total_ignore += n_ignore
        pos_anchor_counts.append(n_pos / imgs.shape[0])

        # logits
        for level in ['cls_P3', 'cls_P4', 'cls_P5']:
            cls_logits = outputs[level][0].numpy()  # (H, W, 9*81)
            H, W = cls_logits.shape[:2]
            cls_logits_flat = cls_logits.reshape(-1, 9, 81)  # (N, 9, 81)
            bg_l = cls_logits_flat[:, :, 0].flatten()
            fg_l = cls_logits_flat[:, :, 1:].max(axis=-1).flatten()
            bg_logits.extend(bg_l.tolist())
            fg_logits.extend(fg_l.tolist())

    print(f"  Loss (avg over {n_iter} batches):")
    print(f"    total = {total_loss_sum/n_iter:.4f}")
    print(f"    cls   = {total_cls_sum/n_iter:.4f}")
    print(f"    box   = {total_box_sum/n_iter:.4f}")

    print(f"\n  Anchor assignment (累计 over {n_iter} batches):")
    print(f"    pos: {total_pos}, neg: {total_neg}, ignore: {total_ignore}")
    print(f"    pos:neg ratio = 1 : {total_neg / max(total_pos, 1):.0f}")
    if total_neg / max(total_pos, 1) > 50:
        print("    ⚠️ 警告：pos:neg 比例过高（>1:50），模型可能 collapse 到预测 background")
        print("    修复建议：添加 OHEM (Online Hard Example Mining) 或提升 focal gamma")

    bg_logits = np.array(bg_logits)
    fg_logits = np.array(fg_logits)
    print(f"\n  Cls logits 分布:")
    print(f"    bg logit: mean={bg_logits.mean():.3f}, std={bg_logits.std():.3f}")
    print(f"    fg logit: mean={fg_logits.mean():.3f}, std={fg_logits.std():.3f}")
    print(f"    bg logit mean >> fg logit mean → 模型倾向预测 background")
    if bg_logits.mean() - fg_logits.mean() > 2.0:
        print("    ⚠️ 警告：bg logit 显著高于 fg logit，模型可能已 collapse")


# ============================================================================
# 4. Anchor 顺序一致性诊断
# ============================================================================
def diagnose_anchor_order(det):
    print()
    print("=" * 70)
    print("4. Anchor 顺序一致性")
    print("=" * 70)
    feature_spec = det.feature_spec
    sorted_levels = sorted(feature_spec.keys())

    print("  训练时 (按 feature_spec 插入顺序):")
    total = 0
    for level in feature_spec:
        n = det.anchor_gen.feature_sizes[level] * det.anchor_gen.feature_sizes[level] * det.anchor_gen.num_anchors_per_cell
        print(f"    {level}: offset=[{total}, {total+n}), n={n}")
        total += n

    print("\n  推理时 (按 sorted 顺序，来自 predict.py/evaluate.py):")
    total = 0
    for level in sorted_levels:
        n = det.anchor_gen.feature_sizes[level] * det.anchor_gen.feature_sizes[level] * det.anchor_gen.num_anchors_per_cell
        print(f"    {level}: offset=[{total}, {total+n}), n={n}")
        total += n

    if list(feature_spec.keys()) != sorted_levels:
        print("\n  ⚠️ 警告：训练和推理的 anchor 顺序不一致")
        print("     训练时使用 P5, P4, P3, P6, P7")
        print("     推理时使用 P3, P4, P5, P6, P7 (sorted)")
        print("     修复：predict.py / evaluate.py 改为按 feature_spec 顺序而非 sorted")


# ============================================================================
# 5. 完整推理对比（sorted vs feature_spec）
# ============================================================================
def diagnose_decode(det, n_samples=5):
    print()
    print("=" * 70)
    print(f"5. Decode 顺序对比 (sorted vs feature_spec) × {n_samples} 张图")
    print("=" * 70)
    val_ds = build_eval_dataset(batch_size=1, input_size=320)
    n_buggy, n_fixed = [], []
    for i, (imgs, lbls) in enumerate(val_ds.take(n_samples)):
        if i >= n_samples:
            break
        outputs = det.predict(imgs, verbose=0)
        h, w = imgs.shape[1], imgs.shape[2]

        # BUGGY: 按 sorted 顺序
        rb, rs = [], []
        for level in sorted(outputs.keys()):
            if not level.startswith("cls_"):
                continue
            cls = np.asarray(outputs[level][0])
            box = np.asarray(outputs[level.replace("cls_", "box_")][0])
            H, W = cls.shape[:2]
            cls = cls.reshape(H, W, -1, config.NUM_CLASSES + 1).reshape(-1, config.NUM_CLASSES + 1)
            box = box.reshape(H, W, -1, 4).reshape(-1, 4)
            rb.append(box)
            rs.append(cls)
        rb = np.concatenate(rb, axis=0)
        rs = np.concatenate(rs, axis=0)
        res_buggy = decode_predictions(
            rb, rs, det.anchors, image_shape=(h, w), input_size=config.INPUT_SIZE,
            score_thresh=0.05, nms_iou_thresh=0.5, num_classes=config.NUM_CLASSES,
        )

        # FIXED: 按 feature_spec 顺序
        rb, rs = [], []
        for level in det.feature_spec.keys():
            cls = np.asarray(outputs[f"cls_{level}"][0])
            box = np.asarray(outputs[f"box_{level}"][0])
            H, W = cls.shape[:2]
            cls = cls.reshape(H, W, -1, config.NUM_CLASSES + 1).reshape(-1, config.NUM_CLASSES + 1)
            box = box.reshape(H, W, -1, 4).reshape(-1, 4)
            rb.append(box)
            rs.append(cls)
        rb = np.concatenate(rb, axis=0)
        rs = np.concatenate(rs, axis=0)
        res_fixed = decode_predictions(
            rb, rs, det.anchors, image_shape=(h, w), input_size=config.INPUT_SIZE,
            score_thresh=0.05, nms_iou_thresh=0.5, num_classes=config.NUM_CLASSES,
        )

        n_buggy.append(len(res_buggy['boxes']))
        n_fixed.append(len(res_fixed['boxes']))

    print(f"  buggy (sorted) 检出数: {n_buggy}")
    print(f"  fixed (feature_spec) 检出数: {n_fixed}")


# ============================================================================
# 主函数
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="训练后诊断")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="权重路径（默认 checkpoints/{experiment}/best.weights.h5）")
    parser.add_argument("--data-only", action="store_true", help="只跑数据诊断")
    parser.add_argument("--no-predict", action="store_true", help="跳过预测分布诊断")
    args = parser.parse_args()

    if args.ckpt is None:
        args.ckpt = config.CHECKPOINT_DIR / "best.weights.h5"
    ckpt = Path(args.ckpt)

    print("=" * 70)
    print(f"诊断模式: experiment={config.MODEL_MODE}_{config.DATASET_MODE}")
    print(f"检查点: {ckpt}")
    print("=" * 70)

    diagnose_data()
    if args.data_only:
        return

    det = DetectionModel()
    if not ckpt.exists():
        print(f"\n❌ 权重不存在: {ckpt}（先跑一次训练）")
        return
    det.load_weights(str(ckpt))

    diagnose_model(det)
    diagnose_anchor_order(det)
    if not args.no_predict:
        diagnose_predictions(det)
    diagnose_decode(det)

    print()
    print("=" * 70)
    print("诊断完成")
    print("=" * 70)


if __name__ == "__main__":
    main()
