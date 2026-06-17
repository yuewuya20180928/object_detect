#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练主入口

用法：
    # 默认配置（从 config.py 读取）
    python train.py

    # 命令行覆盖
    python train.py --epochs 30 --batch-size 16 --lr 1e-4

    # 断点续训
    python train.py --resume

    # 解冻 backbone 微调
    python train.py --unfreeze-backbone --fine-tune-lr 4e-5
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

# 在 import TF 之前设置 GPU（config.py 也会做，这里再确认一次）
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import tensorflow as tf
import config
from utils.logger import get_logger
from data.dataset_builder import build_train_dataset, build_eval_dataset
from models.detector import DetectionModel


# ============================================================================
# 命令行参数
# ============================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="TensorFlow 目标检测训练",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # 基础训练参数
    parser.add_argument("--epochs", type=int, default=config.TOTAL_EPOCHS,
                        help="总训练轮数")
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE,
                        help="batch size")
    parser.add_argument("--input-size", type=int, default=config.INPUT_SIZE,
                        help="输入图像尺寸（512=balanced 默认, 384=省显存, 256=极致）")
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE,
                        help="初始学习率")
    parser.add_argument("--fine-tune-lr", type=float, default=config.FINE_TUNE_LR,
                        help="backbone 微调学习率")

    # Backbone 控制
    parser.add_argument("--unfreeze-backbone", action="store_true",
                        help="解冻 backbone 进行微调")
    parser.add_argument("--unfreeze-layers", type=int, default=40,
                        help="解冻 backbone 最后 N 层")

    # 流程控制
    parser.add_argument("--resume", action="store_true",
                        help="从 latest.weights.h5 断点续训")
    parser.add_argument("--warmup-epochs", type=int, default=3,
                        help="warmup epoch 数（前几轮冻结 backbone）")

    # 评估 & 保存
    parser.add_argument("--eval-every", type=int, default=1,
                        help="每 N 个 epoch 评估一次")
    parser.add_argument("--save-best-only", action="store_true", default=config.SAVE_BEST_ONLY,
                        help="只保存最优 checkpoint")
    parser.add_argument("--early-stop-patience", type=int, default=15,
                        help="早停耐心值")

    # 烟雾测试
    parser.add_argument("--steps-per-epoch", type=int, default=None,
                        help="每个 epoch 的 step 数（默认按 dataset 长度计算；烟雾测试可调小）")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="总 step 上限（烟雾测试用，设了就忽略 --epochs 算出的总 step）")

    return parser.parse_args()


# ============================================================================
# 学习率调度
# ============================================================================
class WarmUpCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Warmup + Cosine Decay 学习率调度
    前 warmup_steps 步线性 warmup，后续 cosine decay
    """

    def __init__(self, initial_lr, warmup_steps, total_steps):
        self.initial_lr = initial_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup_steps = tf.cast(self.warmup_steps, tf.float32)
        total_steps = tf.cast(self.total_steps, tf.float32)

        # Warmup 阶段
        warmup_lr = self.initial_lr * (step / warmup_steps)

        # Cosine 阶段（修复：末尾 lr 降至 initial_lr * min_ratio，而非 0）
        min_lr = self.initial_lr * 0.01
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        progress = tf.clip_by_value(progress, 0.0, 1.0)
        cosine_lr = min_lr + (self.initial_lr - min_lr) * 0.5 * (1.0 + tf.cos(np.pi * progress))

        return tf.where(step < warmup_steps, warmup_lr, cosine_lr)

    def get_config(self):
        return {
            "initial_lr": self.initial_lr,
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps,
        }


import numpy as np


# ============================================================================
# 吞吐监控 Callback
# ============================================================================
class ThroughputLogger(tf.keras.callbacks.Callback):
    """
    每隔 N 个 batch 打印一次吞吐（样本/秒）

    衡量指标：
        - samples_per_sec = batch_size * steps / elapsed_sec
        - 反映 GPU + 数据管道整体吞吐
    """

    def __init__(self, batch_size: int, log_every_steps: int = 50, logger=None):
        super().__init__()
        self.batch_size = batch_size
        self.log_every_steps = log_every_steps
        self.logger = logger
        self.window_start = None
        self.window_steps = 0
        self.global_step = 0
        self.epoch_start = None

    def on_train_begin(self, logs=None):
        import time
        self.t_start = time.time()
        self.window_start = time.time()
        self.window_steps = 0
        self.epoch_start = time.time()

    def on_train_batch_end(self, batch, logs=None):
        import time
        self.global_step += 1
        self.window_steps += 1
        if self.window_steps >= self.log_every_steps:
            now = time.time()
            elapsed = now - self.window_start
            samples = self.window_steps * self.batch_size
            sps = samples / elapsed if elapsed > 0 else 0
            sps_step = 1.0 / (elapsed / self.window_steps)
            total_elapsed = now - self.t_start
            total_samples = self.global_step * self.batch_size
            total_sps = total_samples / total_elapsed
            # ★ 同步打印 loss（修复 TB 全 0 问题）
            loss_str = ""
            if logs:
                loss = logs.get("loss")
                cls_loss = logs.get("cls_loss")
                box_loss = logs.get("box_loss")
                parts = []
                if loss is not None:
                    parts.append(f"loss={float(loss):.4f}")
                if cls_loss is not None:
                    parts.append(f"cls={float(cls_loss):.4f}")
                if box_loss is not None:
                    parts.append(f"box={float(box_loss):.4f}")
                if parts:
                    loss_str = " | " + " ".join(parts)
            msg = (
                f"[吞吐] step {self.global_step}: "
                f"近 {self.log_every_steps} step = {sps:.1f} 张/秒 "
                f"({sps_step*1000:.1f}ms/step) | "
                f"累计 {total_sps:.1f} 张/秒"
                f"{loss_str}"
            )
            if self.logger:
                self.logger.info(msg)
            else:
                print(msg)
            self.window_start = now
            self.window_steps = 0

    def on_epoch_begin(self, epoch, logs=None):
        import time
        self.epoch_start = time.time()

    def on_epoch_end(self, epoch, logs=None):
        import time
        if self.epoch_start is not None:
            epoch_sec = time.time() - self.epoch_start
            self.logger.info(
                f"[Epoch {epoch+1}] 总耗时 {epoch_sec/60:.1f} 分钟"
            )


# ============================================================================
# 训练主函数
# ============================================================================
def main():
    args = parse_args()
    logger = get_logger("train", config.LOG_DIR)
    config.print_config()

    # GPU 设置
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            if config.ENABLE_MEMORY_GROWTH:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            logger.info(f"GPU: {[g.name for g in gpus]}")
        except RuntimeError as e:
            logger.error(f"GPU 设置失败: {e}")

    # 混合精度
    if config.USE_MIXED_PRECISION:
        from tensorflow.keras import mixed_precision
        policy = mixed_precision.Policy("mixed_float16")
        mixed_precision.set_global_policy(policy)
        logger.info(f"混合精度策略: {policy.name}")

    # 构建检测器
    logger.info(f"构建检测器 (input_size={args.input_size})...")
    det = DetectionModel(
        model_mode=config.MODEL_MODE,
        num_classes=config.NUM_CLASSES,
        input_size=args.input_size,  # 动态输入尺寸
        weights="imagenet" if config.BACKBONE_WEIGHTS_PATH is None else None,
        weights_path=config.BACKBONE_WEIGHTS_PATH,
    )

    # Backbone 冻结策略
    # MobileNetV2 backbone 是平铺在主模型里的（非子模型），按层名识别
    backbone_prefixes = ('Conv1', 'bn_Conv1', 'Conv1_relu', 'expanded_',
                          'block_', 'out_relu', 'Conv_1', 'Conv_1_bn')

    if not args.unfreeze_backbone:
        logger.info("Backbone 全部冻结（仅训练 FPN + Head）")
    else:
        n = args.unfreeze_layers
        # 全部 backbone 层先冻结
        backbone_layers = []
        for layer in det.model.layers:
            if any(layer.name.startswith(p) for p in backbone_prefixes):
                layer.trainable = False
                backbone_layers.append(layer)

        # 解冻最后 n 个 backbone 层（跳过 BN）
        for layer in backbone_layers[-n:]:
            if 'bn' not in layer.name.lower() and 'batch' not in layer.name.lower():
                layer.trainable = True

        logger.info(f"Backbone 解冻最后 {n} 层（共 {len(backbone_layers)} 层）")
        logger.info(f"Head LR: {args.lr}, Backbone LR: {args.fine_tune_lr}")

    # 学习率调度
    # 实际训练集样本数（从 record 条目数获取，避免硬编码偏差）
    # 117266 是 convert_to_tfrecord.py 日志中确认的训练集有效条数
    _actual_train_samples = 117266
    _default_steps_per_epoch = _actual_train_samples // args.batch_size  # 与实际 batch_size 一致
    # ★ 烟雾测试：允许 --steps-per-epoch 覆盖
    steps_per_epoch = args.steps_per_epoch or _default_steps_per_epoch
    logger.info(f"steps_per_epoch = {steps_per_epoch} (覆盖自 {_default_steps_per_epoch})" if args.steps_per_epoch
                else f"steps_per_epoch = {steps_per_epoch}")
    total_steps = steps_per_epoch * args.epochs
    # ★ 烟雾测试：允许 --max-steps 覆盖总 step
    if args.max_steps is not None:
        logger.info(f"max_steps 覆盖：{total_steps} → {args.max_steps}")
        total_steps = args.max_steps
    warmup_steps = steps_per_epoch * args.warmup_epochs

    lr_schedule = WarmUpCosineDecay(
        initial_lr=args.lr,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
    )

    # 优化器
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=lr_schedule,
        weight_decay=1e-4,
    )

    # XLA 编译已开启，首次 step 需 3-5 分钟选 cuDNN 算法
    # 选完后每个 step 快 20-30%，总体训练总时间更短
    det.compile(optimizer=optimizer)
    det.summary(print_fn=logger.info)

    # 加载权重（断点续训）
    best_path = config.CHECKPOINT_DIR / "best.weights.h5"
    latest_path = config.CHECKPOINT_DIR / "latest.weights.h5"
    if args.resume and latest_path.exists():
        logger.info(f"加载权重: {latest_path}")
        det.load_weights(str(latest_path))
    elif best_path.exists():
        logger.info(f"发现 best.weights.h5，自动加载")
        det.load_weights(str(best_path))

    # 构建数据 pipeline
    logger.info(f"构建数据 pipeline (input_size={args.input_size})...")
    train_ds = build_train_dataset(
        batch_size=args.batch_size,
        input_size=args.input_size,
        augment=config.AUGMENT_TRAIN,
        augment_color=True,                # 修正: 原本默认 False 颜色增强没开
    )
    val_ds = build_eval_dataset(
        tfrecord_path=config.VAL_RECORD,
        batch_size=args.batch_size,
        input_size=args.input_size,
    )

    # Callbacks
    callbacks = [
        # TensorBoard
        tf.keras.callbacks.TensorBoard(
            log_dir=str(config.LOG_DIR),
            histogram_freq=config.TENSORBOARD_HISTOGRAM_FREQ,
            write_graph=config.TENSORBOARD_WRITE_GRAPH,
            update_freq=config.TENSORBOARD_UPDATE_FREQ,
        ),
        # ModelCheckpoint (best)
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(best_path),
            monitor="val_loss",
            save_best_only=args.save_best_only,
            save_weights_only=True,
            verbose=1,
        ),
        # ModelCheckpoint (latest)
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(latest_path),
            monitor="val_loss",
            save_best_only=False,
            save_weights_only=True,
            verbose=0,
        ),
        # EarlyStopping
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=args.early_stop_patience,
            min_delta=config.EARLY_STOP_MIN_DELTA,
            restore_best_weights=True,
            verbose=1,
        ),
        # 吞吐监控：每 log_every_steps 个 batch 打印一次每秒处理样本数
        ThroughputLogger(
            batch_size=args.batch_size,
            log_every_steps=50,
            logger=logger,
        ),
    ]

    # ============================================================================
    # 分层学习率训练循环（backbone 解冻后 backbone 用更低 LR）
    # ============================================================================
    if args.unfreeze_backbone:
        logger.info("=" * 50)
        logger.info("使用分层学习率训练循环（backbone LR < head LR）")
        logger.info("=" * 50)

        # 分离 backbone 和 head 变量
        # 通过 layer.trainable 状态收集变量（var.name 无层名前缀，需按 layer.name 判断）
        backbone_vars, head_vars = [], []
        for layer in det.model.layers:
            if not layer.trainable:
                continue
            for var in layer.variables:
                if any(layer.name.startswith(p) for p in backbone_prefixes):
                    backbone_vars.append(var)
                else:
                    head_vars.append(var)
        logger.info(f"Backbone 变量: {len(backbone_vars)}, Head 变量: {len(head_vars)}")

        # 两个 LR 调度器（形状相同，幅值不同）
        backbone_lr_schedule = WarmUpCosineDecay(
            initial_lr=args.fine_tune_lr,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
        )
        head_lr_schedule = WarmUpCosineDecay(
            initial_lr=args.lr,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
        )

        backbone_opt = tf.keras.optimizers.AdamW(
            learning_rate=backbone_lr_schedule,
            weight_decay=1e-4,
        )
        head_opt = tf.keras.optimizers.AdamW(
            learning_rate=head_lr_schedule,
            weight_decay=1e-4,
        )

        # 吞吐监控状态
        tput_batch_count = [0]
        tput_start_time = [None]

        best_val_loss = float('inf')
        no_improve_epochs = [0]
        start_time = datetime.now()

        try:
            for epoch in range(args.epochs):
                logger.info(f"\nEpoch {epoch + 1}/{args.epochs}")
                epoch_start = datetime.now()

                # ---- Train ----
                train_losses = []
                for step, (images, labels) in enumerate(train_ds):
                    if tput_start_time[0] is None:
                        tput_start_time[0] = datetime.now()

                    with tf.GradientTape() as tape:
                        preds = det.model(images, training=True)
                        loss_dict = det._compute_loss(preds, labels)
                        total_loss = loss_dict["total"]

                    # 分离梯度（按 variable id 匹配，不依赖顺序）
                    all_vars = backbone_vars + head_vars
                    grads = tape.gradient(total_loss, all_vars)
                    # 建立 var->grad 映射（过滤 None）
                    var_grad_map = {id(v): g for v, g in zip(all_vars, grads) if g is not None}

                    backbone_grads = [(var_grad_map[id(v)], v) for v in backbone_vars if id(v) in var_grad_map]
                    head_grads = [(var_grad_map[id(v)], v) for v in head_vars if id(v) in var_grad_map]

                    if backbone_grads:
                        backbone_opt.apply_gradients(backbone_grads)
                    if head_grads:
                        head_opt.apply_gradients(head_grads)

                    train_losses.append(float(total_loss))

                    # 吞吐
                    tput_batch_count[0] += 1
                    if tput_batch_count[0] % 50 == 0:
                        elapsed = (datetime.now() - tput_start_time[0]).total_seconds()
                        sps = args.batch_size * 50 / elapsed
                        logger.info(f"  [{tput_batch_count[0]} steps] throughput: {sps:.1f} samples/sec")

                    if step + 1 >= steps_per_epoch:
                        break

                train_loss = float(np.mean(train_losses))

                # ---- Validate ----
                val_losses = []
                for images, labels in val_ds:
                    preds = det.model(images, training=False)
                    loss_dict = det._compute_loss(preds, labels)
                    val_losses.append(float(loss_dict["total"]))
                    if len(val_losses) >= 300:  # 最多 300 batch（约 2400 图）做快速验证
                        break
                val_loss = float(np.mean(val_losses))

                epoch_time = (datetime.now() - epoch_start).total_seconds()
                logger.info(
                    f"  train_loss={train_loss:.5f}  val_loss={val_loss:.5f}  "
                    f"head_lr={head_opt.learning_rate.numpy():.2e}  "
                    f"backbone_lr={backbone_opt.learning_rate.numpy():.2e}  "
                    f"time={epoch_time:.0f}s"
                )

                # 保存 best
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    det.save_weights(str(best_path))
                    logger.info(f"  ✓ best model saved (val_loss={val_loss:.5f})")

                # 保存 latest
                det.save_weights(str(latest_path))


                # Early stopping
                if val_loss >= best_val_loss:
                    no_improve_epochs[0] += 1
                    if no_improve_epochs[0] >= args.early_stop_patience:
                        logger.info(f"Early stopping triggered (patience={args.early_stop_patience})")
                        break
                else:
                    no_improve_epochs[0] = 0

        except KeyboardInterrupt:
            logger.warning("训练被用户中断")

        elapsed = datetime.now() - start_time
        logger.info(f"训练完成，耗时: {elapsed}")
        history = None

    else:
        # 标准单优化器训练
        logger.info("=" * 50)
        logger.info(f"开始训练: {args.epochs} epochs, batch_size={args.batch_size}")
        logger.info("=" * 50)
        start_time = datetime.now()

        try:
            history = det.fit(
                train_ds,
                validation_data=val_ds,
                epochs=args.epochs,
                steps_per_epoch=steps_per_epoch,
                callbacks=callbacks,
                verbose=1,
            )
        except KeyboardInterrupt:
            logger.warning("训练被用户中断")
            history = None

        elapsed = datetime.now() - start_time
        logger.info(f"训练完成，耗时: {elapsed}")

        # 保存训练曲线
        if history is not None and history.history:
            from utils.visualization import plot_training_curves
            plot_training_curves(
                history.history,
                save_path=config.OUTPUT_DIR / "training_curves.png",
            )


if __name__ == "__main__":
    main()
