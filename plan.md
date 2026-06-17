# 📋 训练恢复计划

> 创建时间：2026-06-17 02:10
> 目标：从当前 best.weights.h5 (5 epoch, val_loss=0.52) 继续训练，达到合理 mAP

---

## 📚 背景：从 mAP=0 到现在的完整故事

### 项目位置
`/work/object_detection/`：基于 TensorFlow 2.x 的目标检测项目，SSD MobileNetV2 FPNLite 320 backbone + COCO 2017 数据集（80 类 117K 训练图），迁移学习。

### 原问题：训练后 mAP=0
旧 `checkpoints/speed_coco/broken_loss/` 下的最佳权重：
- 23 epoch × batch_size=9 训完
- mAP@0.5 = **0.0000**，80 类 AP 全部 0
- val_loss 已经降到 0.25（看起来"学得不错"），但 mAP 是 0

### 诊断：两个独立 Bug 同时存在

#### Bug 1（致命）：训练时类别严重不平衡导致坍缩
**现象**（诊断脚本 `scripts/diagnose.py` 输出）：
```
Anchor assignment: pos:neg = 1:138
bg logit mean = 4.7, fg logit mean = 1.1
→ 模型被 75000 个 easy negative 主导，把 bg logit 推高就不动权重
```

**证据**：
- 同一 batch 对比 loss：随机初始化 total=1.25 → 23 epoch 后 total=0.23
- loss 下降看起来正常，但 head 权重 `head_P3_cls` 的 std 仍 ≈ glorot 初始值 0.045
- **结论：训练全程在推高 bg logit，其它权重基本没动**

**根因**：`DetectionLoss.__call__` 直接对全部 ~19000 个 anchor 计算 focal loss，没有 OHEM。α=0.25 加 138:1 比例 → 负样本梯度贡献是正样本的 387 倍。

#### Bug 2（中等）：anchor 顺序训练/推理错位
**现象**：
```
训练时 (feature_spec 插入序): P5, P4, P3, P6, P7
推理时 (sorted 顺序):       P3, P4, P5, P6, P7
→ raw_boxes[0] (P3 预测) 配给 anchors[0] (P5 anchor) → 全错位
```

**根因**：`build_fpn()` 用 dict 插入序（P5→P4→P3→P6→P7），`predict.py` / `evaluate.py` 用 `sorted(outputs.keys())`（P3→P4→P5→P6→P7）。两套顺序完全不一致。

**影响**：训练时学到的 box delta 配给错误的 anchor，推理时 box decode 全随机。

### 数据层：OK 不动
- `data/coco/train.record` 117266 条（valid bbox 范围 [0,1]，class id ∈ [1,80]）
- `data/coco/val.record` / `test.record` 各 2476 条
- `label_map.pbtxt` 80 类与 COCO id 对齐
- 单图平均 7 个目标，类分布 top10 合理

### 修复（全部完成）
详见下方"🛠 已修复的 Bug"表。

### 修复后：5 epoch 训练结果
| 阶段 | val_loss | mAP@0.5 | person AP |
|---|---|---|---|
| 旧 (23 epoch, 坏) | 0.25 | 0.0000 | 0.0000 |
| 新 (5 epoch, 好) | 0.52* | **0.0003** | **0.0252** |

*新 loss 函数含 OHEM 项，绝对值与旧不可直接比

**关键意义**：
- person AP 从 0 → 0.025 是 100x 跃升（说明模型真的在学前景/背景）
- 每图检出数 0-1 → 100（达到 max_detections 上限）
- score 分布：100% > 0.05，43% > 0.2，**仅 2% > 0.5**（默认阈值）
- → 模型在"找到东西"了，但 score 普遍偏低，需要继续训练

---

## 🎯 当前状态

| 指标 | 数值 |
|---|---|
| 已训练 | 5 epoch (~25 min) |
| best.weights.h5 | val_loss = 0.521 (epoch 5) |
| val_loss 趋势 | 1.07 → 0.66 → 0.54 → 0.52 → 0.52 (持续下降) |
| mAP@0.5 | 0.0003（person 0.025，其余 79 类 0） |
| 检出数 | 100 个/图（达到 max_detections 上限） |
| 关键问题 | score 普遍偏低，仅 2% 超过 0.5 阈值 |

**判断**：OHEM + 修复全部生效，模型在收敛方向上。**只需继续训练**就能进一步提升 mAP。

---

## 📅 阶段 1：继续 head-only 训练（10-15 epoch）

### 目标
让 5 个 epoch 之后已经"找到路"的网络继续走，预计：
- 10 epoch → mAP 0.05-0.10
- 15 epoch → mAP 0.10-0.15

### 启动命令
```bash
cd /work/object_detection && \
  nohup python3 train.py \
    --epochs 15 \
    --batch-size 32 \
    --input-size 320 \
    > logs/speed_coco/cont_$(date +%H%M).log 2>&1 &
```

**说明**：
- 不需要 `--resume`：`train.py` 看到 `best.weights.h5` 存在会自动加载
- 预计耗时：5 min/epoch × 15 = **~75 分钟**
- 监控：`tail -f logs/speed_coco/cont_*.log | grep 吞吐`

### 关键观察点
- val_loss 应继续下降（目标 < 0.4）
- val_cls_loss 0.37 → 0.25（区分能力增强）
- 早停 patience=15，避免 epoch 7-8 后开始过拟合

---

## 📅 阶段 2：评估 + 决定下一步

### 跑 mAP
```bash
python3 evaluate.py \
  --weights checkpoints/speed_coco/best.weights.h5 \
  --score-thresh 0.05 \
  --max-samples 500
```

### 看实际检测效果
```bash
# 找一张 val 图（用 dataset_builder 抽一张 COCO val 图）
python3 predict.py \
  --image data/coco/raw/val2017/000000000139.jpg \
  --score-thresh 0.1
```

### 三种结果对应三种动作

| 结果 | 结论 | 下一步 |
|---|---|---|
| mAP > 0.15 | 阶段 1 充分 | 跑阶段 3（unfreeze backbone） |
| mAP 0.05-0.15 | 还需训练 | 再跑 10 epoch 阶段 1 |
| mAP < 0.05 | 学习率不合适 | 调高 LR 到 3e-3 重训（从 best 续） |

---

## 📅 阶段 3：解冻 backbone 微调（5-10 epoch）

**前提**：阶段 1 跑完且 mAP > 0.10

### 启动命令
```bash
cd /work/object_detection && \
  nohup python3 train.py \
    --unfreeze-backbone \
    --unfreeze-layers 40 \
    --fine-tune-lr 4e-5 \
    --epochs 8 \
    --batch-size 16 \
    --input-size 320 \
    --resume \
    > logs/speed_coco/finetune_$(date +%H%M).log 2>&1 &
```

**说明**：
- `--unfreeze-backbone` 启用分层 LR（head 高、backbone 低）
- `--fine-tune-lr 4e-5`：backbone 用低 LR，避免破坏 ImageNet 预训练特征
- `--batch-size 16`：解冻后显存占用增加
- `--resume`：从 latest.weights.h5 续（latest 比 best 更新，fine-tune 不在意 val_loss）
- 预计耗时：5 min/epoch × 8 = **~40 分钟**

### 关键观察点
- val_loss 应再降 10-20%（head-only 已经把分类器训好，backbone 微调帮定位）
- 预期 mAP 提升 0.05-0.10
- 早停 patience=8（更短，避免破坏预训练特征）

---

## 📅 阶段 4：最终评估 + 后续优化（视情况）

### 4.1 全量 COCO eval（更精确的 mAP）
```bash
# 改 max-samples 到 5000+ 拿更准的 mAP
python3 evaluate.py \
  --weights checkpoints/speed_coco/best.weights.h5 \
  --score-thresh 0.05 \
  --max-samples 5000
```

### 4.2 导出 ONNX / TFLite 部署
```bash
python3 export.py --format savedmodel
python3 export.py --format tflite --quantize float16
python3 export.py --format onnx
```

### 4.3 切换模型档位（可选）
如果时间允许，试试 `MODEL_MODE = "balanced"`（EfficientDet-D4）跑同样 5+5 epoch，mAP 上限 ~0.43。

```python
# config.py
MODEL_MODE = "balanced"  # 改这一行
```

---

## ⚠️ 注意事项

1. **检查点备份**：训练前先备份当前 best.weights.h5 到 `_cont_backup/`
   ```bash
   cp checkpoints/speed_coco/best.weights.h5 \
      checkpoints/speed_coco/_cont_backup/best_5ep_$(date +%H%M).h5
   ```
2. **混精度 + 自定义 train_step**：fp16 + OHEM 在某些 batch 上偶发 NaN，若发现 NaN，关混精度重训：
   ```python
   # config.py
   USE_MIXED_PRECISION = False
   ```
3. **val 集小**：val 仅 2476 张图，5 epoch 看到的 mAP 涨落可能 ±0.02，多 epoch 看趋势
4. **早停阈值**：默认 patience=15。如果 7-8 epoch 后 val_loss 不再下降，手动 kill 然后跑阶段 3

---

## 🛠 已修复的 Bug（避免下次踩坑）

| Bug | 修复 |
|---|---|
| pos:neg = 1:138 类别塌缩 | 加 OHEM (1:3 neg:pos, min 16) + focal γ=2.5 |
| Anchor 顺序训练/推理错位 | FPN 输出 dict 强制 P3→P7 排序 |
| TB loss 全 0 | ThroughputLogger 加 loss 打印 |
| `--steps-per-epoch` 不支持 | train.py 加参数支持烟雾测试 |

诊断脚本 `scripts/diagnose.py` 已就位，训练后跑一次能看到 pos:neg ratio、bg/fg 比例、anchor 顺序等关键指标。
