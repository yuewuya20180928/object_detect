# finetune_workspace

D1 fine-tune 准备工作目录（**Step 8 跳过，未实际执行**）。

## 状态

❌ **D1 fine-tune 跳过**（2026-06-25 决策）

D1 baseline mAP@0.5 = 0.5056 已超 paper 0.384，fine-tune 涨 1-2 个点价值有限。详见项目根目录 `README.md` 的 "Step 8" section。

## 跳过的原因

尝试用 tensorflow/models（boss 下载到 `/work/tensorflow_model/models-master`）的 `model_main_tf2.py` 跑 D1 fine-tune，遇到 7 处 TF 2.20 兼容性问题：

1. `tensorflow.compat.v1.estimator` 在 TF 2.20 已删除 → patch `inputs.py` / `model_lib.py`
2. `tf.keras.layers.experimental.SyncBatchNormalization` 已删除 → patch `freezable_sync_batch_norm.py`
3. `tensorflow_io` 没装 → patch `deepmac_meta_arch.py`
4. KerasTensor 类型不兼容（keras 3 vs keras 2.4 API 差异）
5. `TF_USE_LEGACY_KERAS=1` 后又遇到 `control_flow_ops.case` 缺失
6. keras-cv 路径：依赖的 `tensorflow-text` 跟 TF 2.20 不兼容
7. 降 keras 到 2.15 跟 TF 2.20 冲突（TF 2.20 要 keras≥3.10）

继续 patch 是无底洞，**Stop-the-world 决策**：跳过 fine-tune，接受现状。

## 文件说明

- **`d1_pipeline.config`**: D1 pipeline.config 副本，改过：
  - `fine_tune_checkpoint`: pretrained/efficientdet_d1_coco17_tpu-32/checkpoint/ckpt-0
  - `num_steps`: 300000 → 3000（短 fine-tune demo）
  - `batch_size`: 88 → 4（RTX 3090 24G）
  - `label_map_path`: 指向 data/coco/label_map.pbtxt
  - `input_path`: 指向 data/coco/train.record + val.record

- **`run_d1_finetune.sh`**: model_main_tf2.py 启动脚本 wrapper

- **`object_detection`** (symlink): 指向 `/work/tensorflow_model/models-master/research/object_detection`

## 复跑 fine-tune 的路径（如果未来要恢复）

按推荐优先级：

1. **降 TF 版本到 2.4-2.10**（推荐，最稳）
   - 用 conda 创建新环境：`conda create -n tf2od python=3.9 tensorflow==2.10`
   - 装 tf-models-official 2.10 + tensorflow-text 2.10
   - tensorflow/models 不需 patch，直接跑
   - 耗时 30-60 min 重装 + 1-2 h fine-tune

2. **写自定义 fine-tune 脚本**（不依赖 tensorflow/models）
   - 加载 OD API D1 saved_model 的 backbone weights
   - 简化训练循环
   - 耗时 2-3 h

3. **接受现状，D1 baseline 0.51 已够用**（当前状态）