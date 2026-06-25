#!/bin/bash
# D1 fine-tune 启动脚本
# 用 tensorflow/models (boss 下载到 /work/tensorflow_model/models-master)
# 跑 OD API model_main_tf2.py

set -e

# 路径
PROJECT_ROOT=/home/yuewuya/Public/object_detection
TF_MODELS=/work/tensorflow_model/models-master
WORKDIR=$PROJECT_ROOT/finetune_workspace

# PYTHONPATH
export PYTHONPATH="$TF_MODELS/research:$TF_MODELS/research/slim:$WORKDIR:$PROJECT_ROOT:$PYTHONPATH"

# Python 解释器
PY=/home/yuewuya/miniconda3/bin/python3.13

# 跑 model_main_tf2.py
cd "$WORKDIR"

$PY $TF_MODELS/research/object_detection/model_main_tf2.py \
    --model_dir="$WORKDIR/d1_finetune_output" \
    --pipeline_config_path="$WORKDIR/d1_pipeline.config" \
    --alsologtostderr \
    "$@"