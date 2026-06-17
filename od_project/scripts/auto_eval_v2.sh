#!/bin/bash
# auto_eval_v2: 直接跑 baseline + TTA 对比（不等 monitor）
# 训练已完成 (09:09)，直接评估 best.weights.h5

set -e
LOG=/work/object_detection/logs/auto_eval_v2_$(date +%Y%m%d_%H%M%S).log
PYTHON=/home/yuewuya/miniconda3/bin/python
PROJECT=/work/object_detection/od_project

echo "[$(date '+%H:%M:%S')] auto_eval_v2 started, log: $LOG" | tee -a "$LOG"

cd "$PROJECT"

# 1) baseline
echo "========================================" | tee -a "$LOG"
echo "[$(date '+%H:%M:%S')] BASELINE: 320 only + per-class thresh + soft-nms + top-k 30" | tee -a "$LOG"
echo "========================================" | tee -a "$LOG"
$PYTHON evaluate_odapi.py \
  --use-per-class-thresh --use-soft-nms --top-k 30 \
  --max-imgs 1000 --score-thresh 0.01 \
  2>&1 | tee -a "$LOG"

# 2) TTA
echo "========================================" | tee -a "$LOG"
echo "[$(date '+%H:%M:%S')] TTA: 320+384+448 + flip + WBF" | tee -a "$LOG"
echo "========================================" | tee -a "$LOG"
$PYTHON evaluate_odapi_tta.py \
  --use-per-class-thresh --use-soft-nms --top-k 30 \
  --max-imgs 1000 --score-thresh 0.01 \
  --tta-scales 320 384 448 --tta-with-flip \
  2>&1 | tee -a "$LOG"

echo "[$(date '+%H:%M:%S')] auto_eval_v2 完成 ✅" | tee -a "$LOG"
