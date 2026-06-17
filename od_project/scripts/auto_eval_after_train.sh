#!/bin/bash
# 等训练完成 → 自动跑 baseline vs TTA 对比
# 用法: bash scripts/auto_eval_after_train.sh
#
# 监控条件: best.weights.h5 在过去 30 分钟内没更新（说明 train.py 退出了）
# 然后跑两组评估: baseline (320) vs TTA (320+384+448+flip)

set -e
LOG=/work/object_detection/logs/auto_eval_$(date +%Y%m%d_%H%M%S).log
CKPT=/work/object_detection/checkpoints/speed_coco/best.weights.h5
PYTHON=/home/yuewuya/miniconda3/bin/python
PROJECT=/work/object_detection/od_project

echo "[$(date '+%H:%M:%S')] auto_eval started, log: $LOG" | tee -a "$LOG"

# 等训练进程退出（最长 5 小时）
echo "[$(date '+%H:%M:%S')] 等待训练进程退出..." | tee -a "$LOG"
TIMEOUT_START=$(date +%s)
while pgrep -f "python.*train.py.*resume" > /dev/null; do
  ELAPSED=$(($(date +%s) - TIMEOUT_START))
  if [ $ELAPSED -gt 18000 ]; then
    echo "[$(date '+%H:%M:%S')] TIMEOUT 5小时, 强制退出" | tee -a "$LOG"
    exit 1
  fi
  # 顺便记录训练最后的 checkpoint 时间
  CKPT_MTIME=$(stat -c '%Y' "$CKPT" 2>/dev/null || echo 0)
  NOW=$(date +%s)
  CKPT_AGE=$((NOW - CKPT_MTIME))
  echo "[$(date '+%H:%M:%S')] 训练还在跑 (checkpoint ${CKPT_AGE}s ago, elapsed=${ELAPSED}s)" | tee -a "$LOG"
  sleep 120
done

echo "[$(date '+%H:%M:%S')] 训练已退出" | tee -a "$LOG"

# 等待 30 秒确保权重写盘完成
sleep 30

echo "========================================" | tee -a "$LOG"
echo "[$(date '+%H:%M:%S')] 跑 baseline (320 only, per-class thresh + soft-nms + top-k 30)" | tee -a "$LOG"
echo "========================================" | tee -a "$LOG"
cd "$PROJECT"
$PYTHON evaluate_odapi.py \
  --use-per-class-thresh --use-soft-nms --top-k 30 \
  --max-imgs 1000 --score-thresh 0.01 \
  2>&1 | tee -a "$LOG"

echo "========================================" | tee -a "$LOG"
echo "[$(date '+%H:%M:%S')] 跑 TTA (320+384+448 + flip + WBF, 同上后处理)" | tee -a "$LOG"
echo "========================================" | tee -a "$LOG"
$PYTHON evaluate_odapi_tta.py \
  --use-per-class-thresh --use-soft-nms --top-k 30 \
  --max-imgs 1000 --score-thresh 0.01 \
  --tta-scales 320 384 448 --tta-with-flip \
  2>&1 | tee -a "$LOG"

echo "[$(date '+%H:%M:%S')] auto_eval 完成 ✅" | tee -a "$LOG"
