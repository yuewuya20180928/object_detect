#!/bin/bash
# 跑 3 个对比:
#   1) baseline (single scale 320, no TTA)       — 之前是 0.0487
#   2) TTA NMS-concat (320+384+448 + flip)        — 新方案
#   3) TTA WBF       (320+384+448 + flip)         — 之前是 0.0201
#
# 用 OD API pretrained (跟训练解耦)，后处理统一开 per-class thresh + soft-NMS + top-k 30
set -e
LOG=/work/object_detection/logs/compare_tta_$(date +%Y%m%d_%H%M%S).log
PY=/home/yuewuya/miniconda3/bin/python
PROJ=/work/object_detection/od_project
cd "$PROJ"

echo "===========================================" | tee "$LOG"
echo "[$(date '+%H:%M:%S')] 1) baseline (single scale, --tta-fusion none)" | tee -a "$LOG"
echo "===========================================" | tee -a "$LOG"
$PY evaluate_odapi_tta.py \
  --tta-fusion none --tta-scales 320 --no-tta-flip \
  --use-per-class-thresh --use-soft-nms --top-k 30 \
  --max-imgs 1000 --score-thresh 0.01 \
  2>&1 | grep -v "Zero area box skipped\|warnings.warn" | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "===========================================" | tee -a "$LOG"
echo "[$(date '+%H:%M:%S')] 2) TTA NMS-concat (320+384+448+flip, iou=0.5)" | tee -a "$LOG"
echo "===========================================" | tee -a "$LOG"
$PY evaluate_odapi_tta.py \
  --tta-fusion nms --tta-scales 320 384 448 --tta-with-flip --nms-iou-thr 0.5 \
  --use-per-class-thresh --use-soft-nms --top-k 30 \
  --max-imgs 1000 --score-thresh 0.01 \
  2>&1 | grep -v "Zero area box skipped\|warnings.warn" | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "===========================================" | tee -a "$LOG"
echo "[$(date '+%H:%M:%S')] 3) TTA WBF (320+384+448+flip, iou=0.55)" | tee -a "$LOG"
echo "===========================================" | tee -a "$LOG"
$PY evaluate_odapi_tta.py \
  --tta-fusion wbf --tta-scales 320 384 448 --tta-with-flip --wbf-iou-thr 0.55 \
  --use-per-class-thresh --use-soft-nms --top-k 30 \
  --max-imgs 1000 --score-thresh 0.01 \
  2>&1 | grep -v "Zero area box skipped\|warnings.warn" | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "[$(date '+%H:%M:%S')] ✅ 全部完成. log: $LOG" | tee -a "$LOG"
