#!/bin/bash
# 优化 TTA: grid search NMS IoU + 更多尺度
# 跑 B-2 (4 个 IoU) + B-3 (5 尺度) 5 个配置
set -e
LOG=/work/object_detection/logs/optimize_tta_$(date +%Y%m%d_%H%M%S).log
PY=/home/yuewuya/miniconda3/bin/python
PROJ=/work/object_detection/od_project
COMMON="--use-per-class-thresh --use-soft-nms --top-k 30 --max-imgs 1000 --score-thresh 0.01"
cd "$PROJ"

echo "===========================================" | tee "$LOG"
echo "B-2: NMS IoU grid search (320+384+448+flip)" | tee -a "$LOG"
echo "===========================================" | tee -a "$LOG"
for IOU in 0.3 0.4 0.5 0.6; do
  echo "" | tee -a "$LOG"
  echo "--- TTA NMS IoU=$IOU ---" | tee -a "$LOG"
  $PY evaluate_odapi_tta.py \
    --tta-fusion nms --tta-scales 320 384 448 --tta-with-flip --nms-iou-thr $IOU \
    $COMMON 2>&1 \
    | grep -v "Zero area box skipped\|warnings.warn" \
    | grep -E "TTA 配置|mAP@|FPS|评估完成|Top 20" | tail -10 | tee -a "$LOG"
done

echo "" | tee -a "$LOG"
echo "===========================================" | tee -a "$LOG"
echo "B-3: 5 尺度 [256 320 384 448 512] + flip + NMS IoU=best" | tee -a "$LOG"
echo "===========================================" | tee -a "$LOG"
$PY evaluate_odapi_tta.py \
  --tta-fusion nms --tta-scales 256 320 384 448 512 --tta-with-flip --nms-iou-thr 0.5 \
  $COMMON 2>&1 \
  | grep -v "Zero area box skipped\|warnings.warn" \
  | grep -E "TTA 配置|mAP@|FPS|评估完成|Top 20" | tail -10 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "[$(date '+%H:%M:%S')] ✅ 全部完成. log: $LOG" | tee -a "$LOG"
