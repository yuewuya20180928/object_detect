#!/bin/bash
# D1 实时 USB 摄像头检测
# 用法:
#   ./run_d1_camera.sh                          # D1 640 默认
#   ./run_d1_camera.sh --list-cameras           # 列出可用摄像头
#   ./run_d1_camera.sh --tta                    # 启用 TTA (慢 ~4x)
#   ./run_d1_camera.sh --camera 1               # 用摄像头 /dev/video1
#   ./run_d1_camera.sh --score-thresh 0.3       # 降低 score 阈值
#   ./run_d1_camera.sh --weights /path/to/other_saved_model --input-size 320
#                                                # 用别的 saved_model (例如 SSD 320)

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# 默认用 D1 (640 input), boss 可通过 --weights / --input-size 覆盖
exec python demo_odapi.py \
    --weights pretrained/efficientdet_d1_coco17_tpu-32/saved_model \
    --input-size 640 \
    "$@"