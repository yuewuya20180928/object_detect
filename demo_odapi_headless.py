#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
USB 摄像头实时目标检测 (TF OD API 官方预训练 + headless 显示版)

适用: 服务器/有 X display / 无 X display 三种环境都能跑
- 摄像头采集 → TF OD API SavedModel (默认 EfficientDet-D1 640) 推理
- 画 bbox + 类名 + 置信度
- 显示方式 (--display) 三选一:
  - imshow (默认) : cv2.namedWindow 弹窗, 按 q 退出 / s 保存
  - png           : 单帧覆盖写 detections/latest.jpg, 用图片浏览器看
  - both          : 弹窗 + 写盘 都要
- 自动探测 X server: Wayland 下用 /run/user/<uid>/.mutter-Xwaylandauth.<random>;
  X11 用 ~/.Xauthority; 自动设 DISPLAY=:0
- 实时 stdout 打印一行汇总 (FPS / Det 数 / Top 3 类别)
- Ctrl+C 优雅退出

用法:
  # D1 640 + 弹窗显示 (默认, 推荐)
  python demo_odapi_headless.py

  # D1 + 只写 PNG (图片浏览器看)
  python demo_odapi_headless.py --display png

  # SSD 320 (快 3x)
  python demo_odapi_headless.py \\
    --weights pretrained/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/saved_model \\
    --input-size 320

  # 跑 60 秒自动停
  python demo_odapi_headless.py --duration 60

  # 跳过帧提速 (每 3 帧检测一次)
  python demo_odapi_headless.py --skip-frames 2

  # 弹窗 + 写盘 都要
  python demo_odapi_headless.py --display both
"""

import os
import sys
import time
import argparse
import signal
from pathlib import Path
from collections import deque, Counter

PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

# ===== TF log / CUDA =====
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("OPENCV_VIDEOIO_PRIORITY_MSMF", "0")

# ===== 探测 X server 并自动配 DISPLAY + XAUTHORITY (Wayland + Xwayland 场景) =====
# 优先看用户/系统是否已经设过; 否则自动找 mutter-Xwaylandauth.<random> 或 ~/.Xauthority
import glob as _glob
def _autodetect_x():
    if "DISPLAY" in os.environ and "XAUTHORITY" in os.environ:
        return
    uid = os.getuid()
    # Wayland (GNOME Mutter) 用的 Xwayland 临时 auth 文件
    mutter = sorted(_glob.glob(f"/run/user/{uid}/.mutter-Xwaylandauth.*"))
    if mutter:
        os.environ.setdefault("XAUTHORITY", mutter[-1])
        os.environ.setdefault("DISPLAY", ":0")
        return
    # 传统 X11: ~/.Xauthority
    xauth = os.path.expanduser("~/.Xauthority")
    if os.path.exists(xauth):
        os.environ.setdefault("XAUTHORITY", xauth)
        os.environ.setdefault("DISPLAY", ":0")
_autodetect_x()

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import mixed_precision

mixed_precision.set_global_policy("mixed_float16")

import config  # noqa: E402

# D1 默认 (主路径)
ODAPI_MODEL_DEFAULT = (
    PROJECT_ROOT / "pretrained" / "efficientdet_d1_coco17_tpu-32" / "saved_model"
)


# ============================================================================
# COCO 90-id 标签 (gapped, 0-indexed = detection_classes - 1)
# 跟 demo_odapi.py 保持一致 (含 None 占位)
# ============================================================================
COCO_NAMES = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck",
    "boat","traffic light","fire hydrant", None,
    "stop sign","parking meter","bench",
    "bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe",
    None,
    "backpack","umbrella",
    None, None,
    "handbag","tie","suitcase","frisbee","skis","snowboard",
    "sports ball","kite","baseball bat","baseball glove","skateboard","surfboard",
    "tennis racket","bottle",
    None,
    "wine glass","cup","fork","knife","spoon","bowl",
    "banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza",
    "donut","cake","chair","couch","potted plant","bed",
    None,
    "dining table",
    None, None,
    "toilet",
    None,
    "tv","laptop","mouse","remote","keyboard","cell phone","microwave",
    "oven","toaster","sink","refrigerator",
    None,
    "book","clock","vase","scissors","teddy bear","hair drier","toothbrush",
]

# 每类固定颜色 (HSV 离散, 黄金角分布)
np.random.seed(42)
CLASS_COLORS = np.zeros((80, 3), dtype=np.uint8)
for i in range(80):
    h = (i * 137) % 180
    s = 200 + (i % 3) * 18
    v = 220 + (i % 2) * 35
    hsv = np.array([[[h, s, v]]], dtype=np.uint8)
    CLASS_COLORS[i] = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]


# ============================================================================
# 几何 / 画框
# ============================================================================
def letterbox(image: np.ndarray, input_size: int) -> tuple:
    h, w = image.shape[:2]
    scale = input_size / max(h, w)
    new_h = int(round(h * scale))
    new_w = int(round(w * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_y = (input_size - new_h) // 2
    pad_x = (input_size - new_w) // 2
    padded = np.full((input_size, input_size, 3), 0, dtype=np.uint8)
    padded[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    return padded, scale, pad_x, pad_y


def unletterbox_boxes(boxes_in: np.ndarray, scale: float, pad_x: int, pad_y: int) -> np.ndarray:
    if len(boxes_in) == 0:
        return boxes_in
    out = boxes_in.astype(np.float32).copy()
    out[:, 0] = (boxes_in[:, 0] - pad_x) / scale
    out[:, 1] = (boxes_in[:, 1] - pad_y) / scale
    out[:, 2] = (boxes_in[:, 2] - pad_x) / scale
    out[:, 3] = (boxes_in[:, 3] - pad_y) / scale
    return out


def cross_class_nms(boxes, scores, class_ids, iou_thresh=0.5):
    """跨类 NMS, 干掉同区域不同 cid 的重复检测"""
    if len(boxes) == 0:
        return boxes, scores, class_ids
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        inds = np.where(iou <= iou_thresh)[0]
        order = order[inds + 1]
    return boxes[keep], scores[keep], class_ids[keep]


def draw_boxes(image: np.ndarray, boxes: np.ndarray, scores: np.ndarray,
               class_ids: np.ndarray, fps: float = None, n_det: int = None,
               threshold: float = None) -> np.ndarray:
    """画 bbox + 标签 + 顶部状态条 (FPS / Det / Thr)"""
    H, W = image.shape[:2]
    for box, score, cid in zip(boxes, scores, class_ids):
        x1, y1, x2, y2 = box.astype(int)
        x1 = max(0, min(W - 1, x1))
        y1 = max(0, min(H - 1, y1))
        x2 = max(0, min(W - 1, x2))
        y2 = max(0, min(H - 1, y2))
        if x2 <= x1 or y2 <= y1:
            continue
        if cid < 0 or cid >= len(COCO_NAMES) or COCO_NAMES[cid] is None:
            continue
        name = COCO_NAMES[cid]
        label = f"{name} {score:.0%}"
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        lbl_x0 = max(2, min(x1, W - tw - 6))
        lbl_x1 = lbl_x0 + tw + 4
        lbl_y1 = y1
        lbl_y0 = lbl_y1 - th - 6
        if lbl_y0 < 2:
            lbl_y0 = y1 + 2
            lbl_y1 = lbl_y0 + th + 6
        color = tuple(int(c) for c in CLASS_COLORS[cid % 80])
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.rectangle(image, (lbl_x0, lbl_y0), (lbl_x1, lbl_y1), color, -1)
        cv2.putText(image, label, (lbl_x0 + 2, lbl_y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    # 顶部状态条
    if fps is not None:
        info = f"FPS: {fps:.1f}  Det: {n_det or 0}  Thr: {threshold or 0:.2f}"
        cv2.putText(image, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0), 2, cv2.LINE_AA)
    return image


# ============================================================================
# 主函数
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="USB 摄像头实时目标检测 headless 版 (TF OD API)",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--camera", type=int, default=0, help="摄像头设备索引 (默认 0)")
    parser.add_argument("--list-cameras", action="store_true", help="列出可用摄像头并退出")
    parser.add_argument("--width", type=int, default=1280, help="采集宽度")
    parser.add_argument("--height", type=int, default=720, help="采集高度")
    parser.add_argument("--score-thresh", type=float, default=0.5,
                        help="score 阈值 (官方预训练权重, 0.5 合理默认)")
    parser.add_argument("--nms-iou", type=float, default=0.5, help="跨类 NMS IoU 阈值")
    parser.add_argument("--skip-frames", type=int, default=0,
                        help="每 N+1 帧跑一次检测 (0=每帧, 2=每 3 帧)")
    parser.add_argument("--input-size", type=int, default=640,
                        help="模型输入尺寸 (D1=640, SSD=320)")
    parser.add_argument("--weights", type=str, default=str(ODAPI_MODEL_DEFAULT),
                        help="OD API SavedModel 路径 (默认 D1)")
    parser.add_argument("--display-path", type=str,
                        default=str(PROJECT_ROOT / "detections" / "latest.jpg"),
                        help="单帧覆盖写出的 PNG 路径 (--display png 模式用)")
    parser.add_argument("--display", type=str, default="imshow",
                        choices=["imshow", "png", "both"],
                        help="显示方式: imshow (cv2 弹窗) | png (覆盖写 latest.jpg) | both")
    parser.add_argument("--save-dir", type=str, default="./detections",
                        help="Ctrl+C 退出时保存最后一帧的目录")
    parser.add_argument("--duration", type=int, default=0,
                        help="运行时长 (秒), 0=无限, 默认 0")
    args = parser.parse_args()

    if args.list_cameras:
        print("可用摄像头:")
        for i in range(5):
            cap = cv2.VideoCapture(i)
            if cap.isOpened() and cap.get(cv2.CAP_PROP_FRAME_WIDTH) > 0:
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                print(f"  /dev/video{i}  ->  {w}x{h}")
            cap.release()
        return

    weights = args.weights
    if not Path(weights).exists():
        print(f"[ERROR] SavedModel 不存在: {weights}")
        sys.exit(1)

    display_path = Path(args.display_path)
    display_path.parent.mkdir(parents=True, exist_ok=True)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ===== 探测 cv2 是否支持 imshow (headless cv2 会抛 NotImplemented) =====
    def _cv2_has_gui():
        try:
            cv2.namedWindow("__probe__", cv2.WINDOW_NORMAL)
            cv2.destroyWindow("__probe__")
            return True
        except Exception:
            return False
    HAS_GUI = _cv2_has_gui()

    # 显示模式处理
    use_imshow = args.display in ("imshow", "both")
    use_png = args.display in ("png", "both")
    if use_imshow and not HAS_GUI:
        print("[WARN] 当前 cv2 是 headless build, imshow 不可用, 自动 fallback 到 png")
        use_imshow = False
        use_png = True  # 兜底: 弹窗不行就用 png
    if not use_imshow and not use_png:
        print("[ERROR] 显示方式都不可用, 请检查环境")
        sys.exit(1)

    # ===== 加载 TF OD API SavedModel =====
    print(f"[INFO] 加载 TF OD API SavedModel: {weights}")
    detect_fn = tf.saved_model.load(str(weights))
    signature = detect_fn.signatures["serving_default"]
    print(f"[INFO] SavedModel 加载完成 | 输入: input_tensor (uint8, H×W×3)")

    # ===== 探测摄像头 =====
    available = []
    for i in range(4):
        cap_test = cv2.VideoCapture(i)
        if cap_test.isOpened() and cap_test.get(cv2.CAP_PROP_FRAME_WIDTH) > 0:
            available.append(i)
            cap_test.release()
    if not available:
        print("[ERROR] 没有可用的摄像头设备")
        sys.exit(1)
    if args.camera not in available:
        print(f"[WARN] /dev/video{args.camera} 不可用, 自动选 /dev/video{available[0]}")
        args.camera = available[0]

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"[ERROR] 无法打开摄像头 {args.camera}")
        sys.exit(1)
    try:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    except Exception as e:
        print(f"[WARN] 设 FOURCC 失败, 用默认: {e}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened() or cap.get(cv2.CAP_PROP_FRAME_WIDTH) == 0:
        print(f"[ERROR] 摄像头 {args.camera} 设参后不可用")
        sys.exit(1)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    mode_str = []
    if use_imshow: mode_str.append("imshow (cv2 弹窗)")
    if use_png:    mode_str.append(f"png ({display_path})")
    print(f"[INFO] 摄像头 {args.camera}: {actual_w}x{actual_h}")
    print(f"[INFO] 显示模式: {' + '.join(mode_str)}")
    print(f"[INFO] 模型输入: {args.input_size}x{args.input_size}, score_thresh={args.score_thresh}")
    if use_imshow:
        print(f"[INFO] DISPLAY={os.environ.get('DISPLAY','(未设)')}  XAUTHORITY={os.environ.get('XAUTHORITY','(未设)')}")
        print(f"[INFO] 弹窗控制: q 退出 | s 保存当前帧")
    print("[INFO] 首次推理会触发 XLA compile (约 2-3 分钟), 之后稳态 20-30ms/帧")
    print()

    # ===== 模型 warmup (触发 XLA compile) =====
    print("[INFO] 模型 warmup 中...")
    t0 = time.perf_counter()
    dummy = tf.zeros((1, args.input_size, args.input_size, 3), dtype=tf.uint8)
    _ = signature(dummy)
    print(f"[INFO] Warmup 完成: {(time.perf_counter() - t0) * 1000:.0f}ms")

    # ===== Ctrl+C 优雅退出 =====
    running = True
    def handle_sigint(sig, frame):
        nonlocal running
        running = False
    signal.signal(signal.SIGINT, handle_sigint)
    signal.signal(signal.SIGTERM, handle_sigint)

    print()
    print("=" * 60)
    print(f"运行中... Ctrl+C 退出")
    print(f"实时画面写到: {display_path}")
    print(f"用图片浏览器打开该路径即可看 (feh/eog/nomacs 等支持自动刷新)")
    print("=" * 60)
    print()

    # ===== 主循环 =====
    fps_history = deque(maxlen=30)
    last_result = None
    class_counter = Counter()
    total_det = 0
    frame_idx = 0
    t_start = time.perf_counter()
    t_last_log = t_start
    t_last_write = 0.0
    last_display = None  # 最新一帧 (供退出时保存)

    # 显示帧的最低刷新频率 (避免每帧都写 PNG 卡 IO)
    WRITE_INTERVAL = 0.05  # 50ms = 20 FPS 写盘, 不影响检测吞吐

    try:
        while running:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.05)
                continue

            t0 = time.perf_counter()

            # 跑检测 (按 skip-frames 决定频率)
            if frame_idx % (args.skip_frames + 1) == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                padded, scale, pad_x, pad_y = letterbox(rgb, args.input_size)
                input_tensor = tf.cast(tf.expand_dims(padded, 0), tf.uint8)
                outputs = signature(input_tensor)
                boxes = outputs["detection_boxes"][0].numpy()
                sc = outputs["detection_scores"][0].numpy()
                cls = outputs["detection_classes"][0].numpy().astype(np.int32)
                num = int(outputs["num_detections"][0].numpy())
                # 先按 score_thresh 滤 (避免后续画空洞 cid)
                keep = sc[:num] >= args.score_thresh
                boxes_320 = np.zeros((num, 4), dtype=np.float32)
                boxes_320[:, 0] = boxes[:num, 1] * args.input_size
                boxes_320[:, 1] = boxes[:num, 0] * args.input_size
                boxes_320[:, 2] = boxes[:num, 3] * args.input_size
                boxes_320[:, 3] = boxes[:num, 2] * args.input_size
                class_ids = cls - 1
                boxes_320 = boxes_320[keep]
                sc_f = sc[:num][keep]
                class_ids_f = class_ids[:num][keep]
                boxes_orig = unletterbox_boxes(boxes_320, scale, pad_x, pad_y)
                # 跨类 NMS
                boxes_orig, sc_f, class_ids_f = cross_class_nms(
                    boxes_orig, sc_f, class_ids_f, iou_thresh=args.nms_iou,
                )
                # 再过一遍 score_thresh (NMS 后有些会被压掉)
                keep_final = sc_f >= args.score_thresh
                last_result = (boxes_orig[keep_final], sc_f[keep_final], class_ids_f[keep_final])
                # 累计统计
                for cid, s in zip(class_ids_f[keep_final], sc_f[keep_final]):
                    if 0 <= cid < len(COCO_NAMES) and COCO_NAMES[cid] is not None:
                        class_counter[cid] += 1
                        total_det += 1

            # 画框
            display_frame = frame
            if last_result is not None:
                boxes, scores, class_ids = last_result
                avg_fps = sum(fps_history) / len(fps_history) if fps_history else 0
                display_frame = draw_boxes(
                    frame.copy(), boxes, scores, class_ids,
                    fps=avg_fps, n_det=len(boxes), threshold=args.score_thresh,
                )
            last_display = display_frame

            t1 = time.perf_counter()
            fps = 1.0 / max(t1 - t0, 1e-6)
            fps_history.append(fps)

            # 节流写 PNG (避免每帧都写卡 IO)
            if use_png and (t1 - t_last_write) >= WRITE_INTERVAL:
                cv2.imwrite(str(display_path), display_frame)
                t_last_write = t1

            # 弹窗显示 (cv2 GUI)
            if use_imshow:
                try:
                    cv2.imshow("USB Camera - D1 Detection (q:quit s:save)", display_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("\n[INFO] 收到 q 键, 退出")
                        running = False
                    elif key == ord('s'):
                        ts = time.strftime("%Y%m%d_%H%M%S")
                        save_path = save_dir / f"manual_{ts}.jpg"
                        cv2.imwrite(str(save_path), display_frame)
                        print(f"[INFO] 保存: {save_path}")
                except cv2.error as e:
                    print(f"[WARN] imshow 失败 ({e}), 关闭弹窗模式, fallback 到 png")
                    use_imshow = False

            # stdout 汇总 (每秒一行)
            t_now = time.perf_counter()
            if t_now - t_last_log > 1.0:
                avg_fps = sum(fps_history) / len(fps_history)
                elapsed = t_now - t_start
                n_det = len(last_result[0]) if last_result else 0
                top3 = class_counter.most_common(3)
                top3_str = ", ".join(
                    f"{COCO_NAMES[c]}({n})" for c, n in top3
                ) if top3 else "(none)"
                print(
                    f"[F{frame_idx:06d} t={elapsed:6.1f}s] "
                    f"FPS={avg_fps:5.1f}  Det={n_det:2d}  "
                    f"TotalDet={total_det:5d}  Top3: {top3_str}"
                )
                t_last_log = t_now

            frame_idx += 1
            if args.duration and (t_now - t_start) >= args.duration:
                print(f"\n[INFO] 已达时长 {args.duration}s, 自动停止")
                break

    finally:
        cap.release()
        if use_imshow:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass
        # 退出时保存最后一帧到 detections/ (不覆盖 latest.jpg, 留个时间戳)
        if last_display is not None:
            ts = time.strftime("%Y%m%d_%H%M%S")
            final_path = save_dir / f"final_{ts}.jpg"
            cv2.imwrite(str(final_path), last_display)
            print(f"[INFO] 最后一帧保存: {final_path.resolve()}")
        elapsed = time.perf_counter() - t_start
        avg_fps = frame_idx / elapsed if elapsed > 0 else 0
        print()
        print("=" * 60)
        print(f"运行结束 | {frame_idx} 帧 | {elapsed:.1f}s | 平均 {avg_fps:.1f} fps")
        print(f"总检测数: {total_det}")
        if class_counter:
            print("Top 10 类别:")
            for cid, n in class_counter.most_common(10):
                name = COCO_NAMES[cid] if 0 <= cid < len(COCO_NAMES) and COCO_NAMES[cid] else f"cls{cid}"
                print(f"  {name:20s} {n:5d}")
        print("=" * 60)


if __name__ == "__main__":
    main()
