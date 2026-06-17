#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可视化工具

提供：
  - draw_boxes(): 在图片上画检测框
  - draw_boxes_with_labels(): 带类别名和置信度
  - save_detection_image(): 保存带检测框的图片
  - plot_training_curves(): 绘制训练曲线
"""

from pathlib import Path
from typing import List, Tuple, Union
import numpy as np
import cv2


# 颜色调色板（22种，循环使用）
_COLORS = [
    (255, 0, 0),     # 红
    (0, 255, 0),     # 绿
    (0, 0, 255),     # 蓝
    (255, 255, 0),   # 黄
    (255, 0, 255),   # 紫
    (0, 255, 255),   # 青
    (128, 0, 0),     # 暗红
    (0, 128, 0),     # 暗绿
    (0, 0, 128),     # 暗蓝
    (128, 128, 0),   # 橄榄
    (128, 0, 128),   # 暗紫
    (0, 128, 128),   # 暗青
    (255, 128, 0),   # 橙
    (255, 0, 128),   # 粉
    (128, 255, 0),   # 黄绿
    (0, 128, 255),   # 天蓝
    (128, 0, 255),   # 紫蓝
    (255, 128, 128), # 浅红
    (128, 255, 128), # 浅绿
    (128, 128, 255), # 浅蓝
    (255, 255, 128), # 浅黄
    (64, 200, 200),  # 青绿
]


def get_color(class_id: int) -> Tuple[int, int, int]:
    """根据类别 ID 获取颜色（BGR 格式，OpenCV 用）"""
    return _COLORS[class_id % len(_COLORS)]


def draw_boxes(
    image: np.ndarray,
    boxes: Union[np.ndarray, List],
    scores: Union[np.ndarray, List, None] = None,
    class_ids: Union[np.ndarray, List, None] = None,
    class_names: Union[List[str], None] = None,
    score_thresh: float = 0.3,
    line_thickness: int = 2,
    font_scale: float = 0.55,
) -> np.ndarray:
    """
    在图片上画检测框

    Args:
        image:        BGR 图片（H, W, 3）
        boxes:        检测框，格式 (N, 4) [x_min, y_min, x_max, y_max]（像素坐标）
        scores:       置信度 (N,)
        class_ids:    类别 ID (N,)
        class_names:  类别名列表（可选）
        score_thresh: 显示的最低置信度
        line_thickness: 框线粗细
        font_scale:   字体大小

    Returns:
        画好框的图片（BGR 格式）
    """
    img = image.copy()
    if len(boxes) == 0:
        return img

    boxes = np.asarray(boxes).reshape(-1, 4)
    if scores is None:
        scores = np.ones(len(boxes))
    else:
        scores = np.asarray(scores)
    if class_ids is None:
        class_ids = np.zeros(len(boxes), dtype=int)
    else:
        class_ids = np.asarray(class_ids, dtype=int)

    for box, score, cls_id in zip(boxes, scores, class_ids):
        if score < score_thresh:
            continue
        x1, y1, x2, y2 = map(int, box)
        color = get_color(int(cls_id))

        # 画矩形
        cv2.rectangle(img, (x1, y1), (x2, y2), color, line_thickness)

        # 文字
        if class_names is not None and int(cls_id) < len(class_names):
            label = f"{class_names[int(cls_id)]}: {score:.2f}"
        else:
            label = f"id={cls_id}: {score:.2f}"
        (tw, th), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
        )
        # 标签背景
        cv2.rectangle(img, (x1, y1 - th - baseline - 4), (x1 + tw + 4, y1), color, -1)
        # 标签文字（黑色，背景上更清晰）
        cv2.putText(
            img, label, (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 1, cv2.LINE_AA
        )

    return img


def save_detection_image(
    image: np.ndarray,
    boxes,
    scores=None,
    class_ids=None,
    class_names=None,
    save_path: Union[str, Path] = None,
    score_thresh: float = 0.3,
):
    """保存带检测框的图片"""
    img_out = draw_boxes(image, boxes, scores, class_ids, class_names, score_thresh)
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_path), img_out)
    return img_out


def plot_training_curves(
    history: dict,
    save_path: Union[str, Path] = None,
    figsize: Tuple[int, int] = (12, 5),
):
    """
    绘制训练曲线

    Args:
        history: 训练历史，键如 'loss', 'val_loss', 'mAP', 'val_mAP'
        save_path: 图片保存路径
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib 未安装，跳过绘图")
        return

    metrics = [k for k in history.keys() if not k.startswith("val_")]
    val_metrics = [k for k in history.keys() if k.startswith("val_")]
    pairs = [(m, f"val_{m}") for m in metrics if f"val_{m}" in val_metrics]

    n = max(len(pairs), 1)
    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]

    for ax, (train_k, val_k) in zip(axes, pairs):
        ax.plot(history[train_k], label=f"train {train_k}", linewidth=2)
        ax.plot(history[val_k],   label=f"val {val_k}",   linewidth=2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(train_k)
        ax.set_title(train_k)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"训练曲线已保存: {save_path}")
    else:
        plt.show()
    plt.close()


if __name__ == "__main__":
    # 自测：画几个随机框
    img = np.zeros((400, 600, 3), dtype=np.uint8)
    img[:] = (50, 50, 50)
    boxes = np.array([[50, 50, 200, 200], [300, 100, 500, 350]])
    scores = np.array([0.95, 0.78])
    class_ids = np.array([0, 2])
    class_names = ["cat", "dog", "person"]
    out = draw_boxes(img, boxes, scores, class_ids, class_names)
    save_detection_image(out, [], save_path=None)  # 不报错就行
    print("可视化自测通过")
