#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
两档数据集配置

支持：
  - COCO 2017 (80类)
  - Objects365 (365类)

切换方式：config.py 中改 DATASET_MODE = "coco" / "objects365"
"""

from pathlib import Path

# ============================================================================
# 数据集元数据
# ============================================================================

DATASET_CONFIGS = {
    # ---------------------- COCO 2017 ----------------------
    "coco": {
        "name": "COCO 2017",
        "num_classes": 80,
        "num_train_images": 118287,
        "num_val_images": 5000,
        # 下载地址（COCO 官方）
        "train_images_url": "http://images.cocodataset.org/zips/train2017.zip",
        "val_images_url":   "http://images.cocodataset.org/zips/val2017.zip",
        "annotations_url":  "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
        "annotation_format": "coco_json",
        "annotation_file": "annotations/instances_train2017.json",
        "val_annotation_file": "annotations/instances_val2017.json",
        # 类别名（80 类，顺序与 COCO id 对齐）
        "category_names": [
            "person", "bicycle", "car", "motorcycle", "airplane",
            "bus", "train", "truck", "boat", "traffic light",
            "fire hydrant", "stop sign", "parking meter", "bench", "bird",
            "cat", "dog", "horse", "sheep", "cow",
            "elephant", "bear", "zebra", "giraffe", "backpack",
            "umbrella", "handbag", "tie", "suitcase", "frisbee",
            "skis", "snowboard", "sports ball", "kite", "baseball bat",
            "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
            "wine glass", "cup", "fork", "knife", "spoon",
            "bowl", "banana", "apple", "sandwich", "orange",
            "broccoli", "carrot", "hot dog", "pizza", "donut",
            "cake", "chair", "couch", "potted plant", "bed",
            "dining table", "toilet", "tv", "laptop", "mouse",
            "remote", "keyboard", "cell phone", "microwave", "oven",
            "toaster", "sink", "refrigerator", "book", "clock",
            "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
        ],
    },

    # ---------------------- Objects365 ----------------------
    "objects365": {
        "name": "Objects365",
        "num_classes": 365,
        "num_train_images": 1700000,        # 官方约 170 万
        "num_val_images": 80000,
        # 下载地址：需在官网注册后获取
        "url": "https://www.objects365.org/download.html",
        "annotation_format": "coco_json",
        "train_annotation_file": "objects365_train.json",
        "val_annotation_file": "objects365_val.json",
        "note": (
            "Objects365 需在 https://www.objects365.org/ 注册邮箱，"
            "获取下载链接。数据集较大（~60GB），建议 SSD 存储。"
        ),
        # 类别名（365 类）
        # 为节省空间此处省略，运行时从 annotation JSON 读取
        "category_names": None,              # 运行时加载
    },
}


# ============================================================================
# 工具函数
# ============================================================================
def get_dataset_config(mode: str) -> dict:
    """根据 mode 获取数据集配置"""
    if mode not in DATASET_CONFIGS:
        raise ValueError(
            f"未知数据集: {mode}，可选: {list(DATASET_CONFIGS.keys())}"
        )
    return DATASET_CONFIGS[mode]


def list_datasets() -> list:
    """列出所有可用数据集"""
    return list(DATASET_CONFIGS.keys())


def generate_label_map(mode: str, save_path: Path = None) -> Path:
    """
    生成 TF Object Detection 标准的 label_map.pbtxt

    格式：
        item {
          id: 1
          name: 'person'
        }
        item {
          id: 2
          name: 'bicycle'
        }
        ...
    """
    cfg = get_dataset_config(mode)

    # 如果类别名是 None（Objects365），从 annotation JSON 加载
    if cfg["category_names"] is None:
        # 留给后续实现：从 annotation 文件读取
        raise NotImplementedError(
            "Objects365 类别名需从 annotation JSON 动态加载"
        )

    lines = []
    for idx, name in enumerate(cfg["category_names"], start=1):
        lines.append("item {\n")
        lines.append(f"  id: {idx}\n")
        lines.append(f"  name: '{name}'\n")
        lines.append("}\n")

    content = "".join(lines)
    if save_path is None:
        save_path = Path(__file__).parent.parent / "data" / mode / "label_map.pbtxt"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text(content, encoding="utf-8")
    print(f"[label_map] {save_path}  ({cfg['num_classes']} 类)")
    return save_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action",
        choices=["list", "labelmap"],
        help="list: 列出数据集；labelmap: 生成 label_map.pbtxt",
    )
    parser.add_argument("--mode", choices=list_datasets(), default="coco")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if args.action == "list":
        for m in list_datasets():
            c = get_dataset_config(m)
            print(f"  {m:12s} | {c['name']:12s} | {c['num_classes']:3d} 类")
    elif args.action == "labelmap":
        generate_label_map(args.mode, args.output)
