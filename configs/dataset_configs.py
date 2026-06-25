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
        # 类别名动态生成：convert_to_tfrecord.py 运行时从 COCO annotations 加载
        # 避免手写 90 项 list 容易错位
        # cat_id 1-90 (含空洞 12, 26, 29, 30, 45, 66, 68, 69, 71, 83)
        "category_names": None,  # None 表示从 COCO JSON 动态加载
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

    # 如果类别名是 None，从 COCO annotations JSON 动态加载（避免手写 list 错位）
    if cfg["category_names"] is None:
        import json
        # 找 annotations JSON
        data_dir = Path(save_path).parent if save_path else Path(__file__).parent.parent / "data" / mode
        # data_dir 是 data/coco/, annotations 在 data/coco/raw/annotations/ 下
        ann_files = [
            data_dir / "raw" / "annotations" / "instances_train2017.json",
            data_dir / "raw" / "annotations" / "instances_val2017.json",
            data_dir / "annotations" / "instances_train2017.json",
            data_dir / "annotations" / "instances_val2017.json",
            data_dir / "raw" / "instances_train2017.json",
            data_dir / "raw" / "instances_val2017.json",
            data_dir / "instances_train2017.json",
            data_dir / "instances_val2017.json",
        ]
        ann_file = None
        for f in ann_files:
            if f.exists():
                ann_file = f
                break
        if ann_file is None:
            raise FileNotFoundError(f"找不到 COCO annotations, 尝试: {[str(f) for f in ann_files]}")

        with open(ann_file, "r", encoding="utf-8") as f:
            coco = json.load(f)

        # COCO cat_id 1-90，含空洞 (12, 26, 29, 30, 45, 66, 68, 69, 71, 83)
        # 生成 list，索引 = cat_id - 1，空洞用空字符串
        names = [""] * 90
        for c in coco["categories"]:
            if 1 <= c["id"] <= 90:
                names[c["id"] - 1] = c["name"]
        category_names_list = names
    else:
        category_names_list = cfg["category_names"]

    lines = []
    for idx, name in enumerate(category_names_list, start=1):
        lines.append("item {\n")
        lines.append(f"  id: {idx}\n")
        lines.append(f"  name: '{name}'\n")
        lines.append("}\n")

    content = "".join(lines)
    if save_path is None:
        save_path = Path(__file__).parent.parent / "data" / mode / "label_map.pbtxt"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text(content, encoding="utf-8")
    print(f"[label_map] {save_path}  ({len(category_names_list)} 项)")
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
