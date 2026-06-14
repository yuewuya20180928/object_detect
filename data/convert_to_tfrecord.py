#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
COCO 标注格式 → TFRecord

支持：
  - COCO 2017 (80 类)
  - Objects365 (365 类)

数据切分：80/10/10 (train/val/test)

输出：
  data/{dataset}/
    ├── train.record
    ├── val.record
    ├── test.record
    └── label_map.pbtxt

用法：
    python data/convert_to_tfrecord.py --dataset coco
    python data/convert_to_tfrecord.py --dataset coco --data-dir data/coco
    python data/convert_to_tfrecord.py --dataset objects365 --data-dir data/objects365
"""

import os
import sys
import json
import random
import hashlib
import io
from pathlib import Path
from typing import List, Dict, Optional

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from configs.dataset_configs import get_dataset_config, generate_label_map
from utils.logger import get_logger
import config  # noqa


# ============================================================================
# 默认路径配置（可通过命令行参数覆盖）
# ============================================================================
_DEFAULTS = {
    "coco": {
        "train_images_dir": "raw/train2017",
        "val_images_dir":   "raw/val2017",
        "train_ann":        "raw/annotations/instances_train2017.json",
        "val_ann":          "raw/annotations/instances_val2017.json",
        "val_split_ratio":  0.5,   # val 集中 50% 作为 val，50% 作为 test
    },
    "objects365": {
        "train_images_dir": "raw/objects365_train/images",
        "val_images_dir":   "raw/objects365_val/images",
        "train_ann":        "raw/objects365_train/objects365_train.json",
        "val_ann":          "raw/objects365_val/objects365_val.json",
        "val_split_ratio":  1.0,   # Objects365 只有 val 公开，test=val
    },
}


# ============================================================================
# COCO 标注解析
# ============================================================================
def load_coco_annotations(
    annotation_file: Path,
    image_dir: Path,
    logger,
    skip_empty: bool = True,
):
    """
    加载 COCO 格式标注

    Args:
        annotation_file: annotations JSON 路径
        image_dir:       图片目录路径
        logger:          日志记录器
        skip_empty:      是否跳过无标注的图片（默认跳过）

    Returns:
        images_info: [{
            "id": int,
            "file_name": str,
            "width": int,
            "height": int,
            "anns": [{"bbox": [x, y, w, h], "category_id": int}, ...]
        }, ...]
    """
    logger.info(f"加载标注: {annotation_file}")
    with open(annotation_file, "r", encoding="utf-8") as f:
        coco = json.load(f)

    # 1. 建立图片 id → 图片信息索引
    images_index = {img["id"]: img for img in coco["images"]}

    # 2. 建立 category_id 映射为连续的 1..N
    #    COCO 原始 category_id 不一定连续，直接用会有空洞
    cat_id_to_continuous = {
        cat["id"]: idx + 1
        for idx, cat in enumerate(sorted(coco["categories"], key=lambda c: c["id"]))
    }

    # 3. 收集每张图的所有标注
    image_anns: Dict[int, List] = {img_id: [] for img_id in images_index}
    for ann in coco["annotations"]:
        img_id = ann["image_id"]
        if img_id in image_anns:
            image_anns[img_id].append(ann)

    # 4. 整合为 images_info
    images_info = []
    missing = 0
    empty = 0
    for img_id, img in images_index.items():
        img_path = image_dir / img["file_name"]
        if not img_path.exists():
            missing += 1
            continue

        anns = []
        for ann in image_anns.get(img_id, []):
            x, y, w, h = ann["bbox"]
            if w <= 0 or h <= 0:
                continue
            anns.append({
                "bbox":       [x, y, w, h],
                "category_id": cat_id_to_continuous[ann["category_id"]],
            })

        if skip_empty and len(anns) == 0:
            empty += 1
            continue

        images_info.append({
            "id":        img_id,
            "file_name": img["file_name"],
            "width":     img["width"],
            "height":    img["height"],
            "anns":      anns,
        })

    logger.info(
        f"  总图片: {len(images_index)}, "
        f"有效(含标注): {len(images_info)}, "
        f"缺失(无图): {missing}, "
        f"跳过(无标注): {empty}"
    )
    return images_info


# ============================================================================
# TFRecord 特征构造
# ============================================================================
def _bytes_feature(value):
    import tensorflow as tf
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_list_feature(value):
    import tensorflow as tf
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _int64_list_feature(value):
    import tensorflow as tf
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def create_tf_example(
    image_info: dict,
    image_dir: Path,
    category_names: Optional[list] = None,
) -> "tf.train.Feature":
    """
    创建单个 TFRecord Example

    Args:
        image_info:       load_coco_annotations() 返回的单张图信息
        image_dir:        图片根目录
        category_names:   类别名列表，按连续 ID 索引（如 category_names[0]="person"）

    Returns:
        tf.train.Example 或 None（图片损坏或无有效标注时）
    """
    import tensorflow as tf
    from PIL import Image

    img_path = image_dir / image_info["file_name"]

    # 读取并校验图片
    with tf.io.gfile.GFile(str(img_path), "rb") as fid:
        encoded_jpg = fid.read()
    try:
        pil_img = Image.open(io.BytesIO(encoded_jpg))
        pil_img.verify()
    except Exception:
        return None

    key = hashlib.sha256(encoded_jpg).hexdigest()[:16]
    height = image_info["height"]
    width = image_info["width"]

    # 收集 bbox 和类别
    xmins, ymins, xmaxs, ymaxs = [], [], [], []
    classes_text, classes_label = [], []

    for ann in image_info["anns"]:
        x, y, w, h = ann["bbox"]
        xmin = max(x / width, 0.0)
        ymin = max(y / height, 0.0)
        xmax = min((x + w) / width, 1.0)
        ymax = min((y + h) / height, 1.0)
        if xmax <= xmin or ymax <= ymin:
            continue

        xmins.append(xmin)
        ymins.append(ymin)
        xmaxs.append(xmax)
        ymaxs.append(ymax)

        # 类别名：优先用真实名称，否则回退到 "object"
        cat_id = ann["category_id"]
        if category_names:
            cat_name = category_names[cat_id - 1].encode("utf-8")
        else:
            cat_name = b"object"
        classes_text.append(cat_name)
        classes_label.append(cat_id)

    if not xmins:
        return None

    # TFRecord Feature（格式与 TF Object Detection API 兼容）
    feature = {
        "image/height":             _int64_list_feature([height]),
        "image/width":              _int64_list_feature([width]),
        "image/filename":           _bytes_feature(image_info["file_name"].encode("utf-8")),
        "image/source_id":          _bytes_feature(str(image_info["id"]).encode("utf-8")),
        "image/key/sha256":         _bytes_feature(key.encode("utf-8")),
        "image/encoded":            _bytes_feature(encoded_jpg),
        "image/format":             _bytes_feature(b"jpeg"),
        "image/object/bbox/xmin":   _float_list_feature(xmins),
        "image/object/bbox/ymin":   _float_list_feature(ymins),
        "image/object/bbox/xmax":   _float_list_feature(xmaxs),
        "image/object/bbox/ymax":   _float_list_feature(ymaxs),
        # classes_text 用 \x00 分隔，支持变长解析
        "image/object/class/text":  _bytes_feature(b"\x00".join(classes_text)),
        "image/object/class/label": _int64_list_feature(classes_label),
    }

    return tf.train.Example(features=tf.train.Features(feature=feature))


def write_tfrecord(records: list, output_path: Path, logger):
    """将 Example 列表写入 TFRecord 文件"""
    import tensorflow as tf
    output_path.parent.mkdir(parents=True, exist_ok=True)
    valid = sum(1 for r in records if r is not None)
    logger.info(f"写入: {output_path} ({valid}/{len(records)} 条有效)")
    with tf.io.TFRecordWriter(str(output_path)) as writer:
        for rec in records:
            if rec is not None:
                writer.write(rec.SerializeToString())


# ============================================================================
# 数据集转换
# ============================================================================
def convert_dataset(
    dataset: str,
    data_dir: Path,
    logger,
    train_images_dir: str = None,
    val_images_dir: str = None,
    train_ann: str = None,
    val_ann: str = None,
    val_split_ratio: float = None,
):
    """
    将 COCO 格式数据集转换为 TFRecord

    Args:
        dataset:         数据集名称（"coco" | "objects365"）
        data_dir:        数据根目录（如 data/coco）
        logger:          日志记录器
        train_images_dir: 训练图片子目录（相对 data_dir）
        val_images_dir:   验证图片子目录
        train_ann:        训练标注文件路径（相对 data_dir）
        val_ann:          验证标注文件路径
        val_split_ratio:  验证集中多少比例作为 val（其余作 test）
    """
    cfg = get_dataset_config(dataset)
    defaults = _DEFAULTS[dataset]

    # 参数解析：命令行参数 > 默认值
    train_images_dir = train_images_dir or defaults["train_images_dir"]
    val_images_dir   = val_images_dir   or defaults["val_images_dir"]
    train_ann        = train_ann        or defaults["train_ann"]
    val_ann          = val_ann          or defaults["val_ann"]
    val_split_ratio  = val_split_ratio  if val_split_ratio is not None else defaults["val_split_ratio"]

    train_img_dir = data_dir / train_images_dir
    val_img_dir   = data_dir / val_images_dir
    train_ann_file = data_dir / train_ann
    val_ann_file   = data_dir / val_ann

    # 1. 生成 label_map.pbtxt
    label_map_path = generate_label_map(dataset, data_dir / "label_map.pbtxt")
    logger.info(f"Label map: {label_map_path} ({cfg['num_classes']} 类)")

    # 2. 加载标注
    train_images = load_coco_annotations(train_ann_file, train_img_dir, logger)
    val_images   = load_coco_annotations(val_ann_file,   val_img_dir,   logger)

    # 3. 切分 val → val + test
    random.seed(config.RANDOM_SEED)
    random.shuffle(val_images)
    split_idx = int(len(val_images) * val_split_ratio)
    val_split  = val_images[:split_idx]
    test_split = val_images[split_idx:]

    logger.info(f"数据切分: train={len(train_images)}, val={len(val_split)}, test={len(test_split)}")

    # 4. 写入 TFRecord
    train_tfrec = config.TRAIN_RECORD
    val_tfrec   = config.VAL_RECORD
    test_tfrec  = config.TEST_RECORD

    category_names = cfg["category_names"]

    logger.info("转换训练集...")
    train_records = [
        create_tf_example(img, train_img_dir, category_names)
        for img in train_images
    ]
    write_tfrecord(train_records, train_tfrec, logger)

    logger.info("转换验证集...")
    val_records = [
        create_tf_example(img, val_img_dir, category_names)
        for img in val_split
    ]
    write_tfrecord(val_records, val_tfrec, logger)

    logger.info("转换测试集...")
    test_records = [
        create_tf_example(img, val_img_dir, category_names)
        for img in test_split
    ]
    write_tfrecord(test_records, test_tfrec, logger)

    # 5. 汇总
    logger.info("=" * 50)
    logger.info("转换完成")
    logger.info(f"  Train: {len(train_records):>6d} -> {train_tfrec}")
    logger.info(f"  Val:   {len(val_records):>6d} -> {val_tfrec}")
    logger.info(f"  Test:  {len(test_records):>6d} -> {test_tfrec}")
    logger.info(f"  Label: {label_map_path}")
    logger.info("=" * 50)


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="COCO/Objects365 数据集转 TFRecord",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  python data/convert_to_tfrecord.py --dataset coco
  python data/convert_to_tfrecord.py --dataset coco --data-dir data/coco
  python data/convert_to_tfrecord.py --dataset objects365 --data-dir data/objects365
        """,
    )
    parser.add_argument(
        "--dataset",
        choices=["coco", "objects365"],
        default=config.DATASET_MODE,
        help="数据集名称（对应 configs/dataset_configs.py 中的 key）",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=config.DATA_DIR,
        help="数据根目录",
    )
    parser.add_argument(
        "--train-images-dir",
        type=str,
        default=None,
        help="训练图片子目录（相对 data-dir），省略则使用数据集默认值",
    )
    parser.add_argument(
        "--val-images-dir",
        type=str,
        default=None,
        help="验证图片子目录（相对 data-dir），省略则使用数据集默认值",
    )
    parser.add_argument(
        "--train-ann",
        type=str,
        default=None,
        help="训练标注文件路径（相对 data-dir），省略则使用数据集默认值",
    )
    parser.add_argument(
        "--val-ann",
        type=str,
        default=None,
        help="验证标注文件路径（相对 data-dir），省略则使用数据集默认值",
    )
    parser.add_argument(
        "--val-split-ratio",
        type=float,
        default=None,
        help="验证集中作为 val 的比例（其余作 test），默认 0.5",
    )
    args = parser.parse_args()

    logger = get_logger("convert", args.data_dir / "logs", verbose=True)
    convert_dataset(
        dataset         = args.dataset,
        data_dir        = args.data_dir,
        logger          = logger,
        train_images_dir = args.train_images_dir,
        val_images_dir  = args.val_images_dir,
        train_ann       = args.train_ann,
        val_ann         = args.val_ann,
        val_split_ratio = args.val_split_ratio,
    )


if __name__ == "__main__":
    main()