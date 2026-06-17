#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据集下载工具

支持：
  - COCO 2017（官方直链，~25GB）
  - Objects365（需注册，本脚本只生成下载指引）

用法：
    python data/download.py coco
    python data/download.py objects365   # 打印注册指引
"""

import os
import sys
import urllib.request
import zipfile
from pathlib import Path

# 把项目根目录加入 path
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from configs.dataset_configs import get_dataset_config, list_datasets
from utils.logger import get_logger


def _download_with_progress(url: str, dest: Path, logger):
    """带进度条的文件下载"""
    if dest.exists():
        logger.info(f"已存在，跳过: {dest.name}")
        return

    logger.info(f"开始下载: {url}")
    logger.info(f"保存到: {dest}")

    # 尝试用 requests + tqdm 获得更友好体验
    try:
        import requests
        from tqdm import tqdm

        resp = requests.get(url, stream=True, timeout=30)
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))

        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=dest.name, ncols=80
        ) as pbar:
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
    except ImportError:
        # 兜底：urllib
        logger.warning("未安装 requests/tqdm，使用 urllib 下载（无进度条）")
        dest.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, dest)


def _extract_zip(zip_path: Path, extract_to: Path, logger):
    """解压 zip"""
    logger.info(f"解压: {zip_path.name} -> {extract_to}")
    extract_to.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)
    logger.info(f"解压完成")


def download_coco(data_dir: Path, logger):
    """下载 COCO 2017"""
    cfg = get_dataset_config("coco")
    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # 1. 训练图片 (~18GB)
    train_zip = raw_dir / "train2017.zip"
    if not (raw_dir / "train2017").exists():
        _download_with_progress(cfg["train_images_url"], train_zip, logger)
        _extract_zip(train_zip, raw_dir, logger)
        train_zip.unlink()  # 节省空间
    else:
        logger.info(f"已存在: {raw_dir / 'train2017'}")

    # 2. 验证图片 (~1GB)
    val_zip = raw_dir / "val2017.zip"
    if not (raw_dir / "val2017").exists():
        _download_with_progress(cfg["val_images_url"], val_zip, logger)
        _extract_zip(val_zip, raw_dir, logger)
        val_zip.unlink()
    else:
        logger.info(f"已存在: {raw_dir / 'val2017'}")

    # 3. 标注文件 (~240MB)
    anno_zip = raw_dir / "annotations.zip"
    if not (raw_dir / "annotations").exists():
        _download_with_progress(cfg["annotations_url"], anno_zip, logger)
        _extract_zip(anno_zip, raw_dir, logger)
        anno_zip.unlink()
    else:
        logger.info(f"已存在: {raw_dir / 'annotations'}")

    logger.info("COCO 2017 下载完成")
    logger.info(f"  训练图片: {raw_dir / 'train2017'}")
    logger.info(f"  验证图片: {raw_dir / 'val2017'}")
    logger.info(f"  标注文件: {raw_dir / 'annotations'}")


def download_objects365(data_dir: Path, logger):
    """Objects365 下载指引（需注册）"""
    cfg = get_dataset_config("objects365")
    logger.warning("Objects365 需在官网注册后才能下载")
    logger.info(f"注册地址: {cfg['url']}")
    logger.info("注册后会收到邮件包含下载链接")
    logger.info("下载后请将文件放置到:")
    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"  {raw_dir}")
    logger.info("目录结构示例：")
    logger.info(f"  {raw_dir}/")
    logger.info(f"  ├── objects365_train/")
    logger.info(f"  │   ├── images/  (训练图片)")
    logger.info(f"  │   └── objects365_train.json  (标注)")
    logger.info(f"  └── objects365_val/")
    logger.info(f"      ├── images/")
    logger.info(f"      └── objects365_val.json")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="数据集下载工具")
    parser.add_argument(
        "dataset",
        choices=list_datasets(),
        help="要下载的数据集",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "coco",
        help="数据保存目录",
    )
    args = parser.parse_args()

    logger = get_logger("download", args.data_dir, verbose=False)
    logger.info(f"目标数据集: {args.dataset}")
    logger.info(f"保存目录: {args.data_dir}")

    if args.dataset == "coco":
        download_coco(args.data_dir, logger)
    elif args.dataset == "objects365":
        # objects365 用 dataset 名字作子目录
        args.data_dir = args.data_dir.parent / "objects365"
        download_objects365(args.data_dir, logger)


if __name__ == "__main__":
    main()
