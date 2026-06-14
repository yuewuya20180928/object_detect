#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
三档模型配置

每档模型独立配置：
  - 预训练权重下载 URL
  - 输入图像尺寸
  - 推荐 batch_size
  - 学习率

切换方式：config.py 中改 MODEL_MODE = "speed" / "balanced" / "accuracy"
"""

from pathlib import Path

# ============================================================================
# 模型元数据
# ============================================================================
# TF 官方预训练权重（基于 COCO 2017 训练）
# 三个模型都从同一档预训练权重起步 → 迁移学习微调

MODEL_CONFIGS = {
    # ---------------------- 速度优先 ----------------------
    "speed": {
        "name": "SSD MobileNetV2 FPNLite 320x320",
        "framework": "tf_official",          # TF Object Detection API
        "pretrained_dir": "ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8",
        "pretrained_url": (
            "http://download.tensorflow.org/models/object_detection/"
            "tf2/20200710/"
            "ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz"
        ),
        # 训练超参
        "input_size": 320,
        "batch_size": 32,
        "learning_rate": 4e-3,
        "fine_tune_lr": 1e-4,
        "warmup_steps": 1000,
        "total_epochs": 30,                  # 小模型需要更多 epoch
        # 推理
        "inference_score_thresh": 0.5,
        "nms_iou_thresh": 0.5,
        # 参考性能
        "coco_map": 29.2,
        "model_size_mb": 14,
    },

    # ---------------------- 均衡档（推荐） ----------------------
    "balanced": {
        "name": "EfficientDet-D4",
        "framework": "tf_official",
        "pretrained_dir": "efficientdet_d4_coco17_tpu-32",
        "pretrained_url": (
            "https://storage.googleapis.com/cloudai/benchmarks/"
            "tf_object_detection/tf2/20200713/"
            "efficientdet_d4_coco17_tpu-32.tar.gz"
        ),
        "input_size": 512,
        "batch_size": 8,                     # 3090 (24G) 跑 D4 的甜点
        "learning_rate": 1e-3,
        "fine_tune_lr": 4e-5,
        "warmup_steps": 500,
        "total_epochs": 20,
        "inference_score_thresh": 0.5,
        "nms_iou_thresh": 0.5,
        "coco_map": 43.0,
        "model_size_mb": 65,
    },

    # ---------------------- 精度优先 ----------------------
    "accuracy": {
        "name": "EfficientDet-D7",
        "framework": "tf_official",
        "pretrained_dir": "efficientdet_d7_coco17_tpu-32",
        "pretrained_url": (
            "https://storage.googleapis.com/cloudai/benchmarks/"
            "tf_object_detection/tf2/20200713/"
            "efficientdet_d7_coco17_tpu-32.tar.gz"
        ),
        "input_size": 1024,                  # D7 必须大输入
        "batch_size": 2,                     # 24G 显存极限
        "learning_rate": 1e-3,
        "fine_tune_lr": 4e-5,
        "warmup_steps": 500,
        "total_epochs": 15,                  # 大模型少 epoch
        "inference_score_thresh": 0.5,
        "nms_iou_thresh": 0.5,
        "coco_map": 52.2,
        "model_size_mb": 256,
    },
}


# ============================================================================
# 工具函数
# ============================================================================
def get_model_config(mode: str) -> dict:
    """根据 mode 获取模型配置"""
    if mode not in MODEL_CONFIGS:
        raise ValueError(
            f"未知模型档位: {mode}，可选: {list(MODEL_CONFIGS.keys())}"
        )
    return MODEL_CONFIGS[mode]


def list_models() -> list:
    """列出所有可用模型档位"""
    return list(MODEL_CONFIGS.keys())


def download_pretrained(mode: str, save_dir: Path = None):
    """
    下载指定档位的预训练权重

    用法：
        from configs.model_configs import download_pretrained
        from pathlib import Path
        download_pretrained("balanced", Path("./pretrained"))
    """
    import urllib.request
    import tarfile
    import sys
    from tqdm import tqdm

    cfg = get_model_config(mode)
    if save_dir is None:
        save_dir = Path(__file__).parent.parent / "pretrained"
    save_dir.mkdir(parents=True, exist_ok=True)

    target_dir = save_dir / cfg["pretrained_dir"]
    if target_dir.exists() and any(target_dir.iterdir()):
        print(f"[跳过] 已存在: {target_dir}")
        return target_dir

    tar_path = save_dir / f"{cfg['pretrained_dir']}.tar.gz"
    if not tar_path.exists():
        url = cfg["pretrained_url"]
        print(f"[下载] {cfg['name']}")
        print(f"  URL: {url}")

        # 带进度条的下载
        def _report(count, block, total):
            pct = min(int(count * block * 100 / total), 100)
            sys.stdout.write(f"\r  进度: {pct}%  ")
            sys.stdout.flush()

        urllib.request.urlretrieve(url, tar_path, reporthook=_report)
        print()

    print(f"[解压] {tar_path.name}")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(save_dir)

    print(f"[完成] {target_dir}")
    return target_dir


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        nargs="?",
        default="balanced",
        choices=list_models(),
        help="要下载的模型档位",
    )
    args = parser.parse_args()
    download_pretrained(args.mode)
