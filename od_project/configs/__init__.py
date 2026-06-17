# 配置文件初始化
from .model_configs import MODEL_CONFIGS, get_model_config
from .dataset_configs import DATASET_CONFIGS, get_dataset_config

__all__ = [
    "MODEL_CONFIGS",
    "DATASET_CONFIGS",
    "get_model_config",
    "get_dataset_config",
]
