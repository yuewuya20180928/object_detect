"""
Backbones 模块

提供：
  - build_ssd_mobilenet_backbone: SSD 风格（速度优先档）
  - build_efficientdet_backbone: EfficientDet 风格（均衡 / 精度档）
  - get_recommended_input_size:  获取推荐输入尺寸
"""

from .ssd_mobilenet import build_ssd_mobilenet_backbone
from .efficientdet import build_efficientdet_backbone, get_recommended_input_size

__all__ = [
    "build_ssd_mobilenet_backbone",
    "build_efficientdet_backbone",
    "get_recommended_input_size",
]
