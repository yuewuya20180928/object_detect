#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一日志工具

提供：
  - get_logger(): 统一格式 logger
  - 彩色输出（INFO/WARN/ERROR）
  - 同时输出到控制台 + 文件
"""

import logging
import sys
from pathlib import Path
from datetime import datetime


# ANSI 颜色码（用 \x1b 代替 \033，兼容 Python 3.12+）
class _Colors:
    RESET   = "\x1b[0m"
    GREY    = "\x1b[90m"
    BLUE    = "\x1b[94m"
    GREEN   = "\x1b[92m"
    YELLOW  = "\x1b[93m"
    RED     = "\x1b[91m"
    BOLD    = "\x1b[1m"


class _ColorFormatter(logging.Formatter):
    """带颜色支持的日志格式化器"""

    LEVEL_COLORS = {
        logging.DEBUG:    _Colors.GREY,
        logging.INFO:     _Colors.GREEN,
        logging.WARNING:  _Colors.YELLOW,
        logging.ERROR:    _Colors.RED,
        logging.CRITICAL: _Colors.RED + _Colors.BOLD,
    }

    def format(self, record):
        color = self.LEVEL_COLORS.get(record.levelno, _Colors.RESET)
        # 简单方式：直接给字符串加颜色，不复制 record（兼容 Python 3.12+）
        # Python 3.14 移除了 logging.makeRecord，改用 makeLogRecord
        try:
            # Python 3.12+ 标准 API
            record_copy = logging.LogRecord(
                record.name, record.levelno, record.pathname, record.lineno,
                record.msg, record.args, record.exc_info,
                func=getattr(record, 'funcName', None),
                sinfo=getattr(record, 'stack_info', None),
            )
        except TypeError:
            # 兜底：使用 makeLogRecord
            record_copy = logging.makeLogRecord({
                'name': record.name,
                'levelno': record.levelno,
                'pathname': record.pathname,
                'lineno': record.lineno,
                'msg': record.msg,
                'args': record.args,
                'exc_info': record.exc_info,
            })
        record_copy.levelname = f"{color}{record.levelname:<7s}{_Colors.RESET}"
        record_copy.asctime = f"{_Colors.GREY}{self.formatTime(record, self.datefmt)}{_Colors.RESET}"
        return super().format(record_copy)


def get_logger(
    name: str = "object_detection",
    log_dir: Path = None,
    level: int = logging.INFO,
    log_to_file: bool = True,
    verbose: bool = True,
) -> logging.Logger:
    """
    获取统一格式的 logger

    Args:
        name:        logger 名称
        log_dir:     日志文件目录（None 则只输出到控制台）
        level:       日志级别
        log_to_file: 是否写文件
        verbose:     是否带颜色（生产环境可关）

    Returns:
        logging.Logger 实例
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()  # 避免重复添加 handler
    logger.propagate = False  # 避免冒泡到 root

    fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    # 控制台输出
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    if verbose:
        console_handler.setFormatter(_ColorFormatter(fmt, datefmt=datefmt))
    else:
        console_handler.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
    logger.addHandler(console_handler)

    # 文件输出
    if log_to_file and log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
        logger.addHandler(file_handler)
        logger.info(f"日志文件: {log_file}")

    return logger


# 全局单例
_logger_singleton = None

def get_global_logger() -> logging.Logger:
    """获取全局 logger（首次调用时从 config.py 加载配置）"""
    global _logger_singleton
    if _logger_singleton is None:
        try:
            import config
            _logger_singleton = get_logger(
                name="object_detection",
                log_dir=config.LOG_DIR if config.LOG_TO_FILE else None,
                level=logging.INFO,
                log_to_file=config.LOG_TO_FILE,
                verbose=config.VERBOSE,
            )
        except (ImportError, AttributeError):
            # 配置未加载时退化
            _logger_singleton = get_logger("object_detection", None, logging.INFO, False, True)
    return _logger_singleton


if __name__ == "__main__":
    # 自测
    logger = get_logger("test", Path("/tmp"), verbose=True)
    logger.debug("调试信息（应不显示）")
    logger.info("这是一条普通信息")
    logger.warning("这是一条警告")
    logger.error("这是一条错误")
    logger.critical("这是一条严重错误")
