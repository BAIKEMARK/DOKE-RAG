"""
Pipeline 工具函数模块

提供文件验证、路径管理、日志记录、元数据保存等共享功能。
"""

import os
import logging
from pathlib import Path
from datetime import datetime
import json
from typing import Any, Dict


# ========== 自定义异常类 ==========

class PipelineError(Exception):
    """Pipeline 基础异常类"""
    pass


class AudioFileError(PipelineError):
    """音频文件错误（格式、大小、损坏）"""
    pass


class PDFParseError(PipelineError):
    """PDF 解析错误（加密、损坏、OCR 失败）"""
    pass


class APIError(PipelineError):
    """API 调用错误"""
    pass


class ConfigError(PipelineError):
    """配置错误"""
    pass


class ValidationError(PipelineError):
    """验证错误"""
    pass


# ========== 文件验证函数 ==========

def validate_file_size(file_path: str, max_size_mb: int) -> bool:
    """验证文件大小是否在限制范围内

    Args:
        file_path: 文件路径
        max_size_mb: 最大文件大小（MB）

    Returns:
        bool: True 表示文件大小符合要求

    Raises:
        FileNotFoundError: 文件不存在
        ValidationError: 文件大小超过限制
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")

    size_mb = path.stat().st_size / (1024 * 1024)

    if size_mb > max_size_mb:
        raise ValidationError(
            f"文件过大: {size_mb:.2f}MB (限制: {max_size_mb}MB)"
        )

    return True


def validate_file_format(file_path: str, allowed_formats: list) -> bool:
    """验证文件格式

    Args:
        file_path: 文件路径
        allowed_formats: 允许的格式列表（如 ['.mp3', '.wav']）

    Returns:
        bool: True 表示格式符合要求

    Raises:
        ValidationError: 文件格式不符合要求
    """
    path = Path(file_path)
    ext = path.suffix.lower()

    if ext not in allowed_formats:
        raise ValidationError(
            f"不支持的文件格式: {ext}，支持的格式: {', '.join(allowed_formats)}"
        )

    return True


def validate_file_exists(file_path: str) -> Path:
    """验证文件是否存在并可访问

    Args:
        file_path: 文件路径

    Returns:
        Path: 文件的 Path 对象

    Raises:
        FileNotFoundError: 文件不存在
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")

    if not path.is_file():
        raise ValueError(f"路径不是文件: {file_path}")

    return path


# ========== 文件名生成函数 ==========

def get_timestamp_filename(prefix: str, extension: str) -> str:
    """生成带时间戳的文件名

    Args:
        prefix: 文件名前缀
        extension: 文件扩展名（不包含点号）

    Returns:
        str: 格式为 "prefix_YYYYMMDD_HHMMSS.extension" 的文件名

    Examples:
        >>> get_timestamp_filename("audio", "md")
        'audio_20250118_143022.md'
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}.{extension}"


def get_safe_filename(filename: str) -> str:
    """生成安全的文件名（移除特殊字符）

    Args:
        filename: 原始文件名

    Returns:
        str: 安全的文件名
    """
    # 移除或替换不安全的字符
    unsafe_chars = ['<', '>', ':', '"', '/', '\\', '|', '?', '*']
    safe_name = filename

    for char in unsafe_chars:
        safe_name = safe_name.replace(char, '_')

    return safe_name


# ========== 元数据处理函数 ==========

def save_metadata(metadata: Dict[str, Any], output_path: str) -> None:
    """保存元数据到 JSON 文件

    Args:
        metadata: 元数据字典
        output_path: 输出文件路径

    Raises:
        IOError: 文件写入失败
    """
    try:
        output_path_obj = Path(output_path)
        ensure_output_dir(output_path_obj.parent)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

    except Exception as e:
        raise IOError(f"保存元数据失败: {str(e)}")


def load_metadata(metadata_path: str) -> Dict[str, Any]:
    """从 JSON 文件加载元数据

    Args:
        metadata_path: 元数据文件路径

    Returns:
        dict: 元数据字典

    Raises:
        FileNotFoundError: 文件不存在
        json.JSONDecodeError: JSON 格式错误
    """
    path = Path(metadata_path)

    if not path.exists():
        raise FileNotFoundError(f"元数据文件不存在: {metadata_path}")

    with open(metadata_path, "r", encoding="utf-8") as f:
        return json.load(f)


# ========== 日志记录函数 ==========

def setup_logger(
    name: str,
    log_file: str = None,
    level: int = logging.INFO
) -> logging.Logger:
    """设置日志记录器

    Args:
        name: 日志记录器名称
        log_file: 日志文件路径（可选）
        level: 日志级别

    Returns:
        logging.Logger: 配置好的日志记录器
    """
    logger = logging.getLogger(name)

    # 避免重复添加 handler
    if logger.handlers:
        return logger

    logger.setLevel(level)

    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)

    # 文件处理器
    if log_file:
        log_path = Path(log_file)
        ensure_output_dir(log_path.parent)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        logger.addHandler(file_handler)

    # 格式化器
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """获取或创建日志记录器

    Args:
        name: 日志记录器名称

    Returns:
        logging.Logger: 日志记录器
    """
    return setup_logger(name)


# ========== 目录管理函数 ==========

def ensure_output_dir(directory: Path) -> None:
    """确保输出目录存在，不存在则创建

    Args:
        directory: 目录路径

    Raises:
        OSError: 目录创建失败
    """
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        raise OSError(f"创建目录失败: {directory}, 错误: {str(e)}")


def clean_directory(directory: Path, pattern: str = "*") -> int:
    """清理目录中的文件

    Args:
        directory: 目录路径
        pattern: 文件匹配模式

    Returns:
        int: 删除的文件数量

    Raises:
        OSError: 删除文件失败
    """
    if not directory.exists():
        return 0

    count = 0
    try:
        for file_path in directory.glob(pattern):
            if file_path.is_file():
                file_path.unlink()
                count += 1
        return count
    except Exception as e:
        raise OSError(f"清理目录失败: {directory}, 错误: {str(e)}")


# ========== 时间处理函数 ==========

def format_timestamp(seconds: float) -> str:
    """将秒数转换为时间戳格式 (MM:SS 或 HH:MM:SS)

    Args:
        seconds: 秒数

    Returns:
        str: 格式化的时间戳

    Examples:
        >>> format_timestamp(65.5)
        '01:05'
        >>> format_timestamp(3665.5)
        '01:01:05'
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    else:
        return f"{minutes:02d}:{secs:02d}"


def get_current_timestamp() -> str:
    """获取当前时间戳字符串

    Returns:
        str: ISO 格式的时间戳
    """
    return datetime.now().isoformat()


# ========== 文件读写辅助函数 ==========

def read_file(file_path: str, encoding: str = "utf-8") -> str:
    """读取文件内容

    Args:
        file_path: 文件路径
        encoding: 文件编码

    Returns:
        str: 文件内容

    Raises:
        FileNotFoundError: 文件不存在
        UnicodeDecodeError: 解码失败
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")

    with open(file_path, "r", encoding=encoding) as f:
        return f.read()


def write_file(
    file_path: str,
    content: str,
    encoding: str = "utf-8"
) -> None:
    """写入文件内容

    Args:
        file_path: 文件路径
        content: 文件内容
        encoding: 文件编码

    Raises:
        IOError: 写入失败
    """
    try:
        path = Path(file_path)
        ensure_output_dir(path.parent)

        with open(file_path, "w", encoding=encoding) as f:
            f.write(content)

    except Exception as e:
        raise IOError(f"写入文件失败: {file_path}, 错误: {str(e)}")


# ========== Markdown 格式化辅助函数 ==========

def create_markdown_header(title: str, level: int = 1) -> str:
    """创建 Markdown 标题

    Args:
        title: 标题文本
        level: 标题级别（1-6）

    Returns:
        str: Markdown 格式的标题
    """
    if not 1 <= level <= 6:
        raise ValueError("标题级别必须在 1-6 之间")

    prefix = "#" * level
    return f"{prefix} {title}"


def create_markdown_table(headers: list, rows: list) -> str:
    """创建 Markdown 表格

    Args:
        headers: 表头列表
        rows: 行数据列表（每行是一个列表）

    Returns:
        str: Markdown 格式的表格

    Examples:
        >>> headers = ["列1", "列2"]
        >>> rows = [["数据1", "数据2"], ["数据3", "数据4"]]
        >>> create_markdown_table(headers, rows)
        '| 列1 | 列2 |\\n|---|---|\\n| 数据1 | 数据2 |\\n| 数据3 | 数据4 |'
    """
    if not headers:
        return ""

    # 表头
    lines = ["| " + " | ".join(headers) + " |"]

    # 分隔线
    separator = "| " + " | ".join(["---"] * len(headers)) + " |"
    lines.append(separator)

    # 表体
    for row in rows:
        if len(row) != len(headers):
            raise ValueError("行数据列数与表头不匹配")
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)
