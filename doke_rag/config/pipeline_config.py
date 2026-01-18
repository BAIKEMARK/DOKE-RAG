"""
Pipeline 配置管理模块

负责加载和管理音频处理、PDF 解析等 pipeline 的配置参数。
所有配置通过环境变量管理。
"""

from dotenv import load_dotenv
import os
from pathlib import Path

# 加载环境变量
load_dotenv()


class PipelineConfig:
    """Pipeline 配置管理类"""

    # ========== 音频处理配置 ==========
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    WHISPER_MODEL = os.getenv("WHISPER_MODEL", "whisper-1")
    WHISPER_LANGUAGE = os.getenv("WHISPER_LANGUAGE", "auto")
    MAX_AUDIO_SIZE_MB = int(os.getenv("MAX_AUDIO_SIZE", "500"))

    # ========== PDF 解析配置 ==========
    PADDLE_MODEL_DIR = Path(os.getenv("PADDLE_MODEL_DIR", "models/paddleocr/"))
    EXTRACT_IMAGES = os.getenv("EXTRACT_IMAGES", "true").lower() == "true"
    IMAGES_OUTPUT_DIR = Path(os.getenv("IMAGES_OUTPUT_DIR", "data/extracted_images/"))
    FORMULA_FORMAT = os.getenv("FORMULA_FORMAT", "latex")  # latex | image
    MAX_PDF_SIZE_MB = int(os.getenv("MAX_PDF_SIZE", "100"))

    # ========== 通用配置 ==========
    OUTPUT_DIR = Path(os.getenv("PIPELINE_OUTPUT_DIR", "data/processed/"))
    MODE = os.getenv("PIPELINE_MODE", "strict")  # strict | tolerant

    @classmethod
    def validate(cls):
        """验证配置有效性

        Raises:
            ValueError: 当必需的配置项缺失时
        """
        # 验证音频处理配置
        if not cls.OPENAI_API_KEY:
            raise ValueError("缺少 OPENAI_API_KEY 配置，请在 .env 文件中设置")

        # 验证模型名称
        valid_models = ["whisper-1"]
        if cls.WHISPER_MODEL not in valid_models:
            raise ValueError(f"不支持的 Whisper 模型: {cls.WHISPER_MODEL}")

        # 验证语言配置
        valid_languages = ["auto", "zh", "en"]
        if cls.WHISPER_LANGUAGE not in valid_languages:
            raise ValueError(f"不支持的语言配置: {cls.WHISPER_LANGUAGE}")

        # 验证文件大小限制
        if cls.MAX_AUDIO_SIZE_MB <= 0:
            raise ValueError("MAX_AUDIO_SIZE 必须大于 0")

        if cls.MAX_PDF_SIZE_MB <= 0:
            raise ValueError("MAX_PDF_SIZE 必须大于 0")

        # 验证模式配置
        if cls.MODE not in ["strict", "tolerant"]:
            raise ValueError(f"PIPELINE_MODE 必须是 'strict' 或 'tolerant'，当前值: {cls.MODE}")

        return True

    @classmethod
    def get_audio_config(cls) -> dict:
        """获取音频处理配置

        Returns:
            dict: 包含音频处理相关配置的字典
        """
        return {
            "api_key": cls.OPENAI_API_KEY,
            "model": cls.WHISPER_MODEL,
            "language": cls.WHISPER_LANGUAGE,
            "max_size_mb": cls.MAX_AUDIO_SIZE_MB
        }

    @classmethod
    def get_pdf_config(cls) -> dict:
        """获取 PDF 解析配置

        Returns:
            dict: 包含 PDF 解析相关配置的字典
        """
        return {
            "model_dir": cls.PADDLE_MODEL_DIR,
            "extract_images": cls.EXTRACT_IMAGES,
            "images_output_dir": cls.IMAGES_OUTPUT_DIR,
            "formula_format": cls.FORMULA_FORMAT,
            "max_size_mb": cls.MAX_PDF_SIZE_MB
        }

    @classmethod
    def is_strict_mode(cls) -> bool:
        """检查是否为严格模式

        Returns:
            bool: True 表示严格模式，False 表示容错模式
        """
        return cls.MODE == "strict"


# 在模块加载时自动验证配置
try:
    PipelineConfig.validate()
except ValueError as e:
    import warnings
    warnings.warn(f"Pipeline 配置验证失败: {str(e)}", UserWarning)
