"""
DOKE-RAG Pipeline 模块

提供音频处理、PDF 解析等数据处理功能。
"""

from doke_rag.pipeline.audio_processor import (
    AudioProcessor,
    transcribe_audio
)

# PDF parser requires PaddleOCR (optional dependency)
try:
    from doke_rag.pipeline.pdf_parser import PDFParser
    _pdf_parser_available = True
except ImportError:
    _pdf_parser_available = False
    PDFParser = None

from doke_rag.pipeline.utils import (
    # 异常类
    PipelineError,
    AudioFileError,
    PDFParseError,
    APIError,
    ConfigError,
    ValidationError,

    # 工具函数
    validate_file_size,
    validate_file_format,
    validate_file_exists,
    get_timestamp_filename,
    save_metadata,
    load_metadata,
    setup_logger,
    get_logger,
    ensure_output_dir,
    format_timestamp,
    read_file,
    write_file,
    create_markdown_header,
    create_markdown_table
)

__all__ = [
    # 音频处理器
    "AudioProcessor",
    "transcribe_audio",

    # PDF 解析器
    "PDFParser",

    # 异常类
    "PipelineError",
    "AudioFileError",
    "PDFParseError",
    "APIError",
    "ConfigError",
    "ValidationError",

    # 工具函数
    "validate_file_size",
    "validate_file_format",
    "validate_file_exists",
    "get_timestamp_filename",
    "save_metadata",
    "load_metadata",
    "setup_logger",
    "get_logger",
    "ensure_output_dir",
    "format_timestamp",
    "read_file",
    "write_file",
    "create_markdown_header",
    "create_markdown_table"
]
