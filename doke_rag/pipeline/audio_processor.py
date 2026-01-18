"""
音频处理器模块

使用 OpenAI Whisper API 将音频文件转录为 Markdown 文本。
支持多种音频格式和多语言转录。
"""

import asyncio
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime

from openai import AsyncOpenAI
from doke_rag.config.pipeline_config import PipelineConfig
from doke_rag.pipeline.utils import (
    AudioFileError,
    APIError,
    validate_file_size,
    validate_file_format,
    validate_file_exists,
    get_timestamp_filename,
    save_metadata,
    write_file,
    format_timestamp,
    setup_logger,
    ensure_output_dir
)


class AudioProcessor:
    """音频转录处理器，使用 Whisper API 将音频转换为 Markdown"""

    # 支持的音频格式
    SUPPORTED_FORMATS = [".mp3", ".wav", ".m4a", ".mp4", ".mpeg", ".mpga", ".ogg", ".flac"]

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "whisper-1",
        strict_mode: bool = True
    ):
        """初始化音频处理器

        Args:
            api_key: OpenAI API 密钥（如果不提供，从配置中读取）
            model: Whisper 模型名称（默认 whisper-1）
            strict_mode: 是否使用严格模式（遇到错误立即抛出异常）

        Raises:
            ConfigError: API 密钥未配置
        """
        # 获取 API 密钥
        self.api_key = api_key or PipelineConfig.OPENAI_API_KEY
        if not self.api_key:
            raise APIError("OPENAI_API_KEY 未配置，请在 .env 文件中设置")

        # 初始化 OpenAI 客户端
        self.client = AsyncOpenAI(api_key=self.api_key)
        self.model = model
        self.strict_mode = strict_mode

        # 设置日志记录器
        self.logger = setup_logger("AudioProcessor")

        self.logger.info(f"音频处理器初始化完成（模型: {self.model}）")

    async def transcribe(
        self,
        audio_path: str,
        output_dir: str,
        language: str = "auto",
        prompt: Optional[str] = None,
        temperature: float = 0.0
    ) -> Dict[str, str]:
        """转录音频文件为 Markdown

        Args:
            audio_path: 音频文件路径
            output_dir: 输出目录
            language: 语言提示（zh/en/auto）
            prompt: 可选的提示文本，用于提高转录准确率
            temperature: 转录温度（0.0-1.0），越低越确定

        Returns:
            dict: 包含 markdown_path, metadata_path 的字典

        Raises:
            AudioFileError: 音频文件无效
            APIError: API 调用失败
            ValidationError: 验证失败
        """
        try:
            self.logger.info(f"开始转录音频: {audio_path}")

            # 1. 验证文件
            self._validate_audio(audio_path)

            # 2. 调用 Whisper API
            self.logger.info("调用 Whisper API...")
            transcript = await self._call_whisper_api(audio_path, language, prompt, temperature)

            # 3. 生成 Markdown
            self.logger.info("生成 Markdown 文件...")
            markdown_content = self._format_markdown(transcript, audio_path)

            # 4. 保存文件
            output_path = Path(output_dir)
            ensure_output_dir(output_path)

            timestamp = get_timestamp_filename("audio", "md")
            markdown_path = output_path / timestamp
            metadata_path = output_path / timestamp.replace(".md", "_metadata.json")

            write_file(str(markdown_path), markdown_content)
            self.logger.info(f"Markdown 已保存: {markdown_path}")

            # 5. 保存元数据
            metadata = self._create_metadata(transcript, audio_path)
            save_metadata(metadata, str(metadata_path))
            self.logger.info(f"元数据已保存: {metadata_path}")

            return {
                "markdown_path": str(markdown_path),
                "metadata_path": str(metadata_path),
                "duration": transcript.get("duration", 0),
                "language": transcript.get("language", "unknown")
            }

        except Exception as e:
            self.logger.error(f"音频转录失败: {str(e)}")

            if self.strict_mode:
                raise
            else:
                # 容错模式：返回错误信息
                return {
                    "error": str(e),
                    "markdown_path": "",
                    "metadata_path": ""
                }

    def _validate_audio(self, file_path: str) -> None:
        """验证音频文件格式、大小、可访问性

        Args:
            file_path: 音频文件路径

        Raises:
            AudioFileError: 音频文件无效
            ValidationError: 验证失败
        """
        # 检查文件是否存在
        validate_file_exists(file_path)

        # 检查格式
        validate_file_format(file_path, self.SUPPORTED_FORMATS)

        # 检查大小
        max_size = PipelineConfig.MAX_AUDIO_SIZE_MB
        validate_file_size(file_path, max_size)

        self.logger.debug(f"音频文件验证通过: {file_path}")

    async def _call_whisper_api(
        self,
        audio_path: str,
        language: str,
        prompt: Optional[str] = None,
        temperature: float = 0.0
    ) -> Dict:
        """调用 OpenAI Whisper API

        Args:
            audio_path: 音频文件路径
            language: 语言提示
            prompt: 可选提示
            temperature: 温度参数

        Returns:
            dict: 转录结果

        Raises:
            APIError: API 调用失败
        """
        try:
            # 打开音频文件
            with open(audio_path, "rb") as audio_file:
                # 构建请求参数
                params = {
                    "model": self.model,
                    "file": audio_file,
                    "response_format": "verbose_json",  # 包含时间戳和详细信息
                    "temperature": temperature
                }

                # 添加语言参数（如果不是 auto）
                if language and language != "auto":
                    params["language"] = language

                # 添加提示（如果提供）
                if prompt:
                    params["prompt"] = prompt

                # 调用 API
                self.logger.debug(f"API 请求参数: model={params['model']}, language={language}")
                transcript = await self.client.audio.transcriptions.create(**params)

                # 转换为字典
                result = transcript.model_dump()

                self.logger.info(
                    f"API 调用成功: "
                    f"语言={result.get('language')}, "
                    f"时长={result.get('duration', 0):.2f}s"
                )

                return result

        except Exception as e:
            error_msg = f"Whisper API 调用失败: {str(e)}"
            self.logger.error(error_msg)

            # 检查是否是认证错误
            if "authentication" in str(e).lower() or "unauthorized" in str(e).lower():
                error_msg = "API 认证失败，请检查 OPENAI_API_KEY 是否正确"

            # 检查是否是配额限制
            elif "quota" in str(e).lower() or "limit" in str(e).lower():
                error_msg = "API 配额不足，请检查账户余额"

            # 检查是否是文件格式问题
            elif "format" in str(e).lower() or "unsupported" in str(e).lower():
                error_msg = f"音频格式不支持: {Path(audio_path).suffix}"

            raise APIError(error_msg)

    def _format_markdown(self, transcript: Dict, audio_path: str) -> str:
        """格式化转录结果为 Markdown

        Args:
            transcript: 转录结果字典
            audio_path: 原始音频文件路径

        Returns:
            str: Markdown 格式的转录文本
        """
        audio_name = Path(audio_path).stem
        duration = transcript.get("duration", 0)
        language = transcript.get("language", "unknown")

        # 构建头部信息
        lines = [
            "# 音频转录：" + audio_name,
            "",
            "## 元数据",
            "",
            f"- **来源文件**: `{audio_path}`",
            f"- **时长**: {format_timestamp(duration)}",
            f"- **语言**: {language}",
            f"- **处理时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "---",
            "",
            "## 转录文本",
            ""
        ]

        # 添加完整文本
        full_text = transcript.get("text", "").strip()
        if full_text:
            lines.append(full_text)
            lines.append("")

        # 添加带时间戳的分段（如果有）
        segments = transcript.get("segments", [])
        if segments and len(segments) > 0:
            lines.append("")
            lines.append("## 时间戳分段")
            lines.append("")

            for segment in segments:
                start = segment.get("start", 0)
                end = segment.get("end", 0)
                text = segment.get("text", "").strip()

                if text:
                    timestamp = f"[{format_timestamp(start)} - {format_timestamp(end)}]"
                    lines.append(f"### {timestamp}")
                    lines.append("")
                    lines.append(text)
                    lines.append("")

        return "\n".join(lines)

    def _create_metadata(self, transcript: Dict, audio_path: str) -> Dict:
        """创建元数据字典

        Args:
            transcript: 转录结果
            audio_path: 音频文件路径

        Returns:
            dict: 元数据字典
        """
        metadata = {
            "source": audio_path,
            "duration": transcript.get("duration", 0),
            "language": transcript.get("language", ""),
            "text": transcript.get("text", ""),
            "model": self.model,
            "segments": transcript.get("segments", []),
            "processed_at": datetime.now().isoformat(),
            "audio_info": {
                "filename": Path(audio_path).name,
                "size_bytes": Path(audio_path).stat().st_size,
                "format": Path(audio_path).suffix.lower()
            }
        }

        # 如果有单词级时间戳，也保存
        if "words" in transcript:
            metadata["words"] = transcript["words"]

        return metadata

    async def batch_transcribe(
        self,
        audio_files: list,
        output_dir: str,
        language: str = "auto",
        max_concurrent: int = 3
    ) -> list:
        """批量转录多个音频文件

        Args:
            audio_files: 音频文件路径列表
            output_dir: 输出目录
            language: 语言提示
            max_concurrent: 最大并发数

        Returns:
            list: 转录结果列表
        """
        self.logger.info(f"开始批量转录 {len(audio_files)} 个音频文件")

        # 创建信号量控制并发数
        semaphore = asyncio.Semaphore(max_concurrent)

        async def transcribe_with_limit(audio_path):
            async with semaphore:
                return await self.transcribe(audio_path, output_dir, language)

        # 并发执行
        tasks = [transcribe_with_limit(audio) for audio in audio_files]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 处理结果
        successful = sum(1 for r in results if isinstance(r, dict) and "markdown_path" in r)
        failed = len(results) - successful

        self.logger.info(
            f"批量转录完成: 成功 {successful} 个, 失败 {failed} 个"
        )

        return results


# 便捷函数
async def transcribe_audio(
    audio_path: str,
    output_dir: str,
    api_key: Optional[str] = None,
    language: str = "auto"
) -> Dict[str, str]:
    """便捷的音频转录函数

    Args:
        audio_path: 音频文件路径
        output_dir: 输出目录
        api_key: OpenAI API 密钥（可选）
        language: 语言提示

    Returns:
        dict: 包含 markdown_path, metadata_path 的字典

    Examples:
        >>> result = await transcribe_audio("lecture.mp3", "output/")
        >>> print(result["markdown_path"])
    """
    processor = AudioProcessor(api_key=api_key)
    return await processor.transcribe(audio_path, output_dir, language)
