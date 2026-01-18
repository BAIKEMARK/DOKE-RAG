# DOKE-RAG 多模态 Pipeline 扩展设计方案

**日期**: 2025-01-18
**作者**: DOKE-RAG Team
**状态**: 待实施

---

## 概述

本文档规划了 DOKE-RAG 框架的多模态数据处理 pipeline 扩展，包括音频转录（Whisper ASR）和 PDF 解析（PaddleOCR-VL）两个核心功能。这两个模块将实现论文 Section 3.1 中描述的多模态数据摄入能力。

---

## 1. 整体架构设计

### 1.1 目录结构

```
doke_rag/pipeline/
├── __init__.py                 # 导出所有处理器
├── audio_processor.py          # Whisper ASR 音频处理
├── pdf_parser.py              # PaddleOCR-VL PDF 解析
├── chunking.py                # 现有的文本分块（不变）
├── chunking_txt.py            # 现有的文本处理（不变）
└── utils.py                   # 共享工具函数（新增）

doke_rag/config/
└── pipeline_config.py         # Pipeline 配置管理（新增）

tests/pipeline/
├── test_audio_processor.py
├── test_pdf_parser.py
└── fixtures/
    ├── sample_audio.mp3
    ├── sample.pdf
    └── sample_complex.pdf
```

### 1.2 模块职责

- **audio_processor.py**: 接收音频文件（mp3/wav/m4a），调用 Whisper API 转录为 Markdown 文本
- **pdf_parser.py**: 接收 PDF 文件，使用 PaddleOCR-VL 提取文本、公式（LaTeX）、表格（Markdown）、图片，生成统一的 Markdown 文档
- **utils.py**: 提供文件验证、路径管理、日志记录等共享功能

### 1.3 数据流

```
原始文件（音频/PDF）
  ↓
对应处理器（独立调用）
  ↓
Markdown 文件 + 元数据 JSON
  ↓
chunking.py（后续集成）
  ↓
知识块 → 知识图谱
```

### 1.4 设计原则

1. **模块独立性**: 每个处理器完全独立，可单独测试和使用
2. **输出标准化**: 统一输出 Markdown 格式，便于后续流程集成
3. **配置一致性**: 所有配置通过环境变量管理，与现有项目保持一致
4. **错误严格性**: 开发阶段采用严格模式，遇到错误立即抛出异常
5. **渐进式集成**: 先独立验证，后期通过 pipeline 统一调度

---

## 2. 音频处理器设计

### 2.1 技术选型

- **Phase 1**: OpenAI Whisper API（whisper-1 模型）
  - 优点：准确率高，支持多语言，无需本地 GPU
  - 缺点：需要 API key，按使用量付费

- **Phase 2**（后续）: 开源 Whisper 本地部署
  - 使用 `openai-whisper` Python 库
  - 可选择 tiny/base/small/medium/large 模型
  - 提供配置选项让用户选择 API 或本地模式

### 2.2 核心类设计

```python
class AudioProcessor:
    """音频转录处理器，使用 Whisper API 将音频转换为 Markdown"""

    def __init__(self, api_key: str, model: str = "whisper-1"):
        """
        Args:
            api_key: OpenAI API 密钥
            model: Whisper 模型名称（默认 whisper-1）
        """
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model
        self.logger = setup_logger("AudioProcessor")

    async def transcribe(
        self,
        audio_path: str,
        output_dir: str,
        language: str = "auto"
    ) -> dict:
        """
        转录音频文件为 Markdown

        Args:
            audio_path: 音频文件路径
            output_dir: 输出目录
            language: 语言提示（zh/en/auto）

        Returns:
            dict: 包含 markdown_path, metadata_path 的字典

        Raises:
            AudioFileError: 音频文件无效
            APIError: API 调用失败
        """
        # 1. 验证文件
        self._validate_audio(audio_path)

        # 2. 调用 Whisper API
        transcript = await self._call_whisper_api(audio_path, language)

        # 3. 生成 Markdown
        markdown_content = self._format_markdown(transcript, audio_path)

        # 4. 保存文件
        timestamp = get_timestamp_filename("audio", "md")
        markdown_path = output_dir / timestamp
        metadata_path = output_dir / timestamp.replace(".md", "_metadata.json")

        self._save_markdown(markdown_content, markdown_path)
        self._save_metadata(transcript, metadata_path)

        return {
            "markdown_path": str(markdown_path),
            "metadata_path": str(metadata_path)
        }

    def _validate_audio(self, file_path: str) -> None:
        """验证音频文件格式、大小、可访问性"""
        # 检查格式
        supported_formats = [".mp3", ".wav", ".m4a", ".mp4", ".mpeg"]
        ext = Path(file_path).suffix.lower()
        if ext not in supported_formats:
            raise AudioFileError(f"不支持的音频格式: {ext}")

        # 检查大小
        size_mb = Path(file_path).stat().st_size / (1024 * 1024)
        if size_mb > PipelineConfig.MAX_AUDIO_SIZE_MB:
            raise AudioFileError(
                f"音频文件过大: {size_mb:.2f}MB "
                f"(限制: {PipelineConfig.MAX_AUDIO_SIZE_MB}MB)"
            )

    async def _call_whisper_api(self, audio_path: str, language: str) -> dict:
        """调用 OpenAI Whisper API"""
        try:
            with open(audio_path, "rb") as audio_file:
                transcript = await self.client.audio.transcriptions.create(
                    model=self.model,
                    file=audio_file,
                    language=language if language != "auto" else None,
                    response_format="verbose_json"  # 包含时间戳
                )
            return transcript.model_dump()
        except Exception as e:
            raise APIError(f"Whisper API 调用失败: {str(e)}")

    def _format_markdown(self, transcript: dict, audio_path: str) -> str:
        """格式化转录结果为 Markdown"""
        audio_name = Path(audio_path).stem
        duration = transcript.get("duration", 0)
        duration_min = int(duration // 60)
        duration_sec = int(duration % 60)

        lines = [
            f"# 音频转录：{audio_name}",
            "",
            f"**来源文件**: `{audio_path}`",
            f"**时长**: {duration_min}:{duration_sec:02d}",
            f"**语言**: {transcript.get('language', 'unknown')}",
            "",
            "---",
            ""
        ]

        # 添加带时间戳的转录文本
        for segment in transcript.get("words", []):
            start = segment.get("start", 0)
            start_min = int(start // 60)
            start_sec = int(start % 60)
            timestamp = f"[{start_min}:{start_sec:02d}]"
            text = segment.get("word", "")

            lines.append(f"{timestamp} {text}")

        return "\n".join(lines)

    def _save_markdown(self, content: str, path: Path) -> None:
        """保存 Markdown 文件"""
        ensure_output_dir(path.parent)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        self.logger.info(f"Markdown 已保存: {path}")

    def _save_metadata(self, transcript: dict, path: Path) -> None:
        """保存元数据 JSON"""
        metadata = {
            "source": transcript.get("source", ""),
            "duration": transcript.get("duration", 0),
            "language": transcript.get("language", ""),
            "text": transcript.get("text", ""),
            "segments": transcript.get("segments", []),
            "processed_at": datetime.now().isoformat()
        }
        save_metadata(metadata, path)
```

### 2.3 关键配置

| 参数 | 环境变量 | 默认值 | 说明 |
|-----|---------|--------|------|
| API 密钥 | `OPENAI_API_KEY` | 必填 | OpenAI API 密钥 |
| 模型名称 | `WHISPER_MODEL` | `whisper-1` | Whisper API 模型 |
| 语言提示 | `WHISPER_LANGUAGE` | `auto` | zh/en/auto |
| 大小限制 | `MAX_AUDIO_SIZE` | 500 | 最大音频文件大小（MB） |

### 2.4 输出示例

**Markdown 文件** (`audio_20250118_143022.md`):
```markdown
# 音频转录：结构力学课程 - 第5讲

**来源文件**: `lecture_05.mp3`
**时长**: 45:23
**语言**: zh

---

[0:00] 大家好，今天我们继续学习超静定结构的分析方法。
[0:08] 上一节课我们介绍了力法的基本概念。
[0:15] 力法是分析超静定结构的基本方法之一。
[0:22] 根据力法，我们可以建立协调方程...
```

**元数据文件** (`audio_20250118_143022_metadata.json`):
```json
{
  "source": "lecture_05.mp3",
  "duration": 2723.5,
  "language": "zh",
  "text": "大家好，今天我们继续学习超静定结构的分析方法...",
  "segments": [...],
  "processed_at": "2025-01-18T14:30:22"
}
```

---

## 3. PDF 解析器设计

### 3.1 技术选型

- **PaddleOCR-VL**: 百度飞桨开发的版面分析+OCR 工具
  - 支持 80+ 种语言识别
  - PPStructure 模型支持版面分析（表格、图片、标题识别）
  - PPStructureV2 支持 LaTeX 公式识别
  - 配合 LaTeX OCR 工具（如 pix2text）实现公式提取

### 3.2 核心类设计

```python
class PDFParser:
    """PDF 解析器，使用 PaddleOCR-VL 提取多模态内容"""

    def __init__(self, model_dir: Path = None):
        """
        Args:
            model_dir: PaddleOCR 模型目录
        """
        self.model_dir = model_dir or PipelineConfig.PADDLE_MODEL_DIR
        self.ocr = PaddleOCR(
            use_angle_cls=True,
            lang='ch',
            det_model_dir=str(self.model_dir / 'det'),
            rec_model_dir=str(self.model_dir / 'rec'),
            use_gpu=False  # 可配置
        )
        self.structure = PPStructure(
            ocr=self.ocr,
            show_log=True
        )
        self.logger = setup_logger("PDFParser")

    async def parse_pdf(
        self,
        pdf_path: str,
        output_dir: str,
        extract_images: bool = True
    ) -> dict:
        """
        解析 PDF 文件为 Markdown

        Args:
            pdf_path: PDF 文件路径
            output_dir: 输出目录
            extract_images: 是否提取图片

        Returns:
            dict: 包含 markdown_path, metadata_path, images_dir 的字典

        Raises:
            PDFParseError: PDF 解析失败
        """
        # 1. 验证文件
        self._validate_pdf(pdf_path)

        # 2. 打开 PDF
        pdf_file = pdfplumber.open(pdf_path)
        total_pages = len(pdf_file.pages)

        # 3. 逐页处理
        markdown_parts = []
        images_info = []

        for page_num, page in enumerate(pdf_file.pages, 1):
            self.logger.info(f"处理第 {page_num}/{total_pages} 页")

            # 提取文本和结构
            page_result = self._process_page(
                page, page_num, output_dir, extract_images
            )

            markdown_parts.append(page_result["markdown"])
            images_info.extend(page_result["images"])

        # 4. 组装完整 Markdown
        full_markdown = "\n\n".join(markdown_parts)

        # 5. 保存文件
        timestamp = get_timestamp_filename("document", "md")
        markdown_path = output_dir / timestamp
        metadata_path = output_dir / timestamp.replace(".md", "_metadata.json")

        self._save_markdown(full_markdown, markdown_path)
        self._save_metadata({
            "source": pdf_path,
            "total_pages": total_pages,
            "images_count": len(images_info),
            "images": images_info,
            "processed_at": datetime.now().isoformat()
        }, metadata_path)

        return {
            "markdown_path": str(markdown_path),
            "metadata_path": str(metadata_path),
            "images_dir": str(output_dir / "extracted_images")
        }

    def _process_page(
        self,
        page,
        page_num: int,
        output_dir: Path,
        extract_images: bool
    ) -> dict:
        """处理单页 PDF"""
        markdown_parts = []
        images = []

        # 使用 PaddleOCR-VL 进行版面分析
        layout_result = self.structure(np.array(page.to_image().convert('RGB')))

        for region in layout_result:
            region_type = region['type']

            if region_type == 'text':
                # 文本段落
                text = region['res']['text']
                markdown_parts.append(text)

            elif region_type == 'title':
                # 标题
                title_level = region.get('level', 1)
                title_text = region['res']['text']
                prefix = "#" * title_level
                markdown_parts.append(f"{prefix} {title_text}")

            elif region_type == 'table':
                # 表格
                table_markdown = self._convert_table_to_markdown(region)
                markdown_parts.append(table_markdown)

            elif region_type == 'figure':
                # 图片
                if extract_images:
                    img_info = self._extract_image(
                        region, page_num, output_dir, len(images)
                    )
                    if img_info:
                        images.append(img_info)
                        img_link = f"![{img_info['caption']}]({img_info['path']})"
                        markdown_parts.append(img_link)

        return {
            "markdown": "\n\n".join(markdown_parts),
            "images": images
        }

    def _convert_table_to_markdown(self, table_region: dict) -> str:
        """将表格转换为 Markdown 格式"""
        # PaddleOCR-VL 已经提供了表格结构
        html_table = table_region['res']['html']

        # 使用 html2markdown 或自定义转换逻辑
        # 这里简化处理，实际需要解析 HTML 表格
        table_data = self._parse_html_table(html_table)

        # 转换为 Markdown 表格
        if len(table_data) == 0:
            return ""

        # 表头
        headers = table_data[0]
        separator = "| " + " | ".join(["---"] * len(headers)) + " |"

        lines = ["| " + " | ".join(headers) + " |", separator]

        # 表体
        for row in table_data[1:]:
            lines.append("| " + " | ".join(row) + " |")

        return "\n".join(lines)

    def _extract_image(
        self,
        region: dict,
        page_num: int,
        output_dir: Path,
        img_index: int
    ) -> dict:
        """提取并保存图片"""
        # 获取图片边界框
        bbox = region['bbox']

        # 从 PDF 页面裁剪图片
        page_img = region['img']  # PaddleOCR 已提取的图片

        # 保存图片
        img_dir = output_dir / "extracted_images" / f"page_{page_num}"
        ensure_output_dir(img_dir)

        img_filename = f"fig_{img_index + 1}.png"
        img_path = img_dir / img_filename

        cv2.imwrite(str(img_path), page_img)

        return {
            "path": str(img_path),
            "page": page_num,
            "caption": f"Figure {page_num}-{img_index + 1}",
            "bbox": bbox
        }

    def _validate_pdf(self, file_path: str) -> None:
        """验证 PDF 文件"""
        # 检查格式
        if not file_path.lower().endswith(".pdf"):
            raise PDFParseError("文件格式错误，必须是 .pdf 文件")

        # 检查大小
        size_mb = Path(file_path).stat().st_size / (1024 * 1024)
        if size_mb > PipelineConfig.MAX_PDF_SIZE_MB:
            raise PDFParseError(
                f"PDF 文件过大: {size_mb:.2f}MB "
                f"(限制: {PipelineConfig.MAX_PDF_SIZE_MB}MB)"
            )

        # 检查是否加密
        try:
            pdf_file = pdfplumber.open(file_path)
            if pdf_file.pages[0].is_encrypted:
                raise PDFParseError("PDF 文件已加密，无法解析")
            pdf_file.close()
        except Exception as e:
            if "encrypted" in str(e).lower():
                raise PDFParseError("PDF 文件已加密，无法解析")
            raise
```

### 3.3 关键配置

| 参数 | 环境变量 | 默认值 | 说明 |
|-----|---------|--------|------|
| 模型目录 | `PADDLE_MODEL_DIR` | `models/paddleocr/` | PaddleOCR 模型路径 |
| 提取图片 | `EXTRACT_IMAGES` | `true` | 是否提取图片 |
| 图片目录 | `IMAGES_OUTPUT_DIR` | `data/extracted_images/` | 图片保存目录 |
| 公式格式 | `FORMULA_FORMAT` | `latex` | 公式格式（latex/image） |
| 大小限制 | `MAX_PDF_SIZE` | 100 | 最大 PDF 文件大小（MB） |

### 3.4 输出示例

**Markdown 文件** (`document_20250118_143022.md`):
```markdown
# 超静定结构分析

## 1. 力法原理

力法是分析超静定结构的基本方法。根据力法，我们可以得到：

$$ M = EI \frac{d^2y}{dx^2} $$

这个公式描述了弯矩与曲率的关系。

下表展示了不同边界条件下的弯矩系数：

| 边界条件 | 固定端 | 铰接端 |
|---------|--------|--------|
| 简支梁   | 0      | 1      |
| 固定梁   | 1      | 0.5    |

![Figure 5-1](extracted_images/page_5/fig_1.png)

如图所示，固定端的弯矩分布呈现出明显的非线性特征。

### 1.1 协调方程

力法的核心是建立位移协调方程...
```

---

## 4. 配置管理

### 4.1 配置模块 (`doke_rag/config/pipeline_config.py`)

```python
from dotenv import load_dotenv
import os
from pathlib import Path

load_dotenv()

class PipelineConfig:
    """Pipeline 配置管理"""

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
        """验证配置有效性"""
        if not cls.OPENAI_API_KEY:
            raise ValueError("缺少 OPENAI_API_KEY 配置")
        # 可添加更多验证逻辑
```

### 4.2 共享工具 (`pipeline/utils.py`)

```python
import os
import logging
from pathlib import Path
from datetime import datetime
import json

def validate_file_size(file_path: str, max_size_mb: int) -> bool:
    """验证文件大小"""
    size_mb = Path(file_path).stat().st_size / (1024 * 1024)
    return size_mb <= max_size_mb

def get_timestamp_filename(prefix: str, extension: str) -> str:
    """生成带时间戳的文件名"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}.{extension}"

def save_metadata(metadata: dict, output_path: str) -> None:
    """保存元数据 JSON"""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

def setup_logger(name: str) -> logging.Logger:
    """设置日志记录器"""
    logger = logging.getLogger(name)

    if not logger.handlers:
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)

        handler = logging.FileHandler(
            log_dir / f"pipeline_{datetime.now().strftime('%Y%m%d')}.log",
            encoding="utf-8"
        )
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    return logger

def ensure_output_dir(directory: Path) -> None:
    """确保输出目录存在"""
    directory.mkdir(parents=True, exist_ok=True)
```

### 4.3 错误处理

```python
class PipelineError(Exception):
    """Pipeline 基础异常"""
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
```

### 4.4 环境变量配置 (`.env.example`)

```bash
# ========== Pipeline Configuration ==========

# Audio Processing (Whisper API)
OPENAI_API_KEY=sk-your-key-here
WHISPER_MODEL=whisper-1
WHISPER_LANGUAGE=auto  # auto | zh | en
MAX_AUDIO_SIZE=500  # MB

# PDF Processing (PaddleOCR-VL)
PADDLE_MODEL_DIR=models/paddleocr/
EXTRACT_IMAGES=true
IMAGES_OUTPUT_DIR=data/extracted_images/
FORMULA_FORMAT=latex  # latex | image
MAX_PDF_SIZE=100  # MB

# Common Settings
PIPELINE_OUTPUT_DIR=data/processed/
PIPELINE_MODE=strict  # strict | tolerant
```

---

## 5. 测试策略

### 5.1 测试文件结构

```
tests/pipeline/
├── __init__.py
├── test_audio_processor.py
├── test_pdf_parser.py
├── fixtures/
│   ├── sample_audio.mp3         # 5秒测试音频（中文）
│   ├── sample_audio_en.mp3      # 5秒测试音频（英文）
│   ├── sample.pdf               # 简单文本 PDF
│   ├── sample_complex.pdf       # 包含公式、表格、图片的 PDF
│   └── sample_encrypted.pdf     # 加密 PDF（用于错误测试）
└── expected_outputs/            # 预期输出示例
    ├── expected_audio.md
    └── expected_pdf.md
```

### 5.2 音频处理器测试用例

```python
import pytest
from doke_rag.pipeline.audio_processor import AudioProcessor
from doke_rag.config.pipeline_config import PipelineConfig

class TestAudioProcessor:
    """音频处理器测试"""

    @pytest.fixture
    def processor(self):
        return AudioProcessor(
            api_key=PipelineConfig.OPENAI_API_KEY
        )

    @pytest.mark.asyncio
    async def test_transcribe_chinese_audio(self, processor):
        """测试中文音频转录"""
        result = await processor.transcribe(
            "tests/fixtures/sample_audio.mp3",
            "tests/output/"
        )

        assert Path(result["markdown_path"]).exists()
        assert Path(result["metadata_path"]).exists()

        # 验证 Markdown 内容
        with open(result["markdown_path"], "r", encoding="utf-8") as f:
            content = f.read()
        assert "# 音频转录" in content
        assert "时长" in content

    @pytest.mark.asyncio
    async def test_transcribe_english_audio(self, processor):
        """测试英文音频转录"""
        result = await processor.transcribe(
            "tests/fixtures/sample_audio_en.mp3",
            "tests/output/"
        )

        with open(result["metadata_path"], "r", encoding="utf-8") as f:
            metadata = json.load(f)

        assert metadata["language"] == "en"

    def test_invalid_audio_format(self, processor):
        """测试不支持的音频格式"""
        with pytest.raises(AudioFileError):
            processor._validate_audio("test.avi")

    def test_oversized_audio(self, processor):
        """测试超大音频文件"""
        # 创建临时大文件
        large_file = "test_large.mp3"
        with open(large_file, "wb") as f:
            f.seek(600 * 1024 * 1024)  # 600MB
            f.write(b"\0")

        with pytest.raises(AudioFileError):
            processor._validate_audio(large_file)

        os.remove(large_file)

    @pytest.mark.asyncio
    async def test_api_failure_handling(self, processor, monkeypatch):
        """测试 API 失败处理"""
        # Mock API 失败
        async def mock_call(*args, **kwargs):
            raise Exception("API Error")

        monkeypatch.setattr(processor, "_call_whisper_api", mock_call)

        with pytest.raises(APIError):
            await processor.transcribe(
                "tests/fixtures/sample_audio.mp3",
                "tests/output/"
            )
```

### 5.3 PDF 解析器测试用例

```python
import pytest
from doke_rag.pipeline.pdf_parser import PDFParser

class TestPDFParser:
    """PDF 解析器测试"""

    @pytest.fixture
    def parser(self):
        return PDFParser()

    @pytest.mark.asyncio
    async def test_parse_text_only_pdf(self, parser):
        """测试纯文本 PDF 解析"""
        result = await parser.parse_pdf(
            "tests/fixtures/sample.pdf",
            "tests/output/",
            extract_images=False
        )

        assert Path(result["markdown_path"]).exists()
        assert Path(result["metadata_path"]).exists()

        with open(result["markdown_path"], "r", encoding="utf-8") as f:
            content = f.read()
        assert len(content) > 0

    @pytest.mark.asyncio
    async def test_parse_complex_pdf(self, parser):
        """测试包含公式、表格、图片的复杂 PDF"""
        result = await parser.parse_pdf(
            "tests/fixtures/sample_complex.pdf",
            "tests/output/",
            extract_images=True
        )

        # 验证图片提取
        assert Path(result["images_dir"]).exists()

        # 验证 Markdown 格式
        with open(result["markdown_path"], "r", encoding="utf-8") as f:
            content = f.read()

        assert "$$" in content or "![Figure](https://" in content  # 公式或图片
        assert "|" in content  # 表格

    def test_encrypted_pdf(self, parser):
        """测试加密 PDF 错误处理"""
        with pytest.raises(PDFParseError, match="加密"):
            parser._validate_pdf("tests/fixtures/sample_encrypted.pdf")

    def test_invalid_pdf_format(self, parser):
        """测试非 PDF 文件"""
        with pytest.raises(PDFParseError, match="格式错误"):
            parser._validate_pdf("test.txt")

    def test_oversized_pdf(self, parser):
        """测试超大 PDF 文件"""
        large_file = "test_large.pdf"
        with open(large_file, "wb") as f:
            f.seek(150 * 1024 * 1024)  # 150MB
            f.write(b"\0")

        with pytest.raises(PDFParseError, match="过大"):
            parser._validate_pdf(large_file)

        os.remove(large_file)
```

### 5.4 集成测试

```python
@pytest.mark.asyncio
async def test_end_to_end_pipeline():
    """端到端 Pipeline 测试"""
    # 1. 处理音频
    audio_processor = AudioProcessor(api_key=PipelineConfig.OPENAI_API_KEY)
    audio_result = await audio_processor.transcribe(
        "data/raw/audio/lecture_01.mp3",
        "data/processed/"
    )

    # 2. 处理 PDF
    pdf_parser = PDFParser()
    pdf_result = await pdf_parser.parse_pdf(
        "data/raw/pdfs/chapter_01.pdf",
        "data/processed/"
    )

    # 3. 验证输出
    assert Path(audio_result["markdown_path"]).exists()
    assert Path(pdf_result["markdown_path"]).exists()

    # 4. 后续集成到 chunking（待实现）
    # from doke_rag.pipeline.chunking import process_markdown
    # chunks = process_markdown(audio_result["markdown_path"])
```

---

## 6. 实施计划

### 6.1 开发阶段

**阶段 1：音频处理器（1-2天）**
- [ ] 创建 `audio_processor.py` 基本框架
- [ ] 实现 OpenAI Whisper API 调用
- [ ] 实现 Markdown 生成和保存
- [ ] 实现元数据生成和保存
- [ ] 编写单元测试
- [ ] 本地测试和调试

**阶段 2：PDF 解析器基础（2-3天）**
- [ ] 安装和配置 PaddleOCR-VL 环境
- [ ] 创建 `pdf_parser.py` 基本框架
- [ ] 实现文本提取和版面分析
- [ ] 编写基础测试用例
- [ ] 本地测试和调试

**阶段 3：PDF 多模态提取（2-3天）**
- [ ] 实现公式识别（LaTeX OCR）
- [ ] 实现表格转 Markdown
- [ ] 实现图片提取和存储
- [ ] 完善 Markdown 组装逻辑
- [ ] 编写完整测试套件

**阶段 4：集成和文档（1天）**
- [ ] 创建配置文件和工具函数
- [ ] 更新 `requirements.txt`
- [ ] 更新 README 和使用文档
- [ ] 添加使用示例和脚本
- [ ] 性能测试和优化

### 6.2 依赖管理

**requirements.txt 更新**:
```txt
# Audio Processing
openai>=1.0.0
aiofiles>=23.0.0

# PDF Processing
paddlepaddle>=2.5.0  # or paddlepaddle-gpu for GPU
paddleocr>=2.7.0
pillow>=10.0.0
pdfplumber>=0.10.0
opencv-python>=4.8.0

# Formula Recognition (optional)
pix2text>=1.1.0  # LaTeX OCR

# Utilities
python-dotenv>=1.0.0
```

### 6.3 已知挑战和缓解措施

| 挑战 | 影响 | 缓解措施 |
|-----|------|---------|
| **PaddleOCR 安装复杂** | 用户难以部署 | 提供详细安装文档；准备 Docker 镜像；提供一键安装脚本 |
| **PDF 解析质量依赖 OCR** | 识别准确率影响输出 | 提供参数调优指南；支持人工校验流程；允许上传纠错数据 |
| **公式识别准确率** | 复杂公式可能识别错误 | 允许用户选择保留图片而非 LaTeX；提供公式校验工具 |
| **大文件处理性能** | 处理时间长 | 实现分页处理和进度条；支持断点续传；提供并发处理选项 |
| **API 成本** | Whisper API 费用 | 清晰标注费用信息；提供本地 Whisper 替代方案 |

---

## 7. 后续集成计划

### 7.1 与现有 Pipeline 集成

**创建统一入口** (`pipeline/main.py`):

```python
from pathlib import Path
from doke_rag.pipeline.audio_processor import AudioProcessor
from doke_rag.pipeline.pdf_parser import PDFParser
from doke_rag.pipeline.chunking import process_markdown

class MultiModalPipeline:
    """统一的多模态数据处理 Pipeline"""

    def __init__(self):
        self.audio_processor = AudioProcessor(...)
        self.pdf_parser = PDFParser(...)
        self.chunker = ChunkingProcessor(...)

    async def process(self, file_path: str) -> list:
        """
        处理任意格式的输入文件

        Args:
            file_path: 音频/PDF/Markdown 文件路径

        Returns:
            list: 知识块列表
        """
        file_path = Path(file_path)

        # 根据文件类型路由
        if file_path.suffix in [".mp3", ".wav", ".m4a"]:
            # 1. 音频转录
            result = await self.audio_processor.transcribe(
                file_path,
                output_dir="data/processed/"
            )
            markdown_path = result["markdown_path"]

        elif file_path.suffix == ".pdf":
            # 2. PDF 解析
            result = await self.pdf_parser.parse_pdf(
                file_path,
                output_dir="data/processed/"
            )
            markdown_path = result["markdown_path"]

        elif file_path.suffix in [".md", ".txt"]:
            # 3. 直接处理 Markdown
            markdown_path = file_path

        else:
            raise ValueError(f"不支持的文件类型: {file_path.suffix}")

        # 4. 后续 Chunking 处理
        chunks = await self.chunker.process(markdown_path)

        return chunks
```

### 7.2 API 接口

**提供简单的 REST API**（可选）:

```python
from fastapi import FastAPI, UploadFile

app = FastAPI()

@app.post("/process/audio")
async def process_audio(file: UploadFile):
    """上传音频文件并返回转录文本"""
    # 保存临时文件
    # 调用 AudioProcessor
    # 返回 Markdown 内容
    pass

@app.post("/process/pdf")
async def process_pdf(file: UploadFile):
    """上传 PDF 文件并返回解析结果"""
    # 保存临时文件
    # 调用 PDFParser
    # 返回 Markdown 内容和图片列表
    pass
```

---

## 8. 文档和示例

### 8.1 README 更新

添加新的使用章节：

```markdown
## Multimodal Processing

DOKE-RAG supports processing audio and PDF files:

### Audio Transcription

```python
from doke_rag.pipeline.audio_processor import AudioProcessor

processor = AudioProcessor(api_key="your-openai-key")
result = await processor.transcribe(
    "lecture.mp3",
    output_dir="data/processed/"
)
```

### PDF Parsing

```python
from doke_rag.pipeline.pdf_parser import PDFParser

parser = PDFParser()
result = await parser.parse_pdf(
    "document.pdf",
    output_dir="data/processed/"
)
```

### Unified Pipeline

```python
from doke_rag.pipeline import MultiModalPipeline

pipeline = MultiModalPipeline()
chunks = await pipeline.process("document.pdf")
```
```

### 8.2 使用示例脚本

创建 `examples/process_multimodal_data.py`:

```python
"""
多模态数据处理示例

演示如何使用 DOKE-RAG 处理音频和 PDF 文件
"""

import asyncio
from doke_rag.pipeline import MultiModalPipeline

async def main():
    pipeline = MultiModalPipeline()

    # 处理音频
    audio_chunks = await pipeline.process("data/raw/audio/lecture_01.mp3")
    print(f"音频已转换为 {len(audio_chunks)} 个知识块")

    # 处理 PDF
    pdf_chunks = await pipeline.process("data/raw/pdfs/chapter_01.pdf")
    print(f"PDF 已转换为 {len(pdf_chunks)} 个知识块")

    # 处理已有 Markdown
    md_chunks = await pipeline.process("data/raw/texts/notes.md")
    print(f"Markdown 已转换为 {len(md_chunks)} 个知识块")

if __name__ == "__main__":
    asyncio.run(main())
```

---

## 9. 总结

本设计方案规划了 DOKE-RAG 框架的多模态数据处理能力扩展，包括：

1. **音频转录**: 使用 Whisper API 将语音转换为文本
2. **PDF 解析**: 使用 PaddleOCR-VL 提取文本、公式、表格、图片
3. **统一输出**: 所有内容转换为统一的 Markdown 格式
4. **独立模块**: 每个处理器可单独使用和测试
5. **渐进式集成**: 先独立验证，后期统一集成到 Pipeline

核心设计原则：
- ✅ 符合论文描述的技术选型
- ✅ 保持与现有项目的一致性
- ✅ 独立可测试的模块化设计
- ✅ 清晰的错误处理和日志记录
- ✅ 灵活的配置管理

实施后，DOKE-RAG 将具备完整的多模态数据摄入能力，实现论文 Section 3.1 中描述的所有功能。
