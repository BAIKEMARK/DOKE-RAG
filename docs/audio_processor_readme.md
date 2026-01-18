# 音频处理器模块 (Audio Processor)

## 概述

音频处理器模块使用 OpenAI Whisper API 将音频文件转录为 Markdown 文本。支持多种音频格式和多语言转录。

## 功能特性

- ✓ 支持多种音频格式（MP3, WAV, M4A, MP4, MPEG, OGG, FLAC）
- ✓ 自动语言检测或指定语言
- ✓ 带时间戳的转录输出
- ✓ 批量处理支持
- ✓ 严格模式和容错模式
- ✓ 完整的元数据保存
- ✓ 异步处理，支持高并发

## 安装

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

关键依赖：
- `openai>=1.0.0` - OpenAI API 客户端
- `python-dotenv>=1.0.0` - 环境变量管理
- `aiofiles>=23.0.0` - 异步文件操作

### 2. 配置环境变量

复制 `.env.example` 到 `.env` 并配置：

```bash
# 必填：OpenAI API 密钥
OPENAI_API_KEY=sk-your-openai-api-key-here

# 可选：Whisper 模型配置
WHISPER_MODEL=whisper-1
WHISPER_LANGUAGE=auto  # auto | zh | en
MAX_AUDIO_SIZE=500  # MB

# 通用配置
PIPELINE_OUTPUT_DIR=data/processed/
PIPELINE_MODE=strict  # strict | tolerant
```

## 快速开始

### 基本使用

```python
import asyncio
from doke_rag.pipeline.audio_processor import AudioProcessor

async def main():
    # 创建处理器
    processor = AudioProcessor()

    # 转录音频
    result = await processor.transcribe(
        audio_path="lecture.mp3",
        output_dir="data/processed/",
        language="auto"  # 自动检测语言
    )

    print(f"Markdown 文件: {result['markdown_path']}")
    print(f"元数据文件: {result['metadata_path']}")
    print(f"时长: {result['duration']} 秒")
    print(f"语言: {result['language']}")

asyncio.run(main())
```

### 使用便捷函数

```python
from doke_rag.pipeline import transcribe_audio

result = await transcribe_audio(
    audio_path="lecture.mp3",
    output_dir="output/",
    language="zh"  # 指定中文
)
```

## 高级用法

### 1. 指定语言和提示

```python
result = await processor.transcribe(
    audio_path="chinese_lecture.mp3",
    output_dir="output/",
    language="zh",  # 明确指定中文，提高准确率
    prompt="这是一段关于结构力学的课程",  # 提供上下文
    temperature=0.0  # 0.0 表示最确定的转录
)
```

### 2. 批量处理

```python
audio_files = [
    "lecture_01.mp3",
    "lecture_02.mp3",
    "lecture_03.mp3"
]

results = await processor.batch_transcribe(
    audio_files=audio_files,
    output_dir="output/",
    language="auto",
    max_concurrent=3  # 最多 3 个并发请求
)

for i, result in enumerate(results):
    print(f"文件 {i+1}: {result.get('markdown_path', '失败')}")
```

### 3. 容错模式

```python
# 创建容错模式处理器
processor = AudioProcessor(strict_mode=False)

result = await processor.transcribe(...)

# 即使失败也不会抛出异常
if "error" in result:
    print(f"处理失败: {result['error']}")
else:
    print(f"成功: {result['markdown_path']}")
```

## 输出格式

### Markdown 文件

```markdown
# 音频转录：lecture_01

## 元数据

- **来源文件**: `lecture_01.mp3`
- **时长**: 45:23
- **语言**: zh
- **处理时间**: 2025-01-18 14:30:22

---

## 转录文本

大家好，今天我们继续学习超静定结构的分析方法...

## 时间戳分段

### [00:00 - 00:08]
大家好，今天我们继续学习超静定结构的分析方法。

### [00:08 - 00:15]
上一节课我们介绍了力法的基本概念。
```

### 元数据文件

```json
{
  "source": "lecture_01.mp3",
  "duration": 2723.5,
  "language": "zh",
  "text": "大家好，今天我们继续学习...",
  "model": "whisper-1",
  "segments": [...],
  "processed_at": "2025-01-18T14:30:22",
  "audio_info": {
    "filename": "lecture_01.mp3",
    "size_bytes": 5242880,
    "format": ".mp3"
  }
}
```

## 支持的音频格式

- `.mp3` - 推荐格式
- `.wav` - 无损格式
- `.m4a` - Apple 音频
- `.mp4` - 音频流
- `.mpeg` - MPEG 音频
- `.mpga` - MPEG 音频流
- `.ogg` - OGG Vorbis
- `.flac` - 无损压缩

## API 参考

### AudioProcessor

#### `__init__(api_key=None, model="whisper-1", strict_mode=True)`

初始化音频处理器。

**参数：**
- `api_key` (str, optional): OpenAI API 密钥
- `model` (str): Whisper 模型名称
- `strict_mode` (bool): 严格模式开关

#### `async transcribe(audio_path, output_dir, language="auto", prompt=None, temperature=0.0)`

转录音频文件。

**参数：**
- `audio_path` (str): 音频文件路径
- `output_dir` (str): 输出目录
- `language` (str): 语言提示（zh/en/auto）
- `prompt` (str, optional): 提示文本
- `temperature` (float): 转录温度（0.0-1.0）

**返回：**
```python
{
    "markdown_path": "path/to/markdown.md",
    "metadata_path": "path/to/metadata.json",
    "duration": 2723.5,
    "language": "zh"
}
```

#### `async batch_transcribe(audio_files, output_dir, language="auto", max_concurrent=3)`

批量转录多个音频文件。

**参数：**
- `audio_files` (list): 音频文件路径列表
- `output_dir` (str): 输出目录
- `language` (str): 语言提示
- `max_concurrent` (int): 最大并发数

**返回：**
- `list`: 转录结果列表

## 错误处理

### 异常类型

- `AudioFileError`: 音频文件无效（格式、大小、损坏）
- `APIError`: API 调用失败（认证、配额、网络）
- `ValidationError`: 验证失败
- `ConfigError`: 配置错误

### 错误示例

```python
from doke_rag.pipeline.utils import AudioFileError, APIError

try:
    result = await processor.transcribe("audio.mp3", "output/")
except AudioFileError as e:
    print(f"音频文件错误: {e}")
except APIError as e:
    print(f"API 调用失败: {e}")
    if "authentication" in str(e):
        print("请检查 OPENAI_API_KEY 是否正确")
    elif "quota" in str(e):
        print("API 配额不足")
```

## 测试

运行测试：

```bash
# 运行所有测试
pytest tests/pipeline/test_audio_processor.py -v

# 运行特定测试
pytest tests/pipeline/test_audio_processor.py::TestAudioProcessor::test_transcribe_success -v
```

## 示例

查看完整示例：

```bash
python examples/audio_processor_example.py
```

## 性能和成本

### Whisper API 定价

- **价格**: $0.006 / 分钟
- **免费额度**: 每月 500 分钟（可能变化）

### 性能优化

1. **批量处理**: 使用 `batch_transcribe` 并发处理多个文件
2. **指定语言**: 设置 `language` 参数可提高准确率和速度
3. **温度设置**: 使用 `temperature=0.0` 获得更确定的转录

## 常见问题

### 1. 如何提高中文转录准确率？

```python
result = await processor.transcribe(
    "chinese_audio.mp3",
    "output/",
    language="zh",  # 明确指定中文
    prompt="相关领域的关键词"  # 提供上下文
)
```

### 2. 如何处理超长音频？

Whisper API 支持最长 25 MB 的音频文件（约 2-3 小时）。对于更长的音频，建议先分割。

### 3. 支持实时转录吗？

当前版本不支持实时转录。如需实时功能，建议使用本地 Whisper 模型。

### 4. 转录速度如何？

通常为实时速度的 0.2x-0.5x（即 10 分钟音频约需 2-5 分钟）。

## 后续计划

- [ ] 支持本地 Whisper 模型（离线使用）
- [ ] 实时转录支持
- [ ] 说话人分离（Diarization）
- [ ] 翻译功能
- [ ] 更多输出格式（SRT, VTT 等）

## 技术支持

如有问题或建议，请提交 Issue 或 Pull Request。

## 许可证

本项目遵循 DOKE-RAG 项目的许可证。
