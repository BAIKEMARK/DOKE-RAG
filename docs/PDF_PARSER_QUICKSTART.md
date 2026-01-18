# PDF 解析器快速参考指南

## 快速开始

### 1. 安装依赖

```bash
# 基础安装
pip install paddlepaddle>=2.5.0
pip install paddleocr>=2.7.0
pip install pdfplumber>=0.10.0 pillow>=10.0.0 opencv-python>=4.8.0

# 安装所有 DOKE-RAG 依赖
pip install -r requirements.txt
```

### 2. 配置环境

编辑 `.env` 文件：

```bash
PADDLE_MODEL_DIR=models/paddleocr/
EXTRACT_IMAGES=true
IMAGES_OUTPUT_DIR=data/extracted_images/
MAX_PDF_SIZE=100
USE_GPU=false
OCR_LANG=ch
PIPELINE_MODE=strict
```

### 3. 基本使用

```python
from doke_rag.pipeline import PDFParser
import asyncio

async def main():
    # 初始化解析器
    parser = PDFParser()

    # 解析 PDF
    result = await parser.parse_pdf(
        pdf_path="document.pdf",
        output_dir="data/processed/",
        extract_images=True
    )

    print(f"Markdown: {result['markdown_path']}")
    print(f"元数据: {result['metadata_path']}")
    print(f"图片: {result['images_dir']}")

asyncio.run(main())
```

## 核心功能

### PDFParser 类

```python
class PDFParser:
    def __init__(model_dir=None)
    async def parse_pdf(pdf_path, output_dir, extract_images)
```

### 配置管理

```python
from doke_rag.config.pipeline_config import PipelineConfig

# 查看配置
config = PipelineConfig.to_dict()

# 验证配置
PipelineConfig.validate()
```

### 共享工具

```python
from doke_rag.pipeline.utils import (
    setup_logger,
    save_metadata,
    get_timestamp_filename,
    ensure_output_dir
)
```

## 输出格式

### Markdown 文件

```markdown
# 文档标题

## 第一章

正文内容...

| 列1 | 列2 |
|-----|-----|
| 数据1 | 数据2 |

![Figure 1-1](extracted_images/page_1/fig_1.png)
```

### 元数据 JSON

```json
{
  "source": "document.pdf",
  "total_pages": 10,
  "images_count": 5,
  "images": [
    {
      "path": "data/extracted_images/page_1/fig_1.png",
      "page": 1,
      "caption": "Figure 1-1",
      "bbox": [100, 200, 300, 400]
    }
  ],
  "processed_at": "2025-01-18T14:30:22"
}
```

## 测试

### 运行测试

```bash
# 1. 安装测试依赖
pip install reportlab pytest

# 2. 创建测试 PDF
cd tests/pipeline
python create_test_pdf.py

# 3. 运行测试
cd ../../
pytest tests/pipeline/test_pdf_parser.py -v
```

### 测试覆盖

- ✅ 文本提取
- ✅ 表格识别
- ✅ 图片提取
- ✅ 多页处理
- ✅ 错误处理
- ✅ 文件验证
- ✅ 端到端测试

## 常见问题

### Q: PaddleOCR 初始化失败?

```bash
# 重新安装
pip uninstall paddlepaddle paddleocr
pip install paddlepaddle>=2.5.0
pip install paddleocr>=2.7.0
```

### Q: 模型下载慢?

```bash
# 使用国内镜像
export HUB_URL=https://hub.paddlepaddle.org.cn
```

### Q: GPU 不可用?

```bash
# 检查 CUDA
nvidia-smi

# 安装 GPU 版本
pip uninstall paddlepaddle
pip install paddlepaddle-gpu
```

### Q: 识别准确率低?

- 提高扫描质量
- 使用 GPU 加速
- 调整 OCR 参数
- 考虑人工校验

## 性能优化

### 使用 GPU

```bash
# 安装 GPU 版本
pip install paddlepaddle-gpu

# 配置 .env
USE_GPU=true
```

### 批量处理

```python
import asyncio

async def process_batch(pdf_files):
    parser = PDFParser()
    tasks = [
        parser.parse_pdf(pdf, "output/", True)
        for pdf in pdf_files
    ]
    results = await asyncio.gather(*tasks)
    return results
```

### 增加批处理大小

```python
parser = PDFParser()
parser.ocr = PaddleOCR(
    use_angle_cls=True,
    lang='ch',
    use_gpu=True,
    max_batch_size=10  # 增加批处理
)
```

## 高级配置

### 自定义模型目录

```python
parser = PDFParser(model_dir=Path("custom_models/"))
```

### 公式格式选择

```bash
# .env
FORMULA_FORMAT=latex  # 或 image
```

### 运行模式

```bash
# .env
PIPELINE_MODE=strict    # 遇到错误立即停止
PIPELINE_MODE=tolerant  # 跳过错误继续处理
```

## 相关文档

- [详细安装指南](PADDLEOCR_INSTALLATION.md)
- [实现总结](pdf_parser_implementation_summary.md)
- [设计文档](../plans/2025-01-18-multimodal-pipeline-design.md)
- [使用示例](../examples/pdf_parser_example.py)

## 技术支持

- 查看 [PaddleOCR 安装指南](PADDLEOCR_INSTALLATION.md)
- 搜索 [PaddleOCR Issues](https://github.com/PaddlePaddle/PaddleOCR/issues)
- 在 DOKE-RAG 仓库提出 Issue
