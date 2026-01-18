# 测试文件说明

本目录包含 PDF 解析器的测试用例和测试数据生成脚本。

## 文件说明

- `test_pdf_parser.py`: PDF 解析器测试用例
- `create_test_pdf.py`: 测试 PDF 生成脚本
- `fixtures/`: 测试 PDF 文件目录（运行 create_test_pdf.py 后生成）
- `expected_outputs/`: 预期输出示例（可选）

## 创建测试数据

### 方法 1: 使用脚本生成测试 PDF

```bash
# 1. 安装 reportlab
pip install reportlab

# 2. 运行生成脚本
python create_test_pdf.py
```

这将生成以下测试文件：
- `fixtures/sample_simple.pdf` - 简单文本
- `fixtures/sample_with_tables.pdf` - 包含表格
- `fixtures/sample_multi_page.pdf` - 多页文档
- `fixtures/sample_complex.pdf` - 复杂文档

### 方法 2: 使用自己的 PDF 文件

将你的测试 PDF 文件放到 `fixtures/` 目录：
```bash
cp your_test.pdf fixtures/
```

## 运行测试

```bash
# 运行所有测试
pytest tests/pipeline/test_pdf_parser.py -v

# 运行特定测试
pytest tests/pipeline/test_pdf_parser.py::TestPDFParser::test_parse_simple_pdf -v

# 显示详细输出
pytest tests/pipeline/test_pdf_parser.py -v -s
```

## 测试覆盖

- ✅ 简单文本 PDF 解析
- ✅ 包含图片的 PDF
- ✅ 包含表格的 PDF
- ✅ 多页 PDF
- ✅ 文件验证（格式、大小、加密）
- ✅ 错误处理（损坏文件、超大文件）
- ✅ 容错模式
- ✅ HTML 表格转 Markdown
- ✅ 表格数据转 Markdown
- ✅ 端到端集成测试

## 注意事项

1. 首次运行测试时，PaddleOCR 会自动下载模型文件（约 10-20MB）
2. 确保已正确安装 PaddleOCR（参见 `docs/PADDLEOCR_INSTALLATION.md`）
3. 如果 GPU 不可用，会自动使用 CPU 模式
4. 某些测试需要实际的 PDF 文件，如果没有会自动跳过

## 自定义测试

要添加自己的测试：

1. 将测试 PDF 放到 `fixtures/` 目录
2. 在 `test_pdf_parser.py` 中添加测试方法
3. 运行测试验证

示例：

```python
@pytest.mark.asyncio
async def test_my_custom_pdf(self, parser, output_dir):
    """测试自定义 PDF"""
    result = await parser.parse_pdf(
        "tests/pipeline/fixtures/my_test.pdf",
        output_dir,
        extract_images=True
    )

    # 验证结果
    assert Path(result["markdown_path"]).exists()
```

## 清理测试数据

```bash
# 删除生成的测试 PDF
rm fixtures/*.pdf

# 删除测试输出
rm -rf tests/output/
```
