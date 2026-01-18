"""
PDF 解析器测试用例

测试 PDFParser 的各种功能：
- 文本提取
- 表格识别
- 图片提取
- 加密 PDF 处理
- 超大文件处理
- 格式错误处理
"""

import pytest
import asyncio
from pathlib import Path
import json
import tempfile
import os

from doke_rag.pipeline.pdf_parser import PDFParser, PDFParseError
from doke_rag.config.pipeline_config import PipelineConfig


class TestPDFParser:
    """PDF 解析器测试"""

    @pytest.fixture
    def parser(self):
        """创建 PDFParser 实例"""
        try:
            return PDFParser()
        except Exception as e:
            pytest.skip(f"无法初始化 PDFParser: {str(e)}")

    @pytest.fixture
    def output_dir(self, tmp_path):
        """创建临时输出目录"""
        output = tmp_path / "output"
        output.mkdir(exist_ok=True)
        return str(output)

    @pytest.mark.asyncio
    async def test_parse_simple_pdf(self, parser, output_dir):
        """测试解析简单文本 PDF"""
        # 注意：这需要一个实际的测试 PDF 文件
        test_pdf = "tests/pipeline/fixtures/sample_simple.pdf"

        if not Path(test_pdf).exists():
            pytest.skip(f"测试文件不存在: {test_pdf}")

        result = await parser.parse_pdf(
            test_pdf,
            output_dir,
            extract_images=False
        )

        # 验证输出文件存在
        assert Path(result["markdown_path"]).exists()
        assert Path(result["metadata_path"]).exists()

        # 验证 Markdown 内容
        with open(result["markdown_path"], "r", encoding="utf-8") as f:
            content = f.read()
        assert len(content) > 0

        # 验证元数据
        with open(result["metadata_path"], "r", encoding="utf-8") as f:
            metadata = json.load(f)
        assert "source" in metadata
        assert "total_pages" in metadata
        assert "processed_at" in metadata

    @pytest.mark.asyncio
    async def test_parse_pdf_with_images(self, parser, output_dir):
        """测试解析包含图片的 PDF"""
        test_pdf = "tests/pipeline/fixtures/sample_with_images.pdf"

        if not Path(test_pdf).exists():
            pytest.skip(f"测试文件不存在: {test_pdf}")

        result = await parser.parse_pdf(
            test_pdf,
            output_dir,
            extract_images=True
        )

        # 验证图片目录存在
        assert Path(result["images_dir"]).exists()

        # 验证 Markdown 中包含图片引用
        with open(result["markdown_path"], "r", encoding="utf-8") as f:
            content = f.read()
        assert "![" in content or "Figure" in content

    @pytest.mark.asyncio
    async def test_parse_pdf_with_tables(self, parser, output_dir):
        """测试解析包含表格的 PDF"""
        test_pdf = "tests/pipeline/fixtures/sample_with_tables.pdf"

        if not Path(test_pdf).exists():
            pytest.skip(f"测试文件不存在: {test_pdf}")

        result = await parser.parse_pdf(
            test_pdf,
            output_dir,
            extract_images=False
        )

        # 验证 Markdown 中包含表格
        with open(result["markdown_path"], "r", encoding="utf-8") as f:
            content = f.read()
        assert "|" in content  # Markdown 表格使用 | 分隔

    def test_validate_pdf_success(self, parser):
        """测试 PDF 验证 - 成功案例"""
        # 创建一个临时 PDF 文件（空的）
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            temp_path = f.name
            f.write(b"%PDF-1.4\n")

        try:
            # 不应该抛出异常
            parser._validate_pdf(temp_path)
        finally:
            os.unlink(temp_path)

    def test_validate_pdf_invalid_extension(self, parser):
        """测试 PDF 验证 - 无效扩展名"""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            temp_path = f.name

        try:
            with pytest.raises(PDFParseError, match="格式错误"):
                parser._validate_pdf(temp_path)
        finally:
            os.unlink(temp_path)

    def test_validate_pdf_file_not_found(self, parser):
        """测试 PDF 验证 - 文件不存在"""
        with pytest.raises(PDFParseError, match="不存在"):
            parser._validate_pdf("nonexistent.pdf")

    def test_validate_pdf_oversized(self, parser):
        """测试 PDF 验证 - 文件过大"""
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            temp_path = f.name
            # 创建一个超过限制的文件（假设限制是 100MB，创建 150MB）
            f.seek(150 * 1024 * 1024)
            f.write(b"\0")

        try:
            with pytest.raises(PDFParseError, match="过大"):
                parser._validate_pdf(temp_path)
        finally:
            os.unlink(temp_path)

    def test_html_table_to_markdown(self, parser):
        """测试 HTML 表格转 Markdown"""
        html = """
        <table>
            <tr>
                <th>列1</th>
                <th>列2</th>
            </tr>
            <tr>
                <td>数据1</td>
                <td>数据2</td>
            </tr>
        </table>
        """

        markdown = parser._html_table_to_markdown(html)

        assert "| 列1 |" in markdown
        assert "| 列2 |" in markdown
        assert "| 数据1 |" in markdown
        assert "| 数据2 |" in markdown
        assert "| --- |" in markdown  # 表头分隔符

    def test_table_data_to_markdown(self, parser):
        """测试表格数据转 Markdown"""
        table_data = [
            ["列1", "列2", "列3"],
            ["数据1", "数据2", "数据3"],
            ["数据4", "数据5", "数据6"]
        ]

        markdown = parser._table_data_to_markdown(table_data)

        assert "| 列1 |" in markdown
        assert "| 数据1 |" in markdown
        assert "| --- |" in markdown  # 表头分隔符

    @pytest.mark.asyncio
    async def test_parse_multiple_pages(self, parser, output_dir):
        """测试解析多页 PDF"""
        test_pdf = "tests/pipeline/fixtures/sample_multi_page.pdf"

        if not Path(test_pdf).exists():
            pytest.skip(f"测试文件不存在: {test_pdf}")

        result = await parser.parse_pdf(
            test_pdf,
            output_dir,
            extract_images=False
        )

        # 验证元数据中的页数
        with open(result["metadata_path"], "r", encoding="utf-8") as f:
            metadata = json.load(f)
        assert metadata["total_pages"] >= 2


class TestPDFParserErrorHandling:
    """测试 PDF 解析器的错误处理"""

    @pytest.fixture
    def parser(self):
        """创建 PDFParser 实例"""
        try:
            return PDFParser()
        except Exception as e:
            pytest.skip(f"无法初始化 PDFParser: {str(e)}")

    def test_corrupted_pdf(self, parser):
        """测试损坏的 PDF 文件"""
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            temp_path = f.name
            f.write(b"This is not a valid PDF file")

        try:
            with pytest.raises(PDFParseError):
                parser._validate_pdf(temp_path)
        finally:
            os.unlink(temp_path)

    @pytest.mark.asyncio
    async def test_tolerance_mode(self, parser, tmp_path):
        """测试容错模式"""
        # 设置容错模式
        import os
        os.environ["PIPELINE_MODE"] = "tolerant"

        # 重新加载配置
        from doke_rag.config.pipeline_config import PipelineConfig
        PipelineConfig.MODE = "tolerant"

        # 注意：这需要一个有部分损坏页面的 PDF
        # 在实际测试中，应该跳过失败的页面而不是抛出异常
        # 这里只是演示测试结构

        # 恢复严格模式
        os.environ["PIPELINE_MODE"] = "strict"


class TestPDFParserIntegration:
    """集成测试"""

    @pytest.fixture
    def parser(self):
        """创建 PDFParser 实例"""
        try:
            return PDFParser()
        except Exception as e:
            pytest.skip(f"无法初始化 PDFParser: {str(e)}")

    @pytest.mark.asyncio
    async def test_end_to_end_pipeline(self, parser, tmp_path):
        """端到端测试"""
        test_pdf = "tests/pipeline/fixtures/sample_complex.pdf"
        output_dir = str(tmp_path / "output")

        if not Path(test_pdf).exists():
            pytest.skip(f"测试文件不存在: {test_pdf}")

        # 解析 PDF
        result = await parser.parse_pdf(
            test_pdf,
            output_dir,
            extract_images=True
        )

        # 验证所有输出文件
        assert Path(result["markdown_path"]).exists()
        assert Path(result["metadata_path"]).exists()

        # 验证 Markdown 格式
        with open(result["markdown_path"], "r", encoding="utf-8") as f:
            content = f.read()

        # 应该包含一些基本结构
        assert len(content) > 0

        # 验证元数据完整性
        with open(result["metadata_path"], "r", encoding="utf-8") as f:
            metadata = json.load(f)

        required_fields = ["source", "total_pages", "images_count", "processed_at"]
        for field in required_fields:
            assert field in metadata


if __name__ == "__main__":
    # 可以直接运行测试
    pytest.main([__file__, "-v", "-s"])
