"""
PDF 解析器模块

使用 PaddleOCR-VL 提取 PDF 中的多模态内容：
- 文本
- 公式（LaTeX）
- 表格（Markdown）
- 图片
"""

import asyncio
import numpy as np
import pdfplumber
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import cv2
import re

from paddleocr import PaddleOCR, PPStructure
from PIL import Image

from doke_rag.config.pipeline_config import PipelineConfig
from doke_rag.pipeline.utils import (
    setup_logger,
    ensure_output_dir,
    get_timestamp_filename,
    save_metadata,
    validate_file_size,
    validate_file_extension,
)


class PDFParseError(Exception):
    """PDF 解析错误"""
    pass


class PDFParser:
    """PDF 解析器，使用 PaddleOCR-VL 提取多模态内容"""

    def __init__(self, model_dir: Optional[Path] = None):
        """
        初始化 PDF 解析器

        Args:
            model_dir: PaddleOCR 模型目录，默认使用配置中的路径
        """
        self.model_dir = model_dir or PipelineConfig.PADDLE_MODEL_DIR
        self.logger = setup_logger("PDFParser")

        # 初始化 PaddleOCR
        self.logger.info("初始化 PaddleOCR...")
        try:
            self.ocr = PaddleOCR(
                use_angle_cls=True,
                lang=PipelineConfig.OCR_LANG,
                use_gpu=PipelineConfig.USE_GPU,
                show_log=False
            )

            # 初始化 PPStructure（版面分析）
            self.structure = PPStructure(
                ocr=self.ocr,
                show_log=False,
                structure_version='PP-StructureV2'
            )

            self.logger.info("PaddleOCR 初始化成功")

        except Exception as e:
            self.logger.error(f"PaddleOCR 初始化失败: {str(e)}")
            raise PDFParseError(f"初始化失败: {str(e)}")

    async def parse_pdf(
        self,
        pdf_path: str,
        output_dir: str,
        extract_images: bool = True
    ) -> Dict[str, str]:
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
        output_dir = Path(output_dir)

        try:
            # 1. 验证文件
            self._validate_pdf(pdf_path)

            # 2. 打开 PDF
            self.logger.info(f"打开 PDF 文件: {pdf_path}")
            pdf_file = pdfplumber.open(pdf_path)
            total_pages = len(pdf_file.pages)

            # 3. 逐页处理
            markdown_parts = []
            images_info = []

            for page_num, page in enumerate(pdf_file.pages, 1):
                self.logger.info(f"处理第 {page_num}/{total_pages} 页")

                try:
                    # 提取文本和结构
                    page_result = await self._process_page(
                        page, page_num, output_dir, extract_images
                    )

                    markdown_parts.append(page_result["markdown"])
                    images_info.extend(page_result["images"])

                except Exception as e:
                    self.logger.error(f"处理第 {page_num} 页失败: {str(e)}")
                    if PipelineConfig.MODE == "strict":
                        raise PDFParseError(f"处理第 {page_num} 页失败: {str(e)}")
                    # tolerant 模式：跳过该页
                    markdown_parts.append(f"\n\n[第 {page_num} 页处理失败]\n\n")

            pdf_file.close()

            # 4. 组装完整 Markdown
            full_markdown = "\n\n".join(markdown_parts)

            # 5. 保存文件
            ensure_output_dir(output_dir)

            timestamp = get_timestamp_filename("document", "md")
            markdown_path = output_dir / timestamp
            metadata_path = output_dir / timestamp.replace(".md", "_metadata.json")

            self._save_markdown(full_markdown, markdown_path)

            # 保存元数据
            metadata = {
                "source": pdf_path,
                "total_pages": total_pages,
                "images_count": len(images_info),
                "images": images_info,
                "processed_at": datetime.now().isoformat()
            }
            save_metadata(metadata, metadata_path)

            self.logger.info(f"PDF 解析完成: {markdown_path}")

            return {
                "markdown_path": str(markdown_path),
                "metadata_path": str(metadata_path),
                "images_dir": str(output_dir / "extracted_images")
            }

        except PDFParseError:
            raise
        except Exception as e:
            self.logger.error(f"PDF 解析失败: {str(e)}")
            raise PDFParseError(f"解析失败: {str(e)}")

    def _validate_pdf(self, file_path: str) -> None:
        """
        验证 PDF 文件

        Args:
            file_path: 文件路径

        Raises:
            PDFParseError: 文件验证失败
        """
        # 检查文件是否存在
        if not Path(file_path).exists():
            raise PDFParseError(f"文件不存在: {file_path}")

        # 检查格式
        if not validate_file_extension(file_path, ['.pdf']):
            raise PDFParseError("文件格式错误，必须是 .pdf 文件")

        # 检查大小
        size_mb = get_file_size_mb(file_path)
        if size_mb > PipelineConfig.MAX_PDF_SIZE_MB:
            raise PDFParseError(
                f"PDF 文件过大: {size_mb:.2f}MB "
                f"(限制: {PipelineConfig.MAX_PDF_SIZE_MB}MB)"
            )

        # 检查是否加密
        try:
            pdf_file = pdfplumber.open(file_path)
            first_page = pdf_file.pages[0]

            # pdfplumber 的 is_encrypted 属性
            if hasattr(first_page, 'is_encrypted') and first_page.is_encrypted:
                pdf_file.close()
                raise PDFParseError("PDF 文件已加密，无法解析")

            pdf_file.close()

        except PDFParseError:
            raise
        except Exception as e:
            error_msg = str(e).lower()
            if "encrypted" in error_msg or "password" in error_msg:
                raise PDFParseError("PDF 文件已加密，无法解析")
            # 其他错误可能在后续处理中暴露
            pass

    async def _process_page(
        self,
        page,
        page_num: int,
        output_dir: Path,
        extract_images: bool
    ) -> Dict[str, Any]:
        """
        处理单页 PDF

        Args:
            page: pdfplumber Page 对象
            page_num: 页码
            output_dir: 输出目录
            extract_images: 是否提取图片

        Returns:
            dict: 包含 markdown 和 images 的字典
        """
        markdown_parts = []
        images = []

        try:
            # 将页面转换为图像
            page_img = page.to_image()

            # 转换为 numpy array (RGB 格式)
            img_array = np.array(page_img.convert('RGB'))

            # 使用 PaddleOCR-VL 进行版面分析
            layout_result = self.structure(img_img=img_array)

            # 处理每个区域
            for region in layout_result:
                region_type = region.get('type', '')

                try:
                    if region_type == 'text':
                        # 文本段落
                        text = self._extract_text(region)
                        if text:
                            markdown_parts.append(text)

                    elif region_type == 'title':
                        # 标题
                        title_text = self._extract_title(region)
                        if title_text:
                            markdown_parts.append(title_text)

                    elif region_type == 'table':
                        # 表格
                        table_markdown = self._convert_table_to_markdown(region)
                        if table_markdown:
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

                except Exception as e:
                    self.logger.warning(f"处理区域失败: {str(e)}")
                    if PipelineConfig.MODE == "strict":
                        raise

        except Exception as e:
            self.logger.error(f"版面分析失败: {str(e)}")
            if PipelineConfig.MODE == "strict":
                raise

        # 如果没有识别到任何内容，尝试提取原始文本
        if not markdown_parts:
            self.logger.warning(f"第 {page_num} 页未识别到结构，尝试提取原始文本")
            original_text = page.extract_text()
            if original_text:
                markdown_parts.append(original_text)

        return {
            "markdown": "\n\n".join(markdown_parts),
            "images": images
        }

    def _extract_text(self, region: Dict) -> str:
        """
        提取文本区域

        Args:
            region: PaddleOCR 识别的区域

        Returns:
            str: 提取的文本
        """
        try:
            res = region.get('res', {})
            if isinstance(res, dict):
                return res.get('text', '')
            elif isinstance(res, list):
                # 合并所有文本块
                texts = []
                for item in res:
                    if isinstance(item, dict):
                        text = item.get('text', '')
                        if text:
                            texts.append(text)
                return ' '.join(texts)
        except Exception as e:
            self.logger.warning(f"提取文本失败: {str(e)}")

        return ''

    def _extract_title(self, region: Dict) -> str:
        """
        提取标题区域

        Args:
            region: PaddleOCR 识别的区域

        Returns:
            str: Markdown 格式的标题
        """
        try:
            res = region.get('res', {})
            title_text = ''

            if isinstance(res, dict):
                title_text = res.get('text', '')
            elif isinstance(res, list) and len(res) > 0:
                title_text = res[0].get('text', '')

            if title_text:
                # 尝试获取标题级别
                level = region.get('level', 2)
                prefix = "#" * min(level, 6)  # 最多 6 级标题
                return f"{prefix} {title_text}"

        except Exception as e:
            self.logger.warning(f"提取标题失败: {str(e)}")

        return ''

    def _convert_table_to_markdown(self, table_region: Dict) -> str:
        """
        将表格转换为 Markdown 格式

        Args:
            table_region: PaddleOCR 识别的表格区域

        Returns:
            str: Markdown 格式的表格
        """
        try:
            res = table_region.get('res', {})

            # 方法1: 从 HTML 表格转换
            html = res.get('html', '')
            if html:
                return self._html_table_to_markdown(html)

            # 方法2: 从表格数据转换
            table_data = res.get('table_data', [])
            if table_data:
                return self._table_data_to_markdown(table_data)

        except Exception as e:
            self.logger.warning(f"表格转换失败: {str(e)}")

        return ''

    def _html_table_to_markdown(self, html: str) -> str:
        """
        将 HTML 表格转换为 Markdown

        Args:
            html: HTML 表格字符串

        Returns:
            str: Markdown 表格
        """
        try:
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(html, 'html.parser')
            table = soup.find('table')

            if not table:
                return ''

            rows = table.find_all('tr')
            if not rows:
                return ''

            markdown_rows = []

            for i, row in enumerate(rows):
                cells = row.find_all(['td', 'th'])
                if not cells:
                    continue

                # 提取单元格文本
                cell_texts = [cell.get_text().strip() for cell in cells]
                row_str = "| " + " | ".join(cell_texts) + " |"
                markdown_rows.append(row_str)

                # 添加表头分隔符
                if i == 0:
                    separator = "| " + " | ".join(["---"] * len(cells)) + " |"
                    markdown_rows.append(separator)

            return "\n".join(markdown_rows)

        except Exception as e:
            self.logger.warning(f"HTML 表格转换失败: {str(e)}")
            return ''

    def _table_data_to_markdown(self, table_data: List) -> str:
        """
        将表格数据转换为 Markdown

        Args:
            table_data: 表格数据（二维列表）

        Returns:
            str: Markdown 表格
        """
        if not table_data or not isinstance(table_data, list):
            return ''

        try:
            markdown_rows = []

            for i, row in enumerate(table_data):
                if not isinstance(row, list):
                    continue

                # 处理单元格数据
                cells = []
                for cell in row:
                    if isinstance(cell, dict):
                        cells.append(cell.get('text', ''))
                    else:
                        cells.append(str(cell))

                row_str = "| " + " | ".join(cells) + " |"
                markdown_rows.append(row_str)

                # 添加表头分隔符
                if i == 0:
                    separator = "| " + " | ".join(["---"] * len(cells)) + " |"
                    markdown_rows.append(separator)

            return "\n".join(markdown_rows)

        except Exception as e:
            self.logger.warning(f"表格数据转换失败: {str(e)}")
            return ''

    def _extract_image(
        self,
        region: Dict,
        page_num: int,
        output_dir: Path,
        img_index: int
    ) -> Optional[Dict[str, Any]]:
        """
        提取并保存图片

        Args:
            region: PaddleOCR 识别的图片区域
            page_num: 页码
            output_dir: 输出目录
            img_index: 图片索引

        Returns:
            dict: 图片信息，包含 path, page, caption, bbox
        """
        try:
            # 获取图片数据
            img_data = region.get('img')
            if img_data is None:
                return None

            # 创建图片目录
            img_dir = output_dir / "extracted_images" / f"page_{page_num}"
            ensure_output_dir(img_dir)

            # 生成文件名
            img_filename = f"fig_{img_index + 1}.png"
            img_path = img_dir / img_filename

            # 保存图片
            if isinstance(img_data, np.ndarray):
                # numpy array 格式
                cv2.imwrite(str(img_path), img_data)
            else:
                # PIL Image 格式
                if hasattr(img_data, 'save'):
                    img_data.save(str(img_path))
                else:
                    # 尝试转换
                    img_pil = Image.fromarray(img_data)
                    img_pil.save(str(img_path))

            # 获取边界框
            bbox = region.get('bbox', [])

            return {
                "path": str(img_path),
                "page": page_num,
                "caption": f"Figure {page_num}-{img_index + 1}",
                "bbox": bbox
            }

        except Exception as e:
            self.logger.warning(f"提取图片失败: {str(e)}")
            return None

    def _save_markdown(self, content: str, path: Path) -> None:
        """
        保存 Markdown 文件

        Args:
            content: Markdown 内容
            path: 保存路径
        """
        ensure_output_dir(path.parent)

        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

        self.logger.info(f"Markdown 已保存: {path}")
