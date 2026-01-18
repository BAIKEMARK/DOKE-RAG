"""
创建测试用的 PDF 文件

这个脚本会生成各种类型的测试 PDF：
- 简单文本 PDF
- 包含表格的 PDF
- 包含图片的 PDF
- 多页 PDF
"""

from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from reportlab.lib.enums import TA_CENTER


def create_simple_text_pdf(output_path):
    """创建简单文本 PDF"""
    doc = SimpleDocTemplate(output_path, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    # 标题
    title = Paragraph("简单文本文档", styles["Heading1"])
    story.append(title)
    story.append(Spacer(1, 12))

    # 正文
    text = """
    这是一个简单的测试文档，用于测试 PDF 解析器的基本文本提取功能。

    DOKE-RAG 框架支持多模态数据处理，包括 PDF 解析、音频转录等功能。

    这个文档包含多个段落，用于验证文本提取的准确性。
    """
    para = Paragraph(text, styles["BodyText"])
    story.append(para)
    story.append(Spacer(1, 12))

    # 更多内容
    more_text = "第二段内容：测试中文和英文混合的文本提取效果。Testing mixed Chinese and English text extraction."
    para2 = Paragraph(more_text, styles["BodyText"])
    story.append(para2)

    doc.build(story)
    print(f"已创建: {output_path}")


def create_table_pdf(output_path):
    """创建包含表格的 PDF"""
    doc = SimpleDocTemplate(output_path, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    # 标题
    title = Paragraph("包含表格的文档", styles["Heading1"])
    story.append(title)
    story.append(Spacer(1, 12))

    # 说明
    text = "下面的表格展示了不同学生的成绩信息："
    para = Paragraph(text, styles["BodyText"])
    story.append(para)
    story.append(Spacer(1, 12))

    # 表格数据
    data = [
        ["姓名", "数学", "语文", "英语", "总分"],
        ["张三", "95", "88", "92", "275"],
        ["李四", "87", "95", "89", "271"],
        ["王五", "92", "90", "94", "276"],
        ["赵六", "89", "92", "88", "269"],
    ]

    # 创建表格
    table = Table(data, colWidths=[1.5*inch, 1*inch, 1*inch, 1*inch, 1*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 14),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))

    story.append(table)
    story.append(Spacer(1, 12))

    # 第二个表格
    title2 = Paragraph("产品价格表", styles["Heading2"])
    story.append(title2)
    story.append(Spacer(1, 12))

    data2 = [
        ["产品名称", "价格", "库存", "类别"],
        ["笔记本电脑", "5999", "50", "电子产品"],
        ["无线鼠标", "99", "200", "配件"],
        ["键盘", "199", "150", "配件"],
    ]

    table2 = Table(data2, colWidths=[2*inch, 1*inch, 1*inch, 1.5*inch])
    table2.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))

    story.append(table2)

    doc.build(story)
    print(f"已创建: {output_path}")


def create_multi_page_pdf(output_path):
    """创建多页 PDF"""
    doc = SimpleDocTemplate(output_path, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    for i in range(1, 6):
        # 页面标题
        title = Paragraph(f"第 {i} 页", styles["Heading1"])
        story.append(title)
        story.append(Spacer(1, 12))

        # 内容
        text = f"""
        这是第 {i} 页的内容。

        DOKE-RAG 是一个领域知识增强的检索生成框架，支持构建专业领域的知识图谱和向量数据库。

        多页文档测试用于验证分页处理和页面整合功能。
        """
        para = Paragraph(text, styles["BodyText"])
        story.append(para)
        story.append(Spacer(1, 12))

        # 添加分页符（除了最后一页）
        if i < 5:
            story.append(PageBreak())

    doc.build(story)
    print(f"已创建: {output_path}")


def create_complex_pdf(output_path):
    """创建复杂 PDF（包含多种元素）"""
    doc = SimpleDocTemplate(output_path, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    # 封面
    title = Paragraph("DOKE-RAG 测试文档", styles["Title"])
    story.append(title)
    story.append(Spacer(1, 24))

    subtitle = Paragraph("包含多种元素的复杂文档", styles["Heading2"])
    story.append(subtitle)
    story.append(Spacer(1, 12))

    # 第一页：文本
    story.append(PageBreak())

    title1 = Paragraph("第一章：简介", styles["Heading1"])
    story.append(title1)
    story.append(Spacer(1, 12))

    text1 = """
    DOKE-RAG 框架提供了完整的多模态数据处理能力。

    主要功能包括：
    • PDF 文档解析
    • 音频转录
    • 知识图谱构建
    • 向量检索

    本文档用于测试这些功能的综合效果。
    """
    para1 = Paragraph(text1, styles["BodyText"])
    story.append(para1)
    story.append(Spacer(1, 12))

    # 第二页：表格
    story.append(PageBreak())

    title2 = Paragraph("第二章：数据统计", styles["Heading1"])
    story.append(title2)
    story.append(Spacer(1, 12))

    data = [
        ["指标", "数值", "单位"],
        ["处理文档数", "1000", "个"],
        ["识别准确率", "95.5", "%"],
        ["平均处理时间", "2.3", "秒"],
    ]

    table = Table(data, colWidths=[2*inch, 1.5*inch, 1*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(table)

    doc.build(story)
    print(f"已创建: {output_path}")


if __name__ == "__main__":
    import os

    # 创建输出目录
    output_dir = "tests/pipeline/fixtures"
    os.makedirs(output_dir, exist_ok=True)

    # 创建各种测试 PDF
    print("开始创建测试 PDF 文件...")

    create_simple_text_pdf(os.path.join(output_dir, "sample_simple.pdf"))
    create_table_pdf(os.path.join(output_dir, "sample_with_tables.pdf"))
    create_multi_page_pdf(os.path.join(output_dir, "sample_multi_page.pdf"))
    create_complex_pdf(os.path.join(output_dir, "sample_complex.pdf"))

    print("\n所有测试 PDF 文件创建完成！")
    print("\n注意：由于 reportlab 库的限制，这些 PDF 主要包含文本和表格。")
    print("对于图片提取测试，建议使用实际包含图片的 PDF 文件。")
