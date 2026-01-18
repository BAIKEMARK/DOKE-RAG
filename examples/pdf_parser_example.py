"""
PDF 解析器使用示例

演示如何使用 DOKE-RAG 的 PDF 解析器提取文档内容
"""

import asyncio
from pathlib import Path
from doke_rag.pipeline import PDFParser
from doke_rag.config.pipeline_config import PipelineConfig


async def example_simple_parsing():
    """示例 1: 简单 PDF 解析"""
    print("=" * 60)
    print("示例 1: 简单 PDF 解析")
    print("=" * 60)

    # 初始化解析器
    parser = PDFParser()

    # 解析 PDF
    result = await parser.parse_pdf(
        pdf_path="data/input/sample.pdf",
        output_dir="data/processed/",
        extract_images=False
    )

    print(f"\nMarkdown 文件: {result['markdown_path']}")
    print(f"元数据文件: {result['metadata_path']}")

    # 读取 Markdown 内容
    with open(result['markdown_path'], 'r', encoding='utf-8') as f:
        content = f.read()
    print(f"\n提取的文本内容（前 200 字符）:\n{content[:200]}...")


async def example_with_images():
    """示例 2: 提取图片"""
    print("\n" + "=" * 60)
    print("示例 2: 提取图片")
    print("=" * 60)

    parser = PDFParser()

    result = await parser.parse_pdf(
        pdf_path="data/input/document_with_figures.pdf",
        output_dir="data/processed/",
        extract_images=True
    )

    print(f"\nMarkdown 文件: {result['markdown_path']}")
    print(f"图片目录: {result['images_dir']}")

    # 检查提取的图片
    images_dir = Path(result['images_dir'])
    if images_dir.exists():
        images = list(images_dir.rglob("*.png"))
        print(f"\n提取了 {len(images)} 张图片")
        for img in images[:5]:  # 显示前 5 张
            print(f"  - {img.relative_to(images_dir.parent)}")


async def example_batch_processing():
    """示例 3: 批量处理多个 PDF"""
    print("\n" + "=" * 60)
    print("示例 3: 批量处理")
    print("=" * 60)

    parser = PDFParser()

    # 获取所有 PDF 文件
    pdf_dir = Path("data/input/")
    pdf_files = list(pdf_dir.glob("*.pdf"))

    print(f"\n找到 {len(pdf_files)} 个 PDF 文件")

    results = []
    for i, pdf_file in enumerate(pdf_files, 1):
        print(f"\n处理 [{i}/{len(pdf_files)}]: {pdf_file.name}")

        try:
            result = await parser.parse_pdf(
                pdf_path=str(pdf_file),
                output_dir="data/processed/",
                extract_images=True
            )
            results.append(result)
            print(f"  ✓ 完成")

        except Exception as e:
            print(f"  ✗ 失败: {str(e)}")

    print(f"\n成功处理 {len(results)} 个文件")


async def example_custom_config():
    """示例 4: 自定义配置"""
    print("\n" + "=" * 60)
    print("示例 4: 自定义配置")
    print("=" * 60)

    # 显示当前配置
    config = PipelineConfig.to_dict()
    print("\n当前配置:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    # 使用自定义模型目录初始化
    parser = PDFParser(model_dir=Path("custom_models/paddleocr/"))

    result = await parser.parse_pdf(
        pdf_path="data/input/sample.pdf",
        output_dir="data/processed/",
        extract_images=False
    )

    print(f"\n解析完成: {result['markdown_path']}")


async def example_error_handling():
    """示例 5: 错误处理"""
    print("\n" + "=" * 60)
    print("示例 5: 错误处理")
    print("=" * 60)

    parser = PDFParser()

    # 测试不存在的文件
    try:
        await parser.parse_pdf(
            pdf_path="nonexistent.pdf",
            output_dir="data/processed/"
        )
    except Exception as e:
        print(f"\n捕获错误: {type(e).__name__}")
        print(f"错误信息: {str(e)}")

    # 测试无效格式
    try:
        await parser.parse_pdf(
            pdf_path="data/input/sample.txt",
            output_dir="data/processed/"
        )
    except Exception as e:
        print(f"\n捕获错误: {type(e).__name__}")
        print(f"错误信息: {str(e)}")


async def example_table_extraction():
    """示例 6: 表格提取"""
    print("\n" + "=" * 60)
    print("示例 6: 表格提取")
    print("=" * 60)

    parser = PDFParser()

    result = await parser.parse_pdf(
        pdf_path="data/input/document_with_tables.pdf",
        output_dir="data/processed/",
        extract_images=False
    )

    # 读取并显示 Markdown
    with open(result['markdown_path'], 'r', encoding='utf-8') as f:
        content = f.read()

    # 查找表格
    lines = content.split('\n')
    print("\n提取的表格（前 20 行）:")
    for i, line in enumerate(lines[:20]):
        if '|' in line:
            print(line)


async def main():
    """运行所有示例"""
    print("\n" + "=" * 60)
    print("DOKE-RAG PDF 解析器示例")
    print("=" * 60)

    # 运行示例
    await example_simple_parsing()
    await example_with_images()
    await example_batch_processing()
    await example_custom_config()
    await example_error_handling()
    await example_table_extraction()

    print("\n" + "=" * 60)
    print("所有示例运行完成！")
    print("=" * 60)


if __name__ == "__main__":
    # 运行示例
    asyncio.run(main())

    # 提示
    print("\n提示：")
    print("1. 确保 PaddleOCR 已正确安装（参见 docs/PADDLEOCR_INSTALLATION.md）")
    print("2. 准备测试 PDF 文件放在 data/input/ 目录")
    print("3. 根据需要修改 .env 配置文件")
