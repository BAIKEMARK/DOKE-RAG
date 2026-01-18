"""
音频处理器使用示例

演示如何使用 AudioProcessor 进行音频转录。
"""

import asyncio
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from doke_rag.pipeline.audio_processor import AudioProcessor
from doke_rag.config.pipeline_config import PipelineConfig


async def main():
    """主函数"""
    print("=" * 60)
    print("DOKE-RAG 音频处理器示例")
    print("=" * 60)
    print()

    # 检查配置
    try:
        PipelineConfig.validate()
        print("✓ 配置验证通过")
    except ValueError as e:
        print(f"✗ 配置错误: {e}")
        print("\n请在 .env 文件中设置 OPENAI_API_KEY")
        return

    # 创建音频处理器
    print("\n1. 初始化音频处理器...")
    processor = AudioProcessor()
    print(f"✓ 处理器初始化完成（模型: {processor.model}）")

    # 示例 1: 单个音频文件转录
    print("\n2. 单个音频文件转录示例")
    print("-" * 60)

    # 替换为实际的音频文件路径
    audio_path = "path/to/your/audio.mp3"
    output_dir = "data/processed/audio"

    # 检查文件是否存在（仅用于演示）
    if Path(audio_path).exists():
        try:
            result = await processor.transcribe(
                audio_path=audio_path,
                output_dir=output_dir,
                language="auto"  # 可以指定 "zh", "en", 或 "auto"
            )

            print(f"✓ 转录成功!")
            print(f"  - Markdown 文件: {result['markdown_path']}")
            print(f"  - 元数据文件: {result['metadata_path']}")
            print(f"  - 时长: {result['duration']:.2f} 秒")
            print(f"  - 语言: {result['language']}")

        except Exception as e:
            print(f"✗ 转录失败: {e}")
    else:
        print(f"⚠ 音频文件不存在: {audio_path}")
        print("  请将音频文件路径替换为实际路径")

    # 示例 2: 批量转录
    print("\n3. 批量音频转录示例")
    print("-" * 60)

    audio_files = [
        "path/to/audio1.mp3",
        "path/to/audio2.mp3",
        "path/to/audio3.mp3"
    ]

    # 检查文件
    existing_files = [f for f in audio_files if Path(f).exists()]

    if existing_files:
        try:
            print(f"开始批量转录 {len(existing_files)} 个文件...")
            results = await processor.batch_transcribe(
                audio_files=existing_files,
                output_dir=output_dir,
                language="auto",
                max_concurrent=3  # 最多 3 个并发
            )

            successful = sum(1 for r in results if isinstance(r, dict) and "markdown_path" in r)
            print(f"✓ 批量转录完成: 成功 {successful}/{len(existing_files)}")

        except Exception as e:
            print(f"✗ 批量转录失败: {e}")
    else:
        print("⚠ 没有找到有效的音频文件")
        print("  请将音频文件路径替换为实际路径")

    # 示例 3: 使用便捷函数
    print("\n4. 使用便捷函数示例")
    print("-" * 60)

    from doke_rag.pipeline import transcribe_audio

    print("""
from doke_rag.pipeline import transcribe_audio

result = await transcribe_audio(
    audio_path="lecture.mp3",
    output_dir="output/",
    language="zh"
)

print(result["markdown_path"])
    """)

    # 示例 4: 指定语言和提示
    print("\n5. 高级选项示例")
    print("-" * 60)

    print("""
# 指定语言提高准确率
result = await processor.transcribe(
    audio_path="chinese_lecture.mp3",
    output_dir="output/",
    language="zh",  # 明确指定中文
    prompt="这是一段关于结构力学的课程",  # 提供上下文
    temperature=0.0  # 使用更确定的转录
)

# 容错模式
processor_tolerant = AudioProcessor(strict_mode=False)
result = await processor_tolerant.transcribe(...)
# 即使失败也不会抛出异常，而是返回错误信息
    """)

    print("\n" + "=" * 60)
    print("示例完成!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
