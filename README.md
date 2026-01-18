# DOKE-RAG: Domain-Oriented, Knowledge-Enhanced Multimodal Graph RAG

**A Multi-modal, Graph-based RAG Framework for Deep Vertical Domains**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

## Overview

DOKE-RAG is a novel framework designed for deep vertical domains that integrates multimodal data processing with expert-curated knowledge graphs. It addresses two fundamental limitations of existing RAG systems: (1) reliance on purely textual corpora, and (2) lack of precision in LLM-automatically constructed knowledge graphs.

## Key Contributions

- **Hi-LVEA Algorithm**: Hierarchical LLM-Validated Entity Alignment for scalable, high-precision entity fusion
- **Hybrid Knowledge Enhancement**: Strategic fusion of LLM-generated and expert-curated knowledge graphs
- **Multimodal Integration**: Unified processing pipeline for text, formulas, tables, and figures
- **Comprehensive Evaluation**: Multi-dimensional assessment framework validated on structural mechanics domain

## Quick Start

### Installation

```bash
git clone https://github.com/BAIKEMARK/DOKE-RAG.git
cd DOKE-RAG
pip install -e .
```

### Configuration

```bash
cp .env.example .env
# Edit .env with your API keys
```

### Basic Usage

#### Core Framework (Graph RAG)

```python
import asyncio
from doke_rag.core import LightRAG, QueryParam

async def main():
    rag = LightRAG(...)

    # Insert documents
    await rag.ainsert("...")

    # Query
    result = await rag.aquery(
        "What is the Force Method?",
        param=QueryParam(mode="hybrid")
    )
    print(result)

asyncio.run(main())
```

#### Multimodal Processing

**Audio Transcription (Whisper API)**

```python
import asyncio
from doke_rag.pipeline import AudioProcessor

async def main():
    processor = AudioProcessor()

    # Transcribe audio to Markdown
    result = await processor.transcribe(
        audio_path="lecture.mp3",
        output_dir="data/processed/",
        language="zh"
    )

    print(f"Markdown: {result['markdown_path']}")

asyncio.run(main())
```

**PDF Parsing (PaddleOCR-VL)**

```python
import asyncio
from doke_rag.pipeline import PDFParser

async def main():
    parser = PDFParser()

    # Parse PDF to Markdown
    result = await parser.parse_pdf(
        pdf_path="document.pdf",
        output_dir="data/processed/",
        extract_images=True
    )

    print(f"Markdown: {result['markdown_path']}")

asyncio.run(main())
```

## Documentation

- [Architecture](docs/ARCHITECTURE.md) - System design and components
- [Usage Guide](docs/USAGE.md) - Detailed usage instructions
- [Audio Processor Guide](docs/audio_processor_readme.md) - Whisper API integration
- [PDF Parser Guide](docs/PDF_PARSER_QUICKSTART.md) - PaddleOCR-VL integration
- [PaddleOCR Installation](docs/PADDLEOCR_INSTALLATION.md) - Setup instructions

## Project Structure

```
DOKE-RAG/
├── doke_rag/                    # Core package
│   ├── core/                   # Core framework (based on LightRAG)
│   ├── pipeline/               # Data processing pipeline
│   ├── kg_construction/        # Knowledge graph construction
│   ├── retrieval/              # Query and retrieval
│   └── config/                 # Configuration management
├── evaluation/                  # Evaluation framework
├── experiments/                 # Experimental scripts
├── scripts/                     # Utility scripts (Neo4j visualization, etc.)
├── data/                        # Data directories
└── docs/                        # Documentation
```

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{doke-rag-2025,
  title={DOKE-RAG: A Multi-modal, Graph-based RAG Framework for Deep Vertical Domains},
  author={...},
  journal={...},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

DOKE-RAG is built upon [LightRAG](https://github.com/HKUDS/LightRAG) (HKUDS, MIT License), which provides the foundational graph RAG framework. We extend our sincere thanks to the LightRAG team for their excellent work.

DOKE-RAG incorporates significant modifications and enhancements:
- Hierarchical LLM-Validated Entity Alignment (Hi-LVEA) algorithm
- Domain-oriented knowledge graph construction
- Multimodal data processing pipeline
- Expert-curated knowledge fusion mechanism

The original LightRAG framework remains available at: https://github.com/HKUDS/LightRAG
