# Usage Guide

This guide provides practical instructions for using the DOKE-RAG framework.

## Installation

```bash
# Clone repository
git clone https://github.com/your-username/DOKE-RAG.git
cd DOKE-RAG

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API keys
```

## Quick Start

### Basic Query

```python
import asyncio
from doke_rag.lightrag import LightRAG, QueryParam

async def main():
    # Initialize RAG
    rag = LightRAG(
        working_dir="./data/run_data",
        llm_model_func=your_llm_func,
        embedding_func=your_embedding_func
    )

    # Query
    result = await rag.aquery(
        "What is the Force Method in structural mechanics?",
        param=QueryParam(mode="hybrid")
    )

    print(result)

asyncio.run(main())
```

## Common Workflows

### 1. Insert Documents

```python
# From text file
with open("document.txt", "r") as f:
    await rag.ainsert(f.read())

# From pre-processed Markdown
await rag.ainsert(markdown_content)
```

### 2. Insert Expert Graph

```python
import json

# Load expert graph
with open("data/expert_graphs/expert_graph.json", "r") as f:
    expert_graph = json.load(f)

# Insert into knowledge base
await rag.ainsert_custom_kg(expert_graph)
```

### 3. Entity Deduplication

```python
# Run Hi-LVEA algorithm
from doke_rag.kg_construction.entity_alignment import OptimizedEntityDeduplicator

deduplicator = OptimizedEntityDeduplicator(
    llm_model_func=your_llm_func,
    embedding_func=your_embedding_func
)

# Load entities from graph storage
entities = await rag.chunk_entity_relation_graph.get_all_entities()

# Run deduplication
merge_groups = await deduplicator.deduplicate_entities(entities)

# Apply merges
for group in merge_groups:
    await rag.amerge_entities(
        source_entities=group["to_merge"],
        target_entity=group["representative"]
    )
```

### 4. Run Evaluation

```bash
# Navigate to evaluation directory
cd evaluation

# Run pairwise evaluation
python evaluate.py --benchmark naive_result.json --challenger doke_result.json

# Run parameter grid search
cd ../experiments
python run_batch_evaluation.py --config config/eval_config.yaml
```

## Query Modes

### Local Mode (Entity-Centric)
Retrieves fine-grained, entity-focused information.

```python
result = await rag.aquery(
    "Define flexibility coefficient",
    param=QueryParam(mode="local", only_need_context=False)
)
```

### Global Mode (Relationship-Centric)
Retrieves community-level, relationship-focused information.

```python
result = await rag.aquery(
    "Compare Force Method and Displacement Method",
    param=QueryParam(mode="global")
)
```

### Hybrid Mode
Combines local and global retrieval.

```python
result = await rag.aquery(
    "Explain the applications of Force Method",
    param=QueryParam(mode="hybrid")
)
```

## Configuration

### LLM Provider Selection

**OpenAI**:
```python
from doke_rag.lightrag.llm.openai import openai_complete_if_cache, openai_embedding

llm_func = lambda prompt, **kwargs: openai_complete_if_cache(
    "gpt-4", prompt, api_key="your-key", **kwargs
)
```

**Ollama (Local)**:
```python
from doke_rag.lightrag.llm.ollama import ollama_model_complete, ollama_embed

llm_func = lambda prompt, **kwargs: ollama_model_complete(
    prompt, model="llama2", host="http://localhost:11434", **kwargs
)
embed_func = lambda texts: ollama_embed(
    texts, embed_model="nomic-embed-text", host="http://localhost:11434"
)
```

**Aliyun (Tongyi/DeepSeek)**:
```python
from openai import AsyncOpenAI

client = AsyncOpenAI(
    api_key="your-aliyun-key",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

llm_func = lambda prompt, **kwargs: client.chat.completions.create(
    model="deepseek-r1", messages=[{"role": "user", "content": prompt}], **kwargs
)
```

### Storage Backend Selection

**Neo4j**:
```python
from doke_rag.lightrag.kg import Neo4JStorage

rag = LightRAG(
    kg_storage=Neo4JStorage(
        namespace="doke_rag",
        global_config=neo4j_config
    )
)
```

**NetworkX (Default)**:
```python
rag = LightRAG(
    working_dir="./data/run_data"  # Uses NetworkX by default
)
```

## Expert Graph Format

Expert graphs should follow this JSON structure:

```json
{
  "chunks": [
    {
      "content": "Markdown content with $$LaTeX$$ formulas and ![](image_urls)",
      "source_id": "chapter6-1"
    }
  ],
  "entities": [
    {
      "entity_name": "Force Method",
      "entity_type": "method",
      "description": "A method for analyzing statically indeterminate structures...",
      "source_id": "chapter6-1"
    }
  ],
  "relationships": [
    {
      "src_id": "Force Method",
      "tgt_id": "Flexibility Coefficient",
      "description": "uses",
      "keywords": "employs, utilizes",
      "source_id": "chapter6-1"
    }
  ]
}
```

## Evaluation Data Format

Evaluation results should be stored as JSON:

```json
{
  "query_id": "Q001",
  "query": "Compare flexibility and stiffness coefficients",
  "answer": "Flexibility coefficients measure displacement...",
  "retrieved_context": ["...", "..."],
  "metadata": {
    "mode": "hybrid",
    "top_k": 40,
    "cosine_threshold": 0.2
  }
}
```

## Troubleshooting

### Issue: Low retrieval quality
**Solution**: Adjust `top_k` and `cosine_threshold` parameters. See Section 4.2 of the paper for optimal configurations.

### Issue: Duplicate entities in results
**Solution**: Run entity deduplication using Hi-LVEA algorithm before querying.

### Issue: Out of memory errors
**Solution**:
1. Reduce `chunk_token_size` (default: 1200)
2. Reduce `llm_model_max_token_size` (default: 32768)
3. Use streaming mode for large documents

### Issue: Slow insertion speed
**Solution**:
1. Increase `llm_model_max_async` (default: 4)
2. Use `insert_batch_size` parameter (default: 50)
3. Consider using faster embedding models

## Best Practices

1. **Preprocessing**: Convert all inputs to Markdown with LaTeX formulas and image URLs before insertion.
2. **Domain Adaptation**: Always run entity type generation on a sample of your domain documents first.
3. **Expert Graph Quality**: Ensure expert graphs have high-quality descriptions to serve as canonical entities during fusion.
4. **Evaluation**: Use the provided evaluation framework to compare different configurations systematically.
5. **Parameter Tuning**: Refer to the paper's RQ2 section for globally optimal configurations.

## Additional Resources

- See `experiments/` directory for complete pipeline scripts
- See `evaluation/` directory for evaluation examples
- See `docs/ARCHITECTURE.md` for system design details
