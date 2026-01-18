# System Architecture

This document describes the high-level architecture of the DOKE-RAG framework.

## Framework Overview

DOKE-RAG is designed as a modular system with three main stages:

```
Input Data → Knowledge Graph Construction → Hybrid Retrieval
```

## Component Architecture

### 1. Data Processing Pipeline
**Location**: `doke_rag/pipeline/`

**Purpose**: Transform raw domain documents into structured knowledge chunks.

**Components**:
- **Chunking**: Token-based text segmentation with sliding window overlap
- **Text Refinement**: Sentence simplification, pronoun resolution, knowledge point extraction

**Input**: Raw text/Markdown documents
**Output**: Structured chunks with preserved formula/figure references

---

### 2. Knowledge Graph Construction
**Location**: `doke_rag/kg_construction/`

**Purpose**: Build hybrid knowledge graph combining LLM-generated and expert-curated knowledge.

**Components**:

#### 2.1 Domain Adaptation (`entity_types.py`)
- Samples representative documents using embedding-based centroid sampling
- Generates domain-specific entity types and prompts
- Optimizes LLM extraction for vertical domains

#### 2.2 Baseline Graph Generation (`graph_builder.py`)
- Extracts entities and relationships from documents using adapted prompts
- Built upon LightRAG with custom modifications
- Supports incremental graph updates

#### 2.3 Expert Graph Management (`graph_builder.py`)
- `ainsert_custom_kg()`: Inserts expert-curated graphs in JSON format
- `aedit_entity()`: Edits entity descriptions
- `amerge_entities()`: Merges duplicate entities with custom strategies

#### 2.4 Entity Alignment (`entity_alignment.py`)
**Hi-LVEA Algorithm** (5 steps):
1. **DBSCAN Clustering**: Coarse-grained semantic grouping
2. **Similarity Scoring**: Fine-grained Levenshtein + cosine similarity
3. **Connected Components**: Pre-groups candidate merge pairs
4. **LLM Validation**: Domain expert validation of entity pairs
5. **Canonical Merging**: Selects representative entity and merges

**Output**: Fused hybrid graph `G_Hybrid` with expert knowledge as backbone

---

### 3. Retrieval System
**Location**: `doke_rag/lightrag/operate.py`

**Purpose**: Query knowledge graph and retrieve relevant context.

**Query Modes**:
- **Local**: Entity-centric retrieval (fine-grained)
- **Global**: Relationship-centric retrieval (community-level)
- **Hybrid**: Combines local and global
- **Naive**: Vector-only retrieval (baseline)

**Retrieval Pipeline**:
1. **Keyword Decomposition**: Extract high-level and low-level keywords
2. **Vector Search**: Embedding-based matching
3. **Graph Traversal**: K-hop expansion for context gathering
4. **Context Assembly**: Combine retrieved entities, relationships, and text blocks

---

### 4. Storage Layer
**Location**: `doke_rag/lightrag/kg/`

**Supported Backends**:
- **Graph Storage**: NetworkX, Neo4j, PostgreSQL (AGE), MongoDB, Redis, Oracle, etc.
- **Vector Storage**: Chroma, Qdrant, Milvus, FAISS, etc.
- **Document Storage**: JSON-based status tracking

**Design**: Pluggable architecture supporting multiple storage combinations.

---

### 5. Evaluation Framework
**Location**: `evaluation/`

**Purpose**: Multi-dimensional assessment of RAG systems.

**Evaluation Pipeline**:
1. **Pairwise Comparison**: Compare benchmark vs. challenger answers
2. **LLM Judge**: DeepSeek-r1 as judge model
3. **Multi-dimensional Scoring**: Comprehensiveness, Diversity, Empowerment, Overall
4. **Aggregation**: Calculate win rates across questions and configurations

**Supported Experiments**:
- Ablation studies (DOKE-RAG variants)
- Parameter sensitivity analysis (top_k, cosine_threshold)
- SOTA comparison (LightRAG, GraphRAG, Naive RAG)

---

## Data Flow

```
┌─────────────────────┐
│   Input Documents   │
│ (Text/Markdown)     │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Pipeline: Chunking │
│  & Refinement       │
└──────────┬──────────┘
           │
           ├──────────────┐
           │              │
           ▼              ▼
┌──────────────────┐  ┌──────────────────┐
│ LLM Graph        │  │ Expert Graph     │
│ (G_LLM)          │  │ (G_Expert)       │
│ Auto-generated   │  │ Manually curated │
└────────┬─────────┘  └────────┬─────────┘
         │                     │
         └──────────┬──────────┘
                    │
                    ▼
         ┌─────────────────────┐
         │   Hi-LVEA Fusion    │
         │   (Alignment)       │
         └──────────┬──────────┘
                    │
                    ▼
         ┌─────────────────────┐
         │  Hybrid Graph       │
         │  (G_Hybrid)         │
         └──────────┬──────────┘
                    │
                    ▼
         ┌─────────────────────┐
         │  Query & Retrieval  │
         │  (Local/Global/     │
         │   Hybrid)           │
         └─────────────────────┘
```

---

## Key Design Decisions

### 1. Modular Pipeline
Each stage (processing, construction, retrieval) is independent and can be used separately.

### 2. Pluggable Storage
Support for 16+ graph databases and vector stores allows flexible deployment.

**Visualization**: Utility scripts for Neo4j graph visualization are available in `scripts/`.

### 3. Format Compatibility
Expert graphs use the same JSON format as LLM-generated graphs, enabling seamless fusion.

### 4. Multi-LLM Support
OpenAI, Ollama, Aliyun (Tongyi), DeepSeek, HuggingFace, and Azure OpenAI supported.

### 5. Async-First Design
All core operations are async for performance and scalability.

---

## Technology Stack

- **Language**: Python 3.9+
- **Graph Framework**: Modified LightRAG
- **LLM Providers**: OpenAI, Ollama, Aliyun, DeepSeek
- **Embedding**: nomic-embed-text, OpenAI embeddings
- **Graph DB**: NetworkX (default), Neo4j, PostgreSQL, etc.
- **Vector DB**: Chroma, Qdrant, Milvus, FAISS, etc.
