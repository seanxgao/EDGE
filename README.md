# EDGE - Embedded Dynamic Graph Engine

EDGE is a semantic memory and retrieval engine designed as a cognitive layer for intelligent memory storage, retrieval, and learning. Inspired by industrial-scale recall and ranking systems, EDGE repurposes these techniques to build trainable, extensible semantic memory infrastructure.

## Overview

At its current stage, EDGE treats an entire book (or large document) as a self-contained Memory Node. Each book is decomposed into sentences, embedded via MiniLM or similar transformer models, indexed using FAISS IVF-PQ, and linked by a lightweight Knowledge Graph that encodes adjacency, paragraph structure, and semantic similarity.

These components together form a local memory module that answers semantic queries not by keyword matching, but by reasoning across meaning and context.

The long-term goal of EDGE is to generalize this architecture into a trainable, extensible semantic memory infrastructure that can:
- Absorb large heterogeneous corpora (books, papers, conversational histories)
- Build interconnected memory graphs that evolve through training
- Support adaptive recall, reranking, and semantic routing
- Serve as a structured, queryable long-term memory backend for intelligent systems or LLMs

In essence, EDGE aims to simulate how an intelligent agent might remember, relate, and re-organize knowledge over time. It combines embedding-based recall, graph-structured memory representation, and trainable ranking modules to achieve this goal.

## Architecture

EDGE follows a modular pipeline architecture:

```
Text Input → Ingestion → Embedding → Indexing → Graph Construction → Query → Ranking
```

### Data Flow

1. **Ingestion**: Parses text into sentence-level chunks with metadata (chapter, paragraph, position)
2. **Embedding**: Encodes sentences via transformer models (default: sentence-transformers/all-MiniLM-L6-v2)
3. **Indexing**: Builds FAISS IVF-PQ index for scalable approximate nearest neighbor search
4. **Graph Construction**: Creates knowledge graph with three edge types:
   - Sequential edges: Adjacent sentences within paragraphs
   - Paragraph edges: Sentences within the same paragraph
   - Semantic edges: High-similarity sentence pairs (threshold-based)
5. **Query Pipeline**: Three-stage hybrid retrieval:
   - Stage 1: Vector retrieval via FAISS ANN search
   - Stage 2: Graph expansion to collect contextual neighbors
   - Stage 3: Fusion ranking combining vector similarity, graph structure, and context coherence
6. **Reranking**: (Planned) Trainable booster (XGBoost) that fuses structural and semantic weights

### Storage Components

- **Vector Store**: Memory-mapped numpy arrays for efficient embedding storage
- **Sentence Store**: JSONL format for sentence metadata and text
- **Graph Store**: NetworkX graph persisted as JSON with schema versioning
- **FAISS Index**: IVF-PQ index for fast vector similarity search

## Components

### Ingestion Pipeline (`pipeline/ingestion.py`)

The `BookIngestionPipeline` class handles end-to-end document processing:

- Parses text with configurable chapter and paragraph patterns
- Generates embeddings in batches for efficiency
- Stores sentences and vectors with row mapping for consistency
- Constructs graph edges (sequential, paragraph, semantic)
- Builds and trains FAISS index

### Embedding (`core/embedder_minilm.py`)

Uses sentence-transformers for local embedding generation:
- Model: `all-MiniLM-L6-v2` (384 dimensions)
- Batch processing support
- No external API dependencies

### Indexing (`core/faiss_index.py`)

FAISS wrapper supporting multiple index types:
- Flat index: Exact search, suitable for small corpora
- IVF-PQ: Approximate search with product quantization, scalable to millions of vectors

### Graph Construction (`graph/graph_builder.py`)

Builds three types of edges:
- **Sequential**: Links consecutive sentences (temporal adjacency)
- **Paragraph**: Links sentences within the same paragraph (structural context)
- **Semantic**: Links sentences above similarity threshold (semantic relationships)

### Query Pipeline (`pipeline/query.py`)

Three-stage retrieval system:

**Stage 1: Vector Retrieval**
- Generates query embedding
- Searches FAISS index for top-k candidates
- Returns candidate IDs and similarity scores

**Stage 2: Graph Expansion**
- Expands candidates by following graph edges
- Collects sequential, paragraph, and semantic neighbors
- Builds node features (degree, neighbor types, context)

**Stage 3: Fusion Ranking**
- Combines multiple signals:
  - Vector similarity (from Stage 1)
  - Graph degree (normalized node connectivity)
  - Context coherence (neighbor overlap with candidates)
- Returns top-k ranked results with metadata

### Reranking (Planned)

Future integration of trainable reranking:
- XGBoost-based booster that learns optimal feature weights
- Training on query-result relevance labels
- Adaptive ranking that improves with usage

## AWS Integration

Planned cloud infrastructure integration:

### Amazon S3
- Store embeddings, graph data, and index snapshots
- Versioned storage for model checkpoints
- Efficient data loading via memory-mapped access patterns

### EC2 / SageMaker
- Run embedding and indexing pipelines at scale
- Distributed processing for large corpora
- Model training and fine-tuning workflows

### CloudWatch
- Log and monitor query performance
- Track embedding generation latency
- Alert on system health metrics

### ECS / Fargate
- Containerized deployment for API server
- Auto-scaling based on query load
- Multi-region deployment support

## Roadmap

### Multi-Book Memory Graph
- Extend from single-book to multi-book memory graphs
- Cross-book semantic relationships
- Hierarchical memory organization (book → chapter → paragraph → sentence)

### LLM Integration
- Serve as long-term memory backend for LLMs
- Context injection for retrieval-augmented generation
- Adaptive memory updates based on LLM feedback

### Adaptive Learning
- Trainable reranking with XGBoost
- Reinforcement learning for query routing
- Continuous graph structure optimization

### Scalability
- Distributed graph storage and querying
- Incremental indexing for streaming data
- Multi-tenant memory isolation

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Basic Usage

```python
from pipeline.ingestion import BookIngestionPipeline
from pipeline.query import QueryPipeline
from core.faiss_index import FaissIndex
from storage.vector_store import VectorStore
from storage.sentence_store import SentenceStore
from graph.graph_store import GraphStore

# Ingest a document
pipeline = BookIngestionPipeline(
    base_path="data/book1",
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    index_type="ivf-pq"
)

with open("3essay.txt", "r", encoding="utf-8") as f:
    text = f.read()

pipeline.ingest_book(
    book_text=text,
    batch_size=16,
    similarity_threshold=0.6
)

# Load components for querying
embedding_dim = pipeline.embedder.get_embedding_dim()
faiss_index = FaissIndex.load("data/book1/faiss.index")
vector_store = VectorStore("data/book1", embedding_dim)
sentence_store = SentenceStore("data/book1/sentences.jsonl")
graph_store = GraphStore("data/book1/graph.json")

# Create query pipeline
query_pipeline = QueryPipeline(
    faiss_index=faiss_index,
    vector_store=vector_store,
    sentence_store=sentence_store,
    graph_store=graph_store
)

# Query
results = query_pipeline.query(
    query_text="What is the main idea?",
    top_k=10,
    initial_k=50,
    expand_sequential=True,
    expand_paragraph=True,
    expand_semantic=True
)

for result in results:
    print(f"Score: {result['score']:.3f}")
    print(f"Text: {result['text']}")
    print(f"Chapter: {result.get('chapter', 'N/A')}\n")
```

### REST API

Start the API server:

```bash
export EDGE_DATA_PATH=data/book1
uvicorn api.server:app --host 0.0.0.0 --port 8000
```

Query endpoint:

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the main idea?",
    "top_k": 10,
    "initial_k": 50
  }'
```

### Testing

All tests are in `test.ipynb`. The notebook includes:
- Dependency verification
- Document ingestion from `3essay.txt`
- Query pipeline execution
- API endpoint testing

Run the notebook:

```bash
jupyter notebook test.ipynb
```

## Project Structure

```
EDGE/
├── api/
│   └── server.py              # FastAPI REST API server
├── core/
│   ├── config.py              # Configuration constants
│   ├── embedder_minilm.py     # Sentence-transformers embedder
│   ├── embedder.py            # Legacy OpenAI embedder
│   ├── faiss_index.py         # FAISS index wrapper
│   └── retriever.py           # Legacy cosine similarity
├── graph/
│   ├── graph_builder.py       # Edge construction logic
│   ├── graph_memory.py        # Legacy graph memory API
│   ├── graph_store.py         # NetworkX graph persistence
│   └── spectral.py            # Legacy spectral functions
├── pipeline/
│   ├── ingestion.py           # Document ingestion pipeline
│   └── query.py               # Three-stage query pipeline
├── storage/
│   ├── sentence_store.py      # Sentence metadata (JSONL)
│   └── vector_store.py        # Embedding storage (numpy mmap)
├── main.py                    # Legacy CLI
├── test.ipynb                 # Comprehensive tests
├── 3essay.txt                 # Sample document
├── README.md                  # This file
├── API.md                     # API reference
└── requirements.txt           # Python dependencies
```

## Vision

EDGE aspires to become a general-purpose semantic memory framework that can ingest, organize, and reason over vast textual memories, from individual books to lifelong AI conversations.

The project extends beyond simple document search. It aims to model how intelligent systems might:
- **Remember**: Persist and organize knowledge in structured, queryable form
- **Relate**: Build semantic connections that capture meaning and context
- **Re-organize**: Adapt memory structure through learning and feedback

By combining embedding-based recall, graph-structured representation, and trainable ranking, EDGE provides a foundation for building systems that can maintain and utilize long-term semantic memory effectively.

## API Reference

See [API.md](API.md) for complete interface documentation.

## License

MIT License
