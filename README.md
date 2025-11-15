# EDGE - Embedded Dynamic Graph Engine

EDGE is a minimal semantic memory system that processes books or documents into structured graph representations. It parses text into sentences, generates OpenAI embeddings, computes contextual statistics, and builds compact NumPy-based graph edges for efficient storage and retrieval. The system emphasizes low maintenance, using OpenAI embeddings and NumPy edge arrays instead of complex indexing infrastructure.

## Overview

EDGE transforms a book (or large document) into a semantic memory graph. Each sentence becomes a node with embeddings and statistics, connected by edges encoding adjacency, question/definition context, and chapter co-occurrence. The output is stored in a compact format suitable for downstream analysis, retrieval, or training expansion policies.

The current implementation focuses on the **memory_bacon** workflow: parse → embed → stats → edges, producing structured data artifacts ready for graph-based reasoning or machine learning pipelines.

## Quickstart

### Installation

```bash
pip install -r requirements.txt
```

### Prepare Input

Create a plain text file with:
- One sentence per line
- Blank lines indicate chapter boundaries

Example (`3essay.txt`):
```
The true end of knowledge is action.

Knowledge is power.

What is the purpose of learning?
...
```

### Run Pipeline

**New unified way (recommended):**

```bash
# Single command processes everything
python process_text.py 3essay.txt data
```

Or in Python:

```python
from storage import process_text_to_memory

with open("3essay.txt", 'r', encoding='utf-8') as f:
    text = f.read()

stats = process_text_to_memory(
    text=text,
    source_name="essay",
    data_dir="data",
)
```

**Legacy three-step way (still supported):**

```bash
# Step 1: Parse text into structured sentences
python parse_essay.py

# Step 2: Generate OpenAI embeddings
python generate_embeddings.py

# Step 3: Build graph edges and convert to NumPy format
python convert_edges_to_numpy.py
```

### Output Structure

All storage files are written to `data/` directory:

```
data/
├── memory.db                # SQLite database (nodes, edges, usage)
├── sentences.jsonl          # Sentence text (append-only)
├── embeddings.npy          # Embedding vectors (N, D) float32
└── graph/
    ├── edge_index.npy       # Edge indices (2, E) int32
    ├── edge_weight.npy      # Edge weights (E,) float32
    └── edge_type.npy        # Edge type IDs (E,) uint8
```

**Storage Architecture:**
- **SQLite** (`memory.db`): Structured data (nodes_context, neighbors, nodes_usage)
- **JSONL** (`sentences.jsonl`): Append-only sentence text
- **NumPy** (`embeddings.npy`, `graph/*.npy`): Efficient binary arrays

**Note:** Old `memory_bacon/` structure is deprecated. Use `data/` directory instead.

## Data Artifacts

| File | Format | Description |
|------|--------|-------------|
| `memory.db` | SQLite | Database with nodes_context, neighbors, nodes_usage tables |
| `sentences.jsonl` | JSONL | One JSON per line: `id`, `chapter`, `position`, `text` |
| `embeddings.npy` | NumPy | `(N, D)` float32 array, where N=sentences, D=embedding dim |
| `graph/edge_index.npy` | NumPy | `(2, E)` int32 array: `[src_indices, dst_indices]` |
| `graph/edge_weight.npy` | NumPy | `(E,)` float32 array: edge weights |
| `graph/edge_type.npy` | NumPy | `(E,)` uint8 array: edge type IDs (0=adjacent, 2=question_context, 3=definition_context) |

**Deprecated files** (no longer generated):
- `id_map.json` - Replaced by database node IDs
- `statistics.jsonl` - Replaced by database nodes_context table
- `embedding_meta.json` - Metadata stored in database
- `summary.txt` - Can be queried from database

## Design Notes

### NumPy Edge Format

Edges are stored as three aligned NumPy arrays (`edge_index`, `edge_weight`, `edge_type`) instead of NetworkX graphs or JSONL. This format:
- **Efficient**: Direct memory mapping, no parsing overhead
- **Compatible**: Works with PyTorch Geometric, DGL, or custom graph libraries
- **Scalable**: Handles millions of edges with minimal memory footprint
- **Type-safe**: Explicit edge types enable filtered traversal

Example usage:
```python
import numpy as np

edge_index = np.load("memory_bacon/graph/edge_index.npy")  # (2, E)
edge_weight = np.load("memory_bacon/graph/edge_weight.npy")  # (E,)
edge_type = np.load("memory_bacon/graph/edge_type.npy")  # (E,)

# Filter edges by type
adjacent_edges = edge_index[:, edge_type == 0]
```

### Extending Edges

To add new edge types (e.g., `semantic_knn`, `keyword_cooccur`):

1. Modify `convert_edges_to_numpy.py`:
   - Add edge type mapping: `edge_type_map["semantic_knn"] = 4`
   - Implement edge construction logic
   - Append to `edge_list` with appropriate weight

2. Edge types are encoded as uint8, supporting up to 255 types.

### Extending Statistics

To add new sentence-level features:

1. Modify `parse_essay.py` or create a post-processing script
2. Add feature computation in the statistics generation loop
3. Append to each sentence's statistics dictionary
4. Update `statistics.jsonl` (overwrite or append)

### Switching Embedding Models

**Current**: OpenAI `text-embedding-3-small` (via `generate_embeddings.py`)

**To use MiniLM** (local, no API):
1. Modify `generate_embeddings.py` to use `sentence-transformers/all-MiniLM-L6-v2`
2. Update batch processing logic (MiniLM supports larger batches)
3. Adjust embedding dimension in metadata (384 vs 1536)

**To use other models**:
- Replace the embedding call in `generate_embeddings.py`
- Ensure output is `(N, D)` float32 NumPy array
- Update `embedding_meta.json` accordingly

## Roadmap

- **XGBoost Expansion Policy**: Train a policy model to predict which edges to add/remove based on query patterns and relevance feedback
- **Optional Re-ranker**: Small MLP that reranks retrieved sentences using graph structure and contextual features
- **Multi-book Merge**: Combine multiple books into a unified graph with cross-book semantic edges
- **FAISS IVF-PQ Index**: Add optional FAISS indexing for fast approximate nearest neighbor search on embeddings

## Project Structure

```
EDGE/
├── core/
│   └── config.py                    # Configuration constants
├── memory_bacon/
│   ├── core/                        # Core data files
│   ├── graph/                       # Graph edge arrays
│   └── meta/                        # Metadata files
├── parse_essay.py                   # Text parsing script
├── generate_embeddings.py           # Embedding generation script
├── convert_edges_to_numpy.py       # Edge construction script
├── main.py                          # Deprecated CLI
├── memory_bacon_exploration.ipynb  # Data exploration notebook
├── test.ipynb                       # Test notebook
├── 3essay.txt                       # Sample input
├── README.md                        # This file
└── requirements.txt                 # Python dependencies
```

## License

MIT License
