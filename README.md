# Project EDGE - Embedded Dynamic Graph Engine

**EDGE** (Embedded Dynamic Graph Engine) is the memory organization layer in a hierarchical cognitive architecture. It acts as the middle brain connecting high-level reasoning systems to the spectral representation foundation provided by SCOPE.

## Overview

Most memory systems store data as linear sequences or flat embedding indexes. EDGE instead represents memory as a self-evolving graph, where each node corresponds to a stored experience and edges encode semantic or temporal relationships.

As new memories arrive, EDGE dynamically connects them to the most relevant existing nodes, forming a semantic topology that reflects meaning, context, and recurrence. The result is a living structure that grows, decays, and reorganizes itself.

## Hierarchical Architecture

Project EDGE is part of a three-layer hierarchical memory architecture:

```
SCOPE → reveals structure (representation & clustering)
   ↓
EDGE → maintains memory (organization & dynamics)
   ↓
Application → utilizes memory (reasoning & generation)
```

### Layer 1: SCOPE
- Defines the spectral and geometric organization of the embedding space
- Performs GPU-accelerated spectral clustering to identify stable semantic clusters
- Serves as the structural foundation for all higher layers

### Layer 2: EDGE (This Project)
- Acts as the Embedded Dynamic Graph Engine for memory
- Builds and maintains a self-evolving graph of memories
- Uses topology provided by SCOPE to organize semantic relationships
- Each memory is a node; edges represent semantic or temporal relationships
- Handles dynamic updates: addition, decay, and reinforcement
- Exposes APIs for retrieval and graph traversal to upper systems

### Layer 3: Application Layer
- Represents cognitive or agent-based systems consuming EDGE as memory backend
- Relies on EDGE for contextual recall, temporal reasoning, and structural persistence
- Does not access embeddings directly; all memory interaction mediated through EDGE

Together they form a hierarchical memory architecture where meaning, structure, and recall are progressively abstracted.

## Quick Start

### 1. Set API Key
Place your OpenAI API key in a file (default: `H:\API\openai_api.txt`).  
To change the path, modify `core/config.py`.

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Use CLI
```bash
# Add a memory
python main.py --add "Graph memory is cool"

# Query memories
python main.py --query "What is graph memory?"

# Show statistics
python main.py --stats

# Interactive mode
python main.py
```

## Project Structure

```
EDGE/
├── main.py                 # CLI interface
├── test.ipynb              # Jupyter notebook tests
├── core/                   # Core utilities
│   ├── config.py          # Configuration constants
│   ├── embedder.py        # Embedding interface (OpenAI/SCOPE)
│   └── retriever.py       # Cosine similarity search
├── graph/                  # Graph memory system
│   ├── graph_memory.py    # High-level API
│   ├── graph_store.py     # NetworkX + JSON persistence
│   └── spectral.py        # SCOPE integration hooks
├── README.md              # Project overview
├── API.md                 # API reference
├── requirements.txt       # Dependencies

## Key Features

### Implemented Features
- Graph Memory Storage: NetworkX-based graph with nodes (memories) and weighted edges (similarity scores)
- Semantic Organization: Automatic connection of new memories to k most similar existing memories
- Dynamic Graph: Self-evolving structure that adapts as memories are added
- Embedding Integration: OpenAI text-embedding-3-small for semantic representation (can be replaced with SCOPE)
- Similarity Retrieval: Cosine similarity search for memory retrieval
- JSON Persistence: Lightweight file-based storage with schema versioning
- API Interface: Clean API for application layer integration
- Custom Embedding Provider: Support for SCOPE embeddings via `set_embedding_provider()`

### Placeholder Features (SCOPE Integration Ready)
- Spectral Clustering: Interface ready for SCOPE's spectral clustering (`spectral_clustering_from_scope()`)
- Optimal Transport: Interface ready for SCOPE's optimal transport updates (`optimal_transport_from_scope()`)
- Laplacian Smoothing: Interface ready for SCOPE's Laplacian smoothing (`laplacian_smoothing_from_scope()`)

## API

See [API.md](API.md) for complete interface reference.

### Quick Example
```python
from graph.graph_memory import GraphMemory

memory = GraphMemory()

# Add memory
node_id = memory.add_memory("User likes coffee")

# Retrieve memories
results = memory.retrieve_memories("What does user like?", k=3)

# Get statistics
stats = memory.show_stats()
```

## CLI Commands

- `python main.py --add "text"` - Add memory
- `python main.py --query "text"` - Query memories
- `python main.py --stats` - Show graph statistics
- `python main.py --verbose` - Enable timing logs
- `python main.py --file path/to/file.json` - Specify memory file
- `python main.py` - Interactive mode

## Requirements

- Python 3.7+
- OpenAI API key (stored in file, see config)
- Network connection (for OpenAI API calls)
- Dependencies: openai, numpy, networkx (see `requirements.txt`)

## Testing

Run tests with the Jupyter notebook:

```bash
# Open and run the notebook
jupyter notebook test.ipynb
```

The test notebook includes:
- Basic operations (add and retrieve memories)
- Store and load from file (persistence verification)
- Store 100 memories and verify graph structure
- Graph edge information and Laplacian matrix computation

All tests use deterministic mock embeddings (no API calls). Tests are reproducible with fixed seeds and validate:
- Graph structure (nodes, edges, density)
- Embedding validity (no NaN or Inf)
- Persistence and retrieval correctness

## Integration with SCOPE

EDGE provides hooks for SCOPE integration:

- Embedding Provider: Use `set_embedding_provider()` to replace OpenAI with SCOPE embeddings
- Spectral Clustering: `spectral_clustering_from_scope()` accepts SCOPE cluster assignments
- Optimal Transport: `optimal_transport_from_scope()` accepts SCOPE updated embeddings
- Graph Smoothing: `laplacian_smoothing_from_scope()` accepts SCOPE smoothed graph data

Current Status: Interfaces are ready but require SCOPE implementation to provide actual results.

Enable SCOPE integration:
```python
from graph.graph_memory import GraphMemory
from core.embedder import set_embedding_provider

# Set SCOPE embedding provider
set_embedding_provider(scope_embedding_function)

# Initialize with SCOPE integration enabled
memory = GraphMemory(scope_integration=True)
```

The `graph/spectral.py` module provides placeholder hooks that accept SCOPE results but do not call SCOPE API directly (awaiting SCOPE implementation).

## Design Philosophy

- Minimal Implementation: Focus on core memory organization and retrieval
- Layer Separation: Applications interact only through EDGE API, not embeddings
- Extensibility: Placeholder hooks for SCOPE integration (spectral clustering, optimal transport)
- Deterministic: Numerical validation and timing checkpoints ensure reproducibility
- Plain Dependencies: NumPy and NetworkX for scientific computing

## Files

- `main.py` - Command-line interface
- `test.ipynb` - Jupyter notebook with interactive tests (deterministic, no API calls)
- `core/` - Core modules (embedder, retriever, config)
- `graph/` - Graph memory system (graph_memory, graph_store, spectral)
- `memory.json` - Persistent storage file (auto-created, gitignored)
- `.gitignore` - Git ignore patterns (excludes `__pycache__`, `memory.json`, etc.)

## License

MIT License