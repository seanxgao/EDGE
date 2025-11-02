# API Reference - Project EDGE

Technical interface reference for Project EDGE (Embedded Dynamic Graph Engine).

EDGE is the memory organization layer in a hierarchical cognitive architecture, positioned between SCOPE (spectral representation) and Application Layer (reasoning systems). It maintains a dynamic graph of memories where nodes represent memories and edges represent semantic relationships.

## Core Module

### `core.embedder`

#### `set_embedding_provider(provider: Callable[[str], np.ndarray])`

Set custom embedding provider (e.g., from SCOPE).

**Parameters:**
- `provider`: Function that takes text and returns normalized embedding vector

**Note:** When set, `get_embedding()` will use this provider instead of OpenAI API. This enables SCOPE integration.

**Example:**
```python
from core.embedder import set_embedding_provider, get_embedding

# Define SCOPE embedding function
def scope_embedder(text: str) -> np.ndarray:
    # Call SCOPE API here
    return normalized_embedding_vector

set_embedding_provider(scope_embedder)
embedding = get_embedding("text")  # Now uses SCOPE
```

#### `get_embedding(text: str) -> np.ndarray`

Generate normalized embedding vector for input text.

**Note:** If `set_embedding_provider()` was called, uses custom provider. Otherwise uses OpenAI API.

**Parameters:**
- `text`: Input text to embed (non-empty string)

**Returns:**
- Normalized embedding vector as numpy array (shape: `(dim,)`)

**Raises:**
- `ValueError`: If API key is invalid or text is empty
- `RuntimeError`: If embedding request fails or contains NaN/Inf values

**Example:**
```python
from core.embedder import get_embedding

embedding = get_embedding("Hello world")
print(embedding.shape)  # e.g., (1536,)
```

#### `cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float`

Calculate cosine similarity between two normalized vectors.

**Parameters:**
- `vec1`: First vector (1D numpy array)
- `vec2`: Second vector (1D numpy array, same shape as vec1)

**Returns:**
- Cosine similarity value in range [-1, 1]

**Raises:**
- `ValueError`: If vectors have incompatible shapes or contain NaN/Inf

**Example:**
```python
from core.embedder import get_embedding, cosine_similarity

vec1 = get_embedding("apple")
vec2 = get_embedding("fruit")
sim = cosine_similarity(vec1, vec2)
```

### `core.retriever`

#### `find_top_k(query_vec: np.ndarray, memory_vecs: List[np.ndarray], k: int = 5) -> List[int]`

Find indices of top-k most similar memory vectors.

**Parameters:**
- `query_vec`: Query embedding vector (1D numpy array)
- `memory_vecs`: List of memory embedding vectors
- `k`: Number of top results to return (default: 5)

**Returns:**
- List of indices (sorted by similarity descending)

**Raises:**
- `ValueError`: If inputs are invalid, shapes mismatch, or contain NaN/Inf

**Example:**
```python
from core.embedder import get_embedding
from core.retriever import find_top_k

query = get_embedding("coffee")
memories = [get_embedding("tea"), get_embedding("caffeine")]
indices = find_top_k(query, memories, k=1)
```

### `core.config`

Configuration constants:

- `DEFAULT_API_KEY_PATH`: Path to OpenAI API key file
- `EMBEDDING_MODEL`: Model name (default: "text-embedding-3-small")
- `DEFAULT_K_NEIGHBORS`: Default number of neighbors for graph connections (default: 3)
- `DEFAULT_RETRIEVAL_K`: Default number of results for retrieval (default: 5)
- `DEFAULT_MEMORY_FILE`: Default memory persistence file (default: "memory.json")
- `SCHEMA_VERSION`: JSON schema version (default: 1)
- `VERBOSE_TIMING`: Enable timing logs (default: False)
- `SCOPE_INTEGRATION_ENABLED`: Enable SCOPE integration (default: False)
- `SCOPE_API_URL`: SCOPE API endpoint (default: None)

## Graph Module

### `graph.graph_store.GraphStore`

Low-level graph storage with NetworkX and JSON persistence.

#### `__init__(filepath: str = None)`

Initialize graph storage.

**Parameters:**
- `filepath`: Path to JSON file (default: from config)

#### `add_node(text: str, embedding: np.ndarray) -> int`

Add new node to graph.

**Parameters:**
- `text`: Node text content (non-empty)
- `embedding`: Node embedding vector (1D numpy array)

**Returns:**
- New node ID (integer)

**Raises:**
- `ValueError`: If inputs are invalid or embedding contains NaN/Inf

#### `add_edge(source_id: int, target_id: int, weight: float = 1.0)`

Add weighted edge between nodes.

**Parameters:**
- `source_id`: Source node ID
- `target_id`: Target node ID
- `weight`: Edge weight (default: 1.0)

#### `get_node_text(node_id: int) -> str`

Get node text by ID.

#### `get_node_embedding(node_id: int) -> np.ndarray`

Get node embedding by ID.

#### `get_all_embeddings() -> List[np.ndarray]`

Get all node embeddings (sorted by node ID).

#### `get_all_node_ids() -> List[int]`

Get all node IDs (sorted).

#### `get_stats() -> Dict[str, Any]`

Return graph statistics.

**Returns:**
- Dictionary with keys:
  - `nodes` (int): Number of nodes in graph
  - `edges` (int): Number of edges in graph
  - `density` (float): Graph density (0.0 to 1.0)

#### `get_graph_data() -> Dict[str, Any]`

Get complete graph data for SCOPE integration.

**Returns:**
- Dictionary with nodes, edges, and graph structure suitable for SCOPE analysis

#### `save_data()`

Save graph data to JSON file (with schema version).

#### `_load_data()`

Load graph data from JSON file (handles schema versioning).

**Note:** This is a private method called during initialization.

### `graph.graph_memory.GraphMemory`

High-level API for graph-based memory system.

#### `__init__(filepath: str = None, scope_integration: bool = False)`

Initialize graph memory system.

**Parameters:**
- `filepath`: Path to memory file (default: from config)
- `scope_integration`: Enable SCOPE integration hooks (default: False)

**Note:** When `scope_integration=True`, SCOPE methods become available but require actual SCOPE implementation to function.

#### `add_memory(text: str) -> int`

Embed text, add as node, connect to k nearest neighbors.

**Parameters:**
- `text`: Memory text content (non-empty)

**Returns:**
- New memory node ID

**Raises:**
- `ValueError`: If text is empty
- `RuntimeError`: If embedding or storage fails

**Example:**
```python
from graph.graph_memory import GraphMemory

memory = GraphMemory()
node_id = memory.add_memory("User likes coffee")
```

#### `retrieve_memories(query: str, k: int = None) -> List[str]`

Return top-k related memory texts.

**Parameters:**
- `query`: Query text
- `k`: Number of memories to return (default: from config)

**Returns:**
- List of related memory texts (sorted by similarity)

**Example:**
```python
results = memory.retrieve_memories("What does user like?", k=3)
for text in results:
    print(text)
```

#### `show_stats() -> Dict[str, Any]`

Return node/edge counts and graph density.

**Returns:**
- Dictionary with keys: `nodes`, `edges`, `density`, `filepath`

#### `get_graph_data() -> Dict[str, Any]`

Get complete graph data for SCOPE integration.

**Returns:**
- Dictionary with keys:
  - `nodes`: Dict mapping node_id to {text, embedding}
  - `edges`: List of (u, v, weight) tuples
  - `graph`: NetworkX graph object

**Example:**
```python
graph_data = memory.get_graph_data()
# Pass to SCOPE for spectral analysis
```

#### `apply_spectral_smoothing()`

Apply spectral smoothing using SCOPE. **Placeholder - requires SCOPE integration.**

**Raises:**
- `RuntimeError`: If `scope_integration=False`

**Note:** Currently a placeholder. Will integrate with SCOPE's spectral smoothing when available.

#### `apply_optimal_transport_update()`

Apply optimal transport updates using SCOPE. **Placeholder - requires SCOPE integration.**

**Raises:**
- `RuntimeError`: If `scope_integration=False`

**Note:** Currently a placeholder. Will integrate with SCOPE's optimal transport when available.

## Configuration

Modify `core/config.py` to adjust:
- API key path
- Embedding model
- Default graph parameters
- Timing verbosity

Enable timing logs:
```python
from core.config import VERBOSE_TIMING
import core.config
core.config.VERBOSE_TIMING = True
```

Or use CLI:
```bash
python main.py --verbose --add "text"
```

## Graph Module: Spectral Functions

### `graph.spectral`

SCOPE integration hooks. Functions accept SCOPE results as parameters but do not call SCOPE API directly (placeholder until SCOPE is integrated).

#### `spectral_clustering_from_scope(graph_data: Dict[str, Any], scope_clusters: Optional[List[int]] = None, k: int = 2) -> List[int]`

Apply spectral clustering using SCOPE results.

**Parameters:**
- `graph_data`: Graph adjacency data or GraphStore instance
- `scope_clusters`: Cluster assignments from SCOPE (if provided, uses them)
- `k`: Number of clusters

**Returns:**
- Cluster assignments (from SCOPE if provided, otherwise empty list)

**Note:** Currently a placeholder. Will call SCOPE API when integrated.

**Example:**
```python
from graph.spectral import spectral_clustering_from_scope

# If SCOPE provides clusters, use them
clusters = spectral_clustering_from_scope(graph_data, scope_clusters=[0, 1, 0, 1])
```

#### `optimal_transport_from_scope(embeddings: List[np.ndarray], scope_weights: Optional[List[float]] = None, scope_updated_embeddings: Optional[List[np.ndarray]] = None) -> List[np.ndarray]`

Apply optimal transport updates using SCOPE.

**Parameters:**
- `embeddings`: Current embeddings
- `scope_weights`: Transport weights from SCOPE (optional)
- `scope_updated_embeddings`: Updated embeddings from SCOPE (optional)

**Returns:**
- Updated embeddings (from SCOPE if provided, otherwise original embeddings)

**Note:** Currently a placeholder. Will call SCOPE optimal transport API when integrated.

#### `laplacian_smoothing_from_scope(graph_data: Dict[str, Any], scope_smoothed_data: Optional[Dict[str, Any]] = None, iterations: int = 10) -> Dict[str, Any]`

Apply Laplacian smoothing using SCOPE.

**Parameters:**
- `graph_data`: Graph data or GraphStore instance
- `scope_smoothed_data`: Smoothed graph data from SCOPE (optional)
- `iterations`: Number of smoothing iterations

**Returns:**
- Smoothed graph data (from SCOPE if provided, otherwise original)

**Note:** Currently a placeholder. Will call SCOPE Laplacian smoothing API when integrated.

#### Legacy Functions (for backward compatibility)

- `spectral_clustering(graph_data: Dict[str, Any], k: int = 2) -> List[int]` - Legacy placeholder
- `optimal_transport_update(embeddings: List[np.ndarray], weights: List[float]) -> List[np.ndarray]` - Legacy placeholder
- `laplacian_smoothing(graph_data: Dict[str, Any], iterations: int = 10) -> Dict[str, Any]` - Legacy placeholder

**Note:** Use `*_from_scope()` functions instead for SCOPE integration.

## Integration Architecture

### SCOPE Integration

EDGE is designed to integrate with SCOPE (spectral and geometric organization layer):

- **Spectral Clustering**: Use SCOPE's clustering to organize memory nodes into semantic clusters
- **Geometric Structure**: Leverage SCOPE's geometric organization of embedding space to guide graph topology
- **Topology Optimization**: Apply SCOPE's spectral analysis to improve graph structure and edge weights

When integrated, SCOPE provides the structural foundation that EDGE uses to organize memories semantically.

### Application Layer Integration

Applications interact with EDGE through the `GraphMemory` API:

```python
from graph.graph_memory import GraphMemory

# Application creates EDGE instance
memory = GraphMemory()

# Application adds memories
node_id = memory.add_memory("User likes coffee")

# Application queries memories
results = memory.retrieve_memories("What does user like?", k=3)

# Application gets statistics
stats = memory.show_stats()
```

**Key Principles:**
- Applications do not access embeddings directly; all memory operations are mediated through EDGE
- EDGE abstracts the graph structure, providing semantic recall without exposing implementation details
- Graph traversal and semantic relationships are handled internally by EDGE

## Error Handling

All functions validate inputs and raise clear errors:
- `ValueError`: Invalid input parameters or shapes
- `RuntimeError`: Computation or I/O failures

Functions check for NaN/Inf values and shape mismatches before operations.

