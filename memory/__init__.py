"""
Memory module for EDGE memory engine.

Provides hybrid storage architecture:
- SQLite for structured data (context, usage, neighbors)
- JSONL for sentence text (append-only)
- NumPy for embeddings and graph arrays (efficient binary format)
"""

from memory.database import (
    init_db,
    NodesContext,
    Neighbors,
    NodesUsage,
    add_node_context,
    add_node_usage,
    add_neighbor,
    get_neighbors,
    get_node_context,
    get_node_usage,
    get_node as get_node_db,
    update_node_usage,
)

from memory.jsonl_utils import (
    load_sentence,
    append_sentence,
    count_sentences,
)

from memory.npy_utils import (
    load_embedding,
    load_edge_arrays,
    compute_embedding_norm,
    get_embedding_shape,
)

from memory.memory_api import (
    create_node,
    get_node,
    update_usage,
    add_graph_edge,
)

from memory.pipeline import (
    process_text_to_memory,
    parse_text_to_sentences,
    generate_embeddings_openai,
    build_graph_edges,
)

from memory.config import (
    DATA_DIR,
    DEFAULT_DB_PATH,
    DEFAULT_SENTENCES_PATH,
    DEFAULT_EMBEDDINGS_PATH,
    DEFAULT_GRAPH_DIR,
    DEFAULT_FAISS_INDEX_PATH,
    DEFAULT_FAISS_INDEX_TYPE,
)

from memory.faiss_index import (
    build_faiss_index,
    load_faiss_index,
    search_faiss_index,
    get_index_stats,
)

__all__ = [
    # Database
    "init_db",
    "NodesContext",
    "Neighbors",
    "NodesUsage",
    "add_node_context",
    "add_node_usage",
    "add_neighbor",
    "get_neighbors",
    "get_node_context",
    "get_node_usage",
    "get_node_db",
    "update_node_usage",
    # JSONL
    "load_sentence",
    "append_sentence",
    "count_sentences",
    # NumPy
    "load_embedding",
    "load_edge_arrays",
    "compute_embedding_norm",
    "get_embedding_shape",
    # High-level API
    "create_node",
    "get_node",
    "update_usage",
    "add_graph_edge",
    # Pipeline
    "process_text_to_memory",
    "parse_text_to_sentences",
    "generate_embeddings_openai",
    "build_graph_edges",
    # Config
    "DATA_DIR",
    "DEFAULT_DB_PATH",
    "DEFAULT_SENTENCES_PATH",
    "DEFAULT_EMBEDDINGS_PATH",
    "DEFAULT_GRAPH_DIR",
    "DEFAULT_FAISS_INDEX_PATH",
    "DEFAULT_FAISS_INDEX_TYPE",
    # FAISS
    "build_faiss_index",
    "load_faiss_index",
    "search_faiss_index",
    "get_index_stats",
]

