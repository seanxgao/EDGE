"""
High-level memory API combining SQLite, JSONL, and NumPy storage.

Provides unified interface for node creation, retrieval, and graph operations.
"""

import os
from datetime import datetime
from typing import Dict, Any, Optional

from memory.database import (
    init_db,
    add_node_context,
    add_node_usage,
    add_neighbor,
    get_node_context,
    get_node_usage,
    get_neighbors,
    update_node_usage,
)
from memory.jsonl_utils import load_sentence, append_sentence
from memory.npy_utils import load_embedding, compute_embedding_norm


from memory.config import (
    DEFAULT_SENTENCES_PATH,
    DEFAULT_EMBEDDINGS_PATH,
    DEFAULT_GRAPH_DIR,
    DEFAULT_DB_PATH,
)


def create_node(
    sentence_text: str,
    embedding_vector,
    metadata_dict: Optional[Dict[str, Any]] = None,
    sentences_path: str = DEFAULT_SENTENCES_PATH,
    embeddings_path: str = DEFAULT_EMBEDDINGS_PATH,
    db_path: str = DEFAULT_DB_PATH,
) -> int:
    """
    Create a new node with context, usage, and embedding.

    Args:
        sentence_text: Text content of the sentence
        embedding_vector: NumPy array or list with embedding vector
        metadata_dict: Optional dict with keys: id, source, tag, language, initial_context
        sentences_path: Path to sentences.jsonl file
        embeddings_path: Path to embeddings.npy file
        db_path: Path to SQLite database

    Returns:
        Node id (INTEGER)
    """
    # Initialize database if needed
    init_db(db_path)

    # Prepare metadata
    metadata = metadata_dict or {}
    node_id = metadata.get("id")
    if node_id is None:
        # Auto-generate node_id from next available id
        # Simple approach: use max existing id + 1
        from memory.database import NodesContext
        try:
            max_id = NodesContext.select().order_by(NodesContext.id.desc()).limit(1).get().id
            node_id = max_id + 1
        except:
            node_id = 1

    # Append sentence to JSONL
    sentence_data = {
        "id": f"node_{node_id:03d}",
        "text": sentence_text,
    }
    if "chapter" in metadata:
        sentence_data["chapter"] = metadata["chapter"]
    if "position" in metadata:
        sentence_data["position"] = metadata["position"]

    sentence_offset = append_sentence(sentences_path, sentence_data)

    # Append embedding to NumPy array
    import numpy as np
    embedding_array = np.array(embedding_vector, dtype=np.float32)
    
    # Load existing embeddings or create new
    if os.path.exists(embeddings_path):
        embeddings = np.load(embeddings_path)
        embeddings = np.vstack([embeddings, embedding_array.reshape(1, -1)])
    else:
        # Ensure directory exists (skip if path is in current directory)
        dirname = os.path.dirname(embeddings_path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        embeddings = embedding_array.reshape(1, -1)
    
    embedding_index = len(embeddings) - 1
    np.save(embeddings_path, embeddings)

    # Compute embedding norm
    embedding_norm = compute_embedding_norm(embedding_array)

    # Add node context
    add_node_context(
        node_id=node_id,
        sentence_offset=sentence_offset,
        embedding_index=embedding_index,
        source=metadata.get("source"),
        tag=metadata.get("tag"),
        language=metadata.get("language"),
        initial_context=metadata.get("initial_context"),
        embedding_norm=embedding_norm,
    )

    # Add node usage (default values)
    add_node_usage(node_id=node_id)

    return node_id


def get_node(
    node_id: int,
    sentences_path: str = DEFAULT_SENTENCES_PATH,
    embeddings_path: str = DEFAULT_EMBEDDINGS_PATH,
    db_path: str = DEFAULT_DB_PATH,
) -> Optional[Dict[str, Any]]:
    """
    Get complete node information (context + usage + neighbors + sentence + embedding).

    Args:
        node_id: Node identifier
        sentences_path: Path to sentences.jsonl file
        embeddings_path: Path to embeddings.npy file
        db_path: Path to SQLite database

    Returns:
        Dict with keys: context, usage, neighbors, sentence, embedding
        Returns None if node doesn't exist
    """
    # Initialize database if needed
    init_db(db_path)

    # Get database records
    context = get_node_context(node_id)
    if context is None:
        return None

    usage = get_node_usage(node_id)
    neighbors = get_neighbors(node_id)

    # Load sentence from JSONL
    sentence = load_sentence(sentences_path, context["sentence_offset"])

    # Load embedding from NumPy
    embedding = load_embedding(embeddings_path, context["embedding_index"])

    return {
        "context": context,
        "usage": usage,
        "neighbors": neighbors,
        "sentence": sentence,
        "embedding": embedding.tolist() if embedding is not None else None,
    }


def update_usage(
    node_id: int,
    access_count: Optional[int] = None,
    last_access_time: Optional[datetime] = None,
    recent_hit_count: Optional[int] = None,
    decay_score: Optional[float] = None,
    popularity: Optional[float] = None,
    db_path: str = DEFAULT_DB_PATH,
) -> bool:
    """
    Update node usage statistics.

    Args:
        node_id: Node identifier
        access_count: Optional new access count
        last_access_time: Optional new last access time (datetime object)
        recent_hit_count: Optional new recent hit count
        decay_score: Optional new decay score
        popularity: Optional new popularity score
        db_path: Path to SQLite database

    Returns:
        True if update succeeded, False if node doesn't exist
    """
    init_db(db_path)
    result = update_node_usage(
        node_id=node_id,
        access_count=access_count,
        last_access_time=last_access_time,
        recent_hit_count=recent_hit_count,
        decay_score=decay_score,
        popularity=popularity,
    )
    return result is not None


def add_graph_edge(
    u: int,
    v: int,
    weight: float,
    edge_type: int,
    db_path: str = DEFAULT_DB_PATH,
) -> bool:
    """
    Add a graph edge between two nodes.

    Args:
        u: Source node id
        v: Target node id
        weight: Edge weight
        edge_type: Edge type code (INTEGER)
        db_path: Path to SQLite database

    Returns:
        True if edge was added/updated successfully
    """
    init_db(db_path)
    try:
        add_neighbor(u, v, weight, edge_type)
        return True
    except Exception:
        return False

