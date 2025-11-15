"""
FAISS index management for fast similarity search.

Provides functions to build and manage FAISS indices for embeddings.
"""

import os
import numpy as np
import faiss
from typing import Optional


def build_faiss_index(
    embeddings: np.ndarray,
    index_type: str = "flat",
    save_path: Optional[str] = None,
) -> faiss.Index:
    """
    Build FAISS index from embeddings.

    Args:
        embeddings: NumPy array of shape (N, D) with embeddings
        index_type: Type of index ("flat" or "ivf-pq")
        save_path: Optional path to save index

    Returns:
        FAISS index object
    """
    num_vectors, embedding_dim = embeddings.shape

    if index_type == "flat":
        # Flat index (exact search)
        index = faiss.IndexFlatL2(embedding_dim)
    elif index_type == "ivf-pq":
        # IVF-PQ index (approximate, faster for large datasets)
        nlist = 100  # Number of clusters
        m = 64  # Number of subquantizers
        nbits = 8  # Bits per subquantizer
        
        quantizer = faiss.IndexFlatL2(embedding_dim)
        index = faiss.IndexIVFPQ(quantizer, embedding_dim, nlist, m, nbits)
        
        # Train index
        print(f"  Training IVF-PQ index with {num_vectors} vectors...")
        index.train(embeddings.astype(np.float32))
    else:
        raise ValueError(f"Unknown index type: {index_type}")

    # Add vectors to index
    print(f"  Adding {num_vectors} vectors to FAISS index...")
    index.add(embeddings.astype(np.float32))

    # Save if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        faiss.write_index(index, save_path)
        print(f"  Saved FAISS index to {save_path}")

    return index


def load_faiss_index(index_path: str) -> faiss.Index:
    """
    Load FAISS index from file.

    Args:
        index_path: Path to FAISS index file

    Returns:
        FAISS index object
    """
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"FAISS index not found: {index_path}")
    
    return faiss.read_index(index_path)


def search_faiss_index(
    index: faiss.Index,
    query_embedding: np.ndarray,
    k: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Search FAISS index for nearest neighbors.

    Args:
        index: FAISS index object
        query_embedding: Query embedding vector (1, D) or (D,)
        k: Number of nearest neighbors to return

    Returns:
        Tuple of (distances, indices) arrays
    """
    # Reshape to (1, D) if needed
    if query_embedding.ndim == 1:
        query_embedding = query_embedding.reshape(1, -1)
    
    query_embedding = query_embedding.astype(np.float32)
    
    distances, indices = index.search(query_embedding, k)
    
    return distances[0], indices[0]


def get_index_stats(index: faiss.Index) -> dict:
    """
    Get statistics about FAISS index.

    Args:
        index: FAISS index object

    Returns:
        Dict with index statistics
    """
    stats = {
        "num_vectors": index.ntotal,
        "embedding_dim": index.d,
        "is_trained": index.is_trained if hasattr(index, "is_trained") else True,
    }
    
    if isinstance(index, faiss.IndexIVFPQ):
        stats["index_type"] = "ivf-pq"
        stats["nlist"] = index.nlist
        stats["m"] = index.pq.m
        stats["nbits"] = index.pq.nbits
    elif isinstance(index, faiss.IndexFlatL2):
        stats["index_type"] = "flat"
    else:
        stats["index_type"] = "unknown"
    
    return stats


