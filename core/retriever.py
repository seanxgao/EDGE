"""
Basic cosine similarity retrieval
"""
import time
import numpy as np
from typing import List
from core.embedder import cosine_similarity
from core.config import VERBOSE_TIMING


def find_top_k(query_vec: np.ndarray, memory_vecs: List[np.ndarray], k: int = 5) -> List[int]:
    """
    Return indices of top-k cosine-similar items.
    
    Args:
        query_vec: Query embedding vector
        memory_vecs: List of memory embedding vectors
        k: Number of top results to return
        
    Returns:
        List of indices of top-k most similar memories
        
    Raises:
        ValueError: If inputs are invalid or empty
    """
    if not memory_vecs:
        return []
    
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if k > len(memory_vecs):
        k = len(memory_vecs)
    
    # Validate query vector
    if query_vec.ndim != 1:
        raise ValueError(f"Expected 1D query vector, got shape {query_vec.shape}")
    if np.any(np.isnan(query_vec)) or np.any(np.isinf(query_vec)):
        raise ValueError("Query vector contains NaN or Inf values")
    
    t_start = time.time() if VERBOSE_TIMING else None
    
    similarities = []
    for i, memory_vec in enumerate(memory_vecs):
        # Validate memory vector shape
        if memory_vec.ndim != 1:
            raise ValueError(f"Memory vector at index {i} is not 1D: {memory_vec.shape}")
        if memory_vec.shape != query_vec.shape:
            raise ValueError(
                f"Shape mismatch at index {i}: {memory_vec.shape} vs {query_vec.shape}"
            )
        
        sim = cosine_similarity(query_vec, memory_vec)
        similarities.append((i, sim))
    
    # Sort by similarity descending
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    # Return top k indices
    result = [idx for idx, _ in similarities[:k]]
    
    if VERBOSE_TIMING:
        t_elapsed = time.time() - t_start
        print(f"[retrieval] duration={t_elapsed:.3f}s, k={k}, candidates={len(memory_vecs)}")
    
    return result