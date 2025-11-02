"""
Embedding interface - currently uses OpenAI API.
Designed to integrate with SCOPE for spectral embeddings.
"""
import time
import openai
import numpy as np
from typing import Optional, Callable
from .config import DEFAULT_API_KEY_PATH, EMBEDDING_MODEL, VERBOSE_TIMING

# Embedding provider function (can be replaced by SCOPE integration)
_embedding_provider: Optional[Callable[[str], np.ndarray]] = None


def set_embedding_provider(provider: Callable[[str], np.ndarray]):
    """
    Set custom embedding provider (e.g., from SCOPE).
    
    Args:
        provider: Function that takes text and returns normalized embedding vector
    """
    global _embedding_provider
    _embedding_provider = provider


def get_embedding(text: str) -> np.ndarray:
    """
    Return OpenAI embedding; no batching, no streaming.
    
    Args:
        text: Input text to embed
        
    Returns:
        Normalized embedding vector as numpy array
        
    Raises:
        ValueError: If API key is invalid or text is empty
        RuntimeError: If embedding request fails
    """
    if not text or not text.strip():
        raise ValueError("Text input cannot be empty")
    
    # Use custom provider if set (e.g., from SCOPE)
    if _embedding_provider is not None:
        return _embedding_provider(text)
    
    # Default: use OpenAI API
    t_start = time.time() if VERBOSE_TIMING else None
    
    # Read API key from file
    try:
        with open(DEFAULT_API_KEY_PATH, 'r', encoding='utf-8') as f:
            api_key = f.read().strip()
        if not api_key:
            raise ValueError("API key file is empty")
    except FileNotFoundError:
        raise ValueError(f"API key file not found at {DEFAULT_API_KEY_PATH}")
    except Exception as e:
        raise ValueError(f"Failed to read API key: {e}")
    
    client = openai.OpenAI(api_key=api_key)
    
    try:
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text
        )
        embedding = np.array(response.data[0].embedding)
        
        # Validate embedding shape and values
        if embedding.size == 0:
            raise RuntimeError("Empty embedding received")
        if np.any(np.isnan(embedding)):
            raise RuntimeError("NaN values detected in embedding")
        if np.any(np.isinf(embedding)):
            raise RuntimeError("Inf values detected in embedding")
        
        # Normalize the vector
        norm = np.linalg.norm(embedding)
        if norm == 0:
            raise RuntimeError("Zero-norm embedding vector")
        normalized = embedding / norm
        
        if VERBOSE_TIMING:
            t_elapsed = time.time() - t_start
            print(f"[embedding] duration={t_elapsed:.3f}s, shape={normalized.shape}")
        
        return normalized
        
    except Exception as e:
        raise RuntimeError(f"Failed to get embedding: {e}")


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Calculate cosine similarity between two vectors.
    
    Args:
        vec1: First vector (must be 1D)
        vec2: Second vector (must be 1D)
        
    Returns:
        Cosine similarity value in [-1, 1]
        
    Raises:
        ValueError: If vectors have incompatible shapes
    """
    # Validate shapes
    if vec1.ndim != 1 or vec2.ndim != 1:
        raise ValueError(f"Expected 1D vectors, got shapes {vec1.shape} and {vec2.shape}")
    if vec1.shape != vec2.shape:
        raise ValueError(f"Vector shape mismatch: {vec1.shape} vs {vec2.shape}")
    
    # Check for NaN/Inf
    if np.any(np.isnan(vec1)) or np.any(np.isnan(vec2)):
        raise ValueError("NaN values detected in input vectors")
    if np.any(np.isinf(vec1)) or np.any(np.isinf(vec2)):
        raise ValueError("Inf values detected in input vectors")
    
    return float(np.dot(vec1, vec2))