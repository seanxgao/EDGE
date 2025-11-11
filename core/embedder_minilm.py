"""
MiniLM embedding interface for book-scale semantic memory.
Supports batch encoding and vector persistence.
"""
import time
import numpy as np
from typing import List, Optional, Union
from sentence_transformers import SentenceTransformer
from core.config import VERBOSE_TIMING, EMBEDDING_MODEL


class MiniLMEmbedder:
    """
    MiniLM-based embedding provider for sentence-level encoding.
    
    Supports:
    - Batch encoding for efficiency
    - CPU/GPU automatic selection
    - Vector normalization
    - Model caching
    """
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[str] = None,
        normalize: bool = True
    ):
        """
        Initialize MiniLM embedder.
        
        Args:
            model_name: HuggingFace model name
                - "all-MiniLM-L6-v2": 384 dimensions (default)
                - "all-MiniLM-L12-v2": 768 dimensions
            device: Device for computation ("cpu", "cuda", or None for auto)
            normalize: Whether to normalize vectors to unit length
        """
        self.model_name = model_name
        self.normalize = normalize
        
        t_start = time.time() if VERBOSE_TIMING else None
        
        # Auto-select device if not specified
        if device is None:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = device
        
        # Load model (cached by sentence-transformers)
        self.model = SentenceTransformer(model_name, device=device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
        if VERBOSE_TIMING:
            t_elapsed = time.time() - t_start
            msg = (
                f"[embedder] Model loaded: {model_name}, "
                f"dim={self.embedding_dim}, device={device}, "
                f"duration={t_elapsed:.3f}s"
            )
            print(msg)
    
    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        show_progress: bool = False
    ) -> np.ndarray:
        """
        Encode text(s) into embedding vectors.
        
        Args:
            texts: Single text string or list of texts
            batch_size: Batch size for encoding
            show_progress: Whether to show progress bar
            
        Returns:
            Embedding vectors as numpy array
            - Single text: shape (dim,)
            - Multiple texts: shape (n, dim)
            
        Raises:
            ValueError: If input is invalid
            RuntimeError: If encoding fails
        """
        # Convert single string to list
        is_single = isinstance(texts, str)
        if is_single:
            texts = [texts]
        
        if not texts:
            raise ValueError("Input texts cannot be empty")
        
        # Validate all texts are non-empty strings
        for i, text in enumerate(texts):
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"Text at index {i} is empty or invalid")
        
        t_start = time.time() if VERBOSE_TIMING else None
        
        try:
            # Encode with batch processing
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
                normalize_embeddings=self.normalize
            )
            
            # Ensure numpy array
            if not isinstance(embeddings, np.ndarray):
                embeddings = np.array(embeddings)
            
            # Validate embeddings
            if embeddings.size == 0:
                raise RuntimeError("Empty embeddings received")
            if np.any(np.isnan(embeddings)):
                raise RuntimeError("NaN values detected in embeddings")
            if np.any(np.isinf(embeddings)):
                raise RuntimeError("Inf values detected in embeddings")
            
            # Check dimension consistency
            if embeddings.ndim == 1:
                if embeddings.shape[0] != self.embedding_dim:
                    raise RuntimeError(
                        f"Dimension mismatch: expected {self.embedding_dim}, "
                        f"got {embeddings.shape[0]}"
                    )
            elif embeddings.ndim == 2:
                if embeddings.shape[1] != self.embedding_dim:
                    raise RuntimeError(
                        f"Dimension mismatch: expected {self.embedding_dim}, "
                        f"got {embeddings.shape[1]}"
                    )
            else:
                raise RuntimeError(f"Unexpected embedding shape: {embeddings.shape}")
            
            # Return single vector for single input
            if is_single:
                embeddings = embeddings[0]
            
            if VERBOSE_TIMING:
                t_elapsed = time.time() - t_start
                n_texts = 1 if is_single else len(texts)
                msg = (
                    f"[embedder] Encoded {n_texts} texts, "
                    f"shape={embeddings.shape}, duration={t_elapsed:.3f}s"
                )
                print(msg)
            
            return embeddings
            
        except Exception as e:
            raise RuntimeError(f"Failed to encode texts: {e}")
    
    def get_embedding_dim(self) -> int:
        """Get embedding dimension"""
        return self.embedding_dim


# Global embedder instance (lazy initialization)
_global_embedder: Optional[MiniLMEmbedder] = None


def get_embedder(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    device: Optional[str] = None
) -> MiniLMEmbedder:
    """
    Get or create global MiniLM embedder instance.
    
    Args:
        model_name: Model name (only used on first call)
        device: Device (only used on first call)
        
    Returns:
        MiniLMEmbedder instance
    """
    global _global_embedder
    
    if _global_embedder is None:
        _global_embedder = MiniLMEmbedder(model_name=model_name, device=device)
    
    return _global_embedder


def encode_batch(
    texts: List[str],
    batch_size: int = 32,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    device: Optional[str] = None
) -> np.ndarray:
    """
    Convenience function for batch encoding.
    
    Args:
        texts: List of texts to encode
        batch_size: Batch size
        model_name: Model name
        device: Device
        
    Returns:
        Embedding matrix of shape (n, dim)
    """
    embedder = get_embedder(model_name=model_name, device=device)
    return embedder.encode(texts, batch_size=batch_size)

