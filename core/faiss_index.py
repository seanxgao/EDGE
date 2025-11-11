"""
Faiss index management for approximate nearest neighbor search.
Supports Flat and IVF-PQ index types.
"""
import time
import numpy as np
import faiss
from typing import Optional, Tuple, List
from pathlib import Path
from core.config import VERBOSE_TIMING


class FaissIndex:
    """
    Faiss index wrapper for vector similarity search.
    
    Supports:
    - Flat index: Exact search, full precision
    - IVF-PQ index: Approximate search, compressed storage
    - Index persistence and loading
    """
    
    def __init__(
        self,
        embedding_dim: int,
        index_type: str = "flat",
        nlist: int = 100,
        m: int = 64,
        nbits: int = 8
    ):
        """
        Initialize Faiss index.
        
        Args:
            embedding_dim: Embedding dimension
            index_type: "flat" or "ivf-pq"
            nlist: Number of clusters for IVF (only for IVF-PQ)
            m: Number of subquantizers for PQ (only for IVF-PQ)
            nbits: Bits per subquantizer (only for IVF-PQ)
        """
        self.embedding_dim = embedding_dim
        self.index_type = index_type.lower()
        self.index: Optional[faiss.Index] = None
        self.is_trained = False
        
        if self.index_type == "flat":
            # Flat index: exact search
            self.index = faiss.IndexFlatIP(embedding_dim)  # Inner product (for normalized vectors)
        elif self.index_type == "ivf-pq":
            # IVF-PQ index: approximate search with compression
            quantizer = faiss.IndexFlatIP(embedding_dim)
            self.index = faiss.IndexIVFPQ(quantizer, embedding_dim, nlist, m, nbits)
            self.nlist = nlist
            self.m = m
            self.nbits = nbits
        else:
            raise ValueError(f"Unknown index type: {index_type}")
    
    def train(self, vectors: np.ndarray):
        """
        Train index on vectors (required for IVF-PQ).
        
        Args:
            vectors: Training vectors, shape (n, dim)
            
        Raises:
            ValueError: If vectors are invalid
        """
        if self.index_type == "flat":
            # Flat index doesn't need training
            return
        
        if not isinstance(vectors, np.ndarray):
            vectors = np.array(vectors)
        
        if vectors.ndim != 2 or vectors.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Expected shape (n, {self.embedding_dim}), got {vectors.shape}"
            )
        
        # Validate vectors
        if np.any(np.isnan(vectors)) or np.any(np.isinf(vectors)):
            raise ValueError("Training vectors contain NaN or Inf values")
        
        # Normalize vectors (required for inner product)
        faiss.normalize_L2(vectors)
        
        t_start = time.time() if VERBOSE_TIMING else None
        
        # Train index
        self.index.train(vectors.astype(np.float32))
        self.is_trained = True
        
        if VERBOSE_TIMING:
            t_elapsed = time.time() - t_start
            msg = (
                f"[faiss_index] Trained {self.index_type} index on "
                f"{len(vectors)} vectors, duration={t_elapsed:.3f}s"
            )
            print(msg)
    
    def add(self, vectors: np.ndarray):
        """
        Add vectors to index.
        
        Args:
            vectors: Vectors to add, shape (n, dim)
            
        Raises:
            ValueError: If vectors are invalid or index not trained (for IVF-PQ)
        """
        if self.index_type == "ivf-pq" and not self.is_trained:
            raise RuntimeError("Index must be trained before adding vectors")
        
        if not isinstance(vectors, np.ndarray):
            vectors = np.array(vectors)
        
        if vectors.ndim != 2 or vectors.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Expected shape (n, {self.embedding_dim}), got {vectors.shape}"
            )
        
        # Validate vectors
        if np.any(np.isnan(vectors)) or np.any(np.isinf(vectors)):
            raise ValueError("Vectors contain NaN or Inf values")
        
        # Normalize vectors
        vectors = vectors.astype(np.float32)
        faiss.normalize_L2(vectors)
        
        t_start = time.time() if VERBOSE_TIMING else None
        
        # Add to index
        self.index.add(vectors)
        
        if VERBOSE_TIMING:
            t_elapsed = time.time() - t_start
            n_vectors = len(vectors)
            msg = (
                f"[faiss_index] Added {n_vectors} vectors, "
                f"total={self.index.ntotal}, duration={t_elapsed:.3f}s"
            )
            print(msg)
    
    def search(
        self,
        query_vectors: np.ndarray,
        k: int,
        nprobe: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for top-k nearest neighbors.
        
        Args:
            query_vectors: Query vectors, shape (n_queries, dim) or (dim,)
            k: Number of neighbors to retrieve
            nprobe: Number of clusters to probe (only for IVF-PQ)
            
        Returns:
            Tuple of (distances, indices):
            - distances: shape (n_queries, k) - similarity scores
            - indices: shape (n_queries, k) - vector indices in index
            
        Raises:
            ValueError: If inputs are invalid
        """
        if self.index.ntotal == 0:
            # Empty index
            if query_vectors.ndim == 1:
                empty_dist = np.empty((0, k), dtype=np.float32)
                empty_idx = np.empty((0, k), dtype=np.int64)
                return empty_dist, empty_idx
            else:
                n_queries = len(query_vectors)
                empty_dist = np.empty((n_queries, k), dtype=np.float32)
                empty_idx = np.empty((n_queries, k), dtype=np.int64)
                return empty_dist, empty_idx
        
        # Handle single query
        is_single = query_vectors.ndim == 1
        if is_single:
            query_vectors = query_vectors.reshape(1, -1)
        
        if query_vectors.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Query dimension mismatch: expected {self.embedding_dim}, "
                f"got {query_vectors.shape[1]}"
            )
        
        # Validate queries
        if np.any(np.isnan(query_vectors)) or np.any(np.isinf(query_vectors)):
            raise ValueError("Query vectors contain NaN or Inf values")
        
        # Normalize queries
        query_vectors = query_vectors.astype(np.float32)
        faiss.normalize_L2(query_vectors)
        
        # Set nprobe for IVF-PQ
        if self.index_type == "ivf-pq":
            self.index.nprobe = min(nprobe, self.nlist)
        
        t_start = time.time() if VERBOSE_TIMING else None
        
        # Search
        distances, indices = self.index.search(query_vectors, min(k, self.index.ntotal))
        
        if VERBOSE_TIMING:
            t_elapsed = time.time() - t_start
            n_queries = 1 if is_single else len(query_vectors)
            print(f"[faiss_index] Searched {n_queries} queries, k={k}, duration={t_elapsed:.3f}s")
        
        # Return single result for single query
        if is_single:
            return distances[0], indices[0]
        
        return distances, indices
    
    def save(self, filepath: str):
        """
        Save index to file.
        
        Args:
            filepath: Path to save index
        """
        if self.index is None:
            raise RuntimeError("Index not initialized")
        
        t_start = time.time() if VERBOSE_TIMING else None
        
        faiss.write_index(self.index, filepath)
        
        if VERBOSE_TIMING:
            t_elapsed = time.time() - t_start
            file_size = Path(filepath).stat().st_size / (1024 * 1024)  # MB
            msg = (
                f"[faiss_index] Saved index to {filepath}, "
                f"size={file_size:.2f}MB, duration={t_elapsed:.3f}s"
            )
            print(msg)
    
    @classmethod
    def load(cls, filepath: str) -> "FaissIndex":
        """
        Load index from file.
        
        Args:
            filepath: Path to index file
            
        Returns:
            FaissIndex instance
        """
        t_start = time.time() if VERBOSE_TIMING else None
        
        index = faiss.read_index(filepath)
        
        # Determine index type and dimension
        embedding_dim = index.d
        if isinstance(index, faiss.IndexFlat):
            index_type = "flat"
            instance = cls(embedding_dim, index_type=index_type)
        elif isinstance(index, faiss.IndexIVFPQ):
            index_type = "ivf-pq"
            nlist = index.nlist
            m = index.pq.m
            nbits = index.pq.nbits
            instance = cls(embedding_dim, index_type=index_type, nlist=nlist, m=m, nbits=nbits)
            instance.is_trained = True
        else:
            raise ValueError(f"Unsupported index type: {type(index)}")
        
        instance.index = index
        
        if VERBOSE_TIMING:
            t_elapsed = time.time() - t_start
            file_size = Path(filepath).stat().st_size / (1024 * 1024)  # MB
            msg = (
                f"[faiss_index] Loaded index from {filepath}, "
                f"size={file_size:.2f}MB, vectors={index.ntotal}, "
                f"duration={t_elapsed:.3f}s"
            )
            print(msg)
        
        return instance
    
    def get_stats(self) -> dict:
        """Get index statistics"""
        return {
            "index_type": self.index_type,
            "embedding_dim": self.embedding_dim,
            "num_vectors": self.index.ntotal if self.index else 0,
            "is_trained": self.is_trained
        }

