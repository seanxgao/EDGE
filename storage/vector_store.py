"""
Binary vector storage with rowmap indexing.
Manages float32 binary files for efficient vector persistence.
"""
import time
import numpy as np
from typing import Optional, Tuple
from pathlib import Path
import json
from core.config import VERBOSE_TIMING


class VectorStore:
    """
    Manages binary vector storage with rowmap indexing.
    
    Format:
    - vectors.bin: Float32 binary file, row-major order
    - rowmap.json: Mapping from sentence ID to row index
    """
    
    def __init__(self, base_path: str, embedding_dim: int):
        """
        Initialize vector store.
        
        Args:
            base_path: Directory path for vector files
            embedding_dim: Embedding dimension (must be consistent)
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        self.embedding_dim = embedding_dim
        self.vector_file = self.base_path / "vectors.bin"
        self.rowmap_file = self.base_path / "rowmap.json"
        
        # In-memory rowmap: sentence_id -> row_index
        self.rowmap: dict = {}
        self.next_row = 0
        
        # Load existing rowmap
        self._load_rowmap()
    
    def _load_rowmap(self):
        """Load rowmap from file"""
        if self.rowmap_file.exists():
            try:
                with open(self.rowmap_file, 'r', encoding='utf-8') as f:
                    self.rowmap = json.load(f)
                    # Convert keys to int
                    self.rowmap = {int(k): int(v) for k, v in self.rowmap.items()}
                    if self.rowmap:
                        self.next_row = max(self.rowmap.values()) + 1
            except Exception as e:
                if VERBOSE_TIMING:
                    print(f"[vector_store] Failed to load rowmap: {e}, starting fresh")
                self.rowmap = {}
                self.next_row = 0
    
    def _save_rowmap(self):
        """Save rowmap to file"""
        with open(self.rowmap_file, 'w', encoding='utf-8') as f:
            json.dump(self.rowmap, f, indent=2)
    
    def add_vectors(
        self,
        sentence_ids: list,
        vectors: np.ndarray,
        append: bool = True
    ) -> dict:
        """
        Add vectors to store and update rowmap.
        
        Args:
            sentence_ids: List of sentence IDs (same length as vectors)
            vectors: Embedding vectors, shape (n, dim) or (dim,)
            append: Whether to append to file (True) or overwrite (False)
            
        Returns:
            Dictionary mapping sentence_id -> row_index
            
        Raises:
            ValueError: If inputs are invalid
        """
        # Validate inputs
        if len(sentence_ids) == 0:
            raise ValueError("sentence_ids cannot be empty")
        
        vectors = np.asarray(vectors)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        
        if vectors.shape[0] != len(sentence_ids):
            raise ValueError(
                f"Length mismatch: {len(sentence_ids)} IDs vs {vectors.shape[0]} vectors"
            )
        
        if vectors.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Dimension mismatch: expected {self.embedding_dim}, got {vectors.shape[1]}"
            )
        
        # Validate vectors
        if np.any(np.isnan(vectors)) or np.any(np.isinf(vectors)):
            raise ValueError("Vectors contain NaN or Inf values")
        
        t_start = time.time() if VERBOSE_TIMING else None
        
        # Convert to float32
        vectors = vectors.astype(np.float32)
        
        # Write to binary file
        mode = 'ab' if append else 'wb'
        with open(self.vector_file, mode) as f:
            vectors.tofile(f)
        
        # Update rowmap
        id_to_row = {}
        for i, sentence_id in enumerate(sentence_ids):
            row_index = self.next_row + i
            self.rowmap[sentence_id] = row_index
            id_to_row[sentence_id] = row_index
        
        self.next_row += len(sentence_ids)
        
        # Save rowmap
        self._save_rowmap()
        
        if VERBOSE_TIMING:
            t_elapsed = time.time() - t_start
            print(f"[vector_store] Added {len(sentence_ids)} vectors, duration={t_elapsed:.3f}s")
        
        return id_to_row
    
    def get_vector(self, sentence_id: int) -> np.ndarray:
        """
        Get vector for a sentence ID.
        
        Args:
            sentence_id: Sentence ID
            
        Returns:
            Embedding vector of shape (dim,)
            
        Raises:
            KeyError: If sentence_id not found
        """
        if sentence_id not in self.rowmap:
            raise KeyError(f"Sentence ID {sentence_id} not found in rowmap")
        
        row_index = self.rowmap[sentence_id]
        return self.get_vector_by_row(row_index)
    
    def get_vector_by_row(self, row_index: int) -> np.ndarray:
        """
        Get vector by row index.
        
        Args:
            row_index: Row index in binary file
            
        Returns:
            Embedding vector of shape (dim,)
        """
        with open(self.vector_file, 'rb') as f:
            # Seek to row position
            offset = row_index * self.embedding_dim * 4  # 4 bytes per float32
            f.seek(offset)
            
            # Read one vector
            vector_bytes = f.read(self.embedding_dim * 4)
            if len(vector_bytes) != self.embedding_dim * 4:
                raise IndexError(f"Row index {row_index} out of bounds")
            
            # Unpack to numpy array
            vector = np.frombuffer(vector_bytes, dtype=np.float32)
            return vector.copy()
    
    def get_vectors_batch(
        self,
        sentence_ids: list,
        use_mmap: bool = True
    ) -> np.ndarray:
        """
        Get multiple vectors efficiently.
        
        Args:
            sentence_ids: List of sentence IDs
            use_mmap: Whether to use memory mapping (faster for large files)
            
        Returns:
            Embedding matrix of shape (n, dim)
        """
        if not sentence_ids:
            return np.empty((0, self.embedding_dim), dtype=np.float32)
        
        # Get row indices
        row_indices = [self.rowmap[sid] for sid in sentence_ids]
        
        if use_mmap and self.vector_file.exists():
            # Use memory mapping for efficiency
            vectors = np.memmap(
                self.vector_file,
                dtype=np.float32,
                mode='r',
                shape=(self.next_row, self.embedding_dim)
            )
            result = vectors[row_indices].copy()
            del vectors  # Close memory map
            return result
        else:
            # Read vectors one by one
            return np.array([self.get_vector_by_row(ri) for ri in row_indices])
    
    def get_all_vectors(self, use_mmap: bool = True) -> np.ndarray:
        """
        Get all vectors (for index building).
        
        Args:
            use_mmap: Whether to use memory mapping
            
        Returns:
            Embedding matrix of shape (n, dim)
        """
        if self.next_row == 0:
            return np.empty((0, self.embedding_dim), dtype=np.float32)
        
        if use_mmap:
            vectors = np.memmap(
                self.vector_file,
                dtype=np.float32,
                mode='r',
                shape=(self.next_row, self.embedding_dim)
            )
            return vectors.copy()
        else:
            # Read all vectors
            with open(self.vector_file, 'rb') as f:
                data = f.read()
            
            vectors = np.frombuffer(data, dtype=np.float32)
            vectors = vectors.reshape(-1, self.embedding_dim)
            return vectors
    
    def get_stats(self) -> dict:
        """Get storage statistics"""
        file_size = self.vector_file.stat().st_size if self.vector_file.exists() else 0
        return {
            "num_vectors": self.next_row,
            "embedding_dim": self.embedding_dim,
            "file_size_bytes": file_size,
            "file_size_mb": round(file_size / (1024 * 1024), 2)
        }

