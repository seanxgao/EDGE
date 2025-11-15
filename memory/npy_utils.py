"""
NumPy array utilities for embedding and graph storage.

Provides helpers to load embeddings and graph arrays from .npy files.
"""

import os
import numpy as np
from typing import Optional, Tuple


def load_embedding(embeddings_path: str, embedding_index: int) -> Optional[np.ndarray]:
    """
    Load a single embedding vector by index from embeddings.npy.

    Args:
        embeddings_path: Path to embeddings.npy file
        embedding_index: Zero-based row index

    Returns:
        NumPy array with embedding vector, or None if index is out of range
    """
    if not os.path.exists(embeddings_path):
        return None

    try:
        embeddings = np.load(embeddings_path, mmap_mode="r")
        if embedding_index < 0 or embedding_index >= len(embeddings):
            return None
        return embeddings[embedding_index].copy()
    except Exception:
        return None


def load_edge_arrays(graph_dir: str) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Load all edge arrays from graph directory.

    Args:
        graph_dir: Directory containing edge_index.npy, edge_weight.npy, edge_type.npy

    Returns:
        Tuple of (edge_index, edge_weight, edge_type) arrays, or None if files don't exist
    """
    edge_index_path = os.path.join(graph_dir, "edge_index.npy")
    edge_weight_path = os.path.join(graph_dir, "edge_weight.npy")
    edge_type_path = os.path.join(graph_dir, "edge_type.npy")

    if not all(
        os.path.exists(p) for p in [edge_index_path, edge_weight_path, edge_type_path]
    ):
        return None

    try:
        edge_index = np.load(edge_index_path, mmap_mode="r")
        edge_weight = np.load(edge_weight_path, mmap_mode="r")
        edge_type = np.load(edge_type_path, mmap_mode="r")
        return edge_index, edge_weight, edge_type
    except Exception:
        return None


def compute_embedding_norm(vec: np.ndarray) -> float:
    """
    Compute L2 norm of embedding vector.

    Args:
        vec: NumPy array with embedding vector

    Returns:
        L2 norm (float)
    """
    return float(np.linalg.norm(vec))


def get_embedding_shape(embeddings_path: str) -> Optional[Tuple[int, int]]:
    """
    Get shape of embeddings array without loading full file.

    Args:
        embeddings_path: Path to embeddings.npy file

    Returns:
        Tuple of (num_embeddings, embedding_dim), or None if file doesn't exist
    """
    if not os.path.exists(embeddings_path):
        return None

    try:
        embeddings = np.load(embeddings_path, mmap_mode="r")
        return embeddings.shape
    except Exception:
        return None

