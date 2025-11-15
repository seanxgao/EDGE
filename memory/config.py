"""
Storage configuration and default paths.

Centralized configuration for all storage paths.
"""

import os

# Base data directory (all storage files go here)
DATA_DIR = "data"

# Database
DEFAULT_DB_PATH = os.path.join(DATA_DIR, "memory.db")

# File storage paths (within data directory)
DEFAULT_SENTENCES_PATH = os.path.join(DATA_DIR, "sentences.jsonl")
DEFAULT_EMBEDDINGS_PATH = os.path.join(DATA_DIR, "embeddings.npy")
DEFAULT_GRAPH_DIR = os.path.join(DATA_DIR, "graph")

# Graph array filenames
EDGE_INDEX_FILE = "edge_index.npy"
EDGE_WEIGHT_FILE = "edge_weight.npy"
EDGE_TYPE_FILE = "edge_type.npy"

# FAISS index
DEFAULT_FAISS_INDEX_PATH = os.path.join(DATA_DIR, "faiss.index")
DEFAULT_FAISS_INDEX_TYPE = "flat"  # "flat" or "ivf-pq"

