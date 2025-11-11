"""
Configuration constants for graph memory system
"""
# API configuration
DEFAULT_API_KEY_PATH = r"H:\API\openai_api.txt"

# Model configuration
EMBEDDING_MODEL = "text-embedding-3-small"  # Legacy OpenAI model
MINILM_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # Default MiniLM model

# Graph configuration
DEFAULT_K_NEIGHBORS = 3
DEFAULT_RETRIEVAL_K = 5

# Persistence configuration
DEFAULT_MEMORY_FILE = "memory.json"
SCHEMA_VERSION = 1
DEFAULT_BASE_PATH = "data"  # Base path for book memory storage

# Faiss index configuration
DEFAULT_INDEX_TYPE = "flat"  # "flat" or "ivf-pq"
IVF_PQ_NLIST = 100  # Number of clusters for IVF-PQ
IVF_PQ_M = 64  # Number of subquantizers
IVF_PQ_NBITS = 8  # Bits per subquantizer

# Graph edge configuration
SEMANTIC_SIMILARITY_THRESHOLD = 0.7  # Threshold for semantic edges
MAX_SEMANTIC_EDGES_PER_NODE = 10  # Maximum semantic edges per node

# Query configuration
DEFAULT_TOP_K = 10  # Default number of results
DEFAULT_INITIAL_K = 50  # Default candidates from Stage 1

# Timing and logging
VERBOSE_TIMING = False  # Set to True to enable timing logs

# SCOPE integration
SCOPE_INTEGRATION_ENABLED = False  # Set to True when SCOPE is available
SCOPE_API_URL = None  # SCOPE API endpoint (when integrated)

