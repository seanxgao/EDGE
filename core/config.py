"""
Configuration constants for graph memory system
"""
# API configuration
DEFAULT_API_KEY_PATH = r"H:\API\openai_api.txt"

# Model configuration
EMBEDDING_MODEL = "text-embedding-3-small"

# Graph configuration
DEFAULT_K_NEIGHBORS = 3
DEFAULT_RETRIEVAL_K = 5

# Persistence configuration
DEFAULT_MEMORY_FILE = "memory.json"
SCHEMA_VERSION = 1

# Timing and logging
VERBOSE_TIMING = False  # Set to True to enable timing logs

# SCOPE integration
SCOPE_INTEGRATION_ENABLED = False  # Set to True when SCOPE is available
SCOPE_API_URL = None  # SCOPE API endpoint (when integrated)

