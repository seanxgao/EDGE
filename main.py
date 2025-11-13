# Deprecated CLI — replaced by direct memory_bacon scripts
# This module uses the old GraphMemory API and is kept for backward compatibility.
# For new workflows, use:
#   - parse_essay.py
#   - generate_embeddings.py
#   - convert_edges_to_numpy.py

"""
Project Edge - Minimal Graph-Based Memory MVP
Simple CLI for testing graph memory functionality

NOTE: This module is DEPRECATED and will not work after cleanup.
The GraphMemory API has been removed. Use memory_bacon scripts instead.
"""
import sys


def main():
    """Main CLI interface - DEPRECATED"""
    print("=" * 60)
    print("DEPRECATED: This CLI has been removed.")
    print("=" * 60)
    print()
    print("The GraphMemory API has been removed as part of project cleanup.")
    print("For the new memory_bacon workflow, use these scripts instead:")
    print()
    print("  1. parse_essay.py")
    print("     - Parse text file into structured sentences")
    print("     - Generates: memory_bacon/core/sentences.jsonl, statistics.jsonl")
    print()
    print("  2. generate_embeddings.py")
    print("     - Generate OpenAI embeddings for all sentences")
    print("     - Generates: memory_bacon/core/embeddings.npy")
    print()
    print("  3. convert_edges_to_numpy.py")
    print("     - Build graph edges and convert to NumPy format")
    print("     - Generates: memory_bacon/graph/edge_*.npy")
    print()
    print("For exploration, see: memory_bacon_exploration.ipynb")
    print()
    sys.exit(1)


if __name__ == "__main__":
    main()
