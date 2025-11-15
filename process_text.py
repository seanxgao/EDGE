#!/usr/bin/env python3
"""
Unified text processing pipeline.

Process a text file into complete memory graph with one command.
"""

import sys
from pathlib import Path
from memory import process_text_to_memory


def main():
    """Main entry point for text processing."""
    if len(sys.argv) < 2:
        print("Usage: python process_text.py <input_file> [data_dir]")
        print("Example: python process_text.py 3essay.txt data")
        sys.exit(1)

    input_file = sys.argv[1]
    data_dir = sys.argv[2] if len(sys.argv) > 2 else "data"

    # Read input text
    if not Path(input_file).exists():
        print(f"Error: File not found: {input_file}")
        sys.exit(1)

    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()

    # Extract source name from filename
    source_name = Path(input_file).stem

    print("=" * 60)
    print("EDGE Memory Processing Pipeline")
    print("=" * 60)
    print(f"Input file: {input_file}")
    print(f"Data directory: {data_dir}")
    print(f"Source name: {source_name}")
    print()

    # Process text
    try:
        stats = process_text_to_memory(
            text=text,
            source_name=source_name,
            data_dir=data_dir,
        )

        print()
        print("=" * 60)
        print("Processing Complete")
        print("=" * 60)
        print(f"Nodes created: {stats['num_nodes']}")
        print(f"Edges created: {stats['num_edges']}")
        print(f"Embeddings shape: {stats['embeddings_shape']}")
        print(f"Data directory: {data_dir}")
        print(f"Database: {stats['db_path']}")
        print()
        print("All data stored in hybrid format:")
        print("  - SQLite: nodes, edges, usage statistics")
        print("  - JSONL: sentence text")
        print("  - NumPy: embeddings and graph arrays")

    except Exception as e:
        print(f"\nError during processing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

