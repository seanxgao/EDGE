"""
JSONL file utilities for sentence storage.

Provides helpers to read and append to sentences.jsonl file.
"""

import json
import os
from typing import Optional, Dict, Any


def load_sentence(sentences_path: str, offset: int) -> Optional[Dict[str, Any]]:
    """
    Load a single sentence by offset (line number) from JSONL file.

    Args:
        sentences_path: Path to sentences.jsonl file
        offset: Zero-based line index

    Returns:
        Dict with sentence data, or None if offset is out of range
    """
    if not os.path.exists(sentences_path):
        return None

    with open(sentences_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == offset:
                return json.loads(line)
    return None


def append_sentence(sentences_path: str, sentence_data: Dict[str, Any]) -> int:
    """
    Append a sentence to JSONL file and return its offset.

    Args:
        sentences_path: Path to sentences.jsonl file
        sentence_data: Dict with sentence fields (id, text, etc.)

    Returns:
        Zero-based line index (offset) of the appended sentence
    """
    # Ensure directory exists (skip if path is in current directory)
    dirname = os.path.dirname(sentences_path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)

    # Count existing lines to get offset
    offset = 0
    if os.path.exists(sentences_path):
        with open(sentences_path, "r", encoding="utf-8") as f:
            for _ in f:
                offset += 1

    # Append new sentence
    with open(sentences_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(sentence_data, ensure_ascii=False) + "\n")

    return offset


def count_sentences(sentences_path: str) -> int:
    """
    Count total number of sentences in JSONL file.

    Args:
        sentences_path: Path to sentences.jsonl file

    Returns:
        Number of lines (sentences) in file
    """
    if not os.path.exists(sentences_path):
        return 0

    count = 0
    with open(sentences_path, "r", encoding="utf-8") as f:
        for _ in f:
            count += 1
    return count

