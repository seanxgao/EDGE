"""
Unified pipeline for processing text into complete memory graph.

This module provides a single entry point to process text files:
1. Parse text into sentences
2. Generate embeddings
3. Build graph edges
4. Store everything in hybrid storage (SQLite + JSONL + NumPy)
"""

import os
import json
import time
import numpy as np
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path

from memory.database import init_db, add_node_context, add_node_usage, add_neighbor
from memory.jsonl_utils import append_sentence, count_sentences
from memory.npy_utils import compute_embedding_norm
from memory.config import (
    DEFAULT_SENTENCES_PATH,
    DEFAULT_EMBEDDINGS_PATH,
    DEFAULT_GRAPH_DIR,
    DEFAULT_DB_PATH,
)


def parse_text_to_sentences(text: str, source_name: str = "unknown") -> List[Dict[str, Any]]:
    """
    Parse text into structured sentences.

    Args:
        text: Input text (one sentence per line, blank lines = chapter boundaries)
        source_name: Source identifier for metadata

    Returns:
        List of sentence dicts with id, chapter, position, text
    """
    lines = text.strip().split('\n')
    sentences = []
    current_chapter = 1
    current_position = 1
    sentence_index = 0

    for line in lines:
        stripped = line.strip()

        # Empty line indicates new chapter
        if not stripped:
            if sentences:  # Only increment if we have previous sentences
                current_chapter += 1
                current_position = 1
            continue

        # Non-empty line is a sentence
        sentence_id = f"{source_name}_{sentence_index + 1:03d}"
        sentence_data = {
            "id": sentence_id,
            "chapter": current_chapter,
            "position": current_position,
            "text": stripped
        }
        sentences.append(sentence_data)

        current_position += 1
        sentence_index += 1

    return sentences


def generate_embeddings_openai(
    sentences: List[str],
    api_key: Optional[str] = None,
    batch_size: int = 100,
    max_retries: int = 3
) -> np.ndarray:
    """
    Generate embeddings using OpenAI API.

    Args:
        sentences: List of sentence texts
        api_key: OpenAI API key (if None, tries env var or default file)
        batch_size: Number of texts per API call
        max_retries: Maximum retries for failed calls

    Returns:
        NumPy array of shape (N, D) with embeddings
    """
    from openai import OpenAI

    # Get API key
    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            default_key_path = r"H:\API\openai_api.txt"
            if os.path.exists(default_key_path):
                with open(default_key_path, 'r', encoding='utf-8') as f:
                    api_key = f.read().strip()
            else:
                raise ValueError("OPENAI_API_KEY not found")

    client = OpenAI(api_key=api_key)
    all_embeds = []
    total_batches = (len(sentences) + batch_size - 1) // batch_size

    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i+batch_size]
        batch_num = i // batch_size + 1

        for attempt in range(max_retries):
            try:
                resp = client.embeddings.create(
                    model="text-embedding-3-small",
                    input=batch
                )
                batch_embeds = [d.embedding for d in resp.data]
                all_embeds.extend(batch_embeds)
                print(f"Processed batch {batch_num}/{total_batches} ({len(batch)} sentences)")
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 2
                    print(f"Error in batch {batch_num}: {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    raise RuntimeError(f"Failed to process batch {batch_num} after {max_retries} attempts: {e}")

    return np.array(all_embeds, dtype=np.float32)


def build_graph_edges(sentences: List[Dict[str, Any]]) -> List[Tuple[int, int, float, int]]:
    """
    Build graph edges from sentences.

    Args:
        sentences: List of sentence dicts with id, chapter, position, text

    Returns:
        List of (u, v, weight, edge_type) tuples
        edge_type: 0=adjacent, 1=same_chapter, 2=question_context, 3=definition_context
    """
    edges = []
    
    # Create id to index mapping
    id_to_idx = {sent["id"]: i for i, sent in enumerate(sentences)}
    
    # Group by chapter
    chapter_groups = {}
    for sent in sentences:
        ch = sent["chapter"]
        if ch not in chapter_groups:
            chapter_groups[ch] = []
        chapter_groups[ch].append(sent)
    
    # Sort within each chapter
    for ch in chapter_groups:
        chapter_groups[ch].sort(key=lambda x: x["position"])
    
    # Build edges
    for i, sent in enumerate(sentences):
        sent_id = sent["id"]
        sent_idx = id_to_idx[sent_id]
        chapter = sent["chapter"]
        position = sent["position"]
        text = sent["text"]
        
        chapter_sents = chapter_groups[chapter]
        current_idx = position - 1  # position is 1-indexed
        
        # (a) Adjacent edges
        if current_idx + 1 < len(chapter_sents):
            next_sent = chapter_sents[current_idx + 1]
            next_id = next_sent["id"]
            next_idx = id_to_idx[next_id]
            edges.append((sent_idx + 1, next_idx + 1, 1.0, 0))  # +1 for 1-based node IDs
        
        # (b) Question context
        if text.strip().endswith("?"):
            for offset in [1, 2]:
                if current_idx + offset < len(chapter_sents):
                    next_sent = chapter_sents[current_idx + offset]
                    next_id = next_sent["id"]
                    next_idx = id_to_idx[next_id]
                    edges.append((sent_idx + 1, next_idx + 1, 0.9, 2))
        
        # (c) Definition context
        text_lower = text.lower()
        if " is " in text_lower or " means " in text_lower or " refers to " in text_lower:
            if current_idx > 0:
                prev_sent = chapter_sents[current_idx - 1]
                prev_id = prev_sent["id"]
                prev_idx = id_to_idx[prev_id]
                edges.append((prev_idx + 1, sent_idx + 1, 0.8, 3))
            
            if current_idx + 1 < len(chapter_sents):
                next_sent = chapter_sents[current_idx + 1]
                next_id = next_sent["id"]
                next_idx = id_to_idx[next_id]
                edges.append((sent_idx + 1, next_idx + 1, 0.8, 3))
    
    return edges


def process_text_to_memory(
    text: str,
    source_name: str = "essay",
    data_dir: str = "data",
    db_path: Optional[str] = None,
    api_key: Optional[str] = None,
    batch_size: int = 100,
) -> Dict[str, Any]:
    """
    Unified pipeline: process text into complete memory graph.

    This function:
    1. Parses text into sentences
    2. Generates embeddings
    3. Builds graph edges
    4. Stores everything in hybrid storage (SQLite + JSONL + NumPy)

    Args:
        text: Input text (one sentence per line, blank lines = chapter boundaries)
        source_name: Source identifier
        data_dir: Data directory for all storage files
        db_path: Path to SQLite database (default: data_dir/memory.db)
        api_key: OpenAI API key (optional)
        batch_size: Batch size for embedding generation

    Returns:
        Dict with statistics: num_nodes, num_edges, embeddings_shape, etc.
    """
    # Setup paths
    os.makedirs(data_dir, exist_ok=True)
    
    if db_path is None:
        db_path = os.path.join(data_dir, "memory.db")
    
    sentences_path = os.path.join(data_dir, "sentences.jsonl")
    embeddings_path = os.path.join(data_dir, "embeddings.npy")
    graph_dir = os.path.join(data_dir, "graph")
    os.makedirs(graph_dir, exist_ok=True)

    # Initialize database
    init_db(db_path)

    # Step 1: Parse text to sentences
    print("Step 1: Parsing text to sentences...")
    sentences = parse_text_to_sentences(text, source_name)
    print(f"  Parsed {len(sentences)} sentences")

    # Step 2: Generate embeddings
    print("\nStep 2: Generating embeddings...")
    sentence_texts = [s["text"] for s in sentences]
    embeddings = generate_embeddings_openai(sentence_texts, api_key=api_key, batch_size=batch_size)
    print(f"  Generated embeddings shape: {embeddings.shape}")

    # Step 3: Store sentences and embeddings, create nodes in database
    print("\nStep 3: Storing data in hybrid storage...")
    
    # Write sentences to JSONL
    if os.path.exists(sentences_path):
        os.remove(sentences_path)  # Start fresh
    
    node_ids = []
    for i, sent in enumerate(sentences):
        sentence_offset = append_sentence(sentences_path, sent)
        embedding = embeddings[i]
        embedding_norm = compute_embedding_norm(embedding)
        
        node_id = i + 1  # 1-based node IDs
        
        # Store in database
        add_node_context(
            node_id=node_id,
            sentence_offset=sentence_offset,
            embedding_index=i,
            source=source_name,
            tag=f"chapter_{sent['chapter']}",
            language="en",
            initial_context=sent["text"][:100],
            embedding_norm=embedding_norm,
        )
        add_node_usage(node_id=node_id)
        
        node_ids.append(node_id)
    
    # Save embeddings
    np.save(embeddings_path, embeddings)
    print(f"  Created {len(node_ids)} nodes in database")
    print(f"  Saved {len(sentences)} sentences to {sentences_path}")
    print(f"  Saved embeddings to {embeddings_path}")

    # Step 3.5: Build FAISS index
    print("\nStep 3.5: Building FAISS index...")
    from memory.faiss_index import build_faiss_index
    from memory.config import DEFAULT_FAISS_INDEX_TYPE
    
    faiss_index_path = os.path.join(data_dir, "faiss.index")
    faiss_index = build_faiss_index(
        embeddings=embeddings,
        index_type=DEFAULT_FAISS_INDEX_TYPE,
        save_path=faiss_index_path,
    )
    print(f"  FAISS index built and saved to {faiss_index_path}")

    # Step 4: Build graph edges
    print("\nStep 4: Building graph edges...")
    edges = build_graph_edges(sentences)
    print(f"  Built {len(edges)} edges")

    # Step 5: Store edges in database and NumPy arrays
    print("\nStep 5: Storing edges...")
    
    # Deduplicate edges (keep first occurrence of each (u, v) pair)
    unique_edges = {}
    for u, v, weight, edge_type in edges:
        key = (u, v)
        if key not in unique_edges:
            unique_edges[key] = (u, v, weight, edge_type)
    
    deduplicated_edges = list(unique_edges.values())
    print(f"  Deduplicated: {len(edges)} -> {len(deduplicated_edges)} edges")
    
    # Store in database
    for u, v, weight, edge_type in deduplicated_edges:
        add_neighbor(u, v, weight, edge_type)
    
    # Store in NumPy arrays
    if deduplicated_edges:
        from memory.config import EDGE_INDEX_FILE, EDGE_WEIGHT_FILE, EDGE_TYPE_FILE
        
        edge_index = np.array([[u, v] for u, v, _, _ in deduplicated_edges], dtype=np.int32).T
        edge_weight = np.array([w for _, _, w, _ in deduplicated_edges], dtype=np.float32)
        edge_type_arr = np.array([t for _, _, _, t in deduplicated_edges], dtype=np.uint8)
        
        np.save(os.path.join(graph_dir, EDGE_INDEX_FILE), edge_index)
        np.save(os.path.join(graph_dir, EDGE_WEIGHT_FILE), edge_weight)
        np.save(os.path.join(graph_dir, EDGE_TYPE_FILE), edge_type_arr)
        
        print(f"  Stored {len(deduplicated_edges)} edges in database")
        print(f"  Saved edge arrays to {graph_dir}/")

    # Return statistics
    return {
        "num_nodes": len(node_ids),
        "num_edges": len(deduplicated_edges),
        "embeddings_shape": embeddings.shape,
        "source": source_name,
        "sentences_path": sentences_path,
        "embeddings_path": embeddings_path,
        "graph_dir": graph_dir,
        "faiss_index_path": faiss_index_path,
        "db_path": db_path,
    }

