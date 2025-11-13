#!/usr/bin/env python3
"""
Convert edge information to efficient NumPy adjacency-list format.
"""

import json
import os
import numpy as np


def load_sentences(sentences_path):
    """Load sentences from JSONL file."""
    sentences = []
    with open(sentences_path, 'r', encoding='utf-8') as f:
        for line in f:
            sentences.append(json.loads(line))
    return sentences


def load_id_map(id_map_path):
    """Load ID to index mapping."""
    with open(id_map_path, 'r', encoding='utf-8') as f:
        id_map = json.load(f)
    # Convert to id -> index mapping
    id2idx = {v: int(k) for k, v in id_map.items()}
    return id2idx


def build_edges(sentences, id2idx):
    """
    Build edge list from sentences.
    
    Returns:
        List of (src_id, dst_id, weight, type_name) tuples
    """
    edge_list = []
    
    # Create index to sentence mapping
    idx2sent = {id2idx[sent["id"]]: sent for sent in sentences}
    
    # Group sentences by chapter
    chapter_groups = {}
    for sent in sentences:
        ch = sent["chapter"]
        if ch not in chapter_groups:
            chapter_groups[ch] = []
        chapter_groups[ch].append(sent)
    
    # Sort sentences within each chapter by position
    for ch in chapter_groups:
        chapter_groups[ch].sort(key=lambda x: x["position"])
    
    # Edge type mapping
    edge_type_map = {
        "adjacent": 0,
        "same_chapter": 1,
        "question_context": 2,
        "definition_context": 3,
        # "semantic_knn": 4,  # placeholder for future use
    }
    
    # Build edges for each sentence
    for sent in sentences:
        sent_id = sent["id"]
        sent_idx = id2idx[sent_id]
        chapter = sent["chapter"]
        position = sent["position"]
        text = sent["text"]
        
        chapter_sents_sorted = chapter_groups[chapter]
        
        # Find current sentence index in sorted list
        current_idx = position - 1  # position is 1-indexed, list is 0-indexed
        
        # (a) Adjacent edges: next sentence in same chapter
        if current_idx + 1 < len(chapter_sents_sorted):
            next_sent = chapter_sents_sorted[current_idx + 1]
            next_id = next_sent["id"]
            edge_list.append((sent_id, next_id, 1.0, "adjacent"))
        
        # (b) Question context: if ends with "?", link to next 2 sentences
        if text.strip().endswith("?"):
            for offset in [1, 2]:
                if current_idx + offset < len(chapter_sents_sorted):
                    next_sent = chapter_sents_sorted[current_idx + offset]
                    next_id = next_sent["id"]
                    edge_list.append((sent_id, next_id, 0.9, "question_context"))
        
        # (c) Definition context: if contains definition patterns, link to prev and next
        text_lower = text.lower()
        is_definition = False
        if " is " in text_lower or " means " in text_lower or " refers to " in text_lower:
            is_definition = True
        
        if is_definition:
            # Link to previous sentence (if exists and same chapter)
            if current_idx > 0:
                prev_sent = chapter_sents_sorted[current_idx - 1]
                prev_id = prev_sent["id"]
                edge_list.append((prev_id, sent_id, 0.8, "definition_context"))
            
            # Link to next sentence (if exists and same chapter)
            if current_idx + 1 < len(chapter_sents_sorted):
                next_sent = chapter_sents_sorted[current_idx + 1]
                next_id = next_sent["id"]
                edge_list.append((sent_id, next_id, 0.8, "definition_context"))
    
    return edge_list


def convert_to_numpy(edge_list, id2idx, graph_dir):
    """
    Convert edge list to NumPy arrays and save.
    
    Args:
        edge_list: List of (src_id, dst_id, weight, type_name) tuples
        id2idx: Dictionary mapping sentence ID to index
        graph_dir: Graph output directory
    """
    edge_type_map = {
        "adjacent": 0,
        "same_chapter": 1,
        "question_context": 2,
        "definition_context": 3,
        # "semantic_knn": 4,  # placeholder for future use
    }
    
    rows = []
    cols = []
    weights = []
    types = []
    
    for src_id, dst_id, weight, type_name in edge_list:
        src_idx = id2idx[src_id]
        dst_idx = id2idx[dst_id]
        type_code = edge_type_map[type_name]
        
        rows.append(src_idx)
        cols.append(dst_idx)
        weights.append(weight)
        types.append(type_code)
    
    # Convert to NumPy arrays
    edge_index = np.vstack([rows, cols])
    edge_weight = np.array(weights, dtype=np.float32)
    edge_type = np.array(types, dtype=np.uint8)
    
    # Save arrays
    os.makedirs(graph_dir, exist_ok=True)
    np.save(os.path.join(graph_dir, "edge_index.npy"), edge_index)
    np.save(os.path.join(graph_dir, "edge_weight.npy"), edge_weight)
    np.save(os.path.join(graph_dir, "edge_type.npy"), edge_type)
    
    return edge_index, edge_weight, edge_type, edge_list


def main():
    """Main conversion function."""
    sentences_path = "memory_bacon/core/sentences.jsonl"
    id_map_path = "memory_bacon/core/id_map.json"
    graph_dir = "memory_bacon/graph"
    
    # Load data
    print("Loading sentences and ID map...")
    sentences = load_sentences(sentences_path)
    id2idx = load_id_map(id_map_path)
    
    print(f"Loaded {len(sentences)} sentences")
    print(f"ID map size: {len(id2idx)}")
    
    # Build edges
    print("\nBuilding edges...")
    edge_list = build_edges(sentences, id2idx)
    print(f"Total edges: {len(edge_list)}")
    
    # Convert to NumPy
    print("\nConverting to NumPy arrays...")
    edge_index, edge_weight, edge_type, _ = convert_to_numpy(edge_list, id2idx, graph_dir)
    
    # Verification
    print("\n" + "=" * 60)
    print("Verification:")
    print("=" * 60)
    print(f"Number of total edges: {len(edge_weight)}")
    print(f"Edge index shape: {edge_index.shape}")
    print(f"Edge weight shape: {edge_weight.shape}")
    print(f"Edge type shape: {edge_type.shape}")
    
    # Show first 5 edges
    print("\nFirst 5 edges (src_id, dst_id, type_name, weight):")
    print("-" * 60)
    edge_type_map = {
        0: "adjacent",
        1: "same_chapter",
        2: "question_context",
        3: "definition_context",
    }
    
    # Create reverse ID mapping
    idx2id = {v: k for k, v in id2idx.items()}
    
    for i in range(min(5, len(edge_list))):
        src_id, dst_id, weight, type_name = edge_list[i]
        print(f"  {src_id} -> {dst_id}, type={type_name}, weight={weight}")
    
    print(f"\nArrays saved to '{graph_dir}/' directory:")
    print(f"  - edge_index.npy")
    print(f"  - edge_weight.npy")
    print(f"  - edge_type.npy")


if __name__ == "__main__":
    main()

