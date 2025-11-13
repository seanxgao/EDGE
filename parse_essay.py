#!/usr/bin/env python3
"""
Parse essay text file and build structured memory dataset.
"""

import json
import os
import re


def compute_statistics(sentences, lines):
    """
    Compute sentence-level statistics.
    
    Args:
        sentences: List of sentence dicts with id, chapter, position, text
        lines: Original lines from file (to detect blank lines)
    
    Returns:
        List of statistics dicts
    """
    # Build chapter length map
    chapter_lengths = {}
    for sent in sentences:
        ch = sent["chapter"]
        chapter_lengths[ch] = chapter_lengths.get(ch, 0) + 1
    
    # Build blank line map for paragraph detection
    blank_line_map = {}
    for i, line in enumerate(lines):
        if not line.strip():
            blank_line_map[i] = True
    
    # Map sentence index to original line index
    line_idx = 0
    sentence_to_line = {}
    for sent_idx, sent in enumerate(sentences):
        while line_idx < len(lines):
            if lines[line_idx].strip():
                sentence_to_line[sent_idx] = line_idx
                line_idx += 1
                break
            line_idx += 1
    
    statistics = []
    
    for i, sent in enumerate(sentences):
        text = sent["text"]
        chapter = sent["chapter"]
        position = sent["position"]
        
        # Basic length features
        len_char = len(text)
        tokens = text.split()
        len_tok = len(tokens)
        
        # Question detection
        is_question = 1 if text.strip().endswith("?") else 0
        
        # Definition-like detection (case-insensitive)
        text_lower = text.lower()
        is_definition_like = 0
        if " is " in text_lower or " means " in text_lower or " refers to " in text_lower:
            is_definition_like = 1
        
        # Punctuation density
        punct_chars = len(re.findall(r'[.,;:!?\-—()\[\]{}"\']', text))
        punct_density = punct_chars / len_char if len_char > 0 else 0.0
        
        # Relative position in chapter
        chapter_length = chapter_lengths[chapter]
        rel_pos_in_chapter = position / chapter_length if chapter_length > 0 else 0.0
        
        # Paragraph start/end detection
        is_paragraph_start = 0
        is_paragraph_end = 0
        
        if position == 1:
            is_paragraph_start = 1
        else:
            # Check if previous line was blank
            sent_line_idx = sentence_to_line.get(i, -1)
            if sent_line_idx > 0 and (sent_line_idx - 1) in blank_line_map:
                is_paragraph_start = 1
        
        if position == chapter_length:
            is_paragraph_end = 1
        else:
            # Check if next line is blank
            sent_line_idx = sentence_to_line.get(i, -1)
            if sent_line_idx >= 0 and (sent_line_idx + 1) < len(lines):
                if not lines[sent_line_idx + 1].strip():
                    is_paragraph_end = 1
        
        stat = {
            "id": sent["id"],
            "chapter": chapter,
            "position": position,
            "len_char": len_char,
            "len_tok": len_tok,
            "is_question": is_question,
            "is_definition_like": is_definition_like,
            "punct_density": round(punct_density, 4),
            "rel_pos_in_chapter": round(rel_pos_in_chapter, 4),
            "chapter_length": chapter_length,
            "is_paragraph_start": is_paragraph_start,
            "is_paragraph_end": is_paragraph_end,
            "local_semantic_density": None
        }
        statistics.append(stat)
    
    return statistics


def build_edges(sentences, statistics):
    """
    Build edge relationships between sentences.
    
    Args:
        sentences: List of sentence dicts
        statistics: List of statistics dicts
    
    Returns:
        List of edge dicts
    """
    edges = []
    
    # Create index maps
    id_to_idx = {sent["id"]: i for i, sent in enumerate(sentences)}
    stat_map = {stat["id"]: stat for stat in statistics}
    
    # (a) Adjacent sentences within same chapter
    for i in range(len(sentences) - 1):
        curr = sentences[i]
        next_sent = sentences[i + 1]
        
        if curr["chapter"] == next_sent["chapter"]:
            edge = {
                "u": curr["id"],
                "v": next_sent["id"],
                "type": "adjacent",
                "weight": 1.0
            }
            edges.append(edge)
    
    # (b) Same chapter connections (all pairs within chapter)
    chapter_groups = {}
    for i, sent in enumerate(sentences):
        ch = sent["chapter"]
        if ch not in chapter_groups:
            chapter_groups[ch] = []
        chapter_groups[ch].append(sent["id"])
    
    for ch, ids in chapter_groups.items():
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                edge = {
                    "u": ids[i],
                    "v": ids[j],
                    "type": "same_chapter",
                    "weight": 1.0
                }
                edges.append(edge)
    
    # (c) Definition context edges
    for i, stat in enumerate(statistics):
        if stat["is_definition_like"] == 1:
            sent_id = stat["id"]
            sent_idx = id_to_idx[sent_id]
            
            # Link to previous sentence (if exists and same chapter)
            if sent_idx > 0:
                prev_sent = sentences[sent_idx - 1]
                if prev_sent["chapter"] == sentences[sent_idx]["chapter"]:
                    edge = {
                        "u": prev_sent["id"],
                        "v": sent_id,
                        "type": "definition_context",
                        "weight": 0.8
                    }
                    edges.append(edge)
            
            # Link to next sentence (if exists and same chapter)
            if sent_idx < len(sentences) - 1:
                next_sent = sentences[sent_idx + 1]
                if next_sent["chapter"] == sentences[sent_idx]["chapter"]:
                    edge = {
                        "u": sent_id,
                        "v": next_sent["id"],
                        "type": "definition_context",
                        "weight": 0.8
                    }
                    edges.append(edge)
    
    # (d) Question context edges
    for i, stat in enumerate(statistics):
        if stat["is_question"] == 1:
            sent_id = stat["id"]
            sent_idx = id_to_idx[sent_id]
            chapter = sentences[sent_idx]["chapter"]
            
            # Link to next two sentences (if exist and same chapter)
            for offset in [1, 2]:
                if sent_idx + offset < len(sentences):
                    next_sent = sentences[sent_idx + offset]
                    if next_sent["chapter"] == chapter:
                        edge = {
                            "u": sent_id,
                            "v": next_sent["id"],
                            "type": "question_context",
                            "weight": 0.9
                        }
                        edges.append(edge)
                    else:
                        break
    
    return edges


def parse_essay_file(input_file: str, output_dir: str):
    """
    Parse essay file and generate structured outputs.
    
    Args:
        input_file: Path to input text file
        output_dir: Output directory for generated files
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Read input file
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Parse sentences and chapters
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
        sentence_id = f"bacon_{sentence_index + 1:03d}"
        sentence_data = {
            "id": sentence_id,
            "chapter": current_chapter,
            "position": current_position,
            "text": stripped
        }
        sentences.append(sentence_data)
        
        current_position += 1
        sentence_index += 1
    
    # Create subdirectories
    core_dir = os.path.join(output_dir, "core")
    os.makedirs(core_dir, exist_ok=True)
    
    # Write sentences.jsonl
    sentences_path = os.path.join(core_dir, "sentences.jsonl")
    with open(sentences_path, 'w', encoding='utf-8') as f:
        for sentence in sentences:
            f.write(json.dumps(sentence, ensure_ascii=False) + '\n')
    
    # Compute and write statistics.jsonl
    statistics = compute_statistics(sentences, lines)
    statistics_path = os.path.join(core_dir, "statistics.jsonl")
    with open(statistics_path, 'w', encoding='utf-8') as f:
        for stat in statistics:
            f.write(json.dumps(stat, ensure_ascii=False) + '\n')
    
    # Note: edges.jsonl is no longer generated (replaced by NumPy arrays)
    # Build edges for verification only
    edges = build_edges(sentences, statistics)
    
    # Print verification
    print("First 3 lines of sentences.jsonl:")
    print("-" * 60)
    with open(sentences_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 3:
                break
            print(line.rstrip())
    
    print("\nFirst 3 lines of statistics.jsonl:")
    print("-" * 60)
    with open(statistics_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 3:
                break
            print(line.rstrip())
    
    print(f"\nTotal sentences: {len(sentences)}")
    print(f"Total statistics: {len(statistics)}")
    print(f"Total edges: {len(edges)} (use convert_edges_to_numpy.py to generate NumPy arrays)")
    print(f"\nOutput files created in '{output_dir}/core/' directory.")


if __name__ == "__main__":
    input_file = "3essay.txt"
    output_dir = "memory_bacon"
    
    parse_essay_file(input_file, output_dir)
