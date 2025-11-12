#!/usr/bin/env python3
"""
Parse essay text file and build structured memory dataset.
"""

import json
import os
from pathlib import Path


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
    
    # Write sentences.jsonl
    sentences_path = os.path.join(output_dir, "sentences.jsonl")
    with open(sentences_path, 'w', encoding='utf-8') as f:
        for sentence in sentences:
            f.write(json.dumps(sentence, ensure_ascii=False) + '\n')
    
    # Write id_map.json
    id_map = {str(i): sentence["id"] for i, sentence in enumerate(sentences)}
    id_map_path = os.path.join(output_dir, "id_map.json")
    with open(id_map_path, 'w', encoding='utf-8') as f:
        json.dump(id_map, f, ensure_ascii=False, indent=2)
    
    # Calculate statistics
    num_chapters = current_chapter
    num_sentences = len(sentences)
    avg_sentences = num_sentences / num_chapters if num_chapters > 0 else 0
    
    # Write meta_summary.txt
    summary_path = os.path.join(output_dir, "meta_summary.txt")
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(f"Number of chapters: {num_chapters}\n")
        f.write(f"Number of total sentences: {num_sentences}\n")
        f.write(f"Average sentences per chapter: {avg_sentences:.2f}\n")
    
    # Print verification
    print("First 5 lines of sentences.jsonl:")
    print("-" * 60)
    with open(sentences_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 5:
                break
            print(line.rstrip())
    
    print("\nMeta summary:")
    print("-" * 60)
    with open(summary_path, 'r', encoding='utf-8') as f:
        print(f.read())
    
    return sentences, id_map, num_chapters, num_sentences, avg_sentences


if __name__ == "__main__":
    input_file = "3essay.txt"
    output_dir = "memory_bacon"
    
    parse_essay_file(input_file, output_dir)
    print(f"\nOutput files created in '{output_dir}/' directory.")

