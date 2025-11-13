#!/usr/bin/env python3
"""
Generate sentence embeddings using OpenAI's text-embedding-3-small model.
"""

import json
import os
import time
import numpy as np
from openai import OpenAI


def load_sentences(sentences_path):
    """Load all sentences from JSONL file."""
    sentences = []
    with open(sentences_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            sentences.append(data["text"])
    return sentences


def get_api_key():
    """Get OpenAI API key from environment variable or file."""
    # Try environment variable first
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        return api_key
    
    # Fallback to file (project default)
    default_key_path = r"H:\API\openai_api.txt"
    if os.path.exists(default_key_path):
        try:
            with open(default_key_path, 'r', encoding='utf-8') as f:
                api_key = f.read().strip()
            if api_key:
                return api_key
        except Exception as e:
            print(f"Warning: Failed to read API key from file: {e}")
    
    raise ValueError(
        "OPENAI_API_KEY not found. "
        "Please set OPENAI_API_KEY environment variable or ensure API key file exists."
    )


def generate_embeddings(sentences, batch_size=100, max_retries=3):
    """
    Generate embeddings for all sentences using OpenAI API.
    
    Args:
        sentences: List of sentence texts
        batch_size: Number of texts per API call (max 100)
        max_retries: Maximum number of retries for failed API calls
    
    Returns:
        List of embedding vectors
    """
    # Initialize OpenAI client
    api_key = get_api_key()
    client = OpenAI(api_key=api_key)
    
    all_embeds = []
    total_batches = (len(sentences) + batch_size - 1) // batch_size
    
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i+batch_size]
        batch_num = i // batch_size + 1
        
        # Retry logic
        for attempt in range(max_retries):
            try:
                resp = client.embeddings.create(
                    model="text-embedding-3-small",
                    input=batch
                )
                
                # Extract embeddings from response
                batch_embeds = [d.embedding for d in resp.data]
                all_embeds.extend(batch_embeds)
                
                print(f"Processed batch {batch_num}/{total_batches} ({len(batch)} sentences)")
                break
                
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 2  # Exponential backoff
                    print(f"Error in batch {batch_num}: {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    raise RuntimeError(f"Failed to process batch {batch_num} after {max_retries} attempts: {e}")
    
    return all_embeds


def main():
    """Main function to generate and save embeddings."""
    sentences_path = "memory_bacon/core/sentences.jsonl"
    core_dir = "memory_bacon/core"
    meta_dir = "memory_bacon/meta"
    
    # Create directories if needed
    os.makedirs(core_dir, exist_ok=True)
    os.makedirs(meta_dir, exist_ok=True)
    
    # Load sentences
    print("Loading sentences...")
    sentences = load_sentences(sentences_path)
    print(f"Loaded {len(sentences)} sentences")
    
    # Generate embeddings
    print("\nGenerating embeddings using OpenAI API...")
    all_embeds = generate_embeddings(sentences, batch_size=100)
    
    # Convert to NumPy array
    embeddings_array = np.array(all_embeds, dtype=np.float32)
    print(f"\nGenerated embeddings shape: {embeddings_array.shape}")
    
    # Save embeddings
    embeddings_path = os.path.join(core_dir, "embeddings.npy")
    np.save(embeddings_path, embeddings_array)
    print(f"Saved embeddings to {embeddings_path}")
    
    # Save metadata
    meta = {
        "model": "text-embedding-3-small",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "num_sentences": len(sentences),
        "embedding_dim": len(all_embeds[0]) if all_embeds else 0
    }
    
    meta_path = os.path.join(meta_dir, "embedding_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved metadata to {meta_path}")
    
    print("\nEmbeddings generation completed successfully!")
    print(f"  - Model: {meta['model']}")
    print(f"  - Sentences: {meta['num_sentences']}")
    print(f"  - Embedding dimension: {meta['embedding_dim']}")


if __name__ == "__main__":
    main()

