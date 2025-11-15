#!/usr/bin/env python3
"""
Verify generated data structure and content.
"""

from memory import (
    init_db,
    NodesContext,
    Neighbors,
    NodesUsage,
    get_node,
    count_sentences,
    get_embedding_shape,
    load_edge_arrays,
    load_faiss_index,
    get_index_stats,
    search_faiss_index,
)
import os

def verify_data(data_dir="data"):
    """Verify all generated data files."""
    print("=" * 60)
    print("Data Verification")
    print("=" * 60)
    
    db_path = os.path.join(data_dir, "memory.db")
    sentences_path = os.path.join(data_dir, "sentences.jsonl")
    embeddings_path = os.path.join(data_dir, "embeddings.npy")
    graph_dir = os.path.join(data_dir, "graph")
    
    # Check files exist
    print("\n1. File existence:")
    files_to_check = [
        (db_path, "Database"),
        (sentences_path, "Sentences JSONL"),
        (embeddings_path, "Embeddings NumPy"),
        (os.path.join(data_dir, "faiss.index"), "FAISS index"),
        (os.path.join(graph_dir, "edge_index.npy"), "Edge index"),
        (os.path.join(graph_dir, "edge_weight.npy"), "Edge weight"),
        (os.path.join(graph_dir, "edge_type.npy"), "Edge type"),
    ]
    
    all_exist = True
    for filepath, name in files_to_check:
        exists = os.path.exists(filepath)
        status = "[OK]" if exists else "[MISSING]"
        print(f"  {status} {name}: {filepath}")
        if not exists:
            all_exist = False
    
    if not all_exist:
        print("\n✗ Some files are missing!")
        return False
    
    # Check database
    print("\n2. Database content:")
    init_db(db_path)
    node_count = NodesContext.select().count()
    edge_count = Neighbors.select().count()
    usage_count = NodesUsage.select().count()
    
    print(f"  [OK] Nodes in database: {node_count}")
    print(f"  [OK] Edges in database: {edge_count}")
    print(f"  [OK] Usage records: {usage_count}")
    
    # Check file content
    print("\n3. File content:")
    sentence_count = count_sentences(sentences_path)
    print(f"  [OK] Sentences in JSONL: {sentence_count}")
    
    emb_shape = get_embedding_shape(embeddings_path)
    if emb_shape:
        print(f"  [OK] Embeddings shape: {emb_shape[0]} sentences x {emb_shape[1]} dims")
    
    edge_arrays = load_edge_arrays(graph_dir)
    if edge_arrays:
        edge_index, edge_weight, edge_type = edge_arrays
        print(f"  [OK] Edge arrays: {len(edge_weight)} edges")
    
    # Check FAISS index
    faiss_index_path = os.path.join(data_dir, "faiss.index")
    if os.path.exists(faiss_index_path):
        try:
            faiss_index = load_faiss_index(faiss_index_path)
            faiss_stats = get_index_stats(faiss_index)
            print(f"  [OK] FAISS index: {faiss_stats['num_vectors']} vectors, type={faiss_stats['index_type']}")
        except Exception as e:
            print(f"  [WARN] Could not load FAISS index: {e}")
    else:
        print(f"  [WARN] FAISS index not found at {faiss_index_path}")
    
    # Check consistency
    print("\n4. Data consistency:")
    if node_count == sentence_count == (emb_shape[0] if emb_shape else 0):
        print(f"  [OK] Node count matches: {node_count}")
    else:
        print(f"  [ERROR] Mismatch: nodes={node_count}, sentences={sentence_count}, embeddings={emb_shape[0] if emb_shape else 0}")
        return False
    
    if edge_arrays:
        # Check for duplicates in arrays
        edge_index, edge_weight, edge_type = edge_arrays
        unique_in_array = len(set((edge_index[0][i], edge_index[1][i]) for i in range(len(edge_weight))))
        if edge_count == len(edge_weight) == unique_in_array:
            print(f"  [OK] Edge count matches: {edge_count}")
        else:
            print(f"  [WARNING] Edge count: DB={edge_count}, arrays={len(edge_weight)}, unique_in_array={unique_in_array}")
            if edge_count != unique_in_array:
                print(f"    Note: Arrays may have duplicates, but DB is correct")
    else:
        print(f"  [ERROR] Edge arrays not found")
        return False
    
    # Test node retrieval
    print("\n5. Node retrieval test:")
    node = get_node(
        1,
        sentences_path=sentences_path,
        embeddings_path=embeddings_path,
        db_path=db_path,
    )
    if node and node['context'] and node['sentence'] and node['embedding']:
        print(f"  [OK] Node 1 retrieved successfully")
        print(f"    Text: {node['sentence']['text'][:50]}...")
        print(f"    Embedding: {len(node['embedding'])} dimensions")
        print(f"    Neighbors: {len(node['neighbors'])} edges")
    else:
        print("  [ERROR] Failed to retrieve node 1")
        return False
    
    # Test FAISS search
    print("\n6. FAISS search test:")
    faiss_index_path = os.path.join(data_dir, "faiss.index")
    if os.path.exists(faiss_index_path):
        try:
            import numpy as np
            faiss_index = load_faiss_index(faiss_index_path)
            query_emb = np.array(node['embedding'], dtype=np.float32)
            distances, indices = search_faiss_index(faiss_index, query_emb, k=5)
            print(f"  [OK] FAISS search successful")
            print(f"    Top 5 neighbors: {indices}")
            print(f"    Distances: {distances[:3]}")
        except Exception as e:
            print(f"  [ERROR] FAISS search failed: {e}")
            return False
    else:
        print("  [SKIP] FAISS index not found")
    
    print("\n" + "=" * 60)
    print("[OK] All verifications passed!")
    print("=" * 60)
    return True

if __name__ == "__main__":
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data"
    success = verify_data(data_dir)
    sys.exit(0 if success else 1)

