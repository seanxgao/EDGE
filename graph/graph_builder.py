"""
Knowledge graph construction for book memory engine.
Builds three types of edges: sequential, paragraph, and semantic.
"""
import time
import numpy as np
from typing import List, Dict, Any, Tuple
from core.config import VERBOSE_TIMING


class GraphBuilder:
    """
    Constructs knowledge graph with three edge types:
    - Sequential: Adjacent sentences
    - Paragraph: Sentences in same paragraph
    - Semantic: High similarity sentences
    """
    
    @staticmethod
    def build_sequential_edges(sentences: List[Dict[str, Any]]) -> List[Tuple[int, int, float]]:
        """
        Build sequential edges from sentence prev/next relationships.
        
        Args:
            sentences: List of sentence nodes
            
        Returns:
            List of (u, v, weight) tuples where weight is 1.0
        """
        edges = []
        
        for sentence in sentences:
            sentence_id = sentence.get('id')
            prev_id = sentence.get('prev_sentence_id')
            next_id = sentence.get('next_sentence_id')
            
            if prev_id is not None:
                edges.append((prev_id, sentence_id, 1.0))
            if next_id is not None:
                edges.append((sentence_id, next_id, 1.0))
        
        if VERBOSE_TIMING:
            print(f"[graph_builder] Built {len(edges)} sequential edges")
        
        return edges
    
    @staticmethod
    def build_paragraph_edges(sentences: List[Dict[str, Any]]) -> List[Tuple[int, int, float]]:
        """
        Build paragraph edges connecting sentences in same paragraph.
        
        Args:
            sentences: List of sentence nodes
            
        Returns:
            List of (u, v, weight) tuples where weight is 1.0
        """
        # Group sentences by (chapter, paragraph_id)
        paragraph_groups: Dict[Tuple[str, int], List[int]] = {}
        
        for sentence in sentences:
            chapter = sentence.get('chapter', '')
            paragraph_id = sentence.get('paragraph_id')
            sentence_id = sentence.get('id')
            
            if paragraph_id is not None:
                key = (chapter, paragraph_id)
                if key not in paragraph_groups:
                    paragraph_groups[key] = []
                paragraph_groups[key].append(sentence_id)
        
        # Connect all sentences within each paragraph
        edges = []
        for group in paragraph_groups.values():
            # Connect each sentence to all others in same paragraph
            for i, u in enumerate(group):
                for v in group[i+1:]:
                    edges.append((u, v, 1.0))
        
        if VERBOSE_TIMING:
            msg = (
                f"[graph_builder] Built {len(edges)} paragraph edges "
                f"from {len(paragraph_groups)} paragraphs"
            )
            print(msg)
        
        return edges
    
    @staticmethod
    def build_semantic_edges(
        sentence_ids: List[int],
        embeddings: np.ndarray,
        similarity_threshold: float = 0.7,
        max_edges_per_node: int = 10
    ) -> List[Tuple[int, int, float]]:
        """
        Build semantic edges based on embedding similarity.
        
        Args:
            sentence_ids: List of sentence IDs (same order as embeddings)
            embeddings: Embedding vectors, shape (n, dim)
            similarity_threshold: Minimum similarity to create edge
            max_edges_per_node: Maximum semantic edges per node
            
        Returns:
            List of (u, v, weight) tuples where weight is similarity score
        """
        if len(sentence_ids) == 0:
            return []
        
        t_start = time.time() if VERBOSE_TIMING else None
        
        # Normalize embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0  # Avoid division by zero
        embeddings_norm = embeddings / norms
        
        # Compute similarity matrix (cosine similarity)
        similarity_matrix = np.dot(embeddings_norm, embeddings_norm.T)
        
        # Remove self-similarity (diagonal)
        np.fill_diagonal(similarity_matrix, 0.0)
        
        edges = []
        id_to_idx = {sid: i for i, sid in enumerate(sentence_ids)}
        
        # For each node, find top similar nodes
        for i, sentence_id in enumerate(sentence_ids):
            similarities = similarity_matrix[i]
            
            # Get top-k similar nodes
            top_indices = np.argsort(similarities)[::-1][:max_edges_per_node]
            
            for idx in top_indices:
                similarity = float(similarities[idx])
                if similarity >= similarity_threshold:
                    target_id = sentence_ids[idx]
                    # Avoid duplicate edges (only add if u < v)
                    if sentence_id < target_id:
                        edges.append((sentence_id, target_id, similarity))
        
        if VERBOSE_TIMING:
            t_elapsed = time.time() - t_start
            msg = (
                f"[graph_builder] Built {len(edges)} semantic edges "
                f"(threshold={similarity_threshold}), duration={t_elapsed:.3f}s"
            )
            print(msg)
        
        return edges
    
    @staticmethod
    def save_edges(edges: List[Tuple[int, int, float]], filepath: str, edge_type: str):
        """
        Save edges to JSONL file.
        
        Args:
            edges: List of (u, v, weight) tuples
            filepath: Path to JSONL file
            edge_type: Edge type ("sequential", "paragraph", or "semantic")
        """
        import json
        from pathlib import Path
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        t_start = time.time() if VERBOSE_TIMING else None
        
        with open(filepath, 'w', encoding='utf-8') as f:
            for u, v, weight in edges:
                edge_data = {
                    "u": u,
                    "v": v,
                    "etype": edge_type,
                    "weight": weight
                }
                json_line = json.dumps(edge_data, ensure_ascii=False)
                f.write(json_line + '\n')
        
        if VERBOSE_TIMING:
            t_elapsed = time.time() - t_start
            msg = (
                f"[graph_builder] Saved {len(edges)} {edge_type} edges "
                f"to {filepath}, duration={t_elapsed:.3f}s"
            )
            print(msg)

