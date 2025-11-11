"""
Three-stage query pipeline for book memory engine.
Stage 1: Vector retrieval (ANN search)
Stage 2: Graph expansion
Stage 3: Fusion ranking
"""
import time
import numpy as np
from typing import List, Dict, Any, Tuple, Set
from core.embedder_minilm import get_embedder
from core.faiss_index import FaissIndex
from storage.vector_store import VectorStore
from storage.sentence_store import SentenceStore
from graph.graph_store import GraphStore
from core.config import VERBOSE_TIMING


class QueryPipeline:
    """
    Three-stage query pipeline for semantic search.
    """
    
    def __init__(
        self,
        faiss_index: FaissIndex,
        vector_store: VectorStore,
        sentence_store: SentenceStore,
        graph_store: GraphStore,
        embedder_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        """
        Initialize query pipeline.
        
        Args:
            faiss_index: Faiss index instance
            vector_store: Vector store instance
            sentence_store: Sentence store instance
            graph_store: Graph store instance
            embedder_model: Embedding model name
        """
        self.faiss_index = faiss_index
        self.vector_store = vector_store
        self.sentence_store = sentence_store
        self.graph_store = graph_store
        self.embedder = get_embedder(model_name=embedder_model)
    
    def stage1_vector_retrieval(
        self,
        query_text: str,
        k: int = 50,
        nprobe: int = 10
    ) -> Tuple[List[int], List[float]]:
        """
        Stage 1: Vector retrieval using ANN search.
        
        Args:
            query_text: Query text
            k: Number of candidates to retrieve
            nprobe: Number of clusters to probe (for IVF-PQ)
            
        Returns:
            Tuple of (candidate_ids, similarity_scores)
        """
        t_start = time.time() if VERBOSE_TIMING else None
        
        # Generate query embedding
        query_embedding = self.embedder.encode(query_text)
        
        # Search Faiss index
        distances, indices = self.faiss_index.search(
            query_embedding.reshape(1, -1),
            k=k,
            nprobe=nprobe
        )
        
        # Convert indices to sentence IDs using rowmap
        # Faiss index position corresponds to row_index in vector_store
        # rowmap: sentence_id -> row_index
        candidate_ids = []
        similarity_scores = []
        
        rowmap = self.vector_store.rowmap
        # Create reverse mapping: row_index -> sentence_id
        row_to_id = {row: sid for sid, row in rowmap.items()}
        
        for idx, dist in zip(indices[0], distances[0]):
            # idx is Faiss index position, which equals row_index
            if idx in row_to_id:
                candidate_ids.append(row_to_id[idx])
                similarity_scores.append(float(dist))
        
        if VERBOSE_TIMING:
            t_elapsed = time.time() - t_start
            msg = (
                f"[query] Stage 1: Retrieved {len(candidate_ids)} candidates, "
                f"duration={t_elapsed:.3f}s"
            )
            print(msg)
        
        return candidate_ids, similarity_scores
    
    def stage2_graph_expansion(
        self,
        candidate_ids: List[int],
        expand_sequential: bool = True,
        expand_paragraph: bool = True,
        expand_semantic: bool = True,
        max_neighbors: int = 5
    ) -> Tuple[List[int], Dict[int, Dict[str, Any]]]:
        """
        Stage 2: Graph expansion to get context.
        
        Args:
            candidate_ids: Candidate sentence IDs from Stage 1
            expand_sequential: Whether to expand sequential neighbors
            expand_paragraph: Whether to expand paragraph neighbors
            expand_semantic: Whether to expand semantic neighbors
            max_neighbors: Maximum neighbors per type per candidate
            
        Returns:
            Tuple of (expanded_ids, node_features)
            - expanded_ids: All unique sentence IDs (candidates + neighbors)
            - node_features: Dictionary mapping sentence_id -> feature dict
        """
        t_start = time.time() if VERBOSE_TIMING else None
        
        expanded_ids: Set[int] = set(candidate_ids)
        node_features: Dict[int, Dict[str, Any]] = {}
        
        # Initialize features for candidates
        for sid in candidate_ids:
            node_features[sid] = {
                "is_candidate": True,
                "sequential_neighbors": [],
                "paragraph_neighbors": [],
                "semantic_neighbors": [],
                "degree": 0
            }
        
        # Expand for each candidate
        for candidate_id in candidate_ids:
            if candidate_id not in self.graph_store.graph:
                continue
            
            neighbors = list(self.graph_store.graph.neighbors(candidate_id))
            node_features[candidate_id]["degree"] = len(neighbors)
            
            # Categorize neighbors by edge type
            sequential_neighbors = []
            paragraph_neighbors = []
            semantic_neighbors = []
            
            for neighbor_id in neighbors:
                edge_data = self.graph_store.graph[candidate_id][neighbor_id]
                etype = edge_data.get('etype', 'semantic')
                
                if etype == 'sequential' and expand_sequential:
                    sequential_neighbors.append(neighbor_id)
                elif etype == 'paragraph' and expand_paragraph:
                    paragraph_neighbors.append(neighbor_id)
                elif etype == 'semantic' and expand_semantic:
                    semantic_neighbors.append(neighbor_id)
            
            # Limit neighbors per type
            sequential_neighbors = sequential_neighbors[:max_neighbors]
            paragraph_neighbors = paragraph_neighbors[:max_neighbors]
            semantic_neighbors = semantic_neighbors[:max_neighbors]
            
            node_features[candidate_id]["sequential_neighbors"] = sequential_neighbors
            node_features[candidate_id]["paragraph_neighbors"] = paragraph_neighbors
            node_features[candidate_id]["semantic_neighbors"] = semantic_neighbors
            
            # Add neighbors to expanded set
            for neighbor_id in sequential_neighbors + paragraph_neighbors + semantic_neighbors:
                expanded_ids.add(neighbor_id)
                if neighbor_id not in node_features:
                    if neighbor_id in self.graph_store.graph:
                        degree = len(list(self.graph_store.graph.neighbors(neighbor_id)))
                    else:
                        degree = 0
                    node_features[neighbor_id] = {
                        "is_candidate": False,
                        "sequential_neighbors": [],
                        "paragraph_neighbors": [],
                        "semantic_neighbors": [],
                        "degree": degree
                    }
        
        if VERBOSE_TIMING:
            t_elapsed = time.time() - t_start
            n_neighbors = len(expanded_ids) - len(candidate_ids)
            msg = (
                f"[query] Stage 2: Expanded to {len(expanded_ids)} nodes "
                f"({n_neighbors} neighbors), duration={t_elapsed:.3f}s"
            )
            print(msg)
        
        return list(expanded_ids), node_features
    
    def stage3_fusion_ranking(
        self,
        candidate_ids: List[int],
        similarity_scores: List[float],
        expanded_ids: List[int],
        node_features: Dict[int, Dict[str, Any]],
        top_k: int = 10,
        weights: Dict[str, float] = None
    ) -> List[Dict[str, Any]]:
        """
        Stage 3: Fusion ranking using multiple features.
        
        Args:
            candidate_ids: Original candidate IDs from Stage 1
            similarity_scores: Similarity scores from Stage 1
            expanded_ids: All expanded IDs from Stage 2
            node_features: Node features from Stage 2
            top_k: Number of final results
            weights: Feature weights for ranking
                - vector_similarity: Weight for vector similarity (default: 0.5)
                - graph_degree: Weight for node degree (default: 0.2)
                - context_coherence: Weight for context coherence (default: 0.3)
            
        Returns:
            List of ranked results, each with:
            - sentence_id: Sentence ID
            - text: Sentence text
            - score: Final ranking score
            - metadata: Additional metadata
        """
        t_start = time.time() if VERBOSE_TIMING else None
        
        # Default weights
        if weights is None:
            weights = {
                "vector_similarity": 0.5,
                "graph_degree": 0.2,
                "context_coherence": 0.3
            }
        
        # Build score map for candidates
        candidate_scores = {
            sid: score for sid, score in zip(candidate_ids, similarity_scores)
        }
        
        # Compute final scores
        final_scores = []
        
        for sid in expanded_ids:
            # Get sentence data
            sentence = self.sentence_store.get_sentence(sid)
            if sentence is None:
                continue
            
            features = node_features.get(sid, {})
            
            # Feature 1: Vector similarity
            vector_score = candidate_scores.get(sid, 0.0)
            
            # Feature 2: Graph degree (normalized)
            max_degree = max([f.get("degree", 0) for f in node_features.values()], default=1)
            degree_score = features.get("degree", 0) / max_degree if max_degree > 0 else 0.0
            
            # Feature 3: Context coherence
            # Check if neighbors are also candidates (indicates coherent context)
            neighbors = (
                features.get("sequential_neighbors", []) +
                features.get("paragraph_neighbors", []) +
                features.get("semantic_neighbors", [])
            )
            neighbor_candidates = sum(1 for nid in neighbors if nid in candidate_scores)
            coherence_score = neighbor_candidates / len(neighbors) if neighbors else 0.0
            
            # Weighted combination
            final_score = (
                weights["vector_similarity"] * vector_score +
                weights["graph_degree"] * degree_score +
                weights["context_coherence"] * coherence_score
            )
            
            final_scores.append({
                "sentence_id": sid,
                "text": sentence.get("text", ""),
                "chapter": sentence.get("chapter", ""),
                "paragraph_id": sentence.get("paragraph_id"),
                "sentence_id_in_para": sentence.get("sentence_id"),
                "score": final_score,
                "vector_similarity": vector_score,
                "graph_degree": features.get("degree", 0),
                "context_coherence": coherence_score,
                "metadata": sentence.get("metadata", {})
            })
        
        # Sort by final score
        final_scores.sort(key=lambda x: x["score"], reverse=True)
        
        # Return top-k
        results = final_scores[:top_k]
        
        if VERBOSE_TIMING:
            t_elapsed = time.time() - t_start
            msg = (
                f"[query] Stage 3: Ranked {len(final_scores)} nodes, "
                f"returned top {len(results)}, duration={t_elapsed:.3f}s"
            )
            print(msg)
        
        return results
    
    def query(
        self,
        query_text: str,
        top_k: int = 10,
        initial_k: int = 50,
        expand_sequential: bool = True,
        expand_paragraph: bool = True,
        expand_semantic: bool = True,
        ranking_weights: Dict[str, float] = None
    ) -> List[Dict[str, Any]]:
        """
        Complete three-stage query pipeline.
        
        Args:
            query_text: Query text
            top_k: Number of final results
            initial_k: Number of candidates from Stage 1
            expand_sequential: Whether to expand sequential neighbors
            expand_paragraph: Whether to expand paragraph neighbors
            expand_semantic: Whether to expand semantic neighbors
            ranking_weights: Feature weights for ranking
            
        Returns:
            List of ranked results
        """
        t_start = time.time() if VERBOSE_TIMING else None
        
        # Stage 1: Vector retrieval
        candidate_ids, similarity_scores = self.stage1_vector_retrieval(
            query_text,
            k=initial_k
        )
        
        if not candidate_ids:
            return []
        
        # Stage 2: Graph expansion
        expanded_ids, node_features = self.stage2_graph_expansion(
            candidate_ids,
            expand_sequential=expand_sequential,
            expand_paragraph=expand_paragraph,
            expand_semantic=expand_semantic
        )
        
        # Stage 3: Fusion ranking
        results = self.stage3_fusion_ranking(
            candidate_ids,
            similarity_scores,
            expanded_ids,
            node_features,
            top_k=top_k,
            weights=ranking_weights
        )
        
        if VERBOSE_TIMING:
            t_elapsed = time.time() - t_start
            print(f"[query] Complete pipeline duration={t_elapsed:.3f}s")
        
        return results

