"""
High-level API for graph-based memory system
"""
import time
import numpy as np
from typing import List, Dict, Any
from .graph_store import GraphStore
from core.embedder import get_embedding
from core.retriever import find_top_k
from core.config import DEFAULT_K_NEIGHBORS, DEFAULT_RETRIEVAL_K, VERBOSE_TIMING


class GraphMemory:
    """
    High-level API for adding/retrieving/updating memories.
    """
    
    def __init__(self, filepath: str = None, scope_integration: bool = False):
        """
        Initialize graph memory system.
        
        Args:
            filepath: Path to memory file (default: from config)
            scope_integration: Enable SCOPE integration hooks (default: False)
        """
        self.store = GraphStore(filepath)
        self.k_neighbors = DEFAULT_K_NEIGHBORS
        self.scope_integration = scope_integration
    
    def add_memory(self, text: str) -> int:
        """
        Embed text, add as node, connect to k nearest neighbors.
        
        Args:
            text: Memory text content
            
        Returns:
            New memory node ID
            
        Raises:
            ValueError: If text is empty
            RuntimeError: If embedding or storage fails
        """
        t_start = time.time() if VERBOSE_TIMING else None
        
        # Get embedding
        embedding = get_embedding(text)
        
        # Add node
        node_id = self.store.add_node(text, embedding)
        
        # Connect to similar existing memories
        if len(self.store.nodes) > 1:
            self._connect_to_similar_memories(node_id, embedding)
        
        # Apply SCOPE integration if enabled
        if self.scope_integration:
            self._apply_scope_organization(node_id, embedding)
        
        # Save data
        self.store.save_data()
        
        if VERBOSE_TIMING:
            t_elapsed = time.time() - t_start
            print(f"[add_memory] duration={t_elapsed:.3f}s, node_id={node_id}")
        
        return node_id
    
    def retrieve_memories(self, query: str, k: int = None) -> List[str]:
        """
        Return top-k related memory texts.
        
        Args:
            query: Query text
            k: Number of memories to return (default from config)
            
        Returns:
            List of related memory texts
        """
        if k is None:
            k = DEFAULT_RETRIEVAL_K
        
        if not self.store.nodes:
            return []
        
        t_start = time.time() if VERBOSE_TIMING else None
        
        # Get query embedding
        query_embedding = get_embedding(query)
        
        # Get all memory embeddings
        memory_embeddings = self.store.get_all_embeddings()
        memory_ids = self.store.get_all_node_ids()
        
        # Find top k similar memories
        top_indices = find_top_k(query_embedding, memory_embeddings, k)
        
        # Return memory texts
        result = [self.store.get_node_text(memory_ids[i]) for i in top_indices]
        
        if VERBOSE_TIMING:
            t_elapsed = time.time() - t_start
            print(f"[retrieve_memories] duration={t_elapsed:.3f}s, k={k}, found={len(result)}")
        
        return result
    
    def show_stats(self) -> Dict[str, Any]:
        """
        Return node/edge counts and graph density.
        
        Returns:
            Dictionary with graph statistics
        """
        stats = self.store.get_stats()
        stats["filepath"] = self.store.filepath
        return stats
    
    def _connect_to_similar_memories(self, new_node_id: int, new_embedding: np.ndarray):
        """Connect new memory to k most similar existing memories"""
        if len(self.store.nodes) <= 1:
            return
        
        # Get all existing embeddings and IDs (excluding the new node)
        existing_ids = [node_id for node_id in self.store.get_all_node_ids() if node_id != new_node_id]
        existing_embeddings = [self.store.get_node_embedding(node_id) for node_id in existing_ids]
        
        # Find top k similar memories
        top_indices = find_top_k(new_embedding, existing_embeddings, self.k_neighbors)
        
        # Add edges with similarity weights
        for idx in top_indices:
            target_id = existing_ids[idx]
            similarity = np.dot(new_embedding, existing_embeddings[idx])
            self.store.add_edge(new_node_id, target_id, weight=similarity)
    
    def _apply_scope_organization(self, node_id: int, embedding: np.ndarray):
        """
        Apply SCOPE organization to new memory node.
        
        This method is called after adding a memory to integrate
        with SCOPE's spectral clustering and geometric organization.
        
        Args:
            node_id: Newly added node ID
            embedding: Node embedding vector
        """
        if not self.scope_integration:
            return
        
        # Hook for SCOPE integration:
        # - Can use SCOPE's spectral clustering to assign node to cluster
        # - Can use SCOPE's geometric structure to optimize edge weights
        # - Can apply SCOPE's optimal transport for embedding updates
        pass
    
    def apply_spectral_smoothing(self):
        """
        Apply spectral smoothing using SCOPE.
        
        Uses SCOPE's spectral analysis to smooth graph structure.
        This method will integrate with SCOPE's spectral clustering
        and Laplacian-based smoothing when SCOPE is available.
        """
        if not self.scope_integration:
            raise RuntimeError("SCOPE integration not enabled")
        # TODO: Integrate with SCOPE's spectral smoothing
        pass
    
    def apply_optimal_transport_update(self):
        """
        Apply optimal transport updates using SCOPE.
        
        Uses SCOPE's optimal transport to update embeddings
        based on graph topology. This method will integrate
        with SCOPE's OT module when available.
        """
        if not self.scope_integration:
            raise RuntimeError("SCOPE integration not enabled")
        # TODO: Integrate with SCOPE's optimal transport
        pass
    
    def get_graph_data(self) -> Dict[str, Any]:
        """
        Get graph data for SCOPE integration.
        
        Returns graph structure and embeddings for SCOPE's
        spectral analysis and geometric organization.
        
        Returns:
            Dictionary with graph structure and embeddings
        """
        return {
            "nodes": {
                node_id: {
                    "text": self.store.get_node_text(node_id),
                    "embedding": self.store.get_node_embedding(node_id).tolist()
                }
                for node_id in self.store.get_all_node_ids()
            },
            "edges": [
                (u, v, self.store.graph[u][v].get("weight", 1.0))
                for u, v in self.store.graph.edges()
            ],
            "graph": self.store.graph
        }
