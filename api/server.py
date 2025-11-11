"""
FastAPI server for book memory engine.
Provides REST API for semantic search.
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import time
from core.faiss_index import FaissIndex
from storage.vector_store import VectorStore
from storage.sentence_store import SentenceStore
from graph.graph_store import GraphStore
from pipeline.query import QueryPipeline
from core.config import VERBOSE_TIMING


app = FastAPI(title="EDGE Book Memory Engine", version="1.0.0")


# Global pipeline instance (initialized on startup)
query_pipeline: Optional[QueryPipeline] = None


class QueryRequest(BaseModel):
    """Query request model"""
    query: str
    top_k: int = 10
    initial_k: int = 50
    expand_sequential: bool = True
    expand_paragraph: bool = True
    expand_semantic: bool = True


class QueryResponse(BaseModel):
    """Query response model"""
    results: List[Dict[str, Any]]
    query_time_ms: float
    num_candidates: int


@app.on_event("startup")
async def startup_event():
    """Initialize query pipeline on startup"""
    global query_pipeline
    
    # Load from environment variable or default path
    import os
    from pathlib import Path
    from core.embedder_minilm import get_embedder
    
    base_path = os.getenv("EDGE_DATA_PATH", "data/book1")
    base_path = Path(base_path)
    
    if not base_path.exists():
        print(f"Warning: Data path {base_path} does not exist. "
              "API will not be available until data is ingested.")
        query_pipeline = None
        return
    
    try:
        # Initialize embedder to get dimension
        embedder = get_embedder()
        embedding_dim = embedder.get_embedding_dim()
        
        # Load all components
        faiss_index = FaissIndex.load(str(base_path / "faiss.index"))
        vector_store = VectorStore(str(base_path), embedding_dim)
        sentence_store = SentenceStore(str(base_path / "sentences.jsonl"))
        graph_store = GraphStore(str(base_path / "graph.json"))
        
        # Create query pipeline
        query_pipeline = QueryPipeline(
            faiss_index=faiss_index,
            vector_store=vector_store,
            sentence_store=sentence_store,
            graph_store=graph_store
        )
        
        print(f"[API] Pipeline initialized with {faiss_index.index.ntotal} vectors")
        
    except Exception as e:
        print(f"[API] Failed to initialize pipeline: {e}")
        import traceback
        traceback.print_exc()
        query_pipeline = None


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "pipeline_initialized": query_pipeline is not None
    }


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """
    Semantic search query endpoint.
    
    Args:
        request: Query request with text and parameters
        
    Returns:
        Query response with ranked results
    """
    if query_pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Query pipeline not initialized. Please load indices and stores first."
        )
    
    t_start = time.time()
    
    try:
        results = query_pipeline.query(
            query_text=request.query,
            top_k=request.top_k,
            initial_k=request.initial_k,
            expand_sequential=request.expand_sequential,
            expand_paragraph=request.expand_paragraph,
            expand_semantic=request.expand_semantic
        )
        
        query_time_ms = (time.time() - t_start) * 1000
        
        return QueryResponse(
            results=results,
            query_time_ms=round(query_time_ms, 2),
            num_candidates=len(results)
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@app.get("/stats")
async def get_stats():
    """Get system statistics"""
    if query_pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    stats = {
        "faiss_index": query_pipeline.faiss_index.get_stats(),
        "vector_store": query_pipeline.vector_store.get_stats(),
        "sentence_store": query_pipeline.sentence_store.get_stats(),
        "graph_store": query_pipeline.graph_store.get_stats()
    }
    
    return stats


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

