"""
Book text ingestion pipeline.
Parses book text into sentences and builds memory structures.
"""
import time
import re
from typing import List, Dict, Any, Optional
from pathlib import Path
from core.embedder_minilm import get_embedder
from storage.sentence_store import SentenceStore
from storage.vector_store import VectorStore
from core.faiss_index import FaissIndex
from graph.graph_builder import GraphBuilder
from graph.graph_store import GraphStore
from core.config import VERBOSE_TIMING


class BookIngestionPipeline:
    """
    Ingests book text and builds memory structures.
    """
    
    def __init__(
        self,
        base_path: str,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        index_type: str = "flat"
    ):
        """
        Initialize ingestion pipeline.
        
        Args:
            base_path: Base directory for storage
            embedding_model: Embedding model name
            index_type: Faiss index type ("flat" or "ivf-pq")
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize embedder
        self.embedder = get_embedder(model_name=embedding_model)
        embedding_dim = self.embedder.get_embedding_dim()
        
        # Initialize stores
        self.sentence_store = SentenceStore(str(self.base_path / "sentences.jsonl"))
        self.vector_store = VectorStore(str(self.base_path), embedding_dim)
        self.faiss_index = FaissIndex(embedding_dim, index_type=index_type)
        self.graph_store = GraphStore(str(self.base_path / "graph.json"))
    
    def parse_book(
        self,
        book_text: str,
        chapter_pattern: str = r"^第[一二三四五六七八九十\d]+章|^Chapter \d+",
        paragraph_separator: str = "\n\n"
    ) -> List[Dict[str, Any]]:
        """
        Parse book text into structured sentences.
        
        Args:
            book_text: Full book text
            chapter_pattern: Regex pattern for chapter headers
            paragraph_separator: Separator between paragraphs
            
        Returns:
            List of sentence dictionaries with metadata
        """
        t_start = time.time() if VERBOSE_TIMING else None
        
        sentences = []
        current_chapter = "Unknown"
        current_paragraph_id = 0
        sentence_id_counter = 1
        
        # Split into paragraphs
        paragraphs = book_text.split(paragraph_separator)
        
        for para_text in paragraphs:
            para_text = para_text.strip()
            if not para_text:
                continue
            
            # Check if paragraph is a chapter header
            if re.match(chapter_pattern, para_text, re.MULTILINE):
                current_chapter = para_text.split('\n')[0].strip()
                current_paragraph_id = 0
                continue
            
            current_paragraph_id += 1
            
            # Split paragraph into sentences
            # Simple sentence splitting (can be improved)
            sentence_texts = re.split(r'[。！？.!?]\s*', para_text)
            sentence_texts = [s.strip() for s in sentence_texts if s.strip()]
            
            prev_sentence_id = None
            
            for sentence_idx, sentence_text in enumerate(sentence_texts):
                sentence_id = sentence_id_counter
                sentence_id_counter += 1
                
                sentence_data = {
                    "id": sentence_id,
                    "text": sentence_text,
                    "chapter": current_chapter,
                    "paragraph_id": current_paragraph_id,
                    "sentence_id": sentence_idx + 1,
                    "prev_sentence_id": prev_sentence_id,
                    "next_sentence_id": None,
                    "embedding_ref": None,
                    "metadata": {}
                }
                
                # Update previous sentence's next_sentence_id
                if prev_sentence_id is not None:
                    for s in sentences:
                        if s["id"] == prev_sentence_id:
                            s["next_sentence_id"] = sentence_id
                            break
                
                sentences.append(sentence_data)
                prev_sentence_id = sentence_id
        
        if VERBOSE_TIMING:
            t_elapsed = time.time() - t_start
            print(f"[ingestion] Parsed {len(sentences)} sentences, duration={t_elapsed:.3f}s")
        
        return sentences
    
    def ingest_book(
        self,
        book_text: str,
        batch_size: int = 32,
        similarity_threshold: float = 0.7
    ):
        """
        Complete book ingestion pipeline.
        
        Args:
            book_text: Full book text
            batch_size: Batch size for embedding
            similarity_threshold: Threshold for semantic edges
        """
        t_start = time.time() if VERBOSE_TIMING else None
        
        # Step 1: Parse book
        sentences = self.parse_book(book_text)
        
        if not sentences:
            raise ValueError("No sentences found in book text")
        
        # Step 2: Generate embeddings
        sentence_texts = [s["text"] for s in sentences]
        embeddings = self.embedder.encode(sentence_texts, batch_size=batch_size)
        
        # Step 3: Store sentences and vectors
        sentence_ids = [s["id"] for s in sentences]
        id_to_row = self.vector_store.add_vectors(sentence_ids, embeddings)
        
        # Update embedding_ref in sentences
        for sentence in sentences:
            sentence["embedding_ref"] = id_to_row[sentence["id"]]
        
        # Save sentences
        for sentence in sentences:
            # Extract parameters for add_sentence (exclude 'id' from kwargs)
            self.sentence_store.add_sentence(
                sentence_id=sentence["id"],
                text=sentence["text"],
                chapter=sentence["chapter"],
                paragraph_id=sentence["paragraph_id"],
                sentence_id_in_para=sentence["sentence_id"],
                prev_sentence_id=sentence.get("prev_sentence_id"),
                next_sentence_id=sentence.get("next_sentence_id"),
                embedding_ref=sentence.get("embedding_ref"),
                metadata=sentence.get("metadata")
            )
        self.sentence_store.save_sentences(append=False)
        
        # Step 4: Build graph edges
        sequential_edges = GraphBuilder.build_sequential_edges(sentences)
        paragraph_edges = GraphBuilder.build_paragraph_edges(sentences)
        semantic_edges = GraphBuilder.build_semantic_edges(
            sentence_ids,
            embeddings,
            similarity_threshold=similarity_threshold
        )
        
        # Add edges to graph store
        for u, v, weight in sequential_edges:
            self.graph_store.add_edge(u, v, weight=weight, etype="sequential")
        for u, v, weight in paragraph_edges:
            self.graph_store.add_edge(u, v, weight=weight, etype="paragraph")
        for u, v, weight in semantic_edges:
            self.graph_store.add_edge(u, v, weight=weight, etype="semantic")
        
        self.graph_store.save_data()
        
        # Step 5: Build Faiss index
        # Load all vectors from vector_store to ensure index matches rowmap
        all_vectors = self.vector_store.get_all_vectors(use_mmap=True)
        
        if len(all_vectors) == 0:
            raise RuntimeError("No vectors found in vector_store after ingestion")
        
        if self.faiss_index.index_type == "ivf-pq":
            # Train on sample (use all if small, sample if large)
            train_size = min(10000, len(all_vectors))
            train_vectors = all_vectors[:train_size]
            self.faiss_index.train(train_vectors)
        
        # Add all vectors to index (ensures index position = row_index)
        self.faiss_index.add(all_vectors)
        self.faiss_index.save(str(self.base_path / "faiss.index"))
        
        if VERBOSE_TIMING:
            t_elapsed = time.time() - t_start
            total_edges = len(sequential_edges) + len(paragraph_edges) + len(semantic_edges)
            print(f"[ingestion] Complete ingestion duration={t_elapsed:.3f}s")
            print(f"[ingestion] Sentences: {len(sentences)}, Edges: {total_edges}")

