"""
JSONL-based sentence storage with metadata.
Manages sentence nodes with chapter, paragraph, and sentence indices.
"""
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
from core.config import VERBOSE_TIMING


class SentenceStore:
    """
    Manages sentence storage in JSONL format.
    
    Format: One sentence node per line in JSON format.
    """
    
    def __init__(self, filepath: str):
        """
        Initialize sentence store.
        
        Args:
            filepath: Path to JSONL file
        """
        self.filepath = Path(filepath)
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # In-memory cache: sentence_id -> sentence_data
        self.sentences: Dict[int, Dict[str, Any]] = {}
        
        # Load existing sentences
        self._load_sentences()
    
    def _load_sentences(self):
        """Load sentences from JSONL file"""
        if not self.filepath.exists():
            return
        
        t_start = time.time() if VERBOSE_TIMING else None
        
        try:
            with open(self.filepath, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        sentence = json.loads(line)
                        sentence_id = sentence.get('id')
                        if sentence_id is not None:
                            self.sentences[int(sentence_id)] = sentence
                    except json.JSONDecodeError as e:
                        if VERBOSE_TIMING:
                            print(f"[sentence_store] Invalid JSON at line {line_num}: {e}")
                        continue
            
            if VERBOSE_TIMING:
                t_elapsed = time.time() - t_start
                msg = (
                    f"[sentence_store] Loaded {len(self.sentences)} sentences, "
                    f"duration={t_elapsed:.3f}s"
                )
                print(msg)
        
        except Exception as e:
            if VERBOSE_TIMING:
                print(f"[sentence_store] Failed to load sentences: {e}")
    
    def add_sentence(
        self,
        sentence_id: int,
        text: str,
        chapter: str,
        paragraph_id: int,
        sentence_id_in_para: int,
        prev_sentence_id: Optional[int] = None,
        next_sentence_id: Optional[int] = None,
        embedding_ref: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Add a sentence node.
        
        Args:
            sentence_id: Unique sentence ID
            text: Sentence text content
            chapter: Chapter name/title
            paragraph_id: Paragraph number within chapter
            sentence_id_in_para: Sentence number within paragraph
            prev_sentence_id: Previous sentence ID (None if first)
            next_sentence_id: Next sentence ID (None if last)
            embedding_ref: Row index in vector file
            metadata: Additional metadata
            
        Returns:
            Sentence data dictionary
        """
        sentence_data = {
            "id": sentence_id,
            "text": text,
            "chapter": chapter,
            "paragraph_id": paragraph_id,
            "sentence_id": sentence_id_in_para,
            "prev_sentence_id": prev_sentence_id,
            "next_sentence_id": next_sentence_id,
            "embedding_ref": embedding_ref,
            "metadata": metadata or {}
        }
        
        self.sentences[sentence_id] = sentence_data
        return sentence_data
    
    def save_sentences(self, append: bool = True):
        """
        Save sentences to JSONL file.
        
        Args:
            append: Whether to append (True) or overwrite (False)
        """
        t_start = time.time() if VERBOSE_TIMING else None
        
        mode = 'a' if append and self.filepath.exists() else 'w'
        
        with open(self.filepath, mode, encoding='utf-8') as f:
            for sentence_id in sorted(self.sentences.keys()):
                sentence = self.sentences[sentence_id]
                json_line = json.dumps(sentence, ensure_ascii=False)
                f.write(json_line + '\n')
        
        if VERBOSE_TIMING:
            t_elapsed = time.time() - t_start
            msg = (
                f"[sentence_store] Saved {len(self.sentences)} sentences, "
                f"duration={t_elapsed:.3f}s"
            )
            print(msg)
    
    def get_sentence(self, sentence_id: int) -> Optional[Dict[str, Any]]:
        """Get sentence by ID"""
        return self.sentences.get(sentence_id)
    
    def get_sentences(self, sentence_ids: List[int]) -> List[Dict[str, Any]]:
        """Get multiple sentences by IDs"""
        return [self.sentences.get(sid) for sid in sentence_ids if sid in self.sentences]
    
    def get_all_sentences(self) -> List[Dict[str, Any]]:
        """Get all sentences, sorted by ID"""
        return [self.sentences[sid] for sid in sorted(self.sentences.keys())]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics"""
        chapters = set()
        paragraphs = set()
        
        for sentence in self.sentences.values():
            chapter = sentence.get('chapter', '')
            paragraph_id = sentence.get('paragraph_id')
            if chapter:
                chapters.add(chapter)
            if paragraph_id is not None:
                paragraphs.add((chapter, paragraph_id))
        
        return {
            "num_sentences": len(self.sentences),
            "num_chapters": len(chapters),
            "num_paragraphs": len(paragraphs)
        }

