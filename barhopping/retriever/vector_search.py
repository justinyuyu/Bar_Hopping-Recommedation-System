import ast
import sqlite3
import numpy as np
from typing import List, Dict, Union
from barhopping.embedding.granite import get_embedding
from .reranker import get_reranker
from barhopping.config import TOP_K, BARS_DB
from barhopping.logger import logger

class VectorSearch:
    def __init__(self, db_path: str = BARS_DB):
        """Initialize the vector search.
        
        Args:
            db_path: Path to the SQLite database
        """
        self.db_path = db_path
        self._load_embeddings()
        
    def _load_embeddings(self):
        """Load all embeddings from the database."""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT id, name, URL, address, photo, summary, embedding FROM bars"
            ).fetchall()
        
        # Initialize storage
        self.ids = []
        self.names = []
        self.URLs = []
        self.addresses = []
        self.photos = []
        self.summaries = []
        self.embeddings = []

        for row in rows:
            bar_id, name, URL, address, photo, summary, embedding_str = row
            self.ids.append(bar_id)
            self.names.append(name)
            self.URLs.append(URL)
            self.addresses.append(address)
            self.photos.append(photo)
            self.summaries.append(summary)
            self.embeddings.append(np.array(ast.literal_eval(embedding_str), dtype=np.float32))
            
        if self.embeddings:
            self.embeddings = np.vstack(self.embeddings)
            logger.info(f"Loaded {len(self.embeddings)} bar embeddings")
        else:
            self.embeddings = np.array([])
            logger.warning("No embeddings found in the database")
            
    def search(self, query: str, rerank: bool = True) -> List[Dict[str, Union[str, float]]]:
        """Search for similar bars using vector search and optional reranking.
        
        Args:
            query: Search query
            top_k: Number of results to return
            rerank: Whether to apply reranking (default: True)
            threshold: Reranking score threshold (optional)
        Returns:
            List of dictionaries containing bar information and scores
        """
        if len(self.embeddings) == 0:
            logger.error("No embeddings available for search")
            return []
            
        # Get query embedding
        query_vec = get_embedding(query).cpu().numpy().reshape(-1)
        sims = self.embeddings @ query_vec
        top_indices = np.argsort(sims)[-2*TOP_K:][::-1]
        
        # Prepare candidates for reranking
        candidates = [
            {
                "id": self.ids[i],
                "name": self.names[i],
                "URL": self.URLs[i],
                "summary": self.summaries[i],
                "address": self.addresses[i],
                "photo": self.photos[i],
                "vector_score": float(sims[i])
            }
            for i in top_indices
        ]
        
        if rerank:
            logger.info("Applying reranking...")
            reranker = get_reranker()
            return reranker.rerank(query, candidates)

        return candidates
            
    def refresh(self):
        """Reload embeddings from the database."""
        logger.info("Refreshing vector search index...")
        self._load_embeddings()
        
# Global instance for reuse
_vector_search = None

def get_vector_search() -> VectorSearch:
    """Return a singleton instance of the vector search engine."""
    global _vector_search
    if _vector_search is None:
        _vector_search = VectorSearch()
    return _vector_search