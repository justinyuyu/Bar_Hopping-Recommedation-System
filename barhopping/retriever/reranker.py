import torch
from typing import List, Dict, Union
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from barhopping.config import TOP_K
from barhopping.logger import logger

class Reranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3", device: str = None):
        """Initialize the reranker model.
        
        Args:
            model_name: Name of the model to use
            device: Device to run the model on (cpu, cuda, mps). If None, uses the default from config.
        """
        self.model_name = model_name
        self.device = device or (
            "mps" if torch.backends.mps.is_available() else
            "cuda" if torch.cuda.is_available() else
            "cpu"
        )
        
        logger.info(f"Loading reranker model {self.model_name} on device {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name).to(self.device)
        self.model.eval()
        
    def rerank(self, query: str, candidates: List[Dict[str, str]], top_k: int = TOP_K, threshold: float = None) -> List[Dict[str, Union[str, float]]]:
        """Rerank candidates based on their relevance to the query.
        
        Args:
            query: The search query
            candidates: List of candidate dictionaries with 'tag_name' and 'summary'
            top_k: Number of top results to return (default: 5)
            threshold: Minimum score threshold (optional)
        Returns:
            List of reranked candidates with scores
        """
        if not candidates:
            return []
        
        # Create input pairs
        pairs = [(query, f"{candidate["name"]}: {candidate["summary"]}") for candidate in candidates]
        
        # Tokenize
        inputs = self.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.device)
        
        # Get scores
        with torch.no_grad():
            scores = self.model(**inputs).logits.squeeze()
        
        # Convert to list and attach to candidates
        if isinstance(scores, torch.Tensor):
            scores = scores.cpu().numpy().tolist()
        for candidate, score in zip(candidates, scores):
            candidate["rerank_score"] = float(score)
        
        # Sort by rerank score
        candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
        if threshold is not None:
            return [c for c in candidates if c["rerank_score"] >= threshold]
        return candidates[:top_k]

# Global instance for reuse
_reranker = None

def get_reranker() -> Reranker:
    """Return a singleton reranker instance."""
    global _reranker
    if _reranker is None:
        _reranker = Reranker()
    return _reranker