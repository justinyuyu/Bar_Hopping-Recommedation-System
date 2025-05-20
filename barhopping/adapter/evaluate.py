import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List

def evaluate(anchors: np.ndarray, positives: np.ndarray, true_ids: np.ndarray, k: int = 20) -> Tuple[float, float]:
    """
    Evaluate retrieval performance using Mean Reciprocal Rank (MRR) and Hit Rate@k.

    Args:
        anchors (np.ndarray): Anchor embeddings matrix (N x D).
        positives (np.ndarray): Positive embeddings matrix (M x D).
        true_ids (np.ndarray): Array of true positive indices (1-based).
        k (int, optional): Cutoff rank for hits. Defaults to 20.

    Returns:
        Tuple[float, float]: Mean Reciprocal Rank (MRR) and Hit Rate@k.
    """
    sims = anchors @ positives.T  # Cosine similarity or dot product matrix
    ranks = np.argsort(-sims, axis=1)  # Descending sort indices per anchor

    reciprocal_ranks: List[float] = []
    hits_at_k: List[bool] = []

    for i, tid in enumerate(true_ids):
        # Adjust for zero-based indexing (assuming true_ids are 1-based)
        target_index = tid - 1

        # Find rank position of the true positive
        pos = np.where(ranks[i] == target_index)[0]
        if pos.size > 0:
            rank_pos = pos[0]
            reciprocal_ranks.append(1.0 / (rank_pos + 1))
            hits_at_k.append(rank_pos < k)
        else:
            reciprocal_ranks.append(0.0)
            hits_at_k.append(False)

    mrr = float(np.mean(reciprocal_ranks))
    hit_rate = float(np.mean(hits_at_k))
    return mrr, hit_rate

def plot_loss(train_losses: List[float], val_losses: List[float]) -> None:
    plt.figure(figsize=(8, 5))
    epochs = range(1, len(train_losses) + 1)

    plt.plot(epochs, train_losses, label='Train')
    plt.plot(epochs, val_losses, label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Adapter Training Loss')
    plt.legend()
    plt.tight_layout()
    plt.show()