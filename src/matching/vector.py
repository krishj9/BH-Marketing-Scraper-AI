"""Vector similarity matching using in-memory cosine similarity.

Phase 1: No vector database. Compute cosine similarity in Python
using NumPy. Embeddings held in memory for the current site only.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Compute cosine similarity between two vectors.

    Args:
        vec_a: First embedding vector.
        vec_b: Second embedding vector.

    Returns:
        Cosine similarity score between -1 and 1.
    """
    a = np.array(vec_a, dtype=np.float32)
    b = np.array(vec_b, dtype=np.float32)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def compute_similarities(
    chunk_embeddings: list[list[float]],
    keyword_embedding: list[float],
) -> list[float]:
    """Compute cosine similarities between chunk embeddings and a keyword embedding.

    Uses vectorized NumPy operations for efficiency.

    Args:
        chunk_embeddings: List of embedding vectors for page chunks.
        keyword_embedding: Single embedding vector for keyword text.

    Returns:
        List of cosine similarity scores, one per chunk.
    """
    if not chunk_embeddings:
        return []

    # Convert to numpy arrays
    chunks_matrix = np.array(chunk_embeddings, dtype=np.float32)
    keyword_vec = np.array(keyword_embedding, dtype=np.float32)

    # Normalize keyword vector
    keyword_norm = np.linalg.norm(keyword_vec)
    if keyword_norm == 0:
        return [0.0] * len(chunk_embeddings)
    keyword_normalized = keyword_vec / keyword_norm

    # Normalize chunk vectors (per-row)
    chunk_norms = np.linalg.norm(chunks_matrix, axis=1, keepdims=True)
    # Avoid division by zero
    chunk_norms = np.where(chunk_norms == 0, 1, chunk_norms)
    chunks_normalized = chunks_matrix / chunk_norms

    # Compute all similarities at once via dot product
    similarities = chunks_normalized @ keyword_normalized

    return similarities.tolist()


def filter_by_threshold(
    similarities: list[float],
    embedding_threshold: float,
) -> list[tuple[int, float]]:
    """Filter similarities by threshold, returning (index, score) pairs.

    Args:
        similarities: List of similarity scores.
        embedding_threshold: Minimum similarity to include.

    Returns:
        List of (chunk_index, similarity) tuples above threshold.
    """
    results = []
    for idx, sim in enumerate(similarities):
        if sim >= embedding_threshold:
            results.append((idx, sim))
    results.sort(key=lambda x: x[1], reverse=True)
    return results
