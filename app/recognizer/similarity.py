from __future__ import annotations

import numpy as np


def cosine_similarity(
    a: np.ndarray,
    b: np.ndarray,
) -> float:
    """
    Calculate cosine similarity between two embeddings.

    Args:
        a: First embedding.
        b: Second embedding.

    Returns:
        Cosine similarity score.
    """

    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)

    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)

    if a_norm == 0 or b_norm == 0:
        raise ValueError("Cannot compare zero-vector embeddings.")

    return float(np.dot(a, b) / (a_norm * b_norm))