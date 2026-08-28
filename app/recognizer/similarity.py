import numpy as np


def cosine_similarity(
    embedding1: np.ndarray,
    embedding2: np.ndarray,
) -> float:

    embedding1 = np.asarray(embedding1, dtype=np.float32)
    embedding2 = np.asarray(embedding2, dtype=np.float32)

    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)

    if norm1 == 0 or norm2 == 0:
        raise ValueError("Cannot calculate cosine similarity for a zero vector")

    return float(
        np.dot(embedding1, embedding2)
        / (norm1 * norm2)
    )