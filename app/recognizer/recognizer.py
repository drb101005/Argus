import numpy as np

from app.recognizer.similarity import cosine_similarity


class FaceRecognizer:
    """
    Matches a face embedding against all embeddings
    stored in the ARGUS database.
    """

    def __init__(self, database, threshold=0.45):
        self.database = database
        self.threshold = threshold

    def recognize(self, embedding: np.ndarray) -> dict:
        """
        Recognize a face using the best similarity
        across all stored embeddings.

        Returns:
            {
                "recognized": bool,
                "name": str | None,
                "similarity": float
            }
        """

        stored_embeddings = self.database.get_all_embeddings()

        # No registered faces
        if not stored_embeddings:
            return {
                "recognized": False,
                "name": None,
                "similarity": 0.0,
            }

        best_match = None
        best_similarity = -1.0

        # Compare against EVERY stored embedding
        for record in stored_embeddings:

            similarity = cosine_similarity(
                embedding,
                record["embedding"],
            )

            if similarity > best_similarity:
                best_similarity = similarity
                best_match = record

        # Apply threshold AFTER finding best match
        if best_similarity >= self.threshold:
            return {
                "recognized": True,
                "name": best_match["name"],
                "similarity": best_similarity,
            }

        return {
            "recognized": False,
            "name": None,
            "similarity": best_similarity,
        }