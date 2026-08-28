from __future__ import annotations

import numpy as np

from app.database.sqlite_db import SQLiteDatabase
from app.recognizer.similarity import cosine_similarity


class FaceRecognizer:
    """Matches face embeddings against registered faces."""

    def __init__(
        self,
        database: SQLiteDatabase,
        threshold: float = 0.45,
    ):
        self.database = database
        self.threshold = threshold

        self.embeddings = self.database.get_all_embeddings()

    def recognize(self, embedding: np.ndarray) -> dict:
        """
        Compare a face embedding against all registered embeddings.

        Returns:
            {
                "name": str | None,
                "similarity": float,
                "recognized": bool
            }
        """

        if not self.embeddings:
            return {
                "name": None,
                "similarity": 0.0,
                "recognized": False,
            }

        best_match = None
        best_similarity = -1.0

        for stored in self.embeddings:
            similarity = cosine_similarity(
                embedding,
                stored["embedding"],
            )

            if similarity > best_similarity:
                best_similarity = similarity
                best_match = stored

        recognized = best_similarity >= self.threshold

        return {
            "name": best_match["name"] if recognized else None,
            "similarity": best_similarity,
            "recognized": recognized,
        }