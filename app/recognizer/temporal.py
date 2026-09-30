from collections import defaultdict, deque


class TemporalRecognizer:
    """
    Stabilizes face recognition results across multiple frames.

    Each track_id maintains its own recent recognition history.
    """

    def __init__(self, window_size=5, min_votes=3):
        self.window_size = window_size
        self.min_votes = min_votes

        self.histories = defaultdict(
            lambda: deque(maxlen=self.window_size)
        )

    def update(self, track_id, result):
        """
        Add a recognition result to a track's history.

        Args:
            track_id: Unique identifier for the detected face.
            result: Result returned by FaceRecognizer.recognize()

        Returns:
            Stabilized recognition result.
        """

        self.histories[track_id].append(result)

        history = self.histories[track_id]

        recognized_results = [
            r for r in history
            if r["recognized"]
        ]

        if not recognized_results:
            return {
                "recognized": False,
                "name": None,
                "similarity": result["similarity"],
            }

        votes = defaultdict(int)

        for r in recognized_results:
            votes[r["name"]] += 1

        best_name = max(votes, key=votes.get)
        best_votes = votes[best_name]

        if best_votes < self.min_votes:
            return {
                "recognized": False,
                "name": None,
                "similarity": result["similarity"],
            }

        matching_results = [
            r for r in recognized_results
            if r["name"] == best_name
        ]

        best_similarity = max(
            r["similarity"]
            for r in matching_results
        )

        return {
            "recognized": True,
            "name": best_name,
            "similarity": best_similarity,
        }

    def remove(self, track_id):
        """
        Remove a track when the face is no longer visible.
        """
        self.histories.pop(track_id, None)

    def clear(self):
        """
        Clear all temporal histories.
        """
        self.histories.clear()