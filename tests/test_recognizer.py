import numpy as np

from app.recognizer.recognizer import FaceRecognizer


class FakeDatabase:
    def __init__(self, embeddings):
        self.embeddings = embeddings

    def get_all_embeddings(self):
        return self.embeddings


def make_record(name, embedding, image_name="test.jpg"):
    return {
        "id": 1,
        "name": name,
        "image_name": image_name,
        "embedding": np.array(embedding, dtype=np.float32),
    }


def test_recognizer_finds_best_embedding():
    database = FakeDatabase(
        [
            make_record("Dhruv", [1.0, 0.0], "dhruv1.jpg"),
            make_record("Dhruv", [0.8, 0.6], "dhruv2.jpg"),
            make_record("Alice", [0.0, 1.0], "alice1.jpg"),
        ]
    )

    recognizer = FaceRecognizer(
        database,
        threshold=0.45,
    )

    result = recognizer.recognize(
        np.array([1.0, 0.0], dtype=np.float32)
    )

    assert result["recognized"] is True
    assert result["name"] == "Dhruv"
    assert result["similarity"] == 1.0


def test_recognizer_selects_best_person():
    database = FakeDatabase(
        [
            make_record("Dhruv", [0.8, 0.6], "dhruv1.jpg"),
            make_record("Alice", [0.0, 1.0], "alice1.jpg"),
        ]
    )

    recognizer = FaceRecognizer(
        database,
        threshold=0.45,
    )

    result = recognizer.recognize(
        np.array([1.0, 0.0], dtype=np.float32)
    )

    assert result["recognized"] is True
    assert result["name"] == "Dhruv"
    assert result["similarity"] > 0.7


def test_recognizer_uses_best_embedding_for_same_person():
    database = FakeDatabase(
        [
            make_record("Dhruv", [0.6, 0.8], "dhruv1.jpg"),
            make_record("Dhruv", [1.0, 0.0], "dhruv2.jpg"),
            make_record("Alice", [0.0, 1.0], "alice1.jpg"),
        ]
    )

    recognizer = FaceRecognizer(
        database,
        threshold=0.45,
    )

    result = recognizer.recognize(
        np.array([1.0, 0.0], dtype=np.float32)
    )

    assert result["recognized"] is True
    assert result["name"] == "Dhruv"
    assert result["similarity"] == 1.0


def test_unknown_face_with_multiple_people():
    database = FakeDatabase(
        [
            make_record("Dhruv", [1.0, 0.0], "dhruv.jpg"),
            make_record("Alice", [0.0, 1.0], "alice.jpg"),
        ]
    )

    recognizer = FaceRecognizer(
        database,
        threshold=0.90,
    )

    result = recognizer.recognize(
        np.array([1.0, 1.0], dtype=np.float32)
    )

    assert result["recognized"] is False
    assert result["name"] is None
    assert result["similarity"] < 0.90