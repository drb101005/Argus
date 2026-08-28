import numpy as np

from app.database.sqlite_db import SQLiteDatabase
from app.recognizer.recognizer import FaceRecognizer


def test_recognizes_matching_embedding(tmp_path):

    db = SQLiteDatabase(str(tmp_path / "faces.db"))

    person_id = db.get_or_create_person("Dhruv")

    stored_embedding = np.array(
        [1.0, 0.0, 0.0],
        dtype=np.float32,
    )

    db.add_embedding(
        person_id,
        "test.jpg",
        stored_embedding,
    )

    recognizer = FaceRecognizer(
        db,
        threshold=0.8,
    )

    result = recognizer.recognize(
        np.array(
            [1.0, 0.0, 0.0],
            dtype=np.float32,
        )
    )

    assert result["recognized"] is True
    assert result["name"] == "Dhruv"
    assert result["similarity"] == 1.0


def test_rejects_unknown_embedding(tmp_path):

    db = SQLiteDatabase(str(tmp_path / "faces.db"))

    person_id = db.get_or_create_person("Dhruv")

    db.add_embedding(
        person_id,
        "test.jpg",
        np.array(
            [1.0, 0.0, 0.0],
            dtype=np.float32,
        ),
    )

    recognizer = FaceRecognizer(
        db,
        threshold=0.8,
    )

    result = recognizer.recognize(
        np.array(
            [0.0, 1.0, 0.0],
            dtype=np.float32,
        )
    )

    assert result["recognized"] is False
    assert result["name"] is None