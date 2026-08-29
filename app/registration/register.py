from pathlib import Path

import cv2

from app.detector.detector import FaceDetector
from app.database.sqlite_db import SQLiteDatabase


class FaceRegistration:
    """Handles registering new people into ARGUS."""

    def __init__(self, database: SQLiteDatabase, detector: FaceDetector):
        self.database = database
        self.detector = detector

    def register(self, name: str, image_path: str) -> None:
        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        image = cv2.imread(str(image_path))

        if image is None:
            raise ValueError(
                f"Could not read image: {image_path}"
            )

        faces = self.detector.detect(image)

        if len(faces) == 0:
            raise ValueError(
                "No face detected in the image."
            )

        if len(faces) > 1:
            raise ValueError(
                "Multiple faces detected. "
                "Please provide an image containing one person."
            )

        face = faces[0]

        embedding = face.embedding

        person_id = self.database.get_or_create_person(name)

        self.database.add_embedding(
            person_id=person_id,
            image_name=image_path.name,
            embedding=embedding,
        )

        print(
            f"Successfully registered '{name}' "
            f"from '{image_path.name}'."
        )