import os

import cv2
from insightface.app import FaceAnalysis

from app.database.sqlite_db import SQLiteDatabase


WATCHLIST_DIR = "watchlist"


def create_face_app():
    app = FaceAnalysis(
        name="buffalo_l",
        providers=[
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ],
    )

    app.prepare(
        ctx_id=0,
        det_size=(640, 640),
    )

    return app


def register_person(face_app, db, person_name, person_dir):
    print(f"\nRegistering: {person_name}")

    person_id = db.get_or_create_person(person_name)

    registered = 0

    for filename in os.listdir(person_dir):

        image_path = os.path.join(
            person_dir,
            filename,
        )

        image = cv2.imread(image_path)

        if image is None:
            print(f"  [SKIP] Cannot read {filename}")
            continue

        faces = face_app.get(image)

        if len(faces) == 0:
            print(f"  [SKIP] No face found: {filename}")
            continue

        if len(faces) > 1:
            print(f"  [SKIP] Multiple faces found: {filename}")
            continue

        face = faces[0]

        embedding = face.embedding

        if embedding is None:
            print(f"  [SKIP] No embedding: {filename}")
            continue

        db.add_embedding(
            person_id=person_id,
            image_name=filename,
            embedding=embedding,
        )

        registered += 1

        print(f"  [OK] {filename}")

    print(
        f"Registered {registered} image(s) for {person_name}"
    )


def main():

    print("Loading InsightFace...")

    face_app = create_face_app()

    db = SQLiteDatabase("data/faces.db")

    for person_name in os.listdir(WATCHLIST_DIR):

        person_dir = os.path.join(
            WATCHLIST_DIR,
            person_name,
        )

        if not os.path.isdir(person_dir):
            continue

        register_person(
            face_app,
            db,
            person_name,
            person_dir,
        )

    db.close()

    print("\nRegistration complete.")


if __name__ == "__main__":
    main()