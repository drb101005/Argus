import argparse

from app.database.sqlite_db import SQLiteDatabase
from app.detector.detector import FaceDetector
from app.registration.register import FaceRegistration


def main():
    parser = argparse.ArgumentParser(
        description="Register a face with ARGUS."
    )

    parser.add_argument(
        "--name",
        required=True,
        help="Person's name",
    )

    parser.add_argument(
        "--image",
        required=True,
        help="Path to the person's image",
    )

    args = parser.parse_args()

    database = SQLiteDatabase("data/faces.db")
    detector = FaceDetector()

    registration = FaceRegistration(
        database=database,
        detector=detector,
    )

    registration.register(
        name=args.name,
        image_path=args.image,
    )


if __name__ == "__main__":
    main()