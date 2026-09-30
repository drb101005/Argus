import cv2

from app.camera.frame_grabber import FrameGrabber
from app.detector.detector import FaceDetector
from app.database.sqlite_db import SQLiteDatabase
from app.recognizer.recognizer import FaceRecognizer
from app.recognizer.temporal import TemporalRecognizer


def main():

    # -----------------------------
    # Initialize ARGUS components
    # -----------------------------

    camera = FrameGrabber(source=0)

    detector = FaceDetector()

    database = SQLiteDatabase("data/faces.db")

    recognizer = FaceRecognizer(
        database,
        threshold=0.45,
    )

    temporal = TemporalRecognizer(
        window_size=5,
        min_votes=3,
    )

    print("ARGUS started.")
    print("Press 'q' to quit.")

    # -----------------------------
    # Main camera loop
    # -----------------------------

    while True:

        success, frame = camera.read()

        if not success:
            print("Failed to read frame.")
            break

        # -----------------------------
        # Detect faces
        # -----------------------------

        faces = detector.detect(frame)

        # -----------------------------
        # Process every detected face
        # -----------------------------

        for track_id, face in enumerate(faces):

            # Face bounding box
            x1, y1, x2, y2 = face.bbox.astype(int)

            # Face embedding
            embedding = face.embedding

            # -----------------------------
            # Normal recognition
            # -----------------------------

            result = recognizer.recognize(embedding)

            # -----------------------------
            # Temporal recognition
            # -----------------------------

            result = temporal.update(
                track_id,
                result,
            )

            # -----------------------------
            # Determine display information
            # -----------------------------

            if result["recognized"]:

                name = result["name"]
                similarity = result["similarity"]

                label = f"{name} ({similarity:.2f})"

            else:

                label = f"UNKNOWN ({result['similarity']:.2f})"

            # -----------------------------
            # Draw bounding box
            # -----------------------------

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2,
            )

            # -----------------------------
            # Draw label
            # -----------------------------

            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

        # -----------------------------
        # Show camera
        # -----------------------------

        cv2.imshow(
            "ARGUS - Face Recognition",
            frame,
        )

        # -----------------------------
        # Quit
        # -----------------------------

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # -----------------------------
    # Cleanup
    # -----------------------------

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()