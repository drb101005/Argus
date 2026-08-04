import cv2

from app.camera.frame_grabber import FrameGrabber
from app.detector.detector import FaceDetector


def main():

    camera = FrameGrabber()
    detector = FaceDetector()

    while True:

        ret, frame = camera.read()

        if not ret:
            break

        faces = detector.detect(frame)

        for face in faces:

            x1, y1, x2, y2 = map(int, face.bbox)

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2,
            )

        cv2.imshow("ARGUS", frame)

        if cv2.waitKey(1) == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()