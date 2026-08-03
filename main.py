import cv2

from app.camera.frame_grabber import FrameGrabber


def main():

    camera = FrameGrabber()

    while True:

        ret, frame = camera.read()

        if not ret:
            break

        cv2.imshow("ARGUS", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()