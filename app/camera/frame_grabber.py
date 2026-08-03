import cv2


class FrameGrabber:
    """
    Handles video input from a webcam or other OpenCV-compatible source.
    """

    def __init__(self, source=0):
        self.source = source
        self.cap = cv2.VideoCapture(source)

        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video source: {source}")

    def read(self):
        """
        Read a single frame.

        Returns:
            tuple(bool, frame)
        """
        return self.cap.read()

    def release(self):
        """
        Release camera resources.
        """
        self.cap.release()