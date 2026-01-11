import cv2
from PIL import Image

class Camera:
    def __init__(self, source=0):
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open camera")

    def capture_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Failed to capture frame")
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(frame_rgb)

    def release(self):
        self.cap.release()
