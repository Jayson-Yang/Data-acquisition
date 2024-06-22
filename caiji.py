# caiji.py
import cv2
from PyQt5.QtCore import QThread, pyqtSignal

class CaptureThread(QThread):
    frameCaptured = pyqtSignal(object)

    def __init__(self):
        super().__init__()
        self.capture = cv2.VideoCapture(0)  # 0表示默认摄像头

    def run(self):
        while True:
            ret, frame = self.capture.read()
            if not ret:
                break
            self.frameCaptured.emit(frame)

    def stop(self):
        self.capture.release()

    def get_current_frame(self):
        ret, frame = self.capture.read()
        if ret:
            return frame
        else:
            return None
