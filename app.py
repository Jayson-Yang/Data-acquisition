# app.py
import sys
import os
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QLabel, QHBoxLayout, QMessageBox
from PyQt5.QtGui import QImage, QPixmap
import cv2
from caiji import CaptureThread
from inference import InferenceEngine

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.capture_thread = CaptureThread()
        self.inference_engine = InferenceEngine()

        self.init_ui()
        self.capture_thread.frameCaptured.connect(self.update_image)

        self.captured_images = []
        self.image_folder = 'image'  # 图片保存文件夹名称

        # 确保图片保存文件夹存在，不存在则创建
        if not os.path.exists(self.image_folder):
            os.makedirs(self.image_folder)

    def init_ui(self):
        self.start_button = QPushButton('开始采集')
        self.stop_button = QPushButton('停止采集')
        self.capture_button = QPushButton('捕获图像')
        self.inference_button = QPushButton('推理图像')

        self.start_button.clicked.connect(self.start_capture)
        self.stop_button.clicked.connect(self.stop_capture)
        self.capture_button.clicked.connect(self.capture_image)
        self.inference_button.clicked.connect(self.run_inference)

        self.image_label = QLabel()
        self.image_label.setFixedSize(640, 480)

        layout = QVBoxLayout()
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)
        button_layout.addWidget(self.capture_button)
        button_layout.addWidget(self.inference_button)
        layout.addLayout(button_layout)
        layout.addWidget(self.image_label)

        self.setLayout(layout)
        self.setWindowTitle('图像采集和推理')

    def start_capture(self):
        self.capture_thread.start()

    def stop_capture(self):
        self.capture_thread.stop()

    def update_image(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(qt_image))

    def capture_image(self):
        if self.capture_thread.isRunning():
            frame = self.capture_thread.get_current_frame()
            if frame is not None:
                image_path = os.path.join(self.image_folder, f'captured_image_{len(self.captured_images)}.jpg')
                cv2.imwrite(image_path, frame)
                self.captured_images.append(image_path)
                print(f"图像已捕获: {image_path}")
            else:
                QMessageBox.warning(self, '警告', '未捕获到图像！')

    def run_inference(self):
        if self.captured_images:
            results = []
            for image_path in self.captured_images:
                result = self.inference_engine.infer(image_path)
                results.extend(result)
            self.inference_engine.save_results(results)
            print("推理完成，结果已保存")
        else:
            QMessageBox.warning(self, '警告', '请先捕获图像！')

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_win = MainWindow()
    main_win.show()
    sys.exit(app.exec_())
