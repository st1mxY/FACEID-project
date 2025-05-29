import cv2
import sys
import time
import numpy as np
import face_recognition

from PySide6.QtWidgets import QWidget, QLabel, QApplication, QPushButton, QVBoxLayout, QHBoxLayout, QTabWidget, \
    QFileDialog, QTextEdit
from PySide6.QtCore import QThread, Qt, Signal, Slot, QSize
from PySide6.QtGui import QImage, QPixmap, QMovie

import recognize

from queue import Queue

pyqtSignal = Signal
pyqtSlot = Slot


class Thread(QThread):
    changePixmap = pyqtSignal(QImage)

    def run(self):
        self.isRunning = True
        self.frames = recognize.get_frames()
        self.Global = next(self.frames)
        while self.isRunning:
            if not self.queue.empty():
                encoding, name = self.queue.get()
                self.Global.known_face_encodings = [encoding] + self.Global.known_face_encodings
                self.Global.known_face_names = [name] + self.Global.known_face_names

            frame = next(self.frames)
            self.update_webcam_image(frame)

    def update_webcam_image(self, frame):
        rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgbImage.shape
        bytesPerLine = ch * w
        convertToQtFormat = QImage(rgbImage.data, w, h, bytesPerLine, QImage.Format_RGB888)
        p = convertToQtFormat.scaled(640, 480, Qt.KeepAspectRatio)
        self.changePixmap.emit(p)

    def stop(self):
        self.isRunning = False
        self.quit()
        self.terminate()


class VideoContainer(QWidget):
    def __init__(self):
        super().__init__()
        self.title = 'Video'
        self.left = 100
        self.top = 100
        self.fwidth = 640
        self.fheight = 480
        self.initUI()

    @pyqtSlot(QImage)
    def setImage(self, image):
        self.label.setPixmap(QPixmap.fromImage(image))

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.fwidth, self.fheight)
        self.setFixedSize(640, 480)
        self.tabs = QTabWidget()

        layout = QVBoxLayout()
        layout.addWidget(self.tabs)
        self.setLayout(layout)
        main_layout = QHBoxLayout()
        main_tab = QWidget()

        self.label = QLabel()
        main_layout.addWidget(self.label)

        main_tab.setLayout(main_layout)

        self.add_person_tab = QWidget()
        layout = QVBoxLayout()

        self.add_person_button = QPushButton("Add image")
        self.send_form = QPushButton("Add")
        self.input_text = QTextEdit()
        self.add_person_button.clicked.connect(self.on_choose_photo_button_clicked)
        self.send_form.clicked.connect(self.on_form_sent)
        layout.addWidget(self.add_person_button)
        layout.addWidget(self.input_text)
        layout.addWidget(self.send_form)
        self.add_person_tab.setLayout(layout)
        # Третья вкладка
        self.rofl_tab = QWidget()
        gif_label = QLabel()
        movie = QMovie("gifka.gif")
        movie.setScaledSize(QSize(600, 400))
        gif_label.setMovie(movie)
        movie.start()

        tab_layout = QVBoxLayout()
        tab_layout.addWidget(gif_label)

        self.rofl_tab.setLayout(tab_layout)

        self.tabs.addTab(main_tab, "WebcamAI")
        self.tabs.addTab(self.add_person_tab, "Add person")
        self.tabs.addTab(self.rofl_tab, "Pascal")

        self.label.resize(640, 480)
        self.th = Thread(self)
        self.th.queue = Queue()
        self.th.changePixmap.connect(self.setImage)
        self.th.start()
        self.show()

    @Slot()
    def on_choose_photo_button_clicked(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Выбрать изображение", "", "Images (*.png *.jpg *.jpeg)")
        if file_path:
            image = face_recognition.load_image_file(file_path)
            self.pending_encoding = face_recognition.face_encodings(image)[0]

    @Slot()
    def on_form_sent(self):
        self.th.queue.put((self.pending_encoding, self.input_text.toPlainText()))


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = VideoContainer()
    sys.exit(app.exec())
