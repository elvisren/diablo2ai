from __future__ import annotations
from PyQt5 import QtWidgets, QtGui, QtCore
import cv2
import numpy as np

def cvimg_to_qimage(img_bgr: np.ndarray) -> QtGui.QImage:
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = img_rgb.shape
    bytes_per_line = ch * w
    return QtGui.QImage(img_rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888).copy()

class ImagePane(QtWidgets.QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScaledContents(False)
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setMinimumSize(320, 180)

    def set_frame(self, img_bgr: np.ndarray):
        qimg = cvimg_to_qimage(img_bgr)
        pix = QtGui.QPixmap.fromImage(qimg)
        self.setPixmap(pix.scaled(self.width(), self.height(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
