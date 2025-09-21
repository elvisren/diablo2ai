from __future__ import annotations
from PyQt5 import QtCore, QtGui, QtWidgets

class ImagePane(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._image = None
        self.setAttribute(QtCore.Qt.WA_OpaquePaintEvent, True)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

    def setImage(self, img: QtGui.QImage):
        self._image = img
        self.update()

    def paintEvent(self, e):
        p = QtGui.QPainter(self)
        r = self.rect()
        p.fillRect(r, QtCore.Qt.black)
        if self._image is None: return
        pm = QtGui.QPixmap.fromImage(self._image)
        pm = pm.scaled(r.size(), QtCore.Qt.KeepAspectRatioByExpanding, QtCore.Qt.SmoothTransformation)
        x = (pm.width() - r.width()) // 2
        y = (pm.height() - r.height()) // 2
        p.drawPixmap(r, pm, QtCore.QRect(x, y, r.width(), r.height()))

class AspectContainer(QtWidgets.QWidget):
    def __init__(self, aspect_w: int, aspect_h: int, child: QtWidgets.QWidget, parent=None):
        super().__init__(parent)
        self._aw, self._ah = aspect_w, aspect_h
        self._child = child
        self._child.setParent(self)
        self.setAttribute(QtCore.Qt.WA_OpaquePaintEvent, True)

    def resizeEvent(self, e: QtGui.QResizeEvent):
        r = self.rect()
        if self._ah == 0:
            self._child.setGeometry(r)
            return
        target = self._aw / self._ah
        cur = r.width() / max(1, r.height())
        if cur > target:
            h = r.height()
            w = int(round(h * target))
            x = r.x() + (r.width() - w) // 2
            y = r.y()
            self._child.setGeometry(QtCore.QRect(x, y, w, h))
        else:
            w = r.width()
            h = int(round(w / target))
            x = r.x()
            y = r.y() + (r.height() - h) // 2
            self._child.setGeometry(QtCore.QRect(x, y, w, h))

    def paintEvent(self, e):
        p = QtGui.QPainter(self)
        p.fillRect(self.rect(), QtCore.Qt.black)
