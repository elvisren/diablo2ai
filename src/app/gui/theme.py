from __future__ import annotations
from PyQt5 import QtWidgets, QtGui, QtCore

def apply_dark_theme(widget: QtWidgets.QWidget):
    QtWidgets.QApplication.setStyle("Fusion")
    pal = QtGui.QPalette()
    pal.setColor(QtGui.QPalette.Window, QtGui.QColor(37, 37, 38))
    pal.setColor(QtGui.QPalette.WindowText, QtCore.Qt.white)
    pal.setColor(QtGui.QPalette.Base, QtGui.QColor(30, 30, 30))
    pal.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor(37, 37, 38))
    pal.setColor(QtGui.QPalette.ToolTipBase, QtCore.Qt.white)
    pal.setColor(QtGui.QPalette.ToolTipText, QtCore.Qt.white)
    pal.setColor(QtGui.QPalette.Text, QtCore.Qt.white)
    pal.setColor(QtGui.QPalette.Button, QtGui.QColor(45, 45, 48))
    pal.setColor(QtGui.QPalette.ButtonText, QtCore.Qt.white)
    pal.setColor(QtGui.QPalette.BrightText, QtCore.Qt.red)
    pal.setColor(QtGui.QPalette.Highlight, QtGui.QColor(14, 99, 156))
    pal.setColor(QtGui.QPalette.HighlightedText, QtCore.Qt.black)
    widget.setPalette(pal)
