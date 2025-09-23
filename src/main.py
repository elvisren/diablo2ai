#!/usr/bin/env python3
import sys, signal
from PyQt5 import QtCore, QtWidgets
from src.backup.common.logging_config import setup_logging
from src.backup.gui.main_window import MainWindow

def main():
    setup_logging()
    app = QtWidgets.QApplication(sys.argv)
    signal.signal(signal.SIGINT, lambda *_: app.quit())
    timer = QtCore.QTimer()
    timer.start(100)
    timer.timeout.connect(lambda: None)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
