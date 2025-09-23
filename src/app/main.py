#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import sys
from PyQt5 import QtWidgets

from src.app.gui.controller import MainWindow


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
