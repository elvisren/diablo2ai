#!/usr/bin/env python3
# yolo_dataset_viewer.py
# Browse a YOLOv11 dataset: shows image list and overlays label bboxes.
# Usage: python yolo_dataset_viewer.py /path/to/dataset_root
#
# Expects structure:
#  dataset_root/
#    classes.txt                 # optional (one class per line)
#    dataset.yaml                # optional (names: [..] or dict)
#    images/
#      train/  *.jpg|png|webp...
#      val/    ...
#    labels/
#      train/  *.txt (YOLO format)
#      val/    *.txt
#
# YOLO label line: <cls_id> <xc> <yc> <w> <h>   (all normalized 0..1)

import sys, os, math, pathlib, traceback
from typing import List, Tuple, Optional, Dict

from PyQt5 import QtCore, QtGui, QtWidgets
import yaml
import cv2
import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def read_classes(dataset_dir: pathlib.Path) -> List[str]:
    # 1) classes.txt (one per line)
    classes_txt = dataset_dir / "classes.txt"
    if classes_txt.exists():
        with classes_txt.open("r", encoding="utf-8") as f:
            names = [line.strip() for line in f if line.strip() != ""]
        if names:
            return names

    # 2) dataset.yaml (YOLO-style)
    data_yaml = dataset_dir / "data.yaml"
    if data_yaml.exists():
        try:
            with data_yaml.open("r", encoding="utf-8") as f:
                y = yaml.safe_load(f)
            # names can be list or dict {id:name}
            if isinstance(y, dict) and "names" in y:
                names_obj = y["names"]
                if isinstance(names_obj, list):
                    return [str(n) for n in names_obj]
                elif isinstance(names_obj, dict):
                    # convert to list in id order
                    max_id = max(int(k) for k in names_obj.keys())
                    res = []
                    for i in range(max_id + 1):
                        res.append(str(names_obj.get(i, names_obj.get(str(i), f"class_{i}"))))
                    return res
        except Exception:
            pass

    return []  # unknown, will display numeric ids

def yolo_txt_to_boxes(txt_path: pathlib.Path) -> List[Tuple[int, float, float, float, float]]:
    """Return list of (cls, xc, yc, w, h) normalized floats."""
    boxes = []
    if not txt_path.exists():
        return boxes
    try:
        with txt_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 5:
                    continue
                cls_id = int(float(parts[0]))
                xc, yc, w, h = list(map(float, parts[1:5]))
                boxes.append((cls_id, xc, yc, w, h))
    except Exception:
        # ignore malformed files
        pass
    return boxes

def norm_to_xyxy(xc, yc, w, h, img_w, img_h):
    x = (xc - w/2.0) * img_w
    y = (yc - h/2.0) * img_h
    bw = w * img_w
    bh = h * img_h
    return int(round(x)), int(round(y)), int(round(x + bw)), int(round(y + bh))

def find_images(split_dir: pathlib.Path) -> List[pathlib.Path]:
    if not split_dir.exists():
        return []
    imgs = []
    for p in split_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            imgs.append(p)
    imgs.sort()
    return imgs

class ImageCanvas(QtWidgets.QGraphicsView):
    """Zoomable, pannable canvas that draws image and boxes."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setRenderHint(QtGui.QPainter.Antialiasing)
        self.setRenderHint(QtGui.QPainter.SmoothPixmapTransform)
        self.setViewportUpdateMode(QtWidgets.QGraphicsView.FullViewportUpdate)
        self.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)

        self._scene = QtWidgets.QGraphicsScene(self)
        self.setScene(self._scene)

        self._pix_item: Optional[QtWidgets.QGraphicsPixmapItem] = None
        self._bbox_items: List[QtWidgets.QGraphicsItem] = []
        self._class_names: List[str] = []
        self._show_labels = True

    def set_class_names(self, names: List[str]):
        self._class_names = names or []

    def clear(self):
        self._scene.clear()
        self._pix_item = None
        self._bbox_items = []

    def fit_to_view(self):
        if self._pix_item:
            self.fitInView(self._pix_item, QtCore.Qt.KeepAspectRatio)

    def wheelEvent(self, event: QtGui.QWheelEvent):
        if event.angleDelta().y() > 0:
            factor = 1.15
        else:
            factor = 1 / 1.15
        self.scale(factor, factor)

    def load_image_with_boxes(self, img_bgr: np.ndarray,
                              boxes: List[Tuple[int, float, float, float, float]]):
        self.clear()

        # Convert BGR to RGB -> QPixmap
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = img_rgb.shape
        bytes_per_line = ch * w
        qimg = QtGui.QImage(img_rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        pm = QtGui.QPixmap.fromImage(qimg)
        self._pix_item = self._scene.addPixmap(pm)
        self._pix_item.setZValue(0)

        # Draw boxes as vector items
        font = QtGui.QFont()
        font.setPointSize(10)
        pen = QtGui.QPen(QtCore.Qt.white)
        pen.setWidth(2)

        for (cls_id, xc, yc, bw, bh) in boxes:
            x1, y1, x2, y2 = norm_to_xyxy(xc, yc, bw, bh, w, h)
            rect_item = self._scene.addRect(QtCore.QRectF(x1, y1, x2 - x1, y2 - y1), pen)
            rect_item.setZValue(1)
            self._bbox_items.append(rect_item)

            if self._show_labels:
                label = self._class_names[cls_id] if 0 <= cls_id < len(self._class_names) else f"{cls_id}"
                text_item = self._scene.addText(label, font)
                text_item.setDefaultTextColor(QtCore.Qt.white)
                # Text background via a rectangle behind text
                br = text_item.boundingRect()
                bg = self._scene.addRect(0, 0, br.width() + 6, br.height() + 4,
                                         QtGui.QPen(QtCore.Qt.NoPen),
                                         QtGui.QBrush(QtGui.QColor(0, 0, 0, 160)))
                # position both
                tx = x1 + 2
                ty = max(0, y1 - br.height() - 6)
                bg.setPos(tx, ty)
                text_item.setPos(tx + 3, ty + 2)
                bg.setZValue(2)
                text_item.setZValue(3)
                self._bbox_items.extend([bg, text_item])

        self.setSceneRect(self._scene.itemsBoundingRect())
        self.fit_to_view()

class DatasetModel(QtCore.QObject):
    """Keeps track of dataset paths and provides image+label resolution."""
    def __init__(self, root: pathlib.Path, class_names: List[str]):
        super().__init__()
        self.root = root
        self.class_names = class_names
        self.images_dir = root / "images"
        self.labels_dir = root / "labels"
        self.splits = ["train", "val"]
        self.image_lists: Dict[str, List[pathlib.Path]] = {s: [] for s in self.splits}
        self._index_images()

    def _index_images(self):
        for s in self.splits:
            self.image_lists[s] = find_images(self.images_dir / s)

    def splits_available(self) -> List[str]:
        return [s for s in self.splits if len(self.image_lists.get(s, [])) > 0]

    def list_images(self, split: str) -> List[pathlib.Path]:
        return self.image_lists.get(split, [])

    def label_path_for(self, image_path: pathlib.Path, split: str) -> pathlib.Path:
        # mirror filename (with .txt) under labels/<split> relative to images/<split>
        # Support images possibly nested in subfolders.
        try:
            rel = image_path.relative_to(self.images_dir / split)
        except Exception:
            rel = image_path.name  # fallback
        rel_txt = pathlib.Path(rel).with_suffix(".txt")
        return self.labels_dir / split / rel_txt

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, dataset_dir: pathlib.Path):
        super().__init__()
        self.setWindowTitle("YOLOv11 Dataset Viewer")
        self.resize(1280, 800)

        # Load classes first
        self.class_names = read_classes(dataset_dir)
        self.model = DatasetModel(dataset_dir, self.class_names)

        # UI
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QHBoxLayout(central)
        layout.setContentsMargins(6,6,6,6)
        layout.setSpacing(8)

        # Left panel: split selector + search + list
        left_panel = QtWidgets.QVBoxLayout()
        layout.addLayout(left_panel, 1)

        header = QtWidgets.QHBoxLayout()
        left_panel.addLayout(header)

        self.split_combo = QtWidgets.QComboBox()
        splits = self.model.splits_available() or self.model.splits
        self.split_combo.addItems(splits)
        header.addWidget(QtWidgets.QLabel("Split:"))
        header.addWidget(self.split_combo)

        self.search_edit = QtWidgets.QLineEdit()
        self.search_edit.setPlaceholderText("Filter (filename substring)...")
        header.addWidget(self.search_edit)

        self.list_widget = QtWidgets.QListWidget()
        self.list_widget.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        left_panel.addWidget(self.list_widget, 1)

        # Status line for counts
        self.count_label = QtWidgets.QLabel("")
        left_panel.addWidget(self.count_label)

        # Right panel: toolbar + canvas
        right_panel = QtWidgets.QVBoxLayout()
        layout.addLayout(right_panel, 3)

        toolrow = QtWidgets.QHBoxLayout()
        right_panel.addLayout(toolrow)

        self.btn_fit = QtWidgets.QPushButton("Fit")
        self.btn_reset = QtWidgets.QPushButton("Reset View")
        self.chk_labels = QtWidgets.QCheckBox("Show Labels")
        self.chk_labels.setChecked(True)
        self.info_label = QtWidgets.QLabel("")
        self.info_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)

        toolrow.addWidget(self.btn_fit)
        toolrow.addWidget(self.btn_reset)
        toolrow.addWidget(self.chk_labels)
        toolrow.addStretch(1)
        toolrow.addWidget(self.info_label)

        self.canvas = ImageCanvas()
        right_panel.addWidget(self.canvas, 1)
        self.canvas.setBackgroundBrush(QtGui.QColor(30,30,30))

        # bottom: class legend (scrollable)
        self.legend = QtWidgets.QScrollArea()
        self.legend.setWidgetResizable(True)
        legend_inner = QtWidgets.QWidget()
        self.legend.setWidget(legend_inner)
        self.legend_layout = QtWidgets.QVBoxLayout(legend_inner)
        self.legend_layout.setContentsMargins(6,6,6,6)
        self.legend_layout.setSpacing(2)
        right_panel.addWidget(self.legend, 0)
        self._refresh_legend()

        # signals
        self.split_combo.currentTextChanged.connect(self.refresh_list)
        self.search_edit.textChanged.connect(self.refresh_list)
        self.list_widget.currentItemChanged.connect(self._on_item_selected)
        self.btn_fit.clicked.connect(self.canvas.fit_to_view)
        self.btn_reset.clicked.connect(self._reset_view)
        self.chk_labels.toggled.connect(self._toggle_labels)

        # init
        self.canvas.set_class_names(self.class_names)
        self.refresh_list()

        # keyboard shortcuts
        QtWidgets.QShortcut(QtGui.QKeySequence("F"), self, activated=self.canvas.fit_to_view)
        QtWidgets.QShortcut(QtGui.QKeySequence("R"), self, activated=self._reset_view)

    def _toggle_labels(self, checked: bool):
        # Simply reload the current item with labels toggled
        self._on_item_selected(self.list_widget.currentItem(), None)

    def _reset_view(self):
        self.canvas.resetTransform()
        self.canvas.fit_to_view()

    def _refresh_legend(self):
        # clear
        while self.legend_layout.count():
            item = self.legend_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()
        if not self.class_names:
            lab = QtWidgets.QLabel("No class names found (classes.txt or data.yaml). Showing numeric ids.")
            self.legend_layout.addWidget(lab)
            return
        title = QtWidgets.QLabel("Classes:")
        title.setStyleSheet("font-weight: 600;")
        self.legend_layout.addWidget(title)
        for i, name in enumerate(self.class_names):
            self.legend_layout.addWidget(QtWidgets.QLabel(f"{i}: {name}"))
        self.legend_layout.addStretch(1)

    def refresh_list(self):
        split = self.split_combo.currentText()
        imgs = self.model.list_images(split)
        q = self.search_edit.text().strip().lower()
        if q:
            imgs = [p for p in imgs if q in str(p.name).lower()]
        self.list_widget.clear()
        for p in imgs:
            item = QtWidgets.QListWidgetItem(str(p.relative_to(self.model.root)))
            item.setData(QtCore.Qt.UserRole, str(p))
            self.list_widget.addItem(item)
        self.count_label.setText(f"{len(imgs)} image(s)")
        if self.list_widget.count() > 0:
            self.list_widget.setCurrentRow(0)
        else:
            self.canvas.clear()
            self.info_label.setText("")

    def _on_item_selected(self, current: Optional[QtWidgets.QListWidgetItem], prev):
        if not current:
            self.canvas.clear()
            self.info_label.setText("")
            return
        try:
            img_path = pathlib.Path(current.data(QtCore.Qt.UserRole))
            split = self.split_combo.currentText()
            lbl_path = self.model.label_path_for(img_path, split)
            img = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)  # handles unicode paths
            if img is None:
                raise RuntimeError(f"Failed to read image: {img_path}")
            boxes = yolo_txt_to_boxes(lbl_path)
            # respect "show labels" checkbox by switching canvas flag then drawing
            self.canvas._show_labels = self.chk_labels.isChecked()
            self.canvas.load_image_with_boxes(img, boxes)
            h, w = img.shape[:2]
            self.info_label.setText(f"{img_path.name}  |  {w}×{h}  |  {len(boxes)} box(es)")
        except Exception as e:
            traceback.print_exc()
            self.canvas.clear()
            self.info_label.setText(f"Error: {e}")

def main():
    app = QtWidgets.QApplication(sys.argv)
    if len(sys.argv) < 2:
        path = QtWidgets.QFileDialog.getExistingDirectory(None, "Select YOLO Dataset Root")
        if not path:
            sys.exit(0)
        dataset_dir = pathlib.Path(path)
    else:
        dataset_dir = pathlib.Path(sys.argv[1])

    if not dataset_dir.exists():
        QtWidgets.QMessageBox.critical(None, "Error", f"Dataset root not found:\n{dataset_dir}")
        sys.exit(1)

    w = MainWindow(dataset_dir)
    w.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
