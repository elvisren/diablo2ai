import time
import torch
from ultralytics import YOLO
import random


class YoloTracker:
    """YOLO object tracker with configurable parameters"""

    def __init__(self,
                 model_path,
                 conf_threshold=0.5,
                 iou_threshold=0.5,
                 device='auto',
                 img_size=640,
                 tracker='botsort.yaml',
                 verbose=False):
        """
        Initialize YOLO tracker with tunable parameters

        Args:
            model_path (str): Path to .pt model file
            conf_threshold (float): Confidence threshold (0-1)
            iou_threshold (float): IOU threshold for NMS (0-1)
            device (str): 'auto', 'cuda', 'mps', or 'cpu'
            img_size (int): Input image size
            tracker (str): Tracking algorithm configuration
            verbose (bool): Show model loading details
        """
        # Device configuration
        if device == 'auto':
            self.device = 'mps' if torch.backends.mps.is_available() else \
                'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        self.model = YOLO(model_path)
        self.model.to(self.device)
        self.model.fuse()

        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.img_size = img_size
        self.tracker = tracker
        self.class_names = self.model.names
        self.track_colors = {}

        if verbose:
            print(f"Loaded YOLO tracker on {self.device}: {model_path}")
            print(f"Tracker: {tracker}")
            print(f"Input size: {img_size}x{img_size}")
            print(f"Confidence: {conf_threshold}, IOU: {iou_threshold}")
            print(f"Classes: {list(self.class_names.values())}")

    def _get_track_color(self, track_id):
        """Get consistent color for a track ID"""
        if track_id not in self.track_colors:
            # Generate a random but consistent color
            self.track_colors[track_id] = (
                random.randint(50, 200),
                random.randint(50, 200),
                random.randint(50, 200)
            )
        return self.track_colors[track_id]

    def track(self, frame, persist=True):
        """Run object tracking on frame and return tracks + inference time"""
        start_time = time.time()

        # Perform tracking
        results = self.model.track(
            frame,  # Assume frame is already RGB
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            imgsz=self.img_size,
            device=self.device,
            persist=persist,
            tracker=self.tracker,
            verbose=False
        )

        inference_time = time.time() - start_time
        tracks = []

        if results and results[0].boxes is not None and results[0].boxes.id is not None:
            for box in results[0].boxes:
                if box.id is not None:
                    tracks.append({
                        'bbox': box.xyxy[0].tolist(),
                        'conf': box.conf.item(),
                        'class': int(box.cls.item()),
                        'class_name': self.class_names[int(box.cls.item())],
                        'track_id': int(box.id.item()),
                        'color': self._get_track_color(int(box.id.item()))
                    })

        return tracks, inference_time