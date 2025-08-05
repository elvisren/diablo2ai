import cv2


class Visualizer:
    """Handles annotation drawing on frames for tracking results"""

    def __init__(self,
                 perf_width=250,
                 perf_height=20,
                 perf_spacing=5,
                 font_scale=0.5,
                 font_thickness=1):
        """
        Initialize visualizer with drawing parameters

        Args:
            perf_width (int): Performance bar width
            perf_height (int): Performance bar height
            perf_spacing (int): Spacing between bars
            font_scale (float): Text font scale
            font_thickness (int): Text thickness
        """
        # Drawing parameters
        self.perf_width = perf_width
        self.perf_height = perf_height
        self.perf_spacing = perf_spacing
        self.font_scale = font_scale
        self.font_thickness = font_thickness
        self.panel_height = 0  # Will calculate dynamically
        self.padding = 10

        # Color scheme
        self.color_map = {
            'high': (0, 255, 0),  # Green: Good
            'medium': (0, 165, 255),  # Orange: Warning
            'low': (0, 0, 255)  # Red: Problem
        }

        # Performance thresholds
        self.fps_thresholds = [25, 15]  # Good, Warning
        self.latency_thresholds = [33, 66]  # ms
        self.proc_thresholds = [15, 30]  # ms

    def _get_status_color(self, value, thresholds):
        """Determine color based on performance thresholds"""
        if value > thresholds[0]:
            return self.color_map['high']
        elif value > thresholds[1]:
            return self.color_map['medium']
        return self.color_map['low']

    def draw_tracks(self, frame, tracks):
        """Draw bounding boxes, track IDs, and labels on frame"""
        for track in tracks:
            x1, y1, x2, y2 = map(int, track['bbox'])
            class_name = track['class_name']
            conf = track['conf']
            track_id = track['track_id']
            color = track['color']

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Draw filled background for labels
            label_text = f"{track_id}: {class_name} {conf:.2f}"
            (label_width, label_height), _ = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, self.font_thickness)

            # Draw background rectangle
            cv2.rectangle(frame,
                          (x1, y1 - label_height - 8),
                          (x1 + label_width + 6, y1),
                          color, -1)

            # Draw label text
            cv2.putText(frame, label_text,
                        (x1 + 3, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        self.font_scale,
                        (255, 255, 255),
                        self.font_thickness)

    def draw_performance(self, frame, fps, latency_ms, proc_ms, source_info,
                         det_time=0, obj_count=0):
        """Draw performance overlay in top-right corner"""
        h, w = frame.shape[:2]

        # Calculate panel dimensions
        bars = 3
        self.panel_height = self.padding * 3 + bars * (self.perf_height + self.perf_spacing)
        self.panel_width = self.perf_width + 120
        panel_x = w - self.panel_width - 10
        panel_y = 10

        # Create semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay,
                      (panel_x, panel_y),
                      (panel_x + self.panel_width, panel_y + self.panel_height),
                      (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Draw header
        cv2.putText(frame, "PERFORMANCE METRICS",
                    (panel_x + 10, panel_y + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 255), 1)

        # Draw FPS bar and text
        fps_bar_ratio = min(fps / 60.0, 1.0)
        fps_color = self._get_status_color(fps, self.fps_thresholds)
        self._draw_bar(frame, panel_x, panel_y, 30,
                       fps_bar_ratio, fps_color,
                       f"FPS: {fps:.1f}", bar_index=0)

        # Draw Latency bar and text
        latency_bar_ratio = min(1.0 - latency_ms / 100.0, 1.0)
        latency_color = self._get_status_color(
            1000 / latency_ms if latency_ms > 0 else 1000,
            [1000 / lt for lt in self.latency_thresholds]
        )
        self._draw_bar(frame, panel_x, panel_y,
                       30 + self.perf_height + self.perf_spacing,
                       latency_bar_ratio, latency_color,
                       f"LAT: {latency_ms:.1f}ms", bar_index=1)

        # Draw Processing Time bar and text
        proc_bar_ratio = min(1.0 - proc_ms / 100.0, 1.0)
        proc_color = self._get_status_color(
            1000 / proc_ms if proc_ms > 0 else 1000,
            [1000 / pt for pt in self.proc_thresholds]
        )
        self._draw_bar(frame, panel_x, panel_y,
                       30 + 2 * (self.perf_height + self.perf_spacing),
                       proc_bar_ratio, proc_color,
                       f"PROC: {proc_ms:.1f}ms", bar_index=2)

        # Draw source info
        cv2.putText(frame, source_info,
                    (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (100, 255, 255), 1)

        # Draw detection info if available
        if det_time > 0:
            det_text = f"TRACK: {det_time:.1f}ms | Objects: {obj_count}"
            cv2.putText(frame, det_text, (10, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    def _draw_bar(self, frame, panel_x, panel_y, y_offset, fill_ratio, color, text, bar_index=0):
        """Draw individual performance bar"""
        # Calculate bar position
        y_pos = panel_y + y_offset + bar_index * (self.perf_height + self.perf_spacing)
        bar_width = int(fill_ratio * self.perf_width)
        bar_x = panel_x + 15

        # Draw filled portion
        cv2.rectangle(frame,
                      (bar_x, y_pos),
                      (bar_x + bar_width, y_pos + self.perf_height),
                      color, -1)

        # Draw outline
        cv2.rectangle(frame,
                      (bar_x, y_pos),
                      (bar_x + self.perf_width, y_pos + self.perf_height),
                      (100, 100, 100), 1)

        # Draw text
        text_x = bar_x + self.perf_width + 10
        cv2.putText(frame, text,
                    (text_x, y_pos + self.perf_height - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, self.font_scale,
                    (255, 255, 255), self.font_thickness)