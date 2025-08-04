import cv2
import time
import numpy as np
import sys


class CameraInspector:
    def __init__(self, camera_index=0):
        self.cap = cv2.VideoCapture(camera_index)
        self.camera_index = camera_index

        if not self.cap.isOpened():
            raise RuntimeError(f"Unable to open camera #{camera_index}")

        # Performance metrics
        self.fps_counter = FPSCounter()
        self.frame_counter = 0
        self.start_time = time.time()
        self.last_frame_time = time.time()
        self.processing_time = 0
        self.latency = 0
        self.frame_drop = 0

        # Camera properties
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fourcc = self._get_fourcc_name()

        # Performance thresholds
        self.fps_thresholds = [25, 15]  # Good, Warning
        self.latency_thresholds = [33, 66]  # ms (Good, Warning)
        self.proc_thresholds = [15, 30]  # ms (Good, Warning)

        # Color encoding for status
        self.color_map = {
            'high': (0, 255, 0),  # Green: Good
            'medium': (0, 165, 255),  # Orange: Warning
            'low': (0, 0, 255)  # Red: Problem
        }

    def _get_fourcc_name(self):
        """Convert fourcc code to readable format"""
        try:
            fourcc_code = int(self.cap.get(cv2.CAP_PROP_FOURCC))
            return "".join([chr((fourcc_code >> 8 * i) & 0xFF) for i in range(4)])
        except:
            return "Unknown"

    def run(self):
        """Run the camera inspection tool"""
        print(f"Camera Inspector started - Camera #{self.camera_index}")
        print(f"Resolution: {self.width}x{self.height}")
        print(f"Format: {self.fourcc}")
        print("Press ESC to exit...")

        window_name = f'Camera #{self.camera_index}'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        try:
            while True:
                # Measure processing start time
                process_start = time.time()

                # Capture frame
                ret, frame = self.cap.read()
                if not ret:
                    self.frame_drop += 1
                    continue

                # Calculate frame timing
                current_time = time.time()
                frame_time = current_time - self.last_frame_time
                self.last_frame_time = current_time

                # Update performance metrics
                self.frame_counter += 1
                self.fps_counter.update()
                self.latency = frame_time
                self.processing_time = time.time() - process_start

                # Add performance overlay to the frame
                self._add_perf_overlay(frame)

                # Display the frame
                cv2.imshow(window_name, frame)

                # Handle ESC key
                if cv2.waitKey(1) == 27:  # ESC key
                    break
        finally:
            # Cleanup and final report
            elapsed = time.time() - self.start_time
            avg_fps = self.frame_counter / elapsed if elapsed > 0 else 0

            print("\n===== Performance Summary =====")
            print(f"Avg FPS: {avg_fps:.2f}")
            print(f"Frames processed: {self.frame_counter}")
            print(f"Frames dropped: {self.frame_drop}")
            print(f"Avg latency: {self.latency * 1000:.1f} ms")
            print(f"Avg processing time: {self.processing_time * 1000:.1f} ms")

            self.cap.release()
            cv2.destroyAllWindows()

    def _add_perf_overlay(self, frame):
        """Add optimized performance bar to the top-right corner"""
        fps = self.fps_counter.get_fps()
        latency_ms = self.latency * 1000
        proc_ms = self.processing_time * 1000

        # Calculate bar positions and widths
        bar_width = 200
        bar_height = 15
        bar_spacing = 5
        text_offset = 10
        panel_height = 3 * (bar_height + bar_spacing) + 30
        panel_width = bar_width + 100

        # Position the panel in top-right corner
        panel_x = frame.shape[1] - panel_width - 10
        panel_y = 10

        # Create semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y),
                      (panel_x + panel_width, panel_y + panel_height),
                      (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Draw header
        cv2.putText(frame, "PERFORMANCE",
                    (panel_x + 10, panel_y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 255), 1)

        # Draw FPS bar and text
        fps_bar_width = int(min(fps / 60.0, 1.0) * bar_width)
        fps_color = self._get_status_color(fps, self.fps_thresholds)
        cv2.rectangle(frame,
                      (panel_x + text_offset, panel_y + 30),
                      (panel_x + text_offset + fps_bar_width, panel_y + 30 + bar_height),
                      fps_color, -1)
        cv2.rectangle(frame,
                      (panel_x + text_offset, panel_y + 30),
                      (panel_x + text_offset + bar_width, panel_y + 30 + bar_height),
                      (100, 100, 100), 1)
        cv2.putText(frame, f"FPS: {fps:.1f}",
                    (panel_x + text_offset + bar_width + 10, panel_y + 30 + bar_height - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Draw Latency bar and text
        latency_bar_width = int(min(1.0 - latency_ms / 100.0, 1.0) * bar_width)
        latency_color = self._get_status_color(1000 / latency_ms if latency_ms > 0 else 1000,
                                               [1000 / self.latency_thresholds[0],
                                                1000 / self.latency_thresholds[1]])
        cv2.rectangle(frame,
                      (panel_x + text_offset, panel_y + 30 + bar_height + bar_spacing),
                      (panel_x + text_offset + latency_bar_width, panel_y + 30 + 2 * bar_height + bar_spacing),
                      latency_color, -1)
        cv2.rectangle(frame,
                      (panel_x + text_offset, panel_y + 30 + bar_height + bar_spacing),
                      (panel_x + text_offset + bar_width, panel_y + 30 + 2 * bar_height + bar_spacing),
                      (100, 100, 100), 1)
        cv2.putText(frame, f"LAT: {latency_ms:.1f}ms",
                    (panel_x + text_offset + bar_width + 10, panel_y + 30 + 2 * bar_height + bar_spacing - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Draw Processing Time bar and text
        proc_bar_width = int(min(1.0 - proc_ms / 100.0, 1.0) * bar_width)
        proc_color = self._get_status_color(1000 / proc_ms if proc_ms > 0 else 1000,
                                            [1000 / self.proc_thresholds[0],
                                             1000 / self.proc_thresholds[1]])
        cv2.rectangle(frame,
                      (panel_x + text_offset, panel_y + 30 + 2 * (bar_height + bar_spacing)),
                      (panel_x + text_offset + proc_bar_width, panel_y + 30 + 3 * bar_height + 2 * bar_spacing),
                      proc_color, -1)
        cv2.rectangle(frame,
                      (panel_x + text_offset, panel_y + 30 + 2 * (bar_height + bar_spacing)),
                      (panel_x + text_offset + bar_width, panel_y + 30 + 3 * bar_height + 2 * bar_spacing),
                      (100, 100, 100), 1)
        cv2.putText(frame, f"PROC: {proc_ms:.1f}ms",
                    (panel_x + text_offset + bar_width + 10, panel_y + 30 + 3 * bar_height + 2 * bar_spacing - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Draw camera info at the bottom of the panel
        cv2.putText(frame, f"{self.width}x{self.height} | {self.fourcc}",
                    (panel_x + 10, panel_y + panel_height + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 255), 1)

    def _get_status_color(self, value, thresholds):
        """Determine color based on performance thresholds"""
        if value > thresholds[0]:
            return self.color_map['high']
        elif value > thresholds[1]:
            return self.color_map['medium']
        return self.color_map['low']


class FPSCounter:
    """Frame rate counter with smoothing"""

    def __init__(self, buffer_size=30):
        self.times = []
        self.buffer_size = buffer_size
        self.last_time = time.time()

    def update(self):
        """Update with current frame"""
        current_time = time.time()
        elapsed = current_time - self.last_time
        self.last_time = current_time

        self.times.append(elapsed)
        if len(self.times) > self.buffer_size:
            self.times.pop(0)

    def get_fps(self):
        """Get current smoothed FPS value"""
        if not self.times:
            return 0.0
        avg_interval = sum(self.times) / len(self.times)
        return 1.0 / avg_interval if avg_interval > 0 else 0.0


if __name__ == "__main__":
    print("===== Camera Performance Monitor =====")
    print("Displays real-time performance metrics in top-right corner")
    print("Press ESC to exit")

    # Get camera index from command line
    camera_index = 0
    if len(sys.argv) > 1:
        try:
            camera_index = int(sys.argv[1])
        except:
            print(f"Invalid camera index: {sys.argv[1]}, using default 0")

    print(f"Starting with camera #{camera_index}...")

    try:
        inspector = CameraInspector(camera_index)
        inspector.run()
    except Exception as e:
        print(f"Error: {str(e)}")
        print("Trying other camera indices...")

        # Try all cameras until one works
        for idx in range(0, 4):
            if idx == camera_index:
                continue
            try:
                print(f"Attempting camera #{idx}...")
                inspector = CameraInspector(idx)
                inspector.run()
                break
            except Exception as e:
                print(f"Camera #{idx} failed: {str(e)}")
        else:
            print("No working cameras found. Please check connections.")