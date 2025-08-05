import cv2
import time


class VideoProcessor:
    """Handles camera/video input with frame processing timing"""

    def __init__(self, source, is_video=False):
        """
        Initialize video source

        Args:
            source (int/str): Camera index or video file path
            is_video (bool): True if source is video file
        """
        self.source = source
        self.is_video = is_video

        # Initialize capture
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Unable to open source: {source}")

        # Get source properties
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.target_fps = self.cap.get(cv2.CAP_PROP_FPS)

        # Validate and set default fps if needed
        if self.target_fps <= 0:
            self.target_fps = 30.0

        self.frame_delay = 1.0 / self.target_fps
        self.source_info = self._get_source_info()

        # Timing control
        self.last_frame_time = 0
        self.processing_time = 0
        self.latency = 0

        # Frame handling
        self.frame_count = 0
        self.dropped_frames = 0
        self.next_frame_time = time.time()

    def _get_source_info(self):
        """Build source information string"""
        if self.is_video:
            return f"Video: {self.source} ({self.width}x{self.height} @ {self.target_fps:.1f}fps)"
        else:
            return f"Camera {self.source} ({self.width}x{self.height} @ {self.target_fps:.1f}fps)"

    def get_frame(self):
        """Get next frame with timing control"""
        current_time = time.time()

        # For camera: Always grab next frame
        if not self.is_video:
            ret, frame = self.cap.read()
            if ret:
                self.frame_count += 1
                self.next_frame_time = current_time + self.frame_delay
                self.latency = time.time() - current_time
                return frame
            return None

        # For video files: Handle frame skipping if behind schedule
        time_diff = current_time - self.next_frame_time
        if time_diff < 0:
            return None

        # Calculate how many frames to skip to catch up
        skip_frames = max(0, int(time_diff * self.target_fps) - 1)
        self.dropped_frames += skip_frames

        # Skip frames if necessary
        for _ in range(skip_frames):
            if not self.cap.grab():
                return None
            self.next_frame_time += self.frame_delay
            self.dropped_frames += 1

        # Read the next frame to show
        ret, frame = self.cap.read()
        if ret:
            self.frame_count += 1
            self.next_frame_time += self.frame_delay
            self.latency = time.time() - current_time
            return frame

        return None

    def release(self):
        """Release video resources"""
        self.cap.release()

    def get_source_info(self):
        """Get information about the video source"""
        return self.source_info

    def get_frame_size(self):
        """Get frame dimensions (width, height)"""
        return (self.width, self.height)