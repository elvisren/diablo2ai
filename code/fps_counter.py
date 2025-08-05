import time
import numpy as np


class FPSCounter:
    """Frame rate counter with smoothing"""

    def __init__(self, buffer_size=30):
        self.times = np.zeros(buffer_size)
        self.buffer_size = buffer_size
        self.last_time = time.time()
        self.index = 0
        self.count = 0

    def update(self):
        """Update with current frame"""
        current_time = time.time()
        elapsed = current_time - self.last_time
        self.last_time = current_time

        self.times[self.index] = elapsed
        self.index = (self.index + 1) % self.buffer_size
        self.count = min(self.count + 1, self.buffer_size)

    def get_fps(self):
        """Get current smoothed FPS value"""
        if self.count == 0:
            return 0.0

        # Only consider non-zero times in the buffer
        non_zero_times = self.times[self.times > 0][:self.count]
        if len(non_zero_times) == 0:
            return 0.0

        avg_interval = np.mean(non_zero_times)
        return 1.0 / avg_interval if avg_interval > 0 else 0.0