from __future__ import annotations
import numpy as np
from .types import FrameResult

class BaseNode:
    def __init__(self, name: str):
        self.name = name
        self.enabled = True

    def process(self, frame: np.ndarray) -> FrameResult:
        raise NotImplementedError
