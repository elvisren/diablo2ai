from __future__ import annotations
from typing import List, Tuple
import time
from .interfaces import FrameBundle, PipelineStage, VideoSource

class Pipeline:
    def __init__(self, source: VideoSource, stages: List[PipelineStage]):
        self.source = source
        self.stages = stages

    def open(self):
        self.source.open()

    def close(self):
        self.source.close()

    def step(self) -> tuple[bool, FrameBundle]:
        ok, frame = self.source.read()
        bundle = FrameBundle(raw_bgr=frame, tstamp=time.perf_counter())
        if not ok:
            return False, bundle
        for s in self.stages:
            bundle = s.process(bundle)
        return True, bundle
