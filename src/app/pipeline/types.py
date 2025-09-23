from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any
import numpy as np

@dataclass
class ObjInstance:
    name: str
    bbox_xyxy: Tuple[int, int, int, int]
    conf: float = 0.0
    source: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)

@dataclass
class FrameResult:
    frame: np.ndarray
    objects: List[ObjInstance]
