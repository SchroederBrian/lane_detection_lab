from __future__ import annotations

from typing import Dict, Any

import numpy as np

from config import Config
from image_processing import build_binary_mask
from perspective import PerspectiveTransformer
from lane_detector import LaneDetector
from renderer import LaneRenderer


class LaneDetectionPipeline:
    def __init__(self, config: Config):
        self.config = config
        self.perspective = PerspectiveTransformer(self.config)
        self.detector = LaneDetector(self.config)
        self.renderer = LaneRenderer(self.config, self.perspective, self.detector)

    def process_frame(self, frame: np.ndarray, fps: float = 0.0) -> tuple[np.ndarray, Dict[str, Any]]:
        return self.renderer.process_frame(frame, build_binary_mask, fps)

    def reset(self) -> None:
        self.perspective.reset()
        self.detector.reset()
