from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple
import numpy as np
import cv2
from PySide6 import QtCore

import mss

from config import Config
from perspective import PerspectiveTransformer
from lane_detector import LaneDetector
from renderer import LaneRenderer
from image_processing import build_binary_mask
# YOLO removed


@dataclass
class MonitorInfo:
    index: int
    left: int
    top: int
    width: int
    height: int

    @property
    def bbox(self) -> Dict[str, int]:
        return {"left": self.left, "top": self.top, "width": self.width, "height": self.height}

    def label(self) -> str:
        return f"Monitor {self.index} ({self.width}x{self.height} @ {self.left},{self.top})"


def list_monitors() -> List[MonitorInfo]:
    monitors: List[MonitorInfo] = []
    with mss.mss() as sct:
        # sct.monitors[0] is virtual bounding box; use 1..N as actual monitors
        for i, mon in enumerate(sct.monitors):
            if i == 0:
                continue
            monitors.append(MonitorInfo(index=i, left=mon["left"], top=mon["top"], width=mon["width"], height=mon["height"]))
    return monitors


class ScreenCaptureWorker(QtCore.QThread):
    frameReady = QtCore.Signal(np.ndarray)
    debugBinaryReady = QtCore.Signal(np.ndarray)
    debugWarpReady = QtCore.Signal(np.ndarray)
    finished = QtCore.Signal()

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self._running = False
        self._pause = False
        self._reset_requested = False
        self._reset_lock = QtCore.QMutex()
        self._monitor_bbox: Optional[Dict[str, int]] = None
        self.perspective = PerspectiveTransformer(self.config)
        self.detector = LaneDetector(self.config)
        self.renderer = LaneRenderer(self.config, self.perspective, self.detector)
        # No YOLO

    def set_monitor_bbox(self, bbox: Dict[str, int]) -> None:
        self._monitor_bbox = bbox

    def stop(self) -> None:
        self._running = False

    def toggle_pause(self) -> None:
        self._pause = not self._pause

    def request_reset(self) -> None:
        locker = QtCore.QMutexLocker(self._reset_lock)
        self._reset_requested = True
        del locker

    def _apply_reset_if_needed(self) -> None:
        need_reset = False
        locker = QtCore.QMutexLocker(self._reset_lock)
        if self._reset_requested:
            need_reset = True
            self._reset_requested = False
        del locker
        if need_reset:
            self.perspective.reset()
            self.detector.reset()

    def run(self) -> None:
        if self._monitor_bbox is None:
            self.finished.emit()
            return
        self._running = True
        with mss.mss() as sct:
            while self._running:
                if self._pause:
                    self.msleep(20)
                    continue
                self._apply_reset_if_needed()
                img = sct.grab(self._monitor_bbox)
                frame = np.array(img)  # BGRA
                if frame.shape[2] == 4:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

                out_frame, debug = self.renderer.process_frame(frame, build_binary_mask)

                # No YOLO overlay

                self.frameReady.emit(out_frame)
                self.debugBinaryReady.emit(debug["binary"])  # type: ignore[arg-type]
                self.debugWarpReady.emit(debug["warped_binary"])  # type: ignore[arg-type]

                # Limit FPS ~30
                self.msleep(15)
        self.finished.emit()


