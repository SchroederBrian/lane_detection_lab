from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Dict

import cv2
import mss
import numpy as np
from PySide6 import QtCore

import time
from queue import Queue, Full

from app.pipeline import LaneDetectionPipeline
from config import Config


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
        for i, mon in enumerate(sct.monitors):
            if i == 0:
                continue
            monitors.append(
                MonitorInfo(index=i, left=mon["left"], top=mon["top"], width=mon["width"], height=mon["height"])
            )
    return monitors


class ScreenCaptureWorker(QtCore.QThread):
    frameReady = QtCore.Signal(np.ndarray)
    debugBinaryReady = QtCore.Signal(np.ndarray)
    debugWarpReady = QtCore.Signal(np.ndarray)
    fpsUpdated = QtCore.Signal(float)  # New signal
    finished = QtCore.Signal()
    debugReady = QtCore.Signal(dict)

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self._running = False
        self._pause = False
        self._reset_requested = False
        self._reset_lock = QtCore.QMutex()
        self._monitor_bbox: Optional[Dict[str, int]] = None
        self.pipeline = LaneDetectionPipeline(self.config)
        self.frame_queue = Queue(maxsize=3)
        self.last_time = time.time()
        self.frame_count = 0
        self.fps = 0.0

    def set_monitor_bbox(self, bbox: Dict[str, int]) -> None:
        self._monitor_bbox = bbox

    def stop(self) -> None:
        self._running = False

    def toggle_pause(self) -> None:
        self._pause = not self._pause

    def request_reset(self) -> None:
        with QtCore.QMutexLocker(self._reset_lock):
            self._reset_requested = True

    def _apply_reset_if_needed(self) -> None:
        need_reset = False
        with QtCore.QMutexLocker(self._reset_lock):
            if self._reset_requested:
                need_reset = True
                self._reset_requested = False
        if need_reset:
            self.pipeline.reset()

    def run(self) -> None:
        try:
            if self._monitor_bbox is None:
                self.finished.emit()
                return
            self._running = True
            with mss.mss() as sct:
                while self._running:
                    if self._pause:
                        self.msleep(20)
                        continue

                    img = sct.grab(self._monitor_bbox)
                    frame = np.array(img)
                    if frame.shape[2] == 4:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

                    try:
                        self.frame_queue.put_nowait(frame)
                    except Full:
                        continue

                    if not self.frame_queue.empty():
                        frame = self.frame_queue.get()
                        self._apply_reset_if_needed()
                        out_frame, debug = self.pipeline.process_frame(frame, self.fps)
                        self.frameReady.emit(out_frame)
                        self.debugBinaryReady.emit(debug["binary"])
                        self.debugWarpReady.emit(debug["warped_binary"])
                        self.debugReady.emit(debug)

                        self.frame_count += 1
                        current_time = time.time()
                        elapsed = current_time - self.last_time
                        if elapsed >= 1.0:
                            self.fps = self.frame_count / elapsed
                            self.fpsUpdated.emit(self.fps)
                            self.frame_count = 0
                            self.last_time = current_time

                    self.msleep(15)
            self.finished.emit()
        except Exception as e:
            import logging
            logging.exception("Exception in ScreenCaptureWorker run")
            self.finished.emit()
