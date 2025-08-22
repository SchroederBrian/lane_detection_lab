from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
from PySide6 import QtCore

from app.pipeline import LaneDetectionPipeline
from config import Config
import time
from queue import Queue, Full
import logging


class VideoWorker(QtCore.QThread):
    frameReady = QtCore.Signal(np.ndarray)
    debugBinaryReady = QtCore.Signal(np.ndarray)
    debugWarpReady = QtCore.Signal(np.ndarray)
    fpsUpdated = QtCore.Signal(float)  # New signal for FPS
    finished = QtCore.Signal()
    debugReady = QtCore.Signal(dict)

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.video_path: Optional[str] = None
        self._running = False
        self._pause = False
        self._reset_requested = False
        self._reset_lock = QtCore.QMutex()
        self.pipeline = LaneDetectionPipeline(self.config)
        self.frame_queue = Queue(maxsize=3)
        self.last_time = time.time()
        self.frame_count = 0
        self.fps = 0.0

    def set_video(self, path: str) -> None:
        self.video_path = path

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
            if not self.video_path:
                self.finished.emit()
                return

            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                logging.error(f"Failed to open video: {self.video_path}")
                self.finished.emit()
                return

            self._running = True
            while self._running:
                if self._pause:
                    self.msleep(20)
                    continue

                ret, frame = cap.read()
                if not ret:
                    break

                try:
                    self.frame_queue.put_nowait(frame)
                except Full:
                    continue  # Drop frame if queue full

                if not self.frame_queue.empty():
                    frame = self.frame_queue.get()
                    self._apply_reset_if_needed()
                    out_frame, debug = self.pipeline.process_frame(frame, self.fps)
                    self.frameReady.emit(out_frame)
                    self.debugBinaryReady.emit(debug["binary"])
                    self.debugWarpReady.emit(debug["warped_binary"])
                    self.debugReady.emit(debug)

                    # Calculate FPS
                    self.frame_count += 1
                    current_time = time.time()
                    elapsed = current_time - self.last_time
                    if elapsed >= 1.0:
                        self.fps = self.frame_count / elapsed
                        self.fpsUpdated.emit(self.fps)
                        self.frame_count = 0
                        self.last_time = current_time

                fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                delay_ms = int(1000.0 / fps)
                self.msleep(max(1, delay_ms // 2))

            cap.release()
            self.finished.emit()
        except Exception as e:
            logging.exception("Exception in VideoWorker run")
            self.finished.emit()
