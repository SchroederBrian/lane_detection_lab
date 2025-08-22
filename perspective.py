from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import cv2
import numpy as np

from config import Config


@dataclass
class PerspectiveTransformer:
    config: Config
    _last_shape: Tuple[int, int] | None = None
    _M: np.ndarray | None = None
    _Minv: np.ndarray | None = None

    def _ensure_matrices(self, frame_shape: Tuple[int, int, int]) -> None:
        h, w = frame_shape[:2]
        if self._last_shape == (h, w) and self._M is not None and self._Minv is not None:
            return
        src, dst = self.config.perspective.compute_src_dst(frame_shape)
        self._M = cv2.getPerspectiveTransform(src, dst)
        self._Minv = cv2.getPerspectiveTransform(dst, src)
        self._last_shape = (h, w)

    def reset(self) -> None:
        self._last_shape = None
        self._M = None
        self._Minv = None

    def warp(self, frame: np.ndarray) -> np.ndarray:
        self._ensure_matrices(frame.shape)
        h, w = frame.shape[:2]

        use_cuda = self.config.cuda_enabled and cv2.cuda.getCudaEnabledDeviceCount() > 0

        if use_cuda:
            gpu_frame = cv2.cuda_GpuMat()
            gpu_frame.upload(frame)
            gpu_warped = cv2.cuda.warpPerspective(gpu_frame, self._M, (w, h), flags=cv2.INTER_LINEAR)
            return gpu_warped.download()
        else:
            return cv2.warpPerspective(frame, self._M, (w, h), flags=cv2.INTER_LINEAR)

    def unwarp(self, frame: np.ndarray) -> np.ndarray:
        # Unwarp uses the inverse matrix; assumes same output size as input
        assert self._Minv is not None, "Call warp at least once to initialize matrices"
        h, w = frame.shape[:2]

        use_cuda = self.config.cuda_enabled and cv2.cuda.getCudaEnabledDeviceCount() > 0

        if use_cuda:
            gpu_frame = cv2.cuda_GpuMat()
            gpu_frame.upload(frame)
            gpu_unwarped = cv2.cuda.warpPerspective(gpu_frame, self._Minv, (w, h), flags=cv2.INTER_LINEAR)
            return gpu_unwarped.download()
        else:
            return cv2.warpPerspective(frame, self._Minv, (w, h), flags=cv2.INTER_LINEAR)


