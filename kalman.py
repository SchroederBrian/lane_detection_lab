from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from typing import Optional

from config import KalmanConfig


@dataclass
class ScalarKalman:
    config: KalmanConfig
    mean: float | None = None
    variance: float | None = None

    def reset(self) -> None:
        self.mean = None
        self.variance = None

    def update(self, measurement: float) -> float:
        if self.mean is None:
            self.mean = measurement
            self.variance = self.config.initial_variance
            return self.mean

        # Prediction: variance increases by process noise
        assert self.variance is not None
        predicted_variance = self.variance + self.config.process_variance

        # Kalman gain
        kalman_gain = predicted_variance / (predicted_variance + self.config.measurement_variance)

        # Update
        self.mean = self.mean + kalman_gain * (measurement - self.mean)
        self.variance = (1 - kalman_gain) * predicted_variance
        return self.mean


class PolyKalman:
    def __init__(self, config: KalmanConfig, degree: int = 2):
        self.config = config
        self.degree = degree
        self.filters = [ScalarKalman(config) for _ in range(degree + 1)]

    def reset(self) -> None:
        for f in self.filters:
            f.reset()

    def update(self, coeffs: np.ndarray) -> np.ndarray:
        # coeffs are highest-order first (np.polyfit style)
        smoothed = []
        for i, c in enumerate(coeffs):
            smoothed.append(self.filters[i].update(float(c)))
        return np.array(smoothed)


