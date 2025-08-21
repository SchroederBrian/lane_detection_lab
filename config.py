from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple
import numpy as np


@dataclass
class PerspectiveConfig:
    # Percent-based trapezoid for perspective src; dst is a rectangle
    src_top_y_pct: float = 0.62
    src_bottom_y_pct: float = 0.95
    src_top_left_x_pct: float = 0.42
    src_top_right_x_pct: float = 0.58
    src_bottom_left_x_pct: float = 0.10
    src_bottom_right_x_pct: float = 0.90

    dst_left_x_pct: float = 0.20
    dst_right_x_pct: float = 0.80
    dst_top_y_pct: float = 0.02
    dst_bottom_y_pct: float = 0.98

    def compute_src_dst(self, frame_shape: Tuple[int, int, int]) -> Tuple[np.ndarray, np.ndarray]:
        h, w = frame_shape[:2]
        src = np.float32([
            [w * self.src_bottom_left_x_pct, h * self.src_bottom_y_pct],
            [w * self.src_bottom_right_x_pct, h * self.src_bottom_y_pct],
            [w * self.src_top_right_x_pct, h * self.src_top_y_pct],
            [w * self.src_top_left_x_pct, h * self.src_top_y_pct],
        ])

        dst = np.float32([
            [w * self.dst_left_x_pct, h * self.dst_bottom_y_pct],
            [w * self.dst_right_x_pct, h * self.dst_bottom_y_pct],
            [w * self.dst_right_x_pct, h * self.dst_top_y_pct],
            [w * self.dst_left_x_pct, h * self.dst_top_y_pct],
        ])

        return src, dst


@dataclass
class CannyConfig:
    gaussian_kernel_size: int = 5
    low_threshold: int = 50
    high_threshold: int = 150


@dataclass
class ColorThresholdConfig:
    # HSV for yellow
    yellow_hsv_lower: Tuple[int, int, int] = (15, 80, 80)
    yellow_hsv_upper: Tuple[int, int, int] = (35, 255, 255)
    # HLS for white
    white_hls_lower: Tuple[int, int, int] = (0, 200, 0)
    white_hls_upper: Tuple[int, int, int] = (255, 255, 40)


@dataclass
class SlidingWindowConfig:
    num_windows: int = 9
    margin: int = 80
    minpix: int = 40


@dataclass
class KalmanConfig:
    process_variance: float = 1e-4
    measurement_variance: float = 1e-2
    initial_variance: float = 1.0


@dataclass
class DrawConfig:
    lane_color_bgr: Tuple[int, int, int] = (0, 255, 0)
    overlay_alpha: float = 0.3
    show_debug_windows: bool = True
    show_binary_overlay: bool = False
    binary_overlay_alpha: float = 0.35
    binary_overlay_color_bgr: Tuple[int, int, int] = (0, 200, 0)
    # HUD panel like the sample image
    show_hud_panel: bool = True
    hud_panel_alpha: float = 0.35
    hud_panel_color_bgr: Tuple[int, int, int] = (50, 80, 180)  # bluish
    hud_panel_border_color_bgr: Tuple[int, int, int] = (255, 255, 255)
    hud_good_offset_m: float = 0.25


@dataclass
class MetricConfig:
    # approximate meters-per-pixel scaling (typical for 720p road scene)
    ym_per_px: float = 30.0 / 720.0
    xm_per_px: float = 3.7 / 700.0


@dataclass
class SteeringConfig:
    max_steering_deg: float = 45.0
    # Simple mapping gains from geometry to angle
    gain_offset: float = 10.0   # deg per meter lateral offset
    gain_curvature: float = 1200.0  # deg per (1/m curvature)


@dataclass
class Config:
    canny: CannyConfig = field(default_factory=CannyConfig)
    color: ColorThresholdConfig = field(default_factory=ColorThresholdConfig)
    perspective: PerspectiveConfig = field(default_factory=PerspectiveConfig)
    sliding: SlidingWindowConfig = field(default_factory=SlidingWindowConfig)
    kalman: KalmanConfig = field(default_factory=KalmanConfig)
    draw: DrawConfig = field(default_factory=DrawConfig)
    metric: MetricConfig = field(default_factory=MetricConfig)
    steering: SteeringConfig = field(default_factory=SteeringConfig)

    # ROI trapezoid percentages (same general shape as perspective src)
    roi_top_y_pct: float = 0.62
    roi_bottom_y_pct: float = 0.98
    roi_top_left_x_pct: float = 0.40
    roi_top_right_x_pct: float = 0.60
    roi_bottom_left_x_pct: float = 0.10
    roi_bottom_right_x_pct: float = 0.90

    def compute_roi_polygon(self, frame_shape: Tuple[int, int, int]) -> np.ndarray:
        h, w = frame_shape[:2]
        polygon = np.array([
            [w * self.roi_bottom_left_x_pct, h * self.roi_bottom_y_pct],
            [w * self.roi_bottom_right_x_pct, h * self.roi_bottom_y_pct],
            [w * self.roi_top_right_x_pct, h * self.roi_top_y_pct],
            [w * self.roi_top_left_x_pct, h * self.roi_top_y_pct],
        ], dtype=np.int32)
        return polygon

    def update_roi_from_points(self, pts_xy: np.ndarray, frame_shape: Tuple[int, int, int]) -> None:
        # pts order: BL, BR, TR, TL (x, y)
        h, w = frame_shape[:2]
        bl, br, tr, tl = pts_xy
        self.roi_bottom_left_x_pct = float(np.clip(bl[0] / w, 0.0, 1.0))
        self.roi_bottom_right_x_pct = float(np.clip(br[0] / w, 0.0, 1.0))
        self.roi_top_right_x_pct = float(np.clip(tr[0] / w, 0.0, 1.0))
        self.roi_top_left_x_pct = float(np.clip(tl[0] / w, 0.0, 1.0))
        self.roi_bottom_y_pct = float(np.clip(bl[1] / h, 0.0, 1.0))
        self.roi_top_y_pct = float(np.clip(tl[1] / h, 0.0, 1.0))


def get_default_config() -> Config:
    return Config()


