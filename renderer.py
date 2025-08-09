from __future__ import annotations

from typing import Optional, Tuple
import cv2
import numpy as np

from config import Config
from lane_detector import LaneDetector
from perspective import PerspectiveTransformer


class LaneRenderer:
    def __init__(self, config: Config, perspective: PerspectiveTransformer, detector: LaneDetector):
        self.config = config
        self.perspective = perspective
        self.detector = detector

    def process_frame(self, frame_bgr: np.ndarray, binary_builder) -> Tuple[np.ndarray, dict]:
        # 1) Binary mask (apply ROI first to constrain detection strictly inside ROI)
        binary = binary_builder(frame_bgr, self.config, apply_roi=True)

        # 2) Warp to bird's-eye
        warped_binary = self.perspective.warp(binary)

        # 3) Detect lanes
        left_fit, right_fit = self.detector.detect(warped_binary)

        # 4) Create lane area in warp space
        color_warp = self.detector.compute_lane_overlay(warped_binary, left_fit, right_fit)

        # 5) Unwarp overlay to original perspective
        unwarped_overlay = self.perspective.unwarp(color_warp)

        # 6) Blend lane overlay
        overlay_alpha = self.config.draw.overlay_alpha
        out = cv2.addWeighted(frame_bgr, 1.0, unwarped_overlay, overlay_alpha, 0)

        # 7) Steering and pose estimation
        pose = self.detector.estimate_vehicle_pose(warped_binary, left_fit, right_fit)
        out = self._draw_steering(out, pose)

        debug = {
            "binary": binary,
            "warped_binary": warped_binary,
            "left_fit": left_fit,
            "right_fit": right_fit,
            "pose": pose,
        }
        return out, debug

    def _draw_steering(self, frame_bgr: np.ndarray, pose: dict) -> np.ndarray:
        h, w = frame_bgr.shape[:2]
        canvas = frame_bgr.copy()

        # Steering gauge
        center = (int(w * 0.1), int(h * 0.85))
        radius = int(min(w, h) * 0.08)
        cv2.circle(canvas, center, radius, (60, 60, 60), 2)

        # Needle for predicted steering
        pred_deg = float(pose.get("steering_deg", 0.0))
        max_deg = self.config.steering.max_steering_deg
        angle_rad = np.deg2rad(90 - (pred_deg / max_deg) * 90)  # map [-max,max] to [180,0]
        needle_len = radius - 6
        end_pt = (
            int(center[0] + needle_len * np.cos(angle_rad)),
            int(center[1] - needle_len * np.sin(angle_rad)),
        )
        cv2.line(canvas, center, end_pt, (0, 140, 255), 3)

        # Current center offset bar
        offset_m = float(pose.get("lateral_offset_m", 0.0))
        bar_w = int(radius * 2)
        bar_h = 8
        bar_x = center[0] - bar_w // 2
        bar_y = center[1] + radius + 12
        cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (80, 80, 80), 1)
        # Map +-1.5m to bar width
        max_m = 1.5
        fill = int(np.clip((offset_m / max_m) * (bar_w // 2), -bar_w // 2, bar_w // 2))
        if fill >= 0:
            cv2.rectangle(canvas, (center[0], bar_y), (center[0] + fill, bar_y + bar_h), (0, 200, 0), -1)
        else:
            cv2.rectangle(canvas, (center[0] + fill, bar_y), (center[0], bar_y + bar_h), (0, 0, 200), -1)

        # Texts
        cv2.putText(canvas, f"Steer: {pred_deg:+.1f} deg", (bar_x, bar_y + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"Offset: {offset_m:+.2f} m", (bar_x, bar_y + 46), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)

        return canvas


