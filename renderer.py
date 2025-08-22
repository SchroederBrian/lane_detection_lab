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

    def process_frame(self, frame_bgr: np.ndarray, binary_builder, fps: float = 0.0) -> Tuple[np.ndarray, dict]:
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

        # Optional: overlay binary edges for visibility
        if self.config.draw.show_binary_overlay:
            bin_color = np.zeros_like(frame_bgr)
            bin_color[binary > 0] = self.config.draw.binary_overlay_color_bgr
            out = cv2.addWeighted(out, 1.0, bin_color, self.config.draw.binary_overlay_alpha, 0)

        # 7) Steering and pose estimation
        pose = self.detector.estimate_vehicle_pose(warped_binary, left_fit, right_fit)
        out = self._draw_steering(out, pose)
        out = self._draw_hud(out, pose, fps)

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

    def _draw_hud(self, frame_bgr: np.ndarray, pose: dict, fps: float) -> np.ndarray:
        if not self.config.draw.show_hud_panel:
            return frame_bgr
        h, w = frame_bgr.shape[:2]
        canvas = frame_bgr.copy()

        # Panel rect on left
        panel_w = int(w * 0.28)
        panel_h = int(h * 0.36)
        x0, y0 = int(w * 0.01), int(h * 0.05)
        x1, y1 = x0 + panel_w, y0 + panel_h

        panel = canvas[y0:y1, x0:x1].copy()
        color = np.array(self.config.draw.hud_panel_color_bgr, dtype=np.uint8)
        overlay = np.full_like(panel, color)
        alpha = self.config.draw.hud_panel_alpha
        panel = cv2.addWeighted(panel, 1 - alpha, overlay, alpha, 0)
        canvas[y0:y1, x0:x1] = panel
        cv2.rectangle(canvas, (x0, y0), (x1, y1), self.config.draw.hud_panel_border_color_bgr, 2)

        # Simple road sign (diamond with right arrow based on curvature sign)
        sign_size = int(min(panel_w, panel_h) * 0.35)
        sign_cx, sign_cy = x0 + sign_size, y0 + sign_size
        sign_pts = np.array([
            [sign_cx, sign_cy - sign_size // 2],
            [sign_cx + sign_size // 2, sign_cy],
            [sign_cx, sign_cy + sign_size // 2],
            [sign_cx - sign_size // 2, sign_cy],
        ], dtype=np.int32)
        cv2.fillPoly(canvas, [sign_pts], (0, 220, 255))
        cv2.polylines(canvas, [sign_pts], True, (0, 0, 0), 2)
        # Arrow
        curvature_inv = float(pose.get("curvature_inv", 0.0))
        right_turn = curvature_inv >= 0
        arr_len = int(sign_size * 0.35)
        arr_th = 4
        if right_turn:
            cv2.arrowedLine(canvas, (sign_cx - arr_len // 2, sign_cy + arr_len // 4), (sign_cx + arr_len // 2, sign_cy - arr_len // 4), (0, 0, 0), arr_th, tipLength=0.4)
        else:
            cv2.arrowedLine(canvas, (sign_cx + arr_len // 2, sign_cy + arr_len // 4), (sign_cx - arr_len // 2, sign_cy - arr_len // 4), (0, 0, 0), arr_th, tipLength=0.4)

        # Text metrics
        curvature_radius_m = (1.0 / max(abs(curvature_inv), 1e-6)) if curvature_inv != 0 else 0.0
        direction = "Right Curve Ahead" if right_turn else "Left Curve Ahead"
        offset_m = float(pose.get("lateral_offset_m", 0.0))
        good = abs(offset_m) <= self.config.draw.hud_good_offset_m

        # Labels
        def put(text: str, x: int, y: int, scale: float = 0.9, color=(240, 240, 240), thick: int = 2):
            cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)

        tx = x0 + int(panel_w * 0.04)
        ty = y0 + int(panel_h * 0.55)
        put(direction, tx, ty, 0.9)
        ty += int(panel_h * 0.10)
        put(f"Curvature = {int(curvature_radius_m)} m", tx, ty, 0.9)

        # Status line
        ty += int(panel_h * 0.16)
        status = "Good Lane Keeping" if good else "Correct Lane Position"
        put(status, tx, ty, 1.0, (50, 255, 80) if good else (0, 200, 255), 3)

        # Offset text
        ty += int(panel_h * 0.14)
        put(f"Vehicle is {abs(offset_m):.2f} m {'right' if offset_m>0 else 'left' if offset_m<0 else 'from'} center", tx, ty, 0.8)

        # Add FPS
        ty += int(panel_h * 0.14)  # After offset text
        put(f"FPS: {fps:.1f}", tx, ty, 0.8)

        return canvas


