from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import numpy as np
import cv2

from config import Config
from kalman import PolyKalman


def compute_histogram_peaks(binary_warped: np.ndarray) -> Tuple[int, int]:
    histogram = np.sum(binary_warped[binary_warped.shape[0] // 2 :, :], axis=0)
    midpoint = histogram.shape[0] // 2
    leftx_base = int(np.argmax(histogram[:midpoint]))
    rightx_base = int(np.argmax(histogram[midpoint:]) + midpoint)
    return leftx_base, rightx_base


@dataclass
class LaneDetectionState:
    left_fit: Optional[np.ndarray] = None  # np.polyfit coeffs (a, b, c)
    right_fit: Optional[np.ndarray] = None


class LaneDetector:
    def __init__(self, config: Config):
        self.config = config
        self.left_kalman = PolyKalman(config.kalman, degree=2)
        self.right_kalman = PolyKalman(config.kalman, degree=2)
        self.state = LaneDetectionState()

    def reset(self) -> None:
        self.left_kalman.reset()
        self.right_kalman.reset()
        self.state = LaneDetectionState()

    def _sliding_window_search(self, binary_warped: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        h, w = binary_warped.shape[:2]
        nwindows = self.config.sliding.num_windows
        margin = self.config.sliding.margin
        minpix = self.config.sliding.minpix

        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        leftx_base, rightx_base = compute_histogram_peaks(binary_warped)

        window_height = int(h / nwindows)
        leftx_current = leftx_base
        rightx_current = rightx_base

        left_lane_inds = []
        right_lane_inds = []

        for window in range(nwindows):
            win_y_low = h - (window + 1) * window_height
            win_y_high = h - window * window_height

            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin

            good_left_inds = (
                (nonzeroy >= win_y_low)
                & (nonzeroy < win_y_high)
                & (nonzerox >= win_xleft_low)
                & (nonzerox < win_xleft_high)
            ).nonzero()[0]
            good_right_inds = (
                (nonzeroy >= win_y_low)
                & (nonzeroy < win_y_high)
                & (nonzerox >= win_xright_low)
                & (nonzerox < win_xright_high)
            ).nonzero()[0]

            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            if good_left_inds.size > minpix:
                leftx_current = int(np.mean(nonzerox[good_left_inds]))
            if good_right_inds.size > minpix:
                rightx_current = int(np.mean(nonzerox[good_right_inds]))

        left_lane_inds = np.concatenate(left_lane_inds) if left_lane_inds else np.array([], dtype=int)
        right_lane_inds = np.concatenate(right_lane_inds) if right_lane_inds else np.array([], dtype=int)

        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        left_fit = None
        right_fit = None
        if leftx.size >= 3:
            left_fit = np.polyfit(lefty, leftx, 2)
        if rightx.size >= 3:
            right_fit = np.polyfit(righty, rightx, 2)

        return left_fit, right_fit

    def _search_around_poly(self, binary_warped: np.ndarray, left_fit: np.ndarray, right_fit: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        margin = self.config.sliding.margin
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        left_lane_inds = (
            (nonzerox > (left_fit[0] * nonzeroy**2 + left_fit[1] * nonzeroy + left_fit[2] - margin))
            & (nonzerox < (left_fit[0] * nonzeroy**2 + left_fit[1] * nonzeroy + left_fit[2] + margin))
        )
        right_lane_inds = (
            (nonzerox > (right_fit[0] * nonzeroy**2 + right_fit[1] * nonzeroy + right_fit[2] - margin))
            & (nonzerox < (right_fit[0] * nonzeroy**2 + right_fit[1] * nonzeroy + right_fit[2] + margin))
        )

        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        new_left_fit = None
        new_right_fit = None
        if leftx.size >= 3:
            new_left_fit = np.polyfit(lefty, leftx, 2)
        if rightx.size >= 3:
            new_right_fit = np.polyfit(righty, rightx, 2)

        return new_left_fit, new_right_fit

    def detect(self, binary_warped: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        left_fit_prev = self.state.left_fit
        right_fit_prev = self.state.right_fit

        if left_fit_prev is not None and right_fit_prev is not None:
            left_fit, right_fit = self._search_around_poly(binary_warped, left_fit_prev, right_fit_prev)
        else:
            left_fit, right_fit = self._sliding_window_search(binary_warped)

        if left_fit is not None:
            left_fit = self.left_kalman.update(left_fit)
            self.state.left_fit = left_fit
        if right_fit is not None:
            right_fit = self.right_kalman.update(right_fit)
            self.state.right_fit = right_fit

        return self.state.left_fit, self.state.right_fit

    @staticmethod
    def compute_lane_overlay(binary_warped: np.ndarray, left_fit: Optional[np.ndarray], right_fit: Optional[np.ndarray]) -> np.ndarray:
        h, w = binary_warped.shape[:2]
        warp_zero = np.zeros((h, w), dtype=np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        ploty = np.linspace(0, h - 1, h)

        if left_fit is not None and right_fit is not None:
            left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
            right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]

            pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
            pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
            pts = np.hstack((pts_left, pts_right)).astype(np.int32)
            cv2.fillPoly(color_warp, [pts], (0, 255, 0))

        return color_warp

    def estimate_vehicle_pose(self, binary_warped: np.ndarray, left_fit: Optional[np.ndarray], right_fit: Optional[np.ndarray]) -> dict:
        # Returns lateral offset (m), curvature (1/m), steering_deg (mapped), and centerline poly x values
        h, w = binary_warped.shape[:2]
        ploty = np.linspace(0, h - 1, h)
        xm_per_px = self.config.metric.xm_per_px
        ym_per_px = self.config.metric.ym_per_px

        center_fitx = None
        curvature_inv = 0.0
        lateral_offset_m = 0.0

        if left_fit is not None and right_fit is not None:
            left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
            right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]
            center_fitx = (left_fitx + right_fitx) / 2.0

            # Vehicle is assumed at image center
            lane_center_px_bottom = center_fitx[-1]
            veh_center_px = w / 2.0
            lateral_offset_px = veh_center_px - lane_center_px_bottom
            lateral_offset_m = lateral_offset_px * xm_per_px

            # Curvature (simple): use derivative of center poly in meters
            # Fit in meters: y_m, x_m
            y_m = ploty * ym_per_px
            x_m = center_fitx * xm_per_px
            fit_cr = np.polyfit(y_m, x_m, 2)
            a, b = fit_cr[0], fit_cr[1]
            y_eval = np.max(y_m)
            curvature_radius_m = ((1 + (2 * a * y_eval + b) ** 2) ** 1.5) / (2 * abs(a) + 1e-6)
            curvature_inv = 1.0 / max(curvature_radius_m, 1e-3)

        # Map to steering degrees
        steer = self.config.steering.gain_offset * lateral_offset_m + self.config.steering.gain_curvature * curvature_inv
        steer = float(np.clip(steer, -self.config.steering.max_steering_deg, self.config.steering.max_steering_deg))

        return {
            "ploty": ploty,
            "center_fitx": center_fitx,
            "lateral_offset_m": lateral_offset_m,
            "curvature_inv": curvature_inv,
            "steering_deg": steer,
        }


