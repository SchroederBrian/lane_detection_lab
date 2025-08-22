from __future__ import annotations

from typing import Tuple
import cv2
import numpy as np

from config import Config


def apply_roi_mask(binary_image: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    if binary_image.ndim != 2:
        return binary_image
    mask = np.zeros_like(binary_image)
    cv2.fillPoly(mask, [polygon], 255)
    masked = cv2.bitwise_and(binary_image, mask)
    return masked


def build_binary_mask(frame_bgr: np.ndarray, config: Config, apply_roi: bool = True) -> np.ndarray:
    h, w = frame_bgr.shape[:2]

    # Color thresholds
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    hls = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HLS)

    yellow_mask = cv2.inRange(
        hsv,
        np.array(config.color.yellow_hsv_lower, dtype=np.uint8),
        np.array(config.color.yellow_hsv_upper, dtype=np.uint8),
    )

    white_mask = cv2.inRange(
        hls,
        np.array(config.color.white_hls_lower, dtype=np.uint8),
        np.array(config.color.white_hls_upper, dtype=np.uint8),
    )

    color_mask = cv2.bitwise_or(yellow_mask, white_mask)

    # Canny edges
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    use_cuda = config.cuda_enabled and cv2.cuda.getCudaEnabledDeviceCount() > 0

    if use_cuda:
        gpu_gray = cv2.cuda_GpuMat()
        gpu_gray.upload(gray)
        gpu_blurred = cv2.cuda_GaussianBlur(gpu_gray, (config.canny.gaussian_kernel_size, config.canny.gaussian_kernel_size), 0)
        gpu_edges = cv2.cuda_Canny(gpu_blurred, config.canny.low_threshold, config.canny.high_threshold)
        edges = gpu_edges.download()
    else:
        blurred = cv2.GaussianBlur(gray, (config.canny.gaussian_kernel_size, config.canny.gaussian_kernel_size), 0)
        edges = cv2.Canny(blurred, config.canny.low_threshold, config.canny.high_threshold)

    # Combine
    combined = cv2.bitwise_or(color_mask, edges)

    # ROI
    if apply_roi:
        roi_polygon = config.compute_roi_polygon(frame_bgr.shape)
        masked = apply_roi_mask(combined, roi_polygon)
    else:
        masked = combined

    # Binary output 0/255
    _, binary = cv2.threshold(masked, 1, 255, cv2.THRESH_BINARY)
    return binary


