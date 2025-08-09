from __future__ import annotations

import argparse
from typing import Optional
import cv2

from config import get_default_config
from image_processing import build_binary_mask
from perspective import PerspectiveTransformer
from lane_detector import LaneDetector
from renderer import LaneRenderer


def main() -> None:
    parser = argparse.ArgumentParser(description="Lane detection with Sliding Window + Polyfit + Kalman")
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument("--save", default=None, help="Optional path to save output video")
    parser.add_argument("--no-debug", action="store_true", help="Disable debug windows")
    args = parser.parse_args()

    config = get_default_config()
    if args.no_debug:
        config.draw.show_debug_windows = False

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {args.video}")

    # Prime shape for perspective matrices
    ret, frame = cap.read()
    if not ret:
        raise SystemExit("Could not read first frame")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    perspective = PerspectiveTransformer(config)
    detector = LaneDetector(config)
    renderer = LaneRenderer(config, perspective, detector)

    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        h, w = frame.shape[:2]
        writer = cv2.VideoWriter(args.save, fourcc, fps, (w, h))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        out_frame, debug = renderer.process_frame(frame, build_binary_mask)

        if writer is not None:
            writer.write(out_frame)

        cv2.imshow("Lane Detection", out_frame)

        if config.draw.show_debug_windows:
            cv2.imshow("Binary", debug["binary"])
            cv2.imshow("Warped Binary", debug["warped_binary"])

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


