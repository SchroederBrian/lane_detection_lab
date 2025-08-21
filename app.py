#!/usr/bin/env python3
from __future__ import annotations

import argparse
from typing import Optional
import cv2

from config_manager import ConfigManager
from image_processing import build_binary_mask
from perspective import PerspectiveTransformer
from lane_detector import LaneDetector
from renderer import LaneRenderer


def main() -> None:
    parser = argparse.ArgumentParser(description="Lane detection with Sliding Window + Polyfit + Kalman")
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument("--save", default=None, help="Optional path to save output video")
    parser.add_argument("--profile", default="default", help="Configuration profile to use")
    # All other arguments are treated as config overrides
    args, unknown = parser.parse_known_args()

    config_manager = ConfigManager()
    
    # Set profile
    config_manager.set_active_profile(args.profile)
    config = config_manager.get_active_config()

    # Handle config overrides
    for arg in unknown:
        if arg.startswith("--"):
            try:
                key, value = arg[2:].split("=")
                keys = key.split(".")
                obj = config
                for k in keys[:-1]:
                    obj = getattr(obj, k)
                
                field_type = type(getattr(obj, keys[-1]))
                if field_type == bool:
                    setattr(obj, keys[-1], value.lower() in ("true", "1", "yes"))
                else:
                    setattr(obj, keys[-1], field_type(value))
                
            except Exception as e:
                print(f"Warning: Could not parse override argument '{arg}': {e}")


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


