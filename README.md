# Lane Detection Lab

## Description

Lane Detection Lab is a Python-based application for real-time lane detection. It uses techniques like sliding window, polynomial fitting, Kalman filtering, and bird's-eye view perspective transformation. The app supports processing video files or live screen captures, with a graphical user interface for configuration, live parameter tuning, and visualization. It also includes gamepad emulation to simulate steering inputs based on detected lane curvature, useful for applications like game automation (e.g., Euro Truck Simulator).

## Features

- Real-time lane detection with binary masking, perspective warping, and lane fitting
- GUI built with PySide6 for interactive preview, parameter sliders, ROI selection, and diagnostic views
- Configurable via YAML with multiple profiles (e.g., default, night, rain) and persistence
- Screen capture support for live input from monitors
- Gamepad emulation using vgamepad for automated steering
- Kalman filter for smooth lane tracking
- Video processing pipeline with optional output saving
- Per-video ROI and settings persistence

## Installation

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

## Usage

Launch the GUI application:

```bash
python app.py
```

In the GUI:
- Use the menu or buttons to open a video file or select a screen for capture.
- Adjust detection parameters using the sliders in the config panel.
- Select a config profile or save/load configurations.
- Enable gamepad emulation to send steering inputs based on lane detection.
- Play/pause the processing and view the output with overlaid lanes and diagnostics.
- Press 'q' or Esc to quit previews if needed.

Configuration can be edited directly in `config.yaml` or through the GUI. ROI points are saved per video/source in the `rois/` directory.

## Dependencies

- opencv-python >= 4.8.0
- numpy >= 1.24.0
- PySide6 >= 6.5.0
- PyYAML
- mss >= 9.0.1
- scipy
- vgamepad
- pynput

These are listed in `requirements.txt` for easy installation.

## Configuration

The app uses `config.yaml` for settings, managed by `config_manager.py`. It supports:
- Multiple profiles for different conditions.
- Versioning and validation of config files.
- GUI persistence for window state, last used video, etc.

Edit `config.yaml` directly or use the GUI to tune and save.

## Development

- Main entry: `app.py`
- GUI: `app/main_window.py`
- Pipeline: `app/pipeline.py`
- Workers: `app/workers/` for video and screen processing
- Core logic: `lane_detector.py`, `kalman.py`, `perspective.py`, `image_processing.py`
- Tests: `tests/` including `test_gamepad.py`

For contributions, see the code and open issues/PRs.

## Test Videos

Sample videos are in `videos/` for testing lane detection.


