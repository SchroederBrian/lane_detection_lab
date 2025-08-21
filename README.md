### Lane Detection: Sliding Window + Polyfit + Kalman + Bird's-Eye Warp

Run on a video file and optionally save output.

#### Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

#### Usage (CLI)

```bash
python app.py --video path/to/input.mp4 --save output.mp4
```

Press `q` or `Esc` to quit.

You can edit thresholds, ROI, and perspective points in `config.py`.

#### Usage (GUI)

```bash
python gui.py
```

In the GUI, open a video, press Play/Pause, and tune parameters live. The top shows the composited output; bottom row shows binary and warped-binary diagnostics.

## New Features in this version
- **Config system upgrade (profiles + persistence):** Load/save Config to YAML/TOML with versioning, validation, and profile selection (e.g., default, night, rain). Allow CLI overrides and GUI Save/Load for the entire config (not just ROI).
- **Persist GUI state and ROI per source:** Use QSettings to remember last video, sliders, window size, and source. Auto-load per-video/per-monitor ROI and warp points; add a “Load ROI…” next to “Save ROI…” in gui.py.

## Usage

### GUI Mode

```bash
python gui.py
```

In the GUI, open a video, press Play/Pause, and tune parameters live. The top shows the composited output; bottom row shows binary and warped-binary diagnostics.

To run the command-line processing script:

```bash
python app.py --video /path/to/your/video.mp4 --save /path/to/output.mp4 --profile night --draw.show_hud_panel=False
```

- `--video`: Path to the input video file.
- `--save`: Optional path to save the processed video.
- `--profile`: Optional. The configuration profile to use (e.g., `default`, `night`).
- You can also override any config parameter using the `--key=value` syntax.

## Configuration
The application now uses a `config.yaml` file to manage settings. This file is created automatically on the first run.

- **Profiles:** You can create and manage different configuration profiles for various conditions (e.g., `day`, `night`, `rain`).
- **Persistence:** All settings, including UI layout and selected profiles, are persisted between sessions.

## Dependencies
- Python 3.8+
- OpenCV (`opencv-python-headless`)
- NumPy
- PySide6 (for the GUI)
- PyYAML
- mss (for screen capture)

Install the dependencies using:
```bash
pip install -r requirements.txt
```


