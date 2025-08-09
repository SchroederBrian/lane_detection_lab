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


