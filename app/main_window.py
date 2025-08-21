from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from app.widgets.interactive_preview_label import InteractivePreviewLabel
from app.workers.screen_capture_worker import ScreenCaptureWorker, list_monitors
from app.workers.video_worker import VideoWorker
from config_manager import ConfigManager
from image_processing import build_binary_mask


def bgr_to_qimage(frame_bgr: np.ndarray) -> QtGui.QImage:
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = frame_rgb.shape
    bytes_per_line = ch * w
    return QtGui.QImage(frame_rgb.data, w, h, bytes_per_line, QtGui.QImage.Format.Format_RGB888).copy()


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Lane Detection - Sliding Window + Polyfit + Kalman")
        self.resize(1280, 820)

        self.config_manager = ConfigManager()
        self.config = self.config_manager.get_active_config()
        self.worker = VideoWorker(self.config)
        self.screenWorker = ScreenCaptureWorker(self.config)

        self._init_ui()
        self._update_profile_combo()
        self._connect_signals()

        self.resetTimer = QtCore.QTimer(self)
        self.resetTimer.setSingleShot(True)
        self.resetTimer.setInterval(150)
        self.resetTimer.timeout.connect(self.worker.request_reset)

        self.settings = QtCore.QSettings("LaneLab", "LaneDetector")
        self.load_settings()

    def _apply_styles(self):
        stylesheet = ""
        try:
            with open("styles.qss", "r") as f:
                stylesheet = f.read()
        except IOError:
            print("Warning: styles.qss not found.")
        self.setStyleSheet(stylesheet)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self.save_settings()
        super().closeEvent(event)

    def _clear_views(self) -> None:
        self.binaryLabel.clear()
        self.warpLabel.clear()
        self.previewLabel.update_image(None, (0, 0))

    def _reset_overlays_to_defaults(self) -> None:
        fresh_config = self.config_manager.get_profile_config(self.config_manager.active_profile_name)
        if not fresh_config:
            print(f"Warning: Could not load profile '{self.config_manager.active_profile_name}' for reset.")
            return

        self.config.roi_top_y_pct = fresh_config.roi_top_y_pct
        self.config.roi_bottom_y_pct = fresh_config.roi_bottom_y_pct
        self.config.roi_top_left_x_pct = fresh_config.roi_top_left_x_pct
        self.config.roi_top_right_x_pct = fresh_config.roi_top_right_x_pct
        self.config.roi_bottom_left_x_pct = fresh_config.roi_bottom_left_x_pct
        self.config.roi_bottom_right_x_pct = fresh_config.roi_bottom_right_x_pct
        p = self.config.perspective
        pd = fresh_config.perspective
        p.src_top_y_pct = pd.src_top_y_pct
        p.src_bottom_y_pct = pd.src_bottom_y_pct
        p.src_top_left_x_pct = pd.src_top_left_x_pct
        p.src_top_right_x_pct = pd.src_top_right_x_pct
        p.src_bottom_left_x_pct = pd.src_bottom_left_x_pct
        p.src_bottom_right_x_pct = pd.src_bottom_right_x_pct
        self.previewLabel.update()

    def save_settings(self):
        self.settings.setValue("geometry", self.saveGeometry())
        self.settings.setValue("cannyLow", self.cannyLow[0].value())
        self.settings.setValue("cannyHigh", self.cannyHigh[0].value())
        self.settings.setValue("margin", self.marginSlider[0].value())
        self.settings.setValue("minpix", self.minpixSlider[0].value())
        self.settings.setValue("source", self.sourceCombo.currentIndex())
        self.settings.setValue("monitor", self.monitorCombo.currentIndex())
        if self.worker.video_path:
            self.settings.setValue("lastVideo", self.worker.video_path)

    def load_settings(self):
        if self.settings.contains("geometry"):
            self.restoreGeometry(self.settings.value("geometry"))
        if self.settings.contains("cannyLow"):
            self.cannyLow[0].setValue(int(self.settings.value("cannyLow")))
        if self.settings.contains("cannyHigh"):
            self.cannyHigh[0].setValue(int(self.settings.value("cannyHigh")))
        if self.settings.contains("margin"):
            self.marginSlider[0].setValue(int(self.settings.value("margin")))
        if self.settings.contains("minpix"):
            self.minpixSlider[0].setValue(int(self.settings.value("minpix")))
        if self.settings.contains("source"):
            self.sourceCombo.setCurrentIndex(int(self.settings.value("source")))
        if self.settings.contains("monitor"):
            self.monitorCombo.setCurrentIndex(int(self.settings.value("monitor")))
        if self.settings.contains("lastVideo"):
            video_path = self.settings.value("lastVideo")
            if Path(video_path).exists():
                self.worker.set_video(video_path)

    def _init_ui(self) -> None:
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        main_layout = QtWidgets.QHBoxLayout(central)

        video_panel = QtWidgets.QVBoxLayout()
        self.previewLabel = InteractivePreviewLabel(self.config)
        self.previewLabel.setMinimumHeight(480)
        self.previewLabel.setMinimumWidth(800)

        debug_previews = QtWidgets.QHBoxLayout()
        self.binaryLabel = QtWidgets.QLabel(alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
        self.warpLabel = QtWidgets.QLabel(alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
        debug_previews.addWidget(self.binaryLabel)
        debug_previews.addWidget(self.warpLabel)

        video_panel.addWidget(self.previewLabel)
        video_panel.addLayout(debug_previews)
        main_layout.addLayout(video_panel, 3)

        controls_panel = QtWidgets.QVBoxLayout()
        controls_panel.setSpacing(15)

        source_group = QtWidgets.QGroupBox("Source")
        source_layout = QtWidgets.QVBoxLayout(source_group)
        self.sourceCombo = QtWidgets.QComboBox()
        self.sourceCombo.addItems(["Video File", "Screen Capture"])
        self.monitorCombo = QtWidgets.QComboBox()
        self._refresh_monitors()
        self.openButton = QtWidgets.QPushButton("Open Video…")
        source_layout.addWidget(self.sourceCombo)
        source_layout.addWidget(self.monitorCombo)
        source_layout.addWidget(self.openButton)
        controls_panel.addWidget(source_group)

        playback_group = QtWidgets.QGroupBox("Playback")
        playback_layout = QtWidgets.QHBoxLayout(playback_group)
        self.playButton = QtWidgets.QPushButton("Play/Pause")
        self.stopButton = QtWidgets.QPushButton("Stop")
        playback_layout.addWidget(self.playButton)
        playback_layout.addWidget(self.stopButton)
        controls_panel.addWidget(playback_group)

        profile_group = QtWidgets.QGroupBox("Configuration Profile")
        profile_layout = QtWidgets.QGridLayout(profile_group)
        self.profileCombo = QtWidgets.QComboBox()
        self.newProfileButton = QtWidgets.QPushButton("New")
        self.deleteProfileButton = QtWidgets.QPushButton("Delete")
        self.saveConfigButton = QtWidgets.QPushButton("Save")
        self.loadConfigButton = QtWidgets.QPushButton("Reload")
        profile_layout.addWidget(QtWidgets.QLabel("Profile:"), 0, 0)
        profile_layout.addWidget(self.profileCombo, 0, 1, 1, 2)
        profile_layout.addWidget(self.newProfileButton, 1, 0)
        profile_layout.addWidget(self.deleteProfileButton, 1, 1)
        profile_layout.addWidget(self.saveConfigButton, 1, 2)
        profile_layout.addWidget(self.loadConfigButton, 1, 3)
        controls_panel.addWidget(profile_group)

        overlays_group = QtWidgets.QGroupBox("Overlays")
        overlays_layout = QtWidgets.QGridLayout(overlays_group)
        self.editModeCombo = QtWidgets.QComboBox()
        self.editModeCombo.addItems(["ROI", "Bird's-Eye Src"])
        self.hideRoiButton = QtWidgets.QPushButton("Toggle ROI")
        self.hideRoiButton.setCheckable(True)
        self.hideWarpButton = QtWidgets.QPushButton("Toggle Bird's-Eye")
        self.hideWarpButton.setCheckable(True)
        self.saveRoiButton = QtWidgets.QPushButton("Save ROI")
        self.loadRoiButton = QtWidgets.QPushButton("Load ROI")
        overlays_layout.addWidget(QtWidgets.QLabel("Edit Mode:"), 0, 0)
        overlays_layout.addWidget(self.editModeCombo, 0, 1)
        overlays_layout.addWidget(self.hideRoiButton, 1, 0)
        overlays_layout.addWidget(self.hideWarpButton, 1, 1)
        overlays_layout.addWidget(self.saveRoiButton, 2, 0)
        overlays_layout.addWidget(self.loadRoiButton, 2, 1)
        controls_panel.addWidget(overlays_group)

        tuning_group = QtWidgets.QGroupBox("Tuning Parameters")
        tuning_layout = QtWidgets.QVBoxLayout(tuning_group)
        self.cannyLow = self._make_slider(0, 255, self.config.canny.low_threshold, "Canny Low")
        self.cannyHigh = self._make_slider(0, 255, self.config.canny.high_threshold, "Canny High")
        self.marginSlider = self._make_slider(20, 200, self.config.sliding.margin, "Search Margin")
        self.minpixSlider = self._make_slider(10, 200, self.config.sliding.minpix, "Min Pixels")
        tuning_layout.addLayout(self.cannyLow[1])
        tuning_layout.addLayout(self.cannyHigh[1])
        tuning_layout.addLayout(self.marginSlider[1])
        tuning_layout.addLayout(self.minpixSlider[1])
        controls_panel.addWidget(tuning_group)

        controls_panel.addStretch(1)
        main_layout.addLayout(controls_panel, 1)

        self._apply_styles()

    def _make_slider(self, minv: int, maxv: int, value: int, label: str):
        slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        slider.setRange(minv, maxv)
        slider.setValue(value)
        lab = QtWidgets.QLabel(f"{label}: {value}")
        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(lab)
        layout.addWidget(slider)
        return slider, layout, lab

    def _connect_signals(self) -> None:
        self.openButton.clicked.connect(self.handle_open)
        self.playButton.clicked.connect(self.handle_play_pause)
        self.stopButton.clicked.connect(self.handle_stop)
        self.saveRoiButton.clicked.connect(self.handle_save_roi)
        self.loadRoiButton.clicked.connect(self.handle_load_roi)
        self.saveConfigButton.clicked.connect(self.handle_save_config)
        self.loadConfigButton.clicked.connect(self.handle_load_config)
        self.profileCombo.currentIndexChanged.connect(self.handle_profile_selected)
        self.newProfileButton.clicked.connect(self.handle_new_profile)
        self.deleteProfileButton.clicked.connect(self.handle_delete_profile)
        self.hideRoiButton.toggled.connect(self.handle_hide_roi)
        self.hideWarpButton.toggled.connect(self.handle_hide_warp)
        self.editModeCombo.currentIndexChanged.connect(self.handle_edit_mode)
        self.sourceCombo.currentIndexChanged.connect(self.handle_source_changed)
        self.monitorCombo.currentIndexChanged.connect(self.handle_monitor_changed)
        QtCore.QTimer.singleShot(300, self.prompt_auto_roi)

        self.cannyLow[0].valueChanged.connect(self.handle_canny_low)
        self.cannyHigh[0].valueChanged.connect(self.handle_canny_high)
        self.marginSlider[0].valueChanged.connect(self.handle_margin)
        self.minpixSlider[0].valueChanged.connect(self.handle_minpix)

        self.worker.frameReady.connect(self.update_preview)
        self.worker.debugBinaryReady.connect(self.update_binary)
        self.worker.debugWarpReady.connect(self.update_warp)
        self.worker.finished.connect(self.on_finished)
        self.screenWorker.frameReady.connect(self.update_preview)
        self.screenWorker.debugBinaryReady.connect(self.update_binary)
        self.screenWorker.debugWarpReady.connect(self.update_warp)
        self.screenWorker.finished.connect(self.on_finished)

        self.previewLabel.requestReset.connect(self.schedule_reset)

        if self.worker.video_path:
            self._load_roi_for_source(self.worker.video_path)

    def _get_source_id(self) -> str | None:
        source_type = self.sourceCombo.currentText()
        if source_type == "Video File":
            if self.worker.video_path:
                return Path(self.worker.video_path).name
        elif source_type == "Screen Capture":
            return f"monitor_{self.monitorCombo.currentIndex()}"
        return None

    def handle_open(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open Video", str(Path.cwd()), "Videos (*.mp4 *.avi *.mov)"
        )
        if not path:
            return
        self.worker.set_video(path)
        loaded = self._load_roi_for_source(path)
        if not loaded:
            self._reset_overlays_to_defaults()
        self.worker.request_reset()
        self._clear_views()
        self.prompt_auto_roi()

    def handle_play_pause(self) -> None:
        source = self.sourceCombo.currentText()
        if source == "Video File":
            if not self.worker.isRunning():
                self.worker.start()
            else:
                self.worker.toggle_pause()
        else:
            if not self.screenWorker.isRunning():
                data = self.monitorCombo.currentData()
                if data is not None:
                    self.screenWorker.set_monitor_bbox(data.bbox)
                self.screenWorker.start()
            else:
                self.screenWorker.toggle_pause()
        self._load_roi_for_source()

    def handle_stop(self) -> None:
        if self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(500)
        if self.screenWorker.isRunning():
            self.screenWorker.stop()
            self.screenWorker.wait(500)

    def handle_hide_roi(self, hidden: bool) -> None:
        self.previewLabel.set_roi_overlay_visible(not hidden)

    def handle_hide_warp(self, hidden: bool) -> None:
        self.previewLabel.set_warp_overlay_visible(not hidden)

    def handle_edit_mode(self, index: int) -> None:
        mode = "roi" if index == 0 else "warp"
        self.previewLabel.set_edit_mode(mode)

    def handle_source_changed(self, index: int) -> None:
        self.handle_stop()
        if self.sourceCombo.currentText() == "Screen Capture":
            self._refresh_monitors()
        self.worker.request_reset()
        self.screenWorker.request_reset()
        loaded = self._load_roi_for_source()
        if not loaded:
            self._reset_overlays_to_defaults()
        self._clear_views()

    def handle_monitor_changed(self, index: int) -> None:
        if self.screenWorker.isRunning():
            data = self.monitorCombo.currentData()
            if data is not None:
                self.screenWorker.set_monitor_bbox(data.bbox)
        self.screenWorker.request_reset()
        loaded = self._load_roi_for_source()
        if not loaded:
            self._reset_overlays_to_defaults()
        self._clear_views()

    def prompt_auto_roi(self) -> None:
        ret = QtWidgets.QMessageBox.question(
            self,
            "Auto ROI",
            "Automatically detect the Region of Interest from the current frame?",
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
        )
        if ret == QtWidgets.QMessageBox.StandardButton.Yes:
            self.compute_auto_roi()

    def _refresh_monitors(self) -> None:
        self.monitorCombo.clear()
        try:
            mons = list_monitors()
        except Exception:
            mons = []
        if not mons:
            self.monitorCombo.addItem("No monitors", userData=None)
            return
        for m in mons:
            self.monitorCombo.addItem(m.label(), userData=m)
        self.monitorCombo.setCurrentIndex(0)

    def compute_auto_roi(self) -> None:
        if not self.worker.video_path:
            return
        cap = cv2.VideoCapture(self.worker.video_path)
        if not cap.isOpened():
            return
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        mid = max(0, total // 2)
        cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            return

        base_mask = build_binary_mask(frame, self.config, apply_roi=False)
        mask = base_mask
        h, w = mask.shape[:2]
        histogram = np.sum(mask[h // 2 :, :], axis=0)
        midpoint = histogram.shape[0] // 2
        left_base = int(np.argmax(histogram[:midpoint]))
        right_base = int(np.argmax(histogram[midpoint:]) + midpoint)

        def follow_up(x_start: int, step: int = 20, win: int = 50) -> int:
            x_curr = x_start
            for y in range(h - 1, int(h * 0.4), -step):
                x0 = max(0, x_curr - win)
                x1 = min(w - 1, x_curr + win)
                col = mask[max(0, y - step) : y, x0:x1]
                if col.size == 0:
                    continue
                xs = np.argmax(np.sum(col, axis=0))
                x_curr = x0 + int(xs)
            return x_curr

        left_top = follow_up(left_base)
        right_top = follow_up(right_base)

        bottom_y = int(h * 0.98)
        top_y = int(h * 0.62)
        bl = max(0, left_base - 60)
        br = min(w - 1, right_base + 60)
        tl = int(0.6 * left_top + 0.4 * right_top)
        tr = int(0.6 * right_top + 0.4 * left_top)
        tl = max(0, min(tl, w - 1))
        tr = max(0, min(tr, w - 1))

        self.config.roi_bottom_left_x_pct = bl / w
        self.config.roi_bottom_right_x_pct = br / w
        self.config.roi_top_left_x_pct = tl / w
        self.config.roi_top_right_x_pct = tr / w
        self.config.roi_bottom_y_pct = bottom_y / h
        self.config.roi_top_y_pct = top_y / h

        self.schedule_reset()

    def handle_save_roi(self) -> None:
        source_id = self._get_source_id()
        if not source_id:
            QtWidgets.QMessageBox.warning(self, "Warning", "No active source to save ROI for.")
            return

        roi_dir = Path.cwd() / "rois"
        roi_dir.mkdir(exist_ok=True)
        path = roi_dir / f"{source_id}.json"

        cfg = self.config
        data = {
            "roi": {
                "top_y_pct": cfg.roi_top_y_pct,
                "bottom_y_pct": cfg.roi_bottom_y_pct,
                "top_left_x_pct": cfg.roi_top_left_x_pct,
                "top_right_x_pct": cfg.roi_top_right_x_pct,
                "bottom_left_x_pct": cfg.roi_bottom_left_x_pct,
                "bottom_right_x_pct": cfg.roi_bottom_right_x_pct,
            },
            "warp": {
                "src_top_y_pct": cfg.perspective.src_top_y_pct,
                "src_bottom_y_pct": cfg.perspective.src_bottom_y_pct,
                "src_top_left_x_pct": cfg.perspective.src_top_left_x_pct,
                "src_top_right_x_pct": cfg.perspective.src_top_right_x_pct,
                "src_bottom_left_x_pct": cfg.perspective.src_bottom_left_x_pct,
                "src_bottom_right_x_pct": cfg.perspective.src_bottom_right_x_pct,
            },
        }
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            QtWidgets.QMessageBox.information(self, "Success", f"ROI saved for {source_id}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to save ROI: {e}")

    def _load_roi_for_source(self, source_path: str | None = None) -> bool:
        source_id = Path(source_path).name if source_path else self._get_source_id()
        if not source_id:
            return False

        roi_dir = Path.cwd() / "rois"
        path = roi_dir / f"{source_id}.json"

        if not path.exists():
            return False

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            self._apply_roi_data(data)

            self.schedule_reset()
            self.previewLabel.update()
            print(f"Loaded ROI for {source_id}")
            return True

        except Exception as e:
            print(f"Could not load ROI for {source_id}: {e}")
            return False

    def handle_load_roi(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load ROI", str(Path.cwd() / "rois"), "JSON (*.json)"
        )
        if not path:
            return

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            self._apply_roi_data(data)

            self.schedule_reset()
            self.previewLabel.update()
            QtWidgets.QMessageBox.information(self, "Success", "ROI and Warp points loaded.")

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to load ROI: {e}")

    def _apply_roi_data(self, data: dict):
        roi_data = data.get("roi")
        if roi_data:
            self.config.roi_top_y_pct = roi_data.get("top_y_pct", self.config.roi_top_y_pct)
            self.config.roi_bottom_y_pct = roi_data.get("bottom_y_pct", self.config.roi_bottom_y_pct)
            self.config.roi_top_left_x_pct = roi_data.get("top_left_x_pct", self.config.roi_top_left_x_pct)
            self.config.roi_top_right_x_pct = roi_data.get("top_right_x_pct", self.config.roi_top_right_x_pct)
            self.config.roi_bottom_left_x_pct = roi_data.get(
                "bottom_left_x_pct", self.config.roi_bottom_left_x_pct
            )
            self.config.roi_bottom_right_x_pct = roi_data.get(
                "bottom_right_x_pct", self.config.roi_bottom_right_x_pct
            )

        warp_data = data.get("warp")
        if warp_data:
            p = self.config.perspective
            p.src_top_y_pct = warp_data.get("src_top_y_pct", p.src_top_y_pct)
            p.src_bottom_y_pct = warp_data.get("src_bottom_y_pct", p.src_bottom_y_pct)
            p.src_top_left_x_pct = warp_data.get("src_top_left_x_pct", p.src_top_left_x_pct)
            p.src_top_right_x_pct = warp_data.get("src_top_right_x_pct", p.src_top_right_x_pct)
            p.src_bottom_left_x_pct = warp_data.get("src_bottom_left_x_pct", p.src_bottom_left_x_pct)
            p.src_bottom_right_x_pct = warp_data.get("src_bottom_right_x_pct", p.src_bottom_right_x_pct)

    def _update_profile_combo(self):
        self.profileCombo.blockSignals(True)
        self.profileCombo.clear()
        profiles = self.config_manager.get_profile_names()
        self.profileCombo.addItems(profiles)
        current_profile = self.config_manager.active_profile_name
        if current_profile in profiles:
            self.profileCombo.setCurrentText(current_profile)
        self.profileCombo.blockSignals(False)

    def handle_profile_selected(self, index: int):
        profile_name = self.profileCombo.itemText(index)
        if profile_name:
            self.handle_save_config(show_message=False)
            self.config_manager.set_active_profile(profile_name)
            self.handle_load_config(show_message=False)
            QtWidgets.QMessageBox.information(self, "Profile Loaded", f"Switched to profile: {profile_name}")

    def handle_new_profile(self):
        text, ok = QtWidgets.QInputDialog.getText(self, "New Profile", "Enter new profile name:")
        if ok and text:
            profile_name = text.strip().lower().replace(" ", "_")
            if profile_name in self.config_manager.get_profile_names():
                QtWidgets.QMessageBox.warning(self, "Profile Exists", "A profile with this name already exists.")
                return

            source_profile = None
            profiles = self.config_manager.get_profile_names()
            if profiles:
                ret = QtWidgets.QMessageBox.question(
                    self,
                    "Copy Profile",
                    "Copy settings from the current profile?",
                    QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
                )
                if ret == QtWidgets.QMessageBox.StandardButton.Yes:
                    source_profile = self.config_manager.active_profile_name

            self.config_manager.create_profile(profile_name, from_profile=source_profile)
            self._update_profile_combo()
            self.profileCombo.setCurrentText(profile_name)

    def handle_delete_profile(self):
        current_profile = self.config_manager.active_profile_name
        if current_profile == "default":
            QtWidgets.QMessageBox.warning(self, "Cannot Delete", "The 'default' profile cannot be deleted.")
            return

        ret = QtWidgets.QMessageBox.warning(
            self,
            "Confirm Delete",
            f"Are you sure you want to delete the profile '{current_profile}'?",
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
        )
        if ret == QtWidgets.QMessageBox.StandardButton.Yes:
            self.config_manager.delete_profile(current_profile)
            self.config_manager.set_active_profile("default")
            self._update_profile_combo()
            self.handle_load_config(show_message=False)
            QtWidgets.QMessageBox.information(
                self, "Profile Deleted", f"Profile '{current_profile}' has been deleted."
            )

    def handle_save_config(self, show_message=True) -> None:
        self.config_manager.update_active_config(self.config)
        self.config_manager.save()
        if show_message:
            QtWidgets.QMessageBox.information(self, "Success", "Configuration saved.")

    def handle_load_config(self, show_message=True) -> None:
        self.config_manager.load_or_create()
        self.config = self.config_manager.get_active_config()
        self.worker.config = self.config
        self.screenWorker.config = self.config
        self.previewLabel.config = self.config
        self.cannyLow[0].setValue(self.config.canny.low_threshold)
        self.cannyHigh[0].setValue(self.config.canny.high_threshold)
        self.marginSlider[0].setValue(self.config.sliding.margin)
        self.minpixSlider[0].setValue(self.config.sliding.minpix)

        self.schedule_reset()
        self.previewLabel.update()
        if show_message:
            QtWidgets.QMessageBox.information(self, "Success", "Configuration loaded.")

    def handle_canny_low(self, value: int) -> None:
        self.cannyLow[2].setText(f"Canny Low: {value}")
        self.config.canny.low_threshold = int(value)
        self.schedule_reset()

    def handle_canny_high(self, value: int) -> None:
        self.cannyHigh[2].setText(f"Canny High: {value}")
        self.config.canny.high_threshold = int(value)
        self.schedule_reset()

    def handle_margin(self, value: int) -> None:
        self.marginSlider[2].setText(f"Search Margin: {value}")
        self.config.sliding.margin = int(value)
        self.schedule_reset()

    def handle_minpix(self, value: int) -> None:
        self.minpixSlider[2].setText(f"Min Pixels: {value}")
        self.config.sliding.minpix = int(value)
        self.schedule_reset()

    def schedule_reset(self) -> None:
        self.resetTimer.start()

    @QtCore.Slot(np.ndarray)
    def update_preview(self, frame_bgr: np.ndarray) -> None:
        img = bgr_to_qimage(frame_bgr)
        h, w = frame_bgr.shape[:2]
        self.previewLabel.update_image(img, (w, h))

    @QtCore.Slot(np.ndarray)
    def update_binary(self, frame: np.ndarray) -> None:
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        img = bgr_to_qimage(frame)
        self.binaryLabel.setPixmap(
            QtGui.QPixmap.fromImage(img).scaled(
                self.binaryLabel.width(),
                self.binaryLabel.height(),
                QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                QtCore.Qt.TransformationMode.SmoothTransformation,
            )
        )

    @QtCore.Slot(np.ndarray)
    def update_warp(self, frame: np.ndarray) -> None:
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        img = bgr_to_qimage(frame)
        self.warpLabel.setPixmap(
            QtGui.QPixmap.fromImage(img).scaled(
                self.warpLabel.width(),
                self.warpLabel.height(),
                QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                QtCore.Qt.TransformationMode.SmoothTransformation,
            )
        )

    def on_finished(self) -> None:
        pass
