from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from config import Config, get_default_config
from image_processing import build_binary_mask
from perspective import PerspectiveTransformer
from lane_detector import LaneDetector
from renderer import LaneRenderer
from screen_capture import ScreenCaptureWorker, list_monitors


def bgr_to_qimage(frame_bgr: np.ndarray) -> QtGui.QImage:
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = frame_rgb.shape
    bytes_per_line = ch * w
    return QtGui.QImage(
        frame_rgb.data, w, h, bytes_per_line, QtGui.QImage.Format.Format_RGB888
    ).copy()


class InteractivePreviewLabel(QtWidgets.QLabel):
    requestReset = QtCore.Signal()

    def __init__(self, config: Config, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.setMouseTracking(True)
        self._pixmap: Optional[QtGui.QPixmap] = None
        self._image_w_h: Optional[Tuple[int, int]] = None  # (w, h)
        self._target_rect: QtCore.QRect = QtCore.QRect()
        self._drag_index: Optional[int] = None
        self._hover_index: Optional[int] = None
        self._handle_radius: int = 8
        # Overlays visibility
        self._show_roi_overlay: bool = True
        self._show_warp_overlay: bool = True
        # Which overlay is being edited: 'roi' or 'warp'
        self._edit_mode: str = "roi"
        self.config = config

    def set_roi_overlay_visible(self, visible: bool) -> None:
        self._show_roi_overlay = visible
        self.update()

    def set_warp_overlay_visible(self, visible: bool) -> None:
        self._show_warp_overlay = visible
        self.update()

    def set_edit_mode(self, mode: str) -> None:
        if mode not in ("roi", "warp"):
            return
        self._edit_mode = mode
        self.update()

    def update_image(self, img: QtGui.QImage, image_size: Tuple[int, int]) -> None:
        self._pixmap = QtGui.QPixmap.fromImage(img)
        self._image_w_h = image_size
        self.update()

    def _compute_target_rect(self) -> QtCore.QRect:
        if not self._pixmap:
            return QtCore.QRect()
        w = self.width()
        h = self.height()
        pm_w = self._pixmap.width()
        pm_h = self._pixmap.height()
        if pm_w <= 0 or pm_h <= 0:
            return QtCore.QRect(0, 0, w, h)
        scale = min(w / pm_w, h / pm_h)
        dw = int(pm_w * scale)
        dh = int(pm_h * scale)
        x = (w - dw) // 2
        y = (h - dh) // 2
        return QtCore.QRect(x, y, dw, dh)

    def _image_to_widget(self, pt: QtCore.QPointF) -> QtCore.QPointF:
        tr = self._target_rect
        if tr.width() == 0 or not self._image_w_h:
            return QtCore.QPointF(pt.x(), pt.y())
        img_w, img_h = self._image_w_h
        sx = tr.width() / img_w
        sy = tr.height() / img_h
        return QtCore.QPointF(tr.x() + pt.x() * sx, tr.y() + pt.y() * sy)

    def _widget_to_image(self, pt: QtCore.QPointF) -> QtCore.QPointF:
        tr = self._target_rect
        if tr.width() == 0 or not self._image_w_h:
            return QtCore.QPointF(pt.x(), pt.y())
        img_w, img_h = self._image_w_h
        sx = img_w / tr.width()
        sy = img_h / tr.height()
        return QtCore.QPointF((pt.x() - tr.x()) * sx, (pt.y() - tr.y()) * sy)

    def _roi_points_image(self) -> list[QtCore.QPointF]:
        if not self._image_w_h:
            return []
        img_w, img_h = self._image_w_h
        cfg = self.config
        pts = [
            QtCore.QPointF(cfg.roi_bottom_left_x_pct * img_w, cfg.roi_bottom_y_pct * img_h),
            QtCore.QPointF(cfg.roi_bottom_right_x_pct * img_w, cfg.roi_bottom_y_pct * img_h),
            QtCore.QPointF(cfg.roi_top_right_x_pct * img_w, cfg.roi_top_y_pct * img_h),
            QtCore.QPointF(cfg.roi_top_left_x_pct * img_w, cfg.roi_top_y_pct * img_h),
        ]
        return pts

    def _set_roi_points_image(self, pts: list[QtCore.QPointF]) -> None:
        if not self._image_w_h or len(pts) != 4:
            return
        img_w, img_h = self._image_w_h
        bl, br, tr, tl = pts  # BL, BR, TR, TL
        def clamp01f(val: float) -> float:
            return float(max(0.0, min(1.0, val)))
        self.config.roi_bottom_left_x_pct = clamp01f(bl.x() / img_w)
        self.config.roi_bottom_right_x_pct = clamp01f(br.x() / img_w)
        self.config.roi_top_right_x_pct = clamp01f(tr.x() / img_w)
        self.config.roi_top_left_x_pct = clamp01f(tl.x() / img_w)
        self.config.roi_bottom_y_pct = clamp01f(bl.y() / img_h)
        self.config.roi_top_y_pct = clamp01f(tl.y() / img_h)
        self.update()

    # Warp source polygon (Bird's-Eye src) helpers
    def _warp_src_points_image(self) -> list[QtCore.QPointF]:
        if not self._image_w_h:
            return []
        img_w, img_h = self._image_w_h
        p = self.config.perspective
        pts = [
            QtCore.QPointF(p.src_bottom_left_x_pct * img_w, p.src_bottom_y_pct * img_h),
            QtCore.QPointF(p.src_bottom_right_x_pct * img_w, p.src_bottom_y_pct * img_h),
            QtCore.QPointF(p.src_top_right_x_pct * img_w, p.src_top_y_pct * img_h),
            QtCore.QPointF(p.src_top_left_x_pct * img_w, p.src_top_y_pct * img_h),
        ]
        return pts

    def _set_warp_src_points_image(self, pts: list[QtCore.QPointF]) -> None:
        if not self._image_w_h or len(pts) != 4:
            return
        img_w, img_h = self._image_w_h
        bl, br, tr, tl = pts
        def clamp01f(val: float) -> float:
            return float(max(0.0, min(1.0, val)))
        p = self.config.perspective
        p.src_bottom_left_x_pct = clamp01f(bl.x() / img_w)
        p.src_bottom_right_x_pct = clamp01f(br.x() / img_w)
        p.src_top_right_x_pct = clamp01f(tr.x() / img_w)
        p.src_top_left_x_pct = clamp01f(tl.x() / img_w)
        p.src_bottom_y_pct = clamp01f(bl.y() / img_h)
        p.src_top_y_pct = clamp01f(tl.y() / img_h)
        self.update()

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        self._target_rect = self._compute_target_rect()

        if self._pixmap:
            painter.drawPixmap(self._target_rect, self._pixmap)

        # Draw ROI overlay
        if self._show_roi_overlay:
            pts_img = self._roi_points_image()
            if len(pts_img) == 4:
                pts_w = [self._image_to_widget(p) for p in pts_img]
                poly = QtGui.QPolygonF(pts_w)
                c_base = QtGui.QColor(0, 255, 0, 50)
                c_active = QtGui.QColor(0, 255, 100, 70)
                fill = c_active if self._edit_mode == "roi" else c_base
                stroke = QtGui.QPen(QtGui.QColor(0, 220, 0), 2)
                painter.setPen(stroke)
                painter.setBrush(fill)
                painter.drawPolygon(poly)

                if self._edit_mode == "roi":
                    for i, p in enumerate(pts_w):
                        r = self._handle_radius
                        color = QtGui.QColor(154, 75, 215) if i != self._hover_index else QtGui.QColor(200, 120, 255)
                        painter.setBrush(color)
                        painter.setPen(QtGui.QPen(QtCore.Qt.GlobalColor.white, 1))
                        painter.drawEllipse(QtCore.QPointF(p.x(), p.y()), r, r)

                    BL, BR, TR, TL = pts_w
                    edge_points = [
                        QtCore.QPointF((BL.x() + TL.x()) / 2.0, (BL.y() + TL.y()) / 2.0),
                        QtCore.QPointF((BR.x() + TR.x()) / 2.0, (BR.y() + TR.y()) / 2.0),
                        QtCore.QPointF((TL.x() + TR.x()) / 2.0, (TL.y() + TR.y()) / 2.0),
                        QtCore.QPointF((BL.x() + BR.x()) / 2.0, (BL.y() + BR.y()) / 2.0),
                    ]
                    for j, p in enumerate(edge_points):
                        r = self._handle_radius
                        color = QtGui.QColor(110, 180, 255) if (self._hover_index == 4 + j) else QtGui.QColor(80, 130, 200)
                        painter.setBrush(color)
                        painter.setPen(QtGui.QPen(QtCore.Qt.GlobalColor.white, 1))
                        painter.drawRect(QtCore.QRectF(p.x() - r, p.y() - r, 2 * r, 2 * r))

        # Draw Warp SRC overlay
        if self._show_warp_overlay:
            pts_img = self._warp_src_points_image()
            if len(pts_img) == 4:
                pts_w = [self._image_to_widget(p) for p in pts_img]
                poly = QtGui.QPolygonF(pts_w)
                c_base = QtGui.QColor(0, 200, 255, 50)
                c_active = QtGui.QColor(50, 220, 255, 80)
                fill = c_active if self._edit_mode == "warp" else c_base
                stroke = QtGui.QPen(QtGui.QColor(0, 180, 220), 2)
                painter.setPen(stroke)
                painter.setBrush(fill)
                painter.drawPolygon(poly)

                if self._edit_mode == "warp":
                    for i, p in enumerate(pts_w):
                        r = self._handle_radius
                        color = QtGui.QColor(255, 120, 0) if i != self._hover_index else QtGui.QColor(255, 170, 80)
                        painter.setBrush(color)
                        painter.setPen(QtGui.QPen(QtCore.Qt.GlobalColor.white, 1))
                        painter.drawEllipse(QtCore.QPointF(p.x(), p.y()), r, r)

        painter.end()

    def _closest_handle(self, pos_w: QtCore.QPointF) -> Optional[int]:
        pts_corners_img = (
            self._roi_points_image() if self._edit_mode == "roi" else self._warp_src_points_image()
        )
        if len(pts_corners_img) != 4:
            return None
        pts_corners_w = [self._image_to_widget(p) for p in pts_corners_img]
        BL, BR, TR, TL = pts_corners_w
        left_center = QtCore.QPointF((BL.x() + TL.x()) / 2.0, (BL.y() + TL.y()) / 2.0)
        right_center = QtCore.QPointF((BR.x() + TR.x()) / 2.0, (BR.y() + TR.y()) / 2.0)
        top_center = QtCore.QPointF((TL.x() + TR.x()) / 2.0, (TL.y() + TR.y()) / 2.0)
        bottom_center = QtCore.QPointF((BL.x() + BR.x()) / 2.0, (BL.y() + BR.y()) / 2.0)
        handle_points = pts_corners_w + [left_center, right_center, top_center, bottom_center]

        min_i = None
        min_d2 = float("inf")
        for i, p in enumerate(handle_points):
            dx = p.x() - pos_w.x()
            dy = p.y() - pos_w.y()
            d2 = dx * dx + dy * dy
            if d2 < min_d2:
                min_d2 = d2
                min_i = i
        if min_i is not None and min_d2 <= (self._handle_radius * 2) ** 2:
            return min_i
        return None

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if (self._edit_mode == "roi" and not self._show_roi_overlay) or (
            self._edit_mode == "warp" and not self._show_warp_overlay
        ):
            super().mousePressEvent(event)
            return
        self._drag_index = self._closest_handle(QtCore.QPointF(event.position()))
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        pos_w = QtCore.QPointF(event.position())
        if self._drag_index is not None and ((self._edit_mode == "roi" and self._show_roi_overlay) or (self._edit_mode == "warp" and self._show_warp_overlay)):
            pos_img = self._widget_to_image(pos_w)
            pts = self._roi_points_image() if self._edit_mode == "roi" else self._warp_src_points_image()
            if len(pts) == 4 and self._image_w_h:
                img_w, img_h = self._image_w_h
                x = max(0.0, min(float(pos_img.x()), float(img_w - 1)))
                y = max(0.0, min(float(pos_img.y()), float(img_h - 1)))

                idx = self._drag_index
                if idx in (0, 1, 2, 3):
                    pts[idx] = QtCore.QPointF(x, y)
                    if idx in (0, 1):
                        pts[0].setY(y)
                        pts[1].setY(y)
                    if idx in (2, 3):
                        pts[2].setY(y)
                        pts[3].setY(y)
                elif idx == 4:
                    pts[0].setX(x)
                    pts[3].setX(x)
                elif idx == 5:
                    pts[1].setX(x)
                    pts[2].setX(x)
                elif idx == 6:
                    pts[2].setY(y)
                    pts[3].setY(y)
                elif idx == 7:
                    pts[0].setY(y)
                    pts[1].setY(y)
                if self._edit_mode == "roi":
                    self._set_roi_points_image(pts)
                else:
                    self._set_warp_src_points_image(pts)
        else:
            self._hover_index = self._closest_handle(pos_w)
            self.update()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._drag_index is not None and ((self._edit_mode == "roi" and self._show_roi_overlay) or (self._edit_mode == "warp" and self._show_warp_overlay)):
            self.requestReset.emit()
        self._drag_index = None
        super().mouseReleaseEvent(event)


class VideoWorker(QtCore.QThread):
    frameReady = QtCore.Signal(np.ndarray)
    debugBinaryReady = QtCore.Signal(np.ndarray)
    debugWarpReady = QtCore.Signal(np.ndarray)
    finished = QtCore.Signal()

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.video_path: Optional[str] = None
        self._running = False
        self._pause = False
        self._reset_requested = False
        self._reset_lock = QtCore.QMutex()
        self.perspective = PerspectiveTransformer(self.config)
        self.detector = LaneDetector(self.config)
        self.renderer = LaneRenderer(self.config, self.perspective, self.detector)
        # No YOLO

    def set_video(self, path: str) -> None:
        self.video_path = path

    def stop(self) -> None:
        self._running = False

    def toggle_pause(self) -> None:
        self._pause = not self._pause

    def request_reset(self) -> None:
        locker = QtCore.QMutexLocker(self._reset_lock)
        self._reset_requested = True
        del locker

    def _apply_reset_if_needed(self) -> None:
        need_reset = False
        # Check + clear under lock
        locker = QtCore.QMutexLocker(self._reset_lock)
        if self._reset_requested:
            need_reset = True
            self._reset_requested = False
        del locker
        if need_reset:
            self.perspective.reset()
            self.detector.reset()

    def run(self) -> None:
        if not self.video_path:
            self.finished.emit()
            return

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            self.finished.emit()
            return

        self._running = True
        while self._running:
            if self._pause:
                self.msleep(20)
                continue

            # Apply deferred reset safely inside worker thread
            self._apply_reset_if_needed()

            ret, frame = cap.read()
            if not ret:
                break

            out_frame, debug = self.renderer.process_frame(frame, build_binary_mask)
            self.frameReady.emit(out_frame)
            self.debugBinaryReady.emit(debug["binary"])  # type: ignore[arg-type]
            self.debugWarpReady.emit(debug["warped_binary"])  # type: ignore[arg-type]

            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            delay_ms = int(1000.0 / fps)
            self.msleep(max(1, delay_ms // 2))

        cap.release()
        self.finished.emit()


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Lane Detection - Sliding Window + Polyfit + Kalman")
        self.resize(1280, 820)

        self.config = get_default_config()
        self.worker = VideoWorker(self.config)
        self.screenWorker = ScreenCaptureWorker(self.config)

        self._init_ui()
        self._connect_signals()
        # Debounce timer for safe, smooth resets on live tuning
        self.resetTimer = QtCore.QTimer(self)
        self.resetTimer.setSingleShot(True)
        self.resetTimer.setInterval(150)
        self.resetTimer.timeout.connect(self.worker.request_reset)

    def _init_ui(self) -> None:
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)

        self.previewLabel = InteractivePreviewLabel(self.config)
        self.previewLabel.setMinimumHeight(420)
        self.binaryLabel = QtWidgets.QLabel(alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
        self.warpLabel = QtWidgets.QLabel(alignment=QtCore.Qt.AlignmentFlag.AlignCenter)

        self.openButton = QtWidgets.QPushButton("Open Video…")
        self.playButton = QtWidgets.QPushButton("Play/Pause")
        self.stopButton = QtWidgets.QPushButton("Stop")
        self.saveRoiButton = QtWidgets.QPushButton("Save ROI…")
        self.hideRoiButton = QtWidgets.QPushButton("Hide ROI")
        self.hideRoiButton.setCheckable(True)
        # Bird's-eye overlay controls
        self.editModeCombo = QtWidgets.QComboBox()
        self.editModeCombo.addItems(["ROI", "Bird's-Eye Src"])
        self.hideWarpButton = QtWidgets.QPushButton("Hide Bird's-Eye")
        self.hideWarpButton.setCheckable(True)
        # Source controls
        self.sourceCombo = QtWidgets.QComboBox()
        self.sourceCombo.addItems(["Video File", "Screen Capture"])
        self.monitorCombo = QtWidgets.QComboBox()
        self._refresh_monitors()

        self.cannyLow = self._make_slider(0, 255, self.config.canny.low_threshold, "Canny Low")
        self.cannyHigh = self._make_slider(0, 255, self.config.canny.high_threshold, "Canny High")
        self.marginSlider = self._make_slider(20, 200, self.config.sliding.margin, "Search Margin")
        self.minpixSlider = self._make_slider(10, 200, self.config.sliding.minpix, "Min Pixels")

        controlBar = QtWidgets.QHBoxLayout()
        controlBar.addWidget(self.openButton)
        controlBar.addWidget(self.playButton)
        controlBar.addWidget(self.stopButton)
        controlBar.addWidget(self.sourceCombo)
        controlBar.addWidget(self.monitorCombo)
        controlBar.addWidget(self.editModeCombo)
        controlBar.addWidget(self.hideWarpButton)
        controlBar.addStretch(1)
        controlBar.addWidget(self.hideRoiButton)
        controlBar.addWidget(self.saveRoiButton)

        sliders = QtWidgets.QGridLayout()
        sliders.addLayout(self.cannyLow[1], 0, 0)
        sliders.addLayout(self.cannyHigh[1], 0, 1)
        sliders.addLayout(self.marginSlider[1], 1, 0)
        sliders.addLayout(self.minpixSlider[1], 1, 1)

        previews = QtWidgets.QGridLayout()
        previews.addWidget(self.previewLabel, 0, 0, 1, 2)
        previews.addWidget(self.binaryLabel, 1, 0)
        previews.addWidget(self.warpLabel, 1, 1)

        root = QtWidgets.QVBoxLayout()
        root.addLayout(controlBar)
        root.addLayout(sliders)
        root.addLayout(previews)
        central.setLayout(root)

        self._apply_dark_theme()

    def _make_slider(self, minv: int, maxv: int, value: int, label: str):
        slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        slider.setRange(minv, maxv)
        slider.setValue(value)
        lab = QtWidgets.QLabel(f"{label}: {value}")
        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(lab)
        layout.addWidget(slider)
        return slider, layout, lab

    def _apply_dark_theme(self) -> None:
        palette = QtGui.QPalette()
        palette.setColor(QtGui.QPalette.ColorRole.Window, QtGui.QColor(30, 30, 30))
        palette.setColor(QtGui.QPalette.ColorRole.WindowText, QtCore.Qt.GlobalColor.white)
        palette.setColor(QtGui.QPalette.ColorRole.Base, QtGui.QColor(25, 25, 25))
        palette.setColor(QtGui.QPalette.ColorRole.AlternateBase, QtGui.QColor(53, 53, 53))
        palette.setColor(QtGui.QPalette.ColorRole.ToolTipBase, QtCore.Qt.GlobalColor.white)
        palette.setColor(QtGui.QPalette.ColorRole.ToolTipText, QtCore.Qt.GlobalColor.white)
        palette.setColor(QtGui.QPalette.ColorRole.Text, QtCore.Qt.GlobalColor.white)
        palette.setColor(QtGui.QPalette.ColorRole.Button, QtGui.QColor(53, 53, 53))
        palette.setColor(QtGui.QPalette.ColorRole.ButtonText, QtCore.Qt.GlobalColor.white)
        palette.setColor(QtGui.QPalette.ColorRole.BrightText, QtCore.Qt.GlobalColor.red)
        palette.setColor(QtGui.QPalette.ColorRole.Highlight, QtGui.QColor(142, 45, 197).lighter())
        palette.setColor(QtGui.QPalette.ColorRole.HighlightedText, QtCore.Qt.GlobalColor.black)
        self.setPalette(palette)

        self.setStyleSheet(
            """
            QLabel { color: #EEE; }
            QPushButton { background: #444; color: #EEE; padding: 6px 12px; border-radius: 4px; }
            QPushButton:hover { background: #555; }
            QSlider::groove:horizontal { height: 6px; background: #555; border-radius: 3px; }
            QSlider::handle:horizontal { background: #9a4bd7; width: 14px; margin: -6px 0; border-radius: 7px; }
            """
        )

    def _connect_signals(self) -> None:
        self.openButton.clicked.connect(self.handle_open)
        self.playButton.clicked.connect(self.handle_play_pause)
        self.stopButton.clicked.connect(self.handle_stop)
        self.saveRoiButton.clicked.connect(self.handle_save_roi)
        self.hideRoiButton.toggled.connect(self.handle_hide_roi)
        self.hideWarpButton.toggled.connect(self.handle_hide_warp)
        self.editModeCombo.currentIndexChanged.connect(self.handle_edit_mode)
        self.sourceCombo.currentIndexChanged.connect(self.handle_source_changed)
        self.monitorCombo.currentIndexChanged.connect(self.handle_monitor_changed)
        # Prompt for auto-ROI on load
        QtCore.QTimer.singleShot(300, self.prompt_auto_roi)

        self.cannyLow[0].valueChanged.connect(self.handle_canny_low)
        self.cannyHigh[0].valueChanged.connect(self.handle_canny_high)
        self.marginSlider[0].valueChanged.connect(self.handle_margin)
        self.minpixSlider[0].valueChanged.connect(self.handle_minpix)

        self.worker.frameReady.connect(self.update_preview)
        self.worker.debugBinaryReady.connect(self.update_binary)
        self.worker.debugWarpReady.connect(self.update_warp)
        self.worker.finished.connect(self.on_finished)
        # Screen worker signals
        self.screenWorker.frameReady.connect(self.update_preview)
        self.screenWorker.debugBinaryReady.connect(self.update_binary)
        self.screenWorker.debugWarpReady.connect(self.update_warp)
        self.screenWorker.finished.connect(self.on_finished)

        self.previewLabel.requestReset.connect(self.schedule_reset)

    def handle_open(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open Video", str(Path.cwd()), "Videos (*.mp4 *.avi *.mov)"
        )
        if not path:
            return
        self.worker.set_video(path)
        # Ask user to auto-detect ROI for this video
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
        # Stop current worker when switching source
        self.handle_stop()
        if self.sourceCombo.currentText() == "Screen Capture":
            self._refresh_monitors()

    def handle_monitor_changed(self, index: int) -> None:
        # Apply new monitor if running
        if self.screenWorker.isRunning():
            data = self.monitorCombo.currentData()
            if data is not None:
                self.screenWorker.set_monitor_bbox(data.bbox)

    def prompt_auto_roi(self) -> None:
        # Ask user if they'd like to compute a dynamic ROI using the current frame
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
        # Grab a single frame to compute ROI
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

        # Optionally refine with YOLO12 segmentation: remove vehicles from lane mask
        union = None
        # YOLO seg removed
        union = None

        base_mask = build_binary_mask(frame, self.config, apply_roi=False)
        mask = base_mask
        if union is not None:
            # Ensure union is 2D uint8 same size
            # Keep only lanes outside union (not used now)
            inv = cv2.bitwise_not(union)
            mask = cv2.bitwise_and(base_mask, inv)
        # Use histogram peaks and Canny edges to find a plausible trapezoid
        h, w = mask.shape[:2]
        # Bottom band histogram
        histogram = np.sum(mask[h // 2 :, :], axis=0)
        midpoint = histogram.shape[0] // 2
        left_base = int(np.argmax(histogram[:midpoint]))
        right_base = int(np.argmax(histogram[midpoint:]) + midpoint)

        # Track lines upward using simple BFS over mask to get top x positions
        def follow_up(x_start: int, step: int = 20, win: int = 50) -> int:
            x_curr = x_start
            for y in range(h - 1, int(h * 0.4), -step):
                x0 = max(0, x_curr - win)
                x1 = min(w - 1, x_curr + win)
                col = mask[max(0, y - step):y, x0:x1]
                if col.size == 0:
                    continue
                xs = np.argmax(np.sum(col, axis=0))
                x_curr = x0 + int(xs)
            return x_curr

        left_top = follow_up(left_base)
        right_top = follow_up(right_base)

        # Compose an ROI trapezoid using these guides, add margins
        bottom_y = int(h * 0.98)
        top_y = int(h * 0.62)
        bl = max(0, left_base - 60)
        br = min(w - 1, right_base + 60)
        tl = int(0.6 * left_top + 0.4 * right_top)
        tr = int(0.6 * right_top + 0.4 * left_top)
        tl = max(0, min(tl, w - 1))
        tr = max(0, min(tr, w - 1))

        # Update config in percentages
        self.config.roi_bottom_left_x_pct = bl / w
        self.config.roi_bottom_right_x_pct = br / w
        self.config.roi_top_left_x_pct = tl / w
        self.config.roi_top_right_x_pct = tr / w
        self.config.roi_bottom_y_pct = bottom_y / h
        self.config.roi_top_y_pct = top_y / h

        # Reset pipeline after dynamic change
        self.schedule_reset()

    def handle_save_roi(self) -> None:
        # Save current ROI percentages to a JSON file
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save ROI", str(Path.cwd() / "roi.json"), "JSON (*.json)"
        )
        if not path:
            return
        cfg = self.config
        data = {
            "roi": {
                "top_y_pct": cfg.roi_top_y_pct,
                "bottom_y_pct": cfg.roi_bottom_y_pct,
                "top_left_x_pct": cfg.roi_top_left_x_pct,
                "top_right_x_pct": cfg.roi_top_right_x_pct,
                "bottom_left_x_pct": cfg.roi_bottom_left_x_pct,
                "bottom_right_x_pct": cfg.roi_bottom_right_x_pct,
            }
        }
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to save ROI: {e}")

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
        # Debounce frequent changes
        self.resetTimer.start()

    @QtCore.Slot(np.ndarray)
    def update_preview(self, frame_bgr: np.ndarray) -> None:
        img = bgr_to_qimage(frame_bgr)
        h, w = frame_bgr.shape[:2]
        self.previewLabel.update_image(img, (w, h))
        # Also show steering text as window title if available via worker? Not necessary here.

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


def main() -> None:
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    rc = app.exec()
    # Ensure thread shutdown if window closed without pressing Stop
    if win.worker.isRunning():
        win.worker.stop()
        win.worker.wait(1000)
    if win.screenWorker.isRunning():
        win.screenWorker.stop()
        win.screenWorker.wait(1000)
    sys.exit(rc)


if __name__ == "__main__":
    main()


