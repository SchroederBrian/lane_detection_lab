from __future__ import annotations

from typing import Optional, Tuple

from PySide6 import QtCore, QtGui, QtWidgets

from config import Config


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

    def update_image(self, img: Optional[QtGui.QImage], image_size: Tuple[int, int]) -> None:
        if img is None:
            self._pixmap = None
        else:
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
