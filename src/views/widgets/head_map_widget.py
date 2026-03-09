from __future__ import annotations

from PyQt5.QtCore import Qt, QPointF, QRectF, pyqtSignal
from PyQt5.QtGui import QBrush, QColor, QFont, QPainter, QPen, QPolygonF
from PyQt5.QtWidgets import QWidget

from src.constants.eeg_constants import PRESET_ELECTRODES

# Azimuthal equidistant projection of MNE standard_1020 montage positions.
# Same projection used by MNE topomaps: origin=vertex (Cz), x=right, y=nose.
# Unit: dimensionless — rho=1.0 corresponds to the ear-level (equator).
# Peripheral electrodes (T7/T8/F7/F8/Fp1/Fp2) reach rho ~1.05-1.09.
# Ultracortex Mark IV — 35 official positions, azimuthal equidistant projection.
# T3/T4/T5/T6 use MNE coordinates of T7/T8/P7/P8 (same physical positions, old naming).
_RAW_POSITIONS: dict[str, tuple[float, float]] = {
    "Fp1": (-0.347538,  0.990749),
    "Fpz": ( 0.001288,  1.012355),
    "Fp2": ( 0.348510,  0.990453),
    "AF3": (-0.338301,  0.771320),
    "AFz": ( 0.002110,  0.736922),
    "AF4": ( 0.350722,  0.763327),
    "F7":  (-0.931084,  0.562845),
    "F3":  (-0.458234,  0.484385),
    "Fz":  ( 0.002452,  0.459555),
    "F4":  ( 0.471587,  0.494045),
    "F8":  ( 0.930257,  0.565743),
    "FC5": (-0.787211,  0.190070),
    "FC1": (-0.248875,  0.190051),
    "FC2": ( 0.256568,  0.195006),
    "FC6": ( 0.791122,  0.198299),
    "T3":  (-1.050320, -0.199911),
    "C3":  (-0.501982, -0.089337),
    "Cz":  ( 0.002539, -0.058055),
    "C4":  ( 0.514622, -0.083577),
    "T4":  ( 1.053360, -0.185964),
    "CP5": (-0.685245, -0.400776),
    "CP1": (-0.219706, -0.292577),
    "CP2": ( 0.237405, -0.291148),
    "CP6": ( 0.698594, -0.386527),
    "T5":  (-0.712928, -0.722951),
    "P3":  (-0.369026, -0.548505),
    "Pz":  ( 0.001978, -0.494166),
    "P4":  ( 0.382663, -0.540037),
    "T6":  ( 0.718109, -0.718233),
    "PO3": (-0.268118, -0.740604),
    "POz": ( 0.001492, -0.707234),
    "PO4": ( 0.271335, -0.743956),
    "O1":  (-0.240830, -0.920705),
    "Oz":  ( 0.000861, -0.919221),
    "O2":  ( 0.244746, -0.919817),
}

# Normalization radius: slightly larger than the most extreme electrode (T8 at rho=1.054)
# so all electrodes fit inside the drawn head circle with a small margin.
_HEAD_RADIUS = 1.12

# Color palette per channel (1-8), consistent with AcquisitionView plot curves
_CHANNEL_COLORS: list[QColor] = [
    QColor("#e06c75"),  # CH1 — rouge
    QColor("#e5c07b"),  # CH2 — jaune
    QColor("#98c379"),  # CH3 — vert
    QColor("#56b6c2"),  # CH4 — teal
    QColor("#61afef"),  # CH5 — bleu
    QColor("#c678dd"),  # CH6 — violet
    QColor("#abb2bf"),  # CH7 — gris
    QColor("#d19a66"),  # CH8 — orange
]


class HeadMapWidget(QWidget):
    """Interactive 2D top-down head diagram for electrode-to-channel assignment.

    Interaction:
    - Click an electrode to select it (highlighted ring).
    - Press 1-8 to assign the selected electrode to that channel number.
    - Press 0, Delete, or Backspace to clear the assignment for the selected electrode.
    - Click the same electrode again or click elsewhere to deselect.

    Emits:
    - assignment_changed(dict[int, str]): whenever an assignment changes.
      Dict maps channel number (1-8) to electrode name.
    """

    assignment_changed = pyqtSignal(dict)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(420, 420)
        self.setFocusPolicy(Qt.StrongFocus)

        self._selected: str | None = None
        self._assignment: dict[int, str] = {}  # channel (1-8) → electrode name
        self._electrode_rects: dict[str, QRectF] = {}  # rebuilt each paintEvent

    # ── Public API ────────────────────────────────────────────────────────────

    def set_assignment(self, assignment: dict[int, str]) -> None:
        """Load a preset assignment and repaint."""
        self._assignment = dict(assignment)
        self._selected = None
        self.update()
        self.assignment_changed.emit(dict(self._assignment))

    def get_assignment(self) -> dict[int, str]:
        return dict(self._assignment)

    def clear_all(self) -> None:
        self._assignment.clear()
        self._selected = None
        self.update()
        self.assignment_changed.emit({})

    # ── Coordinate mapping ────────────────────────────────────────────────────

    def _electrode_center(self, name: str, w: int, h: int) -> QPointF:
        """Map projected (x, y) to widget pixel coordinates.

        Projected coords: x right=positive, y frontal=positive.
        Screen: x right=positive, y down=positive → invert y.
        """
        margin = 0.07 * min(w, h)
        r = (min(w, h) - 2 * margin) / 2
        cx, cy = w / 2.0, h / 2.0
        scale = r / _HEAD_RADIUS
        raw_x, raw_y = _RAW_POSITIONS[name]
        return QPointF(cx + raw_x * scale, cy - raw_y * scale)

    def _head_radius_px(self, w: int, h: int) -> float:
        margin = 0.07 * min(w, h)
        return (min(w, h) - 2 * margin) / 2

    # ── Paint ─────────────────────────────────────────────────────────────────

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        w, h = self.width(), self.height()
        cx, cy = w / 2.0, h / 2.0
        r = self._head_radius_px(w, h)
        elec_to_ch = {v: k for k, v in self._assignment.items()}

        self._draw_head(painter, cx, cy, r)
        self._draw_electrodes(painter, w, h, elec_to_ch)

        # Focus indicator
        if self.hasFocus():
            painter.setPen(QPen(QColor("#4a90d9"), 2, Qt.DashLine))
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(self.rect().adjusted(1, 1, -1, -1))

        painter.end()

    def _draw_head(self, painter: QPainter, cx: float, cy: float, r: float) -> None:
        # Head circle
        painter.setPen(QPen(QColor("#aaaaaa"), 2))
        painter.setBrush(QBrush(QColor("#fafafa")))
        painter.drawEllipse(QPointF(cx, cy), r, r)

        # Nose — small upward triangle, base at circle top
        nose_w = r * 0.08
        nose_h = r * 0.06
        nose_tip_y = cy - r - nose_h
        nose = QPolygonF([
            QPointF(cx - nose_w, cy - r),
            QPointF(cx + nose_w, cy - r),
            QPointF(cx, nose_tip_y),
        ])
        painter.setPen(QPen(QColor("#aaaaaa"), 1.5))
        painter.setBrush(QBrush(QColor("#fafafa")))
        painter.drawPolygon(nose)

        # Ears — small ellipses on each side
        ear_w = r * 0.07
        ear_h = r * 0.13
        painter.setBrush(QBrush(QColor("#fafafa")))
        painter.drawEllipse(QPointF(cx - r - ear_w * 0.4, cy), ear_w, ear_h)
        painter.drawEllipse(QPointF(cx + r + ear_w * 0.4, cy), ear_w, ear_h)

    def _draw_electrodes(
        self,
        painter: QPainter,
        w: int,
        h: int,
        elec_to_ch: dict[str, int],
    ) -> None:
        r = self._head_radius_px(w, h)
        elec_r = max(9, r * 0.075)
        self._electrode_rects.clear()

        font_ch = QFont()
        font_ch.setPointSize(max(6, int(elec_r * 0.80)))
        font_ch.setBold(True)

        font_name = QFont()
        font_name.setPointSize(max(5, int(elec_r * 0.55)))

        for name in PRESET_ELECTRODES:
            center = self._electrode_center(name, w, h)
            rect = QRectF(
                center.x() - elec_r,
                center.y() - elec_r,
                elec_r * 2,
                elec_r * 2,
            )
            self._electrode_rects[name] = rect

            ch = elec_to_ch.get(name)
            is_selected = name == self._selected

            # Outer selection ring
            if is_selected:
                painter.setPen(QPen(QColor("#333333"), 3))
                painter.setBrush(Qt.NoBrush)
                painter.drawEllipse(rect.adjusted(-4, -4, 4, 4))

            # Fill
            fill = _CHANNEL_COLORS[(ch - 1) % len(_CHANNEL_COLORS)] if ch else QColor("#d0d0d0")
            painter.setBrush(QBrush(fill))
            painter.setPen(QPen(QColor("#777777"), 1))
            painter.drawEllipse(rect)

            # Label
            if ch is not None:
                painter.setFont(font_ch)
                painter.setPen(QPen(QColor("white"), 1))
                painter.drawText(rect, Qt.AlignCenter, str(ch))
            else:
                painter.setFont(font_name)
                painter.setPen(QPen(QColor("#444444"), 1))
                painter.drawText(rect, Qt.AlignCenter, name)

    # ── Mouse ─────────────────────────────────────────────────────────────────

    def mousePressEvent(self, event) -> None:
        if event.button() != Qt.LeftButton:
            return
        pos = QPointF(event.pos())
        for name, rect in self._electrode_rects.items():
            if rect.adjusted(-5, -5, 5, 5).contains(pos):
                self._selected = None if self._selected == name else name
                self.update()
                self.setFocus()
                return
        self._selected = None
        self.update()

    # ── Keyboard ──────────────────────────────────────────────────────────────

    def keyPressEvent(self, event) -> None:
        if self._selected is None:
            super().keyPressEvent(event)
            return

        key = event.key()

        if Qt.Key_1 <= key <= Qt.Key_8:
            ch = key - Qt.Key_0  # 1-8
            # Remove this electrode from any previous channel
            new = {k: v for k, v in self._assignment.items() if v != self._selected}
            # Remove whatever was in this channel slot
            new.pop(ch, None)
            new[ch] = self._selected
            self._assignment = new
            self.update()
            self.assignment_changed.emit(dict(self._assignment))

        elif key in (Qt.Key_0, Qt.Key_Delete, Qt.Key_Backspace):
            self._assignment = {
                k: v for k, v in self._assignment.items() if v != self._selected
            }
            self.update()
            self.assignment_changed.emit(dict(self._assignment))

        else:
            super().keyPressEvent(event)
