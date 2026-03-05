"""
Widget de sélection de la visibilité des panels de visualisation.
"""

from __future__ import annotations

from PyQt5.QtWidgets import QGroupBox, QVBoxLayout, QCheckBox
from PyQt5.QtCore import pyqtSignal

_PANEL_LABELS = ['Time Series', 'PSD', 'SNR', 'Spectrogram']


class PanelSelector(QGroupBox):
    """
    QGroupBox avec une QCheckBox par panel de visualisation.

    Signaux :
        panel_toggled(int, bool)  : index du panel, nouvel état
        visibility_changed(list)  : liste complète panel_visible[bool]
    """

    panel_toggled      = pyqtSignal(int, bool)
    visibility_changed = pyqtSignal(list)

    def __init__(self, labels: list[str] | None = None, parent=None) -> None:
        super().__init__('Panels', parent=parent)

        _labels = labels if labels is not None else _PANEL_LABELS
        self._panel_visible: list[bool] = [True] * len(_labels)
        self._checkboxes: list[QCheckBox] = []

        layout = QVBoxLayout(self)
        layout.setSpacing(4)
        layout.setContentsMargins(8, 12, 8, 8)

        for i, label in enumerate(_labels):
            cb = QCheckBox(label)
            cb.setChecked(True)
            cb.stateChanged.connect(self._make_handler(i))
            layout.addWidget(cb)
            self._checkboxes.append(cb)

    def _make_handler(self, idx: int):
        def handler(state: int) -> None:
            checked = bool(state)
            self._panel_visible[idx] = checked
            self.panel_toggled.emit(idx, checked)
            self.visibility_changed.emit(list(self._panel_visible))
        return handler

    @property
    def panel_visible(self) -> list[bool]:
        return list(self._panel_visible)

    def set_panel_visible(self, idx: int, visible: bool) -> None:
        """Modification programmatique sans émettre de signal."""
        self._checkboxes[idx].blockSignals(True)
        self._checkboxes[idx].setChecked(visible)
        self._checkboxes[idx].blockSignals(False)
        self._panel_visible[idx] = visible
