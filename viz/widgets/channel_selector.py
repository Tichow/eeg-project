"""
Widget de sélection des canaux EEG : une checkbox par canal.
"""

from __future__ import annotations

from PyQt5.QtWidgets import QGroupBox, QVBoxLayout, QCheckBox
from PyQt5.QtCore import pyqtSignal

_COLORS = [
    '#e6194b', '#3cb44b', '#4363d8', '#f58231',
    '#911eb4', '#42d4f4', '#f032e6', '#bfef45',
]


class ChannelSelector(QGroupBox):
    """
    QGroupBox avec une QCheckBox par canal EEG.

    Signaux :
        channel_toggled(int, bool) : index du canal, nouvel état
        visibility_changed(list)   : liste complète ch_visible[bool]
    """

    channel_toggled    = pyqtSignal(int, bool)
    visibility_changed = pyqtSignal(list)

    def __init__(
        self,
        ch_labels: list[str],
        initial_visible: list[bool] | None = None,
        parent=None,
    ) -> None:
        super().__init__('Canaux', parent=parent)
        self._ch_visible: list[bool] = (
            list(initial_visible) if initial_visible is not None
            else [True] * len(ch_labels)
        )
        self._checkboxes: list[QCheckBox] = []

        layout = QVBoxLayout(self)
        layout.setSpacing(4)
        layout.setContentsMargins(8, 12, 8, 8)

        for i, label in enumerate(ch_labels):
            cb = QCheckBox(label)
            cb.setChecked(self._ch_visible[i])
            color = _COLORS[i % len(_COLORS)]
            cb.setStyleSheet(
                f'QCheckBox {{ color: {color}; font-weight: bold; }}'
            )
            cb.stateChanged.connect(self._make_handler(i))
            layout.addWidget(cb)
            self._checkboxes.append(cb)

    def _make_handler(self, idx: int):
        def handler(state: int) -> None:
            checked = bool(state)
            self._ch_visible[idx] = checked
            self.channel_toggled.emit(idx, checked)
            self.visibility_changed.emit(list(self._ch_visible))
        return handler

    @property
    def ch_visible(self) -> list[bool]:
        return list(self._ch_visible)

    def set_channel_visible(self, idx: int, visible: bool) -> None:
        """Modification programmatique sans émettre de signal."""
        self._checkboxes[idx].blockSignals(True)
        self._checkboxes[idx].setChecked(visible)
        self._checkboxes[idx].blockSignals(False)
        self._ch_visible[idx] = visible
