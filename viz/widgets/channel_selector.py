"""
Widget de sélection des canaux EEG : une checkbox par canal.

Supporte un nombre arbitraire de canaux (scroll area) et des boutons
preset pour sélection rapide (Occipital, Moteur, Tout, Aucun).
"""

from __future__ import annotations

from PyQt5.QtWidgets import (
    QGroupBox, QVBoxLayout, QHBoxLayout, QCheckBox,
    QPushButton, QScrollArea, QWidget,
)
from PyQt5.QtCore import pyqtSignal, Qt

_COLORS = [
    '#e6194b', '#3cb44b', '#4363d8', '#f58231',
    '#911eb4', '#42d4f4', '#f032e6', '#bfef45',
]

# Presets de canaux 10-10 (noms en majuscules, comme PhysioNet après nettoyage)
_PRESETS: dict[str, list[str]] = {
    'Occipital': ['O1', 'OZ', 'O2', 'P3', 'PZ', 'P4', 'C3', 'C4'],
    'Moteur':    ['C3', 'CZ', 'C4', 'FC3', 'FCZ', 'FC4', 'CP3', 'CP4'],
}


class ChannelSelector(QGroupBox):
    """
    QGroupBox avec une QCheckBox par canal EEG, dans un QScrollArea.

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
        self._ch_labels   = list(ch_labels)
        self._ch_visible: list[bool] = (
            list(initial_visible) if initial_visible is not None
            else [True] * len(ch_labels)
        )
        self._checkboxes: list[QCheckBox] = []

        outer = QVBoxLayout(self)
        outer.setSpacing(4)
        outer.setContentsMargins(8, 12, 8, 8)

        # --- Boutons globaux (Tout / Aucun) ---
        row_all = QHBoxLayout()
        btn_all  = QPushButton('Tout')
        btn_none = QPushButton('Aucun')
        btn_all.setFixedHeight(22)
        btn_none.setFixedHeight(22)
        btn_all.clicked.connect(self._select_all)
        btn_none.clicked.connect(self._select_none)
        row_all.addWidget(btn_all)
        row_all.addWidget(btn_none)
        outer.addLayout(row_all)

        # --- Boutons preset (toujours affichés) ---
        for preset_name, preset_chs in _PRESETS.items():
            btn = QPushButton(preset_name)
            btn.setFixedHeight(22)
            btn.clicked.connect(self._make_preset_handler(preset_chs))
            outer.addWidget(btn)

        # --- Scroll area avec les checkboxes ---
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setMaximumHeight(300)

        inner_widget = QWidget()
        inner_layout = QVBoxLayout(inner_widget)
        inner_layout.setSpacing(2)
        inner_layout.setContentsMargins(0, 0, 0, 0)

        for i, label in enumerate(ch_labels):
            cb = QCheckBox(label)
            cb.setChecked(self._ch_visible[i])
            color = _COLORS[i % len(_COLORS)]
            cb.setStyleSheet(
                f'QCheckBox {{ color: {color}; font-weight: bold; }}'
            )
            cb.stateChanged.connect(self._make_handler(i))
            inner_layout.addWidget(cb)
            self._checkboxes.append(cb)

        inner_layout.addStretch()
        scroll.setWidget(inner_widget)
        outer.addWidget(scroll)

    # ------------------------------------------------------------------
    # Handlers
    # ------------------------------------------------------------------

    def _make_handler(self, idx: int):
        def handler(state: int) -> None:
            checked = bool(state)
            self._ch_visible[idx] = checked
            self.channel_toggled.emit(idx, checked)
            self.visibility_changed.emit(list(self._ch_visible))
        return handler

    def _make_preset_handler(self, preset_chs: list[str]):
        def handler() -> None:
            ch_upper = [c.upper() for c in self._ch_labels]
            for i, label_upper in enumerate(ch_upper):
                visible = label_upper in preset_chs
                self._set_checked_silent(i, visible)
            self.visibility_changed.emit(list(self._ch_visible))
        return handler

    def _select_all(self) -> None:
        for i in range(len(self._ch_labels)):
            self._set_checked_silent(i, True)
        self.visibility_changed.emit(list(self._ch_visible))

    def _select_none(self) -> None:
        for i in range(len(self._ch_labels)):
            self._set_checked_silent(i, False)
        self.visibility_changed.emit(list(self._ch_visible))

    def _set_checked_silent(self, idx: int, visible: bool) -> None:
        self._checkboxes[idx].blockSignals(True)
        self._checkboxes[idx].setChecked(visible)
        self._checkboxes[idx].blockSignals(False)
        self._ch_visible[idx] = visible

    # ------------------------------------------------------------------
    # API publique
    # ------------------------------------------------------------------

    @property
    def ch_visible(self) -> list[bool]:
        return list(self._ch_visible)

    def set_channel_visible(self, idx: int, visible: bool) -> None:
        """Modification programmatique sans émettre de signal."""
        self._set_checked_silent(idx, visible)
