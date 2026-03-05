"""
Widget de contrôle de lecture pour le mode offline.

Affiche un bouton play/pause, un slider de position, un label de temps
et un sélecteur de vitesse. Se met à jour toutes les 200 ms via un QTimer
indépendant du timer d'acquisition du Dashboard.
"""

from __future__ import annotations

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QComboBox, QGroupBox, QHBoxLayout, QLabel,
    QPushButton, QSlider, QVBoxLayout, QWidget,
)

from viz.offline_player import OfflinePlayer

_SPEEDS = [('0.5×', 0.5), ('1×', 1.0), ('2×', 2.0), ('4×', 4.0)]


def _fmt_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f'{m:02d}:{s:02d}'


class PlaybackWidget(QGroupBox):
    """
    Contrôles de lecture : ▶/⏸, slider, temps, vitesse.

    Usage :
        player = OfflinePlayer(data_uv, sfreq)
        widget = PlaybackWidget(player)
        sidebar_layout.addWidget(widget)
    """

    def __init__(
        self,
        player: OfflinePlayer,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__('Lecture', parent=parent)
        self._player = player

        layout = QVBoxLayout(self)
        layout.setSpacing(6)
        layout.setContentsMargins(8, 12, 8, 8)

        # -- Bouton play/pause --
        self._btn = QPushButton('⏸')
        self._btn.setFixedHeight(28)
        self._btn.clicked.connect(self._toggle_pause)
        layout.addWidget(self._btn)

        # -- Slider de position --
        self._slider = QSlider(Qt.Horizontal)
        self._slider.setMinimum(0)
        self._slider.setMaximum(player.n_total - 1)
        self._slider.setValue(0)
        self._slider.sliderPressed.connect(self._on_slider_pressed)
        self._slider.sliderReleased.connect(self._on_slider_released)
        layout.addWidget(self._slider)

        # -- Label temps --
        total_sec = player.n_total / player.sfreq
        self._lbl = QLabel(f'00:00 / {_fmt_time(total_sec)}')
        self._lbl.setAlignment(Qt.AlignCenter)
        layout.addWidget(self._lbl)

        # -- Vitesse --
        speed_row = QHBoxLayout()
        speed_row.addWidget(QLabel('Vitesse'))
        self._speed_box = QComboBox()
        for label, _ in _SPEEDS:
            self._speed_box.addItem(label)
        self._speed_box.setCurrentIndex(1)  # 1× par défaut
        self._speed_box.currentIndexChanged.connect(self._on_speed_changed)
        speed_row.addWidget(self._speed_box)
        layout.addLayout(speed_row)

        # -- Timer de refresh --
        self._slider_dragging = False
        self._refresh_timer = QTimer(self)
        self._refresh_timer.setInterval(200)
        self._refresh_timer.timeout.connect(self._refresh)
        self._refresh_timer.start()

    # ------------------------------------------------------------------
    # Slots privés
    # ------------------------------------------------------------------

    def _toggle_pause(self) -> None:
        if self._player.is_paused:
            self._player.resume()
            self._btn.setText('⏸')
        else:
            self._player.pause()
            self._btn.setText('▶')

    def _on_slider_pressed(self) -> None:
        self._slider_dragging = True
        self._player.pause()
        self._btn.setText('▶')

    def _on_slider_released(self) -> None:
        self._player.seek(self._slider.value())
        self._slider_dragging = False
        self._player.resume()
        self._btn.setText('⏸')

    def _on_speed_changed(self, idx: int) -> None:
        self._player.set_speed(_SPEEDS[idx][1])

    def _refresh(self) -> None:
        """Met à jour le slider et le label avec la position courante."""
        if self._slider_dragging:
            return
        pos = self._player.current_pos
        self._slider.blockSignals(True)
        self._slider.setValue(min(pos, self._player.n_total - 1))
        self._slider.blockSignals(False)
        cur_sec   = pos / self._player.sfreq
        total_sec = self._player.n_total / self._player.sfreq
        self._lbl.setText(f'{_fmt_time(cur_sec)} / {_fmt_time(total_sec)}')
