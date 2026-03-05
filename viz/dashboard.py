"""
Dashboard EEG temps réel — QMainWindow qui orchestre panels et widgets.
"""

from __future__ import annotations

import json
import os
from datetime import datetime

import numpy as np
from PyQt5.QtCore import QEvent, Qt, QTimer
from PyQt5.QtWidgets import (
    QMainWindow, QScrollArea, QSplitter, QVBoxLayout, QWidget,
)

from processing import filter_signal, apply_car
from .panels.base import BasePanel, DashboardState
from .panels.timeseries import TimeSeriesPanel
from .panels.psd import PSDPanel
from .panels.snr_bar import SNRBarPanel
from .panels.spectrogram import SpectrogramPanel
from .widgets.channel_selector import ChannelSelector
from .widgets.processing_selector import ProcessingSelector
from .widgets.panel_selector import PanelSelector

_L_FREQ     = 1.0
_H_FREQ     = 50.0
_NOTCH_FREQ = 50.0
_RECORDINGS_DIR = 'recordings'


class Dashboard(QMainWindow):
    """
    Fenêtre principale du dashboard EEG.

    Layout :
        QSplitter horizontal
        ├── QSplitter vertical  (panels empilés, redimensionnables)
        │   ├── TimeSeriesPanel
        │   ├── PSDPanel
        │   ├── SNRBarPanel
        │   └── SpectrogramPanel
        └── QScrollArea (sidebar fixe 200 px)
            ├── ChannelSelector
            └── ProcessingSelector

    La touche R démarre / arrête l'enregistrement.
    """

    def __init__(
        self,
        board,
        channels: list[int],
        ch_labels: list[str],
        sfreq: float,
        n_samples: int,
        window_sec: float = 5.0,
        update_ms: int = 100,
        sidebar_extra: QWidget | None = None,
        ref_psd: dict | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle('OpenBCI Cyton — EEG Temps Réel')

        self._board      = board
        self._channels   = channels
        self._ch_labels  = ch_labels
        self._sfreq      = sfreq
        self._n_samples  = n_samples
        self._update_ms  = update_ms
        self._n_ch       = len(channels)
        self._ref_psd    = ref_psd

        self._proc_state: dict = {'bandpass': True, 'notch': True, 'car': False, 'show_snr': False}
        self._rec_active = False
        self._rec_buffer: list[np.ndarray] = []
        self._sidebar_extra = sidebar_extra

        self._panels: list[BasePanel] = []
        self._ts_panel:  TimeSeriesPanel | None = None

        self._build_ui(window_sec)

        # Intercepte R même si un widget enfant a le focus
        from PyQt5.QtWidgets import QApplication
        QApplication.instance().installEventFilter(self)

        self._timer = QTimer(self)
        self._timer.setInterval(update_ms)
        self._timer.timeout.connect(self._update)
        self._timer.start()

    # ------------------------------------------------------------------
    # Construction UI
    # ------------------------------------------------------------------

    def _build_ui(self, window_sec: float) -> None:
        root = QSplitter(Qt.Horizontal, self)
        self.setCentralWidget(root)

        # -- Zone panels (gauche) --
        self._panels_splitter = QSplitter(Qt.Vertical)
        root.addWidget(self._panels_splitter)

        self._ts_panel   = TimeSeriesPanel(self._ch_labels, self._n_samples, window_sec)
        self._psd_panel  = PSDPanel(self._ch_labels)
        self._snr_panel  = SNRBarPanel(self._ch_labels)
        self._spec_panel = SpectrogramPanel(self._ch_labels, update_ms=self._update_ms)

        self.add_panel(self._ts_panel)
        self.add_panel(self._psd_panel)
        self.add_panel(self._snr_panel)
        self.add_panel(self._spec_panel)
        self._panels_splitter.setSizes([400, 280, 150, 270])

        # -- Sidebar (droite) --
        sidebar_scroll = QScrollArea()
        sidebar_scroll.setWidgetResizable(True)
        sidebar_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        sidebar_scroll.setFixedWidth(200)

        sidebar_widget = QWidget()
        sidebar_layout = QVBoxLayout(sidebar_widget)
        sidebar_layout.setAlignment(Qt.AlignTop)
        sidebar_layout.setSpacing(12)
        sidebar_layout.setContentsMargins(8, 8, 8, 8)

        self._ch_selector    = ChannelSelector(self._ch_labels)
        self._proc_selector  = ProcessingSelector()
        self._panel_selector = PanelSelector()  # ordre : [ts, psd, snr, spec]
        sidebar_layout.addWidget(self._ch_selector)
        sidebar_layout.addWidget(self._proc_selector)
        sidebar_layout.addWidget(self._panel_selector)

        if self._sidebar_extra is not None:
            sidebar_layout.addWidget(self._sidebar_extra)

        sidebar_scroll.setWidget(sidebar_widget)
        root.addWidget(sidebar_scroll)
        root.setSizes([800, 200])

        # Connexions signaux
        self._ch_selector.visibility_changed.connect(self._on_channels_changed)
        self._proc_selector.processing_changed.connect(self._on_processing_changed)
        self._panel_selector.panel_toggled.connect(self._on_panel_toggled)

    # ------------------------------------------------------------------
    # API publique
    # ------------------------------------------------------------------

    def add_panel(self, panel: BasePanel) -> None:
        """Ajoute un panel dans le splitter vertical."""
        self._panels_splitter.addWidget(panel.widget)
        self._panels.append(panel)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_channels_changed(self, ch_visible: list[bool]) -> None:
        for panel in self._panels:
            panel.on_channels_changed(ch_visible)

    def _on_processing_changed(self, state: dict) -> None:
        self._proc_state = state
        self._refresh_title()

    def _on_panel_toggled(self, idx: int, visible: bool) -> None:
        panel = self._panels[idx]
        if visible:
            panel.widget.show()
        else:
            panel.widget.hide()

    def _refresh_title(self) -> None:
        parts = []
        if self._proc_state.get('bandpass'):
            parts.append(f'1–{int(_H_FREQ)} Hz')
        if self._proc_state.get('notch'):
            parts.append(f'notch {int(_NOTCH_FREQ)} Hz')
        if self._proc_state.get('car'):
            parts.append('CAR')
        title = (
            f"Signal filtré ({', '.join(parts)})"
            if parts else 'Signal brut'
        )
        if self._ts_panel:
            self._ts_panel.set_title(title)

    # ------------------------------------------------------------------
    # Boucle d'acquisition
    # ------------------------------------------------------------------

    def _update(self) -> None:
        raw = self._board.get_current_board_data(self._n_samples)
        if raw.shape[1] < self._n_samples:
            return

        data_uv = np.array([raw[ch] for ch in self._channels])
        data_v  = data_uv * 1e-6

        if self._rec_active:
            chunk = max(1, int(self._update_ms / 1000.0 * self._sfreq))
            self._rec_buffer.append(data_uv[:, -chunk:].copy())

        data_filt = filter_signal(
            data_v, self._sfreq,
            l_freq=_L_FREQ, h_freq=_H_FREQ, notch_freq=_NOTCH_FREQ,
            causal=False,
            apply_bandpass=self._proc_state['bandpass'],
            apply_notch=self._proc_state['notch'],
        )

        if self._proc_state['car']:
            data_filt = apply_car(data_filt)

        state = DashboardState(
            ch_visible=self._ch_selector.ch_visible,
            show_snr=self._proc_state['show_snr'],
            ref_psd=self._ref_psd,
        )

        for panel in self._panels:
            if panel.widget.isVisible():
                panel.update(data_filt, self._sfreq, state)

    # ------------------------------------------------------------------
    # Enregistrement
    # ------------------------------------------------------------------

    def _toggle_recording(self) -> None:
        if not self._rec_active:
            self._rec_active = True
            self._rec_buffer = []
            self.setWindowTitle('OpenBCI Cyton — EEG Temps Réel  [● REC]')
            print('\n  [REC] Enregistrement démarré — appuie sur R pour arrêter')
        else:
            self._rec_active = False
            self.setWindowTitle('OpenBCI Cyton — EEG Temps Réel')
            if self._rec_buffer:
                recorded = np.concatenate(self._rec_buffer, axis=1)
                self._save_recording(recorded)
            self._rec_buffer = []

    def _save_recording(self, data: np.ndarray) -> None:
        os.makedirs(_RECORDINGS_DIR, exist_ok=True)
        ts        = datetime.now().strftime('%Y%m%d_%H%M%S')
        npy_path  = os.path.join(_RECORDINGS_DIR, f'{ts}.npy')
        meta_path = os.path.join(_RECORDINGS_DIR, f'{ts}.json')
        np.save(npy_path, data)
        meta = {
            'timestamp':    ts,
            'sfreq':        self._sfreq,
            'channels':     self._ch_labels,
            'n_samples':    data.shape[1],
            'duration_sec': round(data.shape[1] / self._sfreq, 2),
        }
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)
        print(f'\n  [REC] Sauvegardé : {npy_path} ({meta["duration_sec"]} s)')

    # ------------------------------------------------------------------
    # Événements Qt
    # ------------------------------------------------------------------

    def eventFilter(self, obj, event) -> bool:
        """Intercepte R globalement (même si un widget enfant a le focus)."""
        if event.type() == QEvent.KeyPress and event.text().lower() == 'r':
            self._toggle_recording()
            return True
        return super().eventFilter(obj, event)

    def closeEvent(self, event) -> None:
        self._timer.stop()
        super().closeEvent(event)
