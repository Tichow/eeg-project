from __future__ import annotations

import os
import re

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from src.constants.eeg_constants import ACQUISITION_PROTOCOLS
from src.views.base_view import BaseView

# Standard 10-20 electrode names for QComboBox items
_EEG_ELECTRODES = [
    "Fp1", "Fpz", "Fp2", "AF7", "AF3", "AFz", "AF4", "AF8",
    "F7", "F5", "F3", "F1", "Fz", "F2", "F4", "F6", "F8",
    "FT7", "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6", "FT8",
    "T3", "T7", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "T8", "T4",
    "TP7", "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6", "TP8",
    "T5", "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8", "T6",
    "PO7", "PO5", "PO3", "POz", "PO4", "PO6", "PO8",
    "O1", "Oz", "O2",
]

_DEFAULT_CHANNELS = ["Fz", "C3", "Cz", "C4", "Pz", "PO3", "Oz", "PO4"]
_N_CHANNELS = 8
_PREVIEW_SECS = 4
_SFREQ = 250
_CHANNEL_OFFSET_V = 100e-6  # 100 µV vertical spacing in preview
_VREF = 4.5  # ADS1299 reference voltage (V)
_GAIN_VALUES = [1, 2, 4, 6, 8, 12, 24]
_HP_ALPHA: float = 1.0 - 2.0 * np.pi * 0.5 / _SFREQ  # DC-blocking ~0.5 Hz HP filter

_BTN_BLUE = """
    QPushButton {
        background-color: #4a90d9; color: white;
        border: none; border-radius: 4px; padding: 6px 12px;
    }
    QPushButton:hover { background-color: #357abd; }
    QPushButton:pressed { background-color: #2a6099; }
    QPushButton:disabled { background-color: #aaaaaa; }
"""
_BTN_RED = """
    QPushButton {
        background-color: #d94a4a; color: white;
        border: none; border-radius: 4px; padding: 6px 12px;
    }
    QPushButton:hover { background-color: #bd3535; }
    QPushButton:pressed { background-color: #992020; }
"""
_BTN_GRAY = """
    QPushButton {
        background-color: #e0e0e0; color: #333;
        border: 1px solid #bbb; border-radius: 4px; padding: 6px 12px;
    }
    QPushButton:hover { background-color: #d0d0d0; }
"""


class _CueFullscreenWindow(QWidget):
    """Fenêtre plein écran affichant les indications au sujet pendant l'acquisition."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowFlags(Qt.Window | Qt.FramelessWindowHint)
        self.setStyleSheet("background: #000;")
        layout = QVBoxLayout(self)
        self._label = QLabel("✛")
        font = QFont()
        font.setPointSize(120)
        font.setBold(True)
        self._label.setFont(font)
        self._label.setAlignment(Qt.AlignCenter)
        self._label.setStyleSheet("color: #4a90d9;")
        layout.addWidget(self._label)

    def set_text(self, text: str) -> None:
        self._label.setText(text)

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key_Escape:
            self.close()


class AcquisitionView(BaseView):
    """Vue d'acquisition EEG temps réel avec un casque OpenBCI Cyton."""

    def setup_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # --- Header ---
        header = QWidget()
        header.setFixedHeight(48)
        header.setStyleSheet("background:#2c2c2c;")
        h_layout = QHBoxLayout(header)
        h_layout.setContentsMargins(12, 0, 12, 0)

        back_btn = QPushButton("← Retour")
        back_btn.setStyleSheet(_BTN_GRAY)
        back_btn.setCursor(Qt.PointingHandCursor)
        back_btn.clicked.connect(lambda: self.navigate.emit("home"))

        title = QLabel("Acquisition EEG — OpenBCI Cyton")
        title.setStyleSheet("color: white; font-size: 15px; font-weight: bold;")

        h_layout.addWidget(back_btn)
        h_layout.addSpacing(12)
        h_layout.addWidget(title)
        h_layout.addStretch()
        root.addWidget(header)

        # --- State (must be initialized before building panels) ---
        self._railed_labels: list[QLabel] = []
        self._plot_curves: list = []

        # --- Main splitter ---
        splitter = QSplitter(Qt.Horizontal)
        root.addWidget(splitter)

        splitter.addWidget(self._build_left_panel())
        splitter.addWidget(self._build_right_panel())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([300, 900])

        # --- State ---
        self._board = None
        self._stream_worker = None
        self._record_worker = None
        self._connected = False

        # Ring buffer for 4s of 8-channel data (stores HP-filtered signal for display)
        self._buffer = np.zeros((_N_CHANNELS, _PREVIEW_SECS * _SFREQ), dtype=np.float64)
        self._plot_x = np.linspace(0, _PREVIEW_SECS, _PREVIEW_SECS * _SFREQ)
        # HP filter state (reset on disconnect to avoid transition artefacts)
        self._hp_x_prev = np.zeros(_N_CHANNELS, dtype=np.float64)
        self._hp_y_prev = np.zeros(_N_CHANNELS, dtype=np.float64)
        self._cached_ch_names: list[str] = []
        self._indicator_states: list[str] = ["" ] * _N_CHANNELS  # "flat"|"railed"|"ok"|""
        self._srb2_warning_shown: bool = False
        self._init_plot_curves()

        # Gain state (default Cyton hardware gain)
        self._current_gain: int = 24
        self._railed_threshold_v: float = _VREF / 24 * 0.95  # 95% of full-scale for robust detection

        # Cue display map — updated when a protocol preset is selected
        self._cue_display_map: dict[str, str] = dict(ACQUISITION_PROTOCOLS[0].cue_display_map)
        self._cue_window: _CueFullscreenWindow | None = None

        # Initial UI state
        self._set_connected(False)
        self._auto_fill_subject_number()

    # ------------------------------------------------------------------
    # Panel builders
    # ------------------------------------------------------------------

    def _build_left_panel(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(300)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        layout.addWidget(self._build_connection_group())
        layout.addWidget(self._build_gain_group())
        layout.addWidget(self._build_channels_group())
        layout.addWidget(self._build_subject_group())
        layout.addWidget(self._build_protocol_group())

        self._start_btn = QPushButton("▶ Démarrer l'enregistrement")
        self._start_btn.setStyleSheet(_BTN_BLUE)
        self._start_btn.setCursor(Qt.PointingHandCursor)
        self._start_btn.setEnabled(False)
        self._start_btn.clicked.connect(self._on_start_record)
        layout.addWidget(self._start_btn)

        layout.addStretch()
        scroll.setWidget(container)
        return scroll

    def _build_connection_group(self) -> QGroupBox:
        grp = QGroupBox("Connexion")
        layout = QVBoxLayout(grp)

        port_row = QHBoxLayout()
        self._port_combo = QComboBox()
        self._port_combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        refresh_btn = QPushButton("↻")
        refresh_btn.setFixedWidth(30)
        refresh_btn.setToolTip("Rafraîchir les ports série")
        refresh_btn.setCursor(Qt.PointingHandCursor)
        refresh_btn.clicked.connect(self._refresh_ports)
        port_row.addWidget(self._port_combo, 1)
        port_row.addWidget(refresh_btn)
        layout.addLayout(port_row)

        self._connect_btn = QPushButton("Connecter")
        self._connect_btn.setStyleSheet(_BTN_BLUE)
        self._connect_btn.setCursor(Qt.PointingHandCursor)
        self._connect_btn.clicked.connect(self._on_connect_toggle)
        layout.addWidget(self._connect_btn)

        self._status_label = QLabel("Non connecté")
        self._status_label.setStyleSheet("color: #888; font-size: 11px;")
        layout.addWidget(self._status_label)

        self._refresh_ports()
        return grp

    def _build_gain_group(self) -> QGroupBox:
        grp = QGroupBox("Gain")
        layout = QVBoxLayout(grp)

        row = QHBoxLayout()
        row.addWidget(QLabel("Gain global :"))
        self._gain_combo = QComboBox()
        for g in _GAIN_VALUES:
            self._gain_combo.addItem(f"{g}x")
        self._gain_combo.setCurrentText("24x")
        self._gain_combo.currentTextChanged.connect(self._update_gain_range_label)
        row.addWidget(self._gain_combo, 1)

        self._gain_apply_btn = QPushButton("Appliquer")
        self._gain_apply_btn.setStyleSheet(_BTN_BLUE)
        self._gain_apply_btn.setCursor(Qt.PointingHandCursor)
        self._gain_apply_btn.setEnabled(False)
        self._gain_apply_btn.clicked.connect(self._on_gain_apply)
        row.addWidget(self._gain_apply_btn)
        layout.addLayout(row)

        self._gain_auto_btn = QPushButton("⚡ Détection automatique")
        self._gain_auto_btn.setStyleSheet(_BTN_GRAY)
        self._gain_auto_btn.setCursor(Qt.PointingHandCursor)
        self._gain_auto_btn.setEnabled(False)
        self._gain_auto_btn.clicked.connect(self._on_gain_auto_detect)
        layout.addWidget(self._gain_auto_btn)

        self._gain_range_label = QLabel(f"Plage : ± {_VREF / 24 * 1000:.1f} mV")
        self._gain_range_label.setStyleSheet("color: #888; font-size: 11px;")
        layout.addWidget(self._gain_range_label)

        return grp

    def _build_channels_group(self) -> QGroupBox:
        grp = QGroupBox("Canaux (10-20)")
        layout = QVBoxLayout(grp)

        # Preset selector row
        preset_row = QHBoxLayout()
        self._elec_preset_combo = QComboBox()
        self._elec_preset_combo.setPlaceholderText("Préréglage…")
        self._elec_preset_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        apply_btn = QPushButton("Appliquer")
        apply_btn.setStyleSheet(_BTN_BLUE)
        apply_btn.setFixedWidth(72)
        apply_btn.setCursor(Qt.PointingHandCursor)
        apply_btn.clicked.connect(self._apply_electrode_preset)
        preset_row.addWidget(self._elec_preset_combo, 1)
        preset_row.addWidget(apply_btn)
        layout.addLayout(preset_row)

        self._ch_combos: list[QComboBox] = []
        for i in range(_N_CHANNELS):
            row = QHBoxLayout()
            lbl = QLabel(f"CH{i + 1}")
            lbl.setFixedWidth(32)
            combo = QComboBox()
            combo.addItems(_EEG_ELECTRODES)
            default = _DEFAULT_CHANNELS[i] if i < len(_DEFAULT_CHANNELS) else _EEG_ELECTRODES[0]
            idx = combo.findText(default)
            if idx >= 0:
                combo.setCurrentIndex(idx)
            indicator = QLabel("●")
            indicator.setFixedWidth(16)
            indicator.setStyleSheet("color: #555555; font-size: 14px;")
            indicator.setToolTip("Non connecté")
            row.addWidget(lbl)
            row.addWidget(combo, 1)
            row.addWidget(indicator)
            layout.addLayout(row)
            self._ch_combos.append(combo)
            self._railed_labels.append(indicator)
        return grp

    def _apply_electrode_preset(self) -> None:
        from src.services.preset_service import PresetService
        name = self._elec_preset_combo.currentText()
        if not name:
            return
        preset = PresetService.get_preset(name)
        if preset is None:
            return
        for combo, elec in zip(self._ch_combos, preset.get_ordered(n=_N_CHANNELS)):
            if not elec:
                continue
            idx = combo.findText(elec)
            if idx >= 0:
                combo.setCurrentIndex(idx)

    def on_navigate_to(self) -> None:
        from src.services.preset_service import PresetService
        current = self._elec_preset_combo.currentText()
        self._elec_preset_combo.blockSignals(True)
        self._elec_preset_combo.clear()
        for p in PresetService.load_all():
            self._elec_preset_combo.addItem(p.name)
        idx = self._elec_preset_combo.findText(current)
        if idx >= 0:
            self._elec_preset_combo.setCurrentIndex(idx)
        self._elec_preset_combo.blockSignals(False)

    def _build_subject_group(self) -> QGroupBox:
        grp = QGroupBox("Sujet & Sortie")
        layout = QVBoxLayout(grp)

        layout.addWidget(QLabel("Sujet :"))
        name_row = QHBoxLayout()
        self._name_combo = QComboBox()
        self._name_combo.addItems(["MATTEO", "FABIEN", "NEMO", "AUTRE"])
        self._number_spin = QSpinBox()
        self._number_spin.setRange(1, 99)
        self._number_spin.setValue(1)
        self._number_spin.setFixedWidth(55)
        name_row.addWidget(self._name_combo, 1)
        name_row.addWidget(self._number_spin)
        layout.addLayout(name_row)

        layout.addWidget(QLabel("Dossier de sortie :"))
        dir_row = QHBoxLayout()
        self._output_edit = QLineEdit("data/custom/")
        browse_btn = QPushButton("…")
        browse_btn.setFixedWidth(30)
        browse_btn.setCursor(Qt.PointingHandCursor)
        browse_btn.clicked.connect(self._browse_output_dir)
        dir_row.addWidget(self._output_edit, 1)
        dir_row.addWidget(browse_btn)
        layout.addLayout(dir_row)

        layout.addWidget(QLabel("Label run :"))
        self._run_label_edit = QLineEdit("R01")
        layout.addWidget(self._run_label_edit)

        # Auto-fill connections
        self._name_combo.currentTextChanged.connect(self._auto_fill_subject_number)
        self._output_edit.textChanged.connect(self._auto_fill_subject_number)
        self._number_spin.valueChanged.connect(self._auto_fill_run_label)

        return grp

    def _build_protocol_group(self) -> QGroupBox:
        grp = QGroupBox("Protocole")
        layout = QVBoxLayout(grp)

        layout.addWidget(QLabel("Préréglage :"))
        self._preset_combo = QComboBox()
        for p in ACQUISITION_PROTOCOLS:
            self._preset_combo.addItem(p.name)
        self._preset_combo.currentIndexChanged.connect(self._on_preset_changed)
        layout.addWidget(self._preset_combo)

        trials_row = QHBoxLayout()
        trials_row.addWidget(QLabel("Essais / classe :"))
        self._n_trials_spin = QSpinBox()
        self._n_trials_spin.setRange(1, 100)
        self._n_trials_spin.setValue(20)
        trials_row.addWidget(self._n_trials_spin)
        layout.addLayout(trials_row)

        for attr, label, default in [
            ("_t_baseline_spin", "Baseline (s) :", 2.0),
            ("_t_cue_spin", "Cue (s) :", 4.0),
            ("_t_rest_spin", "Repos (s) :", 1.5),
        ]:
            row = QHBoxLayout()
            row.addWidget(QLabel(label))
            spin = QDoubleSpinBox()
            spin.setRange(0.0, 120.0)
            spin.setSingleStep(0.5)
            spin.setValue(default)
            row.addWidget(spin)
            setattr(self, attr, spin)
            layout.addLayout(row)

        layout.addWidget(QLabel("Classes :"))
        self._class_left_cb = QCheckBox("Gauche (T1)")
        self._class_left_cb.setChecked(True)
        self._class_right_cb = QCheckBox("Droite (T2)")
        self._class_right_cb.setChecked(True)
        layout.addWidget(self._class_left_cb)
        layout.addWidget(self._class_right_cb)

        return grp

    def _build_right_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # Real-time signal preview
        self._plot_widget = pg.PlotWidget()
        self._plot_widget.setBackground("#1a1a2e")
        self._plot_widget.showGrid(x=True, y=False, alpha=0.2)
        self._plot_widget.setLabel("bottom", "Temps (s)")
        self._plot_widget.setLabel("left", "Canaux")
        self._plot_widget.setMouseEnabled(x=False, y=False)
        self._plot_widget.disableAutoRange()
        self._plot_widget.setXRange(0, _PREVIEW_SECS, padding=0)
        y_span = _N_CHANNELS * _CHANNEL_OFFSET_V
        self._plot_widget.setYRange(-_CHANNEL_OFFSET_V, y_span, padding=0.05)
        self._plot_widget.setMinimumHeight(200)
        layout.addWidget(self._plot_widget, 3)

        # Cue label
        self._cue_label = QLabel("✛")
        font = QFont()
        font.setPointSize(48)
        font.setBold(True)
        self._cue_label.setFont(font)
        self._cue_label.setAlignment(Qt.AlignCenter)
        self._cue_label.setStyleSheet("color: #4a90d9; background: #0d0d1a; border-radius: 8px;")
        self._cue_label.setMinimumHeight(100)
        layout.addWidget(self._cue_label, 1)

        # Progress bar
        self._progress_bar = QProgressBar()
        self._progress_bar.setValue(0)
        self._progress_bar.setFormat("Essai %v / %m")
        layout.addWidget(self._progress_bar)

        # Log
        self._log = QTextEdit()
        self._log.setReadOnly(True)
        mono = QFont("Monospace")
        mono.setStyleHint(QFont.Monospace)
        self._log.setFont(mono)
        self._log.setMaximumHeight(120)
        self._log.setStyleSheet("background: #111; color: #ccc;")
        layout.addWidget(self._log)

        return panel

    # ------------------------------------------------------------------
    # Plot initialisation
    # ------------------------------------------------------------------

    def _init_plot_curves(self) -> None:
        """Create 8 color-coded curves in the preview PlotWidget."""
        colors = [
            "#e06c75", "#e5c07b", "#98c379", "#56b6c2",
            "#61afef", "#c678dd", "#abb2bf", "#d19a66",
        ]
        self._plot_curves = []
        x = np.linspace(0, _PREVIEW_SECS, _PREVIEW_SECS * _SFREQ)
        ch_names = _DEFAULT_CHANNELS
        for i in range(_N_CHANNELS):
            offset = (_N_CHANNELS - 1 - i) * _CHANNEL_OFFSET_V
            name = ch_names[i] if i < len(ch_names) else f"CH{i + 1}"
            curve = self._plot_widget.plot(
                x,
                np.zeros(_PREVIEW_SECS * _SFREQ) + offset,
                pen=pg.mkPen(colors[i % len(colors)], width=1),
                name=name,
            )
            self._plot_curves.append((curve, offset))

        # Y-axis ticks = channel names
        ticks = [
            (_N_CHANNELS - 1 - i) * _CHANNEL_OFFSET_V
            for i in range(_N_CHANNELS)
        ]
        tick_labels = [
            [(v, n) for v, n in zip(ticks, ch_names)]
        ]
        self._plot_widget.getAxis("left").setTicks(tick_labels)
        self._plot_widget.setYRange(-_CHANNEL_OFFSET_V, _N_CHANNELS * _CHANNEL_OFFSET_V, padding=0.05)

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def _refresh_ports(self) -> None:
        from src.services.acquisition_service import AcquisitionService

        self._port_combo.clear()
        try:
            ports = AcquisitionService.list_serial_ports()
        except Exception as exc:
            self._append_log(f"[ERREUR] Lecture ports : {exc}")
            return
        if ports:
            self._port_combo.addItems(ports)
        else:
            self._port_combo.addItem("(aucun port détecté)")

    def _on_connect_toggle(self) -> None:
        if self._connected:
            self._disconnect()
        else:
            self._connect()

    def _connect(self) -> None:
        from src.services.acquisition_service import AcquisitionService

        port = self._port_combo.currentText()
        if not port or port.startswith("("):
            self._append_log("[ERREUR] Sélectionnez un port valide.")
            return

        self._connect_btn.setEnabled(False)
        self._status_label.setText("Connexion en cours…")
        self._append_log(f"[INFO] Connexion sur {port}…")

        try:
            board, sfreq = AcquisitionService.connect(port)
        except Exception as exc:
            self._append_log(f"[ERREUR] Connexion échouée : {exc}")
            self._connect_btn.setEnabled(True)
            self._status_label.setText("Échec de connexion")
            return

        self._board = board
        self._append_log(f"[INFO] Connecté — {sfreq:.0f} Hz")
        self._set_connected(True)
        self._start_stream()

    def _disconnect(self) -> None:
        from src.services.acquisition_service import AcquisitionService

        self._stop_stream()
        if self._board is not None:
            try:
                AcquisitionService.disconnect(self._board)
            except Exception as exc:
                self._append_log(f"[WARN] Déconnexion : {exc}")
            self._board = None
        self._hp_x_prev[:] = 0.0
        self._hp_y_prev[:] = 0.0
        self._set_connected(False)
        self._append_log("[INFO] Déconnecté.")

    def _set_connected(self, connected: bool) -> None:
        self._connected = connected
        if connected:
            self._connect_btn.setText("Déconnecter")
            self._connect_btn.setStyleSheet(_BTN_RED)
            self._status_label.setText("Connecté ✓")
            self._status_label.setStyleSheet("color: #98c379; font-size: 11px;")
            self._start_btn.setEnabled(True)
            self._gain_apply_btn.setEnabled(True)
            self._gain_auto_btn.setEnabled(True)
            self._on_gain_apply()  # apply selected gain to hardware immediately on connect
        else:
            self._connect_btn.setText("Connecter")
            self._connect_btn.setStyleSheet(_BTN_BLUE)
            self._status_label.setText("Non connecté")
            self._status_label.setStyleSheet("color: #888; font-size: 11px;")
            self._start_btn.setEnabled(False)
            self._gain_apply_btn.setEnabled(False)
            self._gain_auto_btn.setEnabled(False)
            for lbl in self._railed_labels:
                lbl.setStyleSheet("color: #555555; font-size: 14px;")
                lbl.setToolTip("Non connecté")
        self._connect_btn.setEnabled(True)

    # ------------------------------------------------------------------
    # Stream worker
    # ------------------------------------------------------------------

    def _start_stream(self) -> None:
        if self._board is None:
            return
        from src.workers.acquisition_stream_worker import AcquisitionStreamWorker

        self._stream_worker = AcquisitionStreamWorker(self._board)
        self._stream_worker.chunk_ready.connect(self._on_chunk_ready)
        self._stream_worker.error.connect(self._on_stream_error)
        self._stream_worker.start()

    def _stop_stream(self) -> None:
        if self._stream_worker is not None:
            try:
                self._stream_worker.chunk_ready.disconnect()
                self._stream_worker.error.disconnect()
            except (RuntimeError, TypeError):
                pass
            self._stream_worker.stop()
            self._stream_worker.quit()
            self._stream_worker.wait()
            self._stream_worker = None

    def _hp_filter(self, chunk: np.ndarray) -> np.ndarray:
        """Apply per-channel DC-blocking HP filter (~0.5 Hz) to a chunk (8, n).

        y[n] = x[n] - x[n-1] + α·y[n-1]   removes DC and slow electrode drift.
        """
        out = np.empty_like(chunk)
        for s in range(chunk.shape[1]):
            x = chunk[:, s]
            y = x - self._hp_x_prev + _HP_ALPHA * self._hp_y_prev
            out[:, s] = y
            self._hp_x_prev = x
            self._hp_y_prev = y
        return out

    def _on_chunk_ready(self, chunk: np.ndarray) -> None:
        """Update the ring buffer and redraw preview curves."""
        filtered = self._hp_filter(chunk)
        n = filtered.shape[1]
        buf_len = self._buffer.shape[1]
        if n >= buf_len:
            self._buffer[:] = filtered[:, -buf_len:]
        else:
            self._buffer = np.roll(self._buffer, -n, axis=1)
            self._buffer[:, -n:] = filtered

        ch_names = [combo.currentText() for combo in self._ch_combos]
        if ch_names != self._cached_ch_names:
            ticks_y = [(_N_CHANNELS - 1 - i) * _CHANNEL_OFFSET_V for i in range(_N_CHANNELS)]
            self._plot_widget.getAxis("left").setTicks([list(zip(ticks_y, ch_names))])
            self._cached_ch_names = ch_names

        for i, (curve, offset) in enumerate(self._plot_curves):
            if i < self._buffer.shape[0]:
                ch_data = self._buffer[i]
                curve.setData(self._plot_x, ch_data + offset)

        for i, lbl in enumerate(self._railed_labels):
            if i < self._buffer.shape[0]:
                ch_buf = self._buffer[i]
                is_flat = np.std(ch_buf) < 2e-6
                is_railed = not is_flat and np.max(np.abs(ch_buf)) > self._railed_threshold_v
                state = "flat" if is_flat else ("railed" if is_railed else "ok")
                if state != self._indicator_states[i]:
                    self._indicator_states[i] = state
                    if state == "flat":
                        lbl.setStyleSheet("color: #e5c07b; font-size: 14px;")
                        lbl.setToolTip("Canal plat — électrode déconnectée ou mauvais contact")
                    elif state == "railed":
                        lbl.setStyleSheet("color: #e06c75; font-size: 14px;")
                        lbl.setToolTip("Raillé — saturation ADC, vérifier électrode et référence")
                        self._append_log(f"[WARN] Canal {i + 1} raillé — saturation ADC")
                    else:
                        lbl.setStyleSheet("color: #98c379; font-size: 14px;")
                        lbl.setToolTip("Signal OK")

        # Detect all channels railed simultaneously → likely missing SRB2 reference electrode
        n_railed = sum(1 for s in self._indicator_states if s == "railed")
        if n_railed >= 6 and not self._srb2_warning_shown:
            self._srb2_warning_shown = True
            self._append_log(
                "[WARN] ≥6 canaux raillés simultanément — référence SRB2 absente ? "
                "Vérifier l'électrode earlobe branchée sur le pin SRB2 du Cyton."
            )
        elif n_railed < 6:
            self._srb2_warning_shown = False

    def _on_stream_error(self, msg: str) -> None:
        self._append_log(f"[ERREUR stream] {msg}")

    # ------------------------------------------------------------------
    # Gain management
    # ------------------------------------------------------------------

    def _update_gain_range_label(self) -> None:
        gain = int(self._gain_combo.currentText().replace("x", ""))
        self._gain_range_label.setText(f"Plage : ± {_VREF / gain * 1000:.1f} mV")

    def _on_gain_apply(self) -> None:
        if self._board is None:
            return
        from src.services.acquisition_service import AcquisitionService

        gain = int(self._gain_combo.currentText().replace("x", ""))
        try:
            AcquisitionService.set_gain(self._board, gain)
        except Exception as exc:
            self._append_log(f"[ERREUR gain] {exc}")
            return
        self._current_gain = gain
        self._railed_threshold_v = _VREF / gain * 0.95
        self._update_gain_range_label()
        for lbl in self._railed_labels:
            lbl.setStyleSheet("color: #98c379; font-size: 14px;")
            lbl.setToolTip("Non raillé")
        self._append_log(f"[INFO] Gain appliqué : {gain}x (plage ± {_VREF / gain * 1000:.1f} mV)")

    def _on_gain_auto_detect(self) -> None:
        # Buffer contains HP-filtered signal (DC and drift removed).
        # 99th percentile of |values| = robust peak amplitude (ignores isolated spikes).
        max_amp_v = float(np.percentile(np.abs(self._buffer), 99))
        if max_amp_v < 1e-9:
            self._append_log("[WARN] Buffer vide — attendre ~1 s après connexion")
            return
        _SAFETY = 0.85
        optimal = 1
        for g in sorted(_GAIN_VALUES, reverse=True):
            if max_amp_v < (_VREF / g) * _SAFETY:
                optimal = g
                break
        idx = self._gain_combo.findText(f"{optimal}x")
        self._gain_combo.setCurrentIndex(idx)
        self._on_gain_apply()
        self._append_log(
            f"[AUTO] Amplitude AC max = {max_amp_v * 1e6:.1f} µV → gain optimal : {optimal}x"
        )

    # ------------------------------------------------------------------
    # Record worker
    # ------------------------------------------------------------------

    def _on_start_record(self) -> None:
        if not self._connected or self._board is None:
            return

        # Warn if any channel is railed — data quality will be compromised
        n_railed = sum(1 for s in self._indicator_states if s == "railed")
        n_flat = sum(1 for s in self._indicator_states if s == "flat")
        if n_railed > 0 or n_flat > 0:
            problems = []
            if n_railed:
                problems.append(f"{n_railed} canal(aux) saturé(s) (rouge ❌)")
            if n_flat:
                problems.append(f"{n_flat} canal(aux) plat(s) (jaune ⚠️)")
            msg = QMessageBox(self)
            msg.setWindowTitle("Qualité du signal insuffisante")
            msg.setIcon(QMessageBox.Warning)
            msg.setText(
                "Des problèmes de signal ont été détectés :\n"
                + "\n".join(f"  • {p}" for p in problems)
                + "\n\nCauses possibles :\n"
                "  • Électrode de référence SRB2 (earlobe) déconnectée\n"
                "  • Électrode(s) mal posée(s) ou gel insuffisant\n\n"
                "Continuer quand même ?"
            )
            msg.setStandardButtons(QMessageBox.Yes | QMessageBox.Cancel)
            msg.setDefaultButton(QMessageBox.Cancel)
            if msg.exec_() != QMessageBox.Yes:
                return

        preset = ACQUISITION_PROTOCOLS[self._preset_combo.currentIndex()]
        self._cue_display_map = dict(preset.cue_display_map)

        if len(preset.classes) == 1:
            classes = list(preset.classes)
        else:
            classes = []
            if self._class_left_cb.isChecked():
                classes.append(preset.classes[0])
            if self._class_right_cb.isChecked():
                classes.append(preset.classes[1])
            if not classes:
                self._append_log("[ERREUR] Cochez au moins une classe.")
                return

        from src.models.acquisition_config import AcquisitionConfig

        config = AcquisitionConfig(
            serial_port=self._port_combo.currentText(),
            ch_names=[c.currentText() for c in self._ch_combos],
            subject_id=self._current_subject_id(),
            run_label=self._run_label_edit.text().strip() or "R01",
            output_dir=self._output_edit.text().strip() or "data/custom/",
            n_trials_per_class=self._n_trials_spin.value(),
            t_baseline_s=self._t_baseline_spin.value(),
            t_cue_s=self._t_cue_spin.value(),
            t_rest_s=self._t_rest_spin.value(),
            classes=classes,
            annotation_labels=list(preset.annotation_labels),
        )

        total = config.n_trials_per_class * len(config.classes)
        self._progress_bar.setMaximum(total)
        self._progress_bar.setValue(0)

        self._stop_stream()

        self._start_btn.setEnabled(False)
        self._connect_btn.setEnabled(False)
        self._cue_label.setText("✛")
        self._cue_window = _CueFullscreenWindow()
        self._cue_window.showFullScreen()
        self._append_log(
            f"[INFO] Démarrage — {total} essais, sujet {config.subject_id}, run {config.run_label}"
        )

        from src.workers.acquisition_record_worker import AcquisitionRecordWorker

        self._record_worker = AcquisitionRecordWorker(self._board, config)
        self._record_worker.trial_update.connect(self._on_trial_update)
        self._record_worker.phase_update.connect(self._on_phase_update)
        self._record_worker.finished.connect(self._on_record_finished)
        self._record_worker.error.connect(self._on_record_error)
        self._record_worker.start()

    def _stop_record(self) -> None:
        if self._record_worker is not None:
            try:
                self._record_worker.trial_update.disconnect()
                self._record_worker.phase_update.disconnect()
                self._record_worker.finished.disconnect()
                self._record_worker.error.disconnect()
            except (RuntimeError, TypeError):
                pass
            self._record_worker.quit()
            self._record_worker.wait()
            self._record_worker = None

    def _on_trial_update(self, i: int, total: int, cue: str) -> None:
        self._progress_bar.setValue(i)
        display = self._cue_display_map.get(cue, cue)
        self._cue_label.setText(display)
        if self._cue_window is not None:
            self._cue_window.set_text(display)
        self._append_log(f"  Essai {i}/{total} — {display}")

    def _on_phase_update(self, phase: str) -> None:
        if phase in ("baseline", "rest"):
            self._cue_label.setText("✛")
            if self._cue_window is not None:
                self._cue_window.set_text("✛")

    def _on_record_finished(self, signal_data) -> None:
        if self._cue_window is not None:
            self._cue_window.close()
            self._cue_window = None
        self._stop_record()

        # Export to EDF
        from src.services.edf_export_service import EdfExportService

        # Rebuild config for export path
        from src.models.acquisition_config import AcquisitionConfig

        config = AcquisitionConfig(
            serial_port="",
            ch_names=signal_data.ch_names,
            subject_id=self._current_subject_id(),
            run_label=self._run_label_edit.text().strip() or "R01",
            output_dir=self._output_edit.text().strip() or "data/custom/",
        )

        try:
            path = EdfExportService.export(signal_data, config)
            self._append_log(f"[OK] EDF exporté → {path}")
            self._auto_fill_run_label()
        except Exception as exc:
            self._append_log(f"[ERREUR] Export EDF : {exc}")

        self._cue_label.setText("✛")
        self._progress_bar.setValue(self._progress_bar.maximum())
        self._start_btn.setEnabled(True)
        self._connect_btn.setEnabled(True)

        # Restart stream for live preview
        self._start_stream()

    def _on_record_error(self, msg: str) -> None:
        if self._cue_window is not None:
            self._cue_window.close()
            self._cue_window = None
        self._stop_record()
        self._append_log(f"[ERREUR enregistrement] {msg}")
        self._cue_label.setText("✛")
        self._start_btn.setEnabled(True)
        self._connect_btn.setEnabled(True)
        self._start_stream()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _current_subject_id(self) -> str:
        return f"{self._name_combo.currentText()}{self._number_spin.value()}"

    def _auto_fill_subject_number(self) -> None:
        """Scan output_dir to find the next available number for the chosen name."""
        name = self._name_combo.currentText()
        output_dir = self._output_edit.text().strip() or "data/custom/"
        pattern = re.compile(rf"^{re.escape(name)}(\d+)$", re.IGNORECASE)
        max_n = 0
        if os.path.isdir(output_dir):
            for entry in os.listdir(output_dir):
                m = pattern.match(entry)
                if m and os.path.isdir(os.path.join(output_dir, entry)):
                    max_n = max(max_n, int(m.group(1)))
        # Block signals to avoid triggering _auto_fill_run_label twice
        self._number_spin.blockSignals(True)
        self._number_spin.setValue(max_n + 1)
        self._number_spin.blockSignals(False)
        self._auto_fill_run_label()

    def _auto_fill_run_label(self) -> None:
        """Propose the next available R{NN} for the current subject in output_dir."""
        subject_id = self._current_subject_id()
        output_dir = self._output_edit.text().strip() or "data/custom/"
        subject_dir = os.path.join(output_dir, subject_id)
        pattern = re.compile(rf"^{re.escape(subject_id)}R(\d{{2}})\.edf$", re.IGNORECASE)
        max_r = 0
        if os.path.isdir(subject_dir):
            for fname in os.listdir(subject_dir):
                m = pattern.match(fname)
                if m:
                    max_r = max(max_r, int(m.group(1)))
        self._run_label_edit.setText(f"R{max_r + 1:02d}")

    def _on_preset_changed(self, index: int) -> None:
        preset = ACQUISITION_PROTOCOLS[index]
        is_custom = (index == 0)

        self._cue_display_map = dict(preset.cue_display_map)

        if not is_custom:
            self._n_trials_spin.setValue(preset.n_trials_per_class)
            self._t_baseline_spin.setValue(preset.t_baseline_s)
            self._t_cue_spin.setValue(preset.t_cue_s)
            self._t_rest_spin.setValue(preset.t_rest_s)
            self._auto_fill_run_label()

        for spin in (self._n_trials_spin, self._t_baseline_spin, self._t_cue_spin, self._t_rest_spin):
            spin.setEnabled(is_custom)

        # Update class checkboxes
        self._class_left_cb.setText(preset.class_labels[0])
        self._class_left_cb.setChecked(True)
        self._class_left_cb.setEnabled(is_custom)

        if len(preset.classes) >= 2:
            self._class_right_cb.setText(preset.class_labels[1])
            self._class_right_cb.setChecked(True)
            self._class_right_cb.setVisible(True)
            self._class_right_cb.setEnabled(is_custom)
        else:
            self._class_right_cb.setChecked(False)
            self._class_right_cb.setVisible(False)

    def _browse_output_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Choisir le dossier de sortie")
        if path:
            self._output_edit.setText(path)

    def _append_log(self, text: str) -> None:
        self._log.append(text)
        self._log.verticalScrollBar().setValue(self._log.verticalScrollBar().maximum())
