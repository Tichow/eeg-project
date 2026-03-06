import numpy as np
import pyqtgraph as pg

from PyQt5.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QListWidget,
    QListWidgetItem, QGroupBox, QProgressBar, QSplitter, QSizePolicy,
    QStackedWidget, QWidget, QCheckBox, QDoubleSpinBox, QComboBox,
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

from src.views.base_view import BaseView
from src.constants.eeg_constants import RUN_DESCRIPTIONS
from src.models.preprocess_config import PreprocessConfig

pg.setConfigOptions(antialias=True)

_ANNOTATION_COLORS = {
    "T0": (150, 150, 150, 128),
    "T1": (70, 130, 180, 128),
    "T2": (255, 99, 71, 128),
}
_DEFAULT_N_CHANNELS = 10


class SignalView(BaseView):
    def setup_ui(self):
        self._worker = None
        self._preprocess_worker = None
        self._signal_data = None
        self._raw_signal = None
        self._processed_signal = None
        self._show_processed = False
        self._loaded_path: str | None = None
        self._pending_path: str | None = None
        self._pending_subject: int = 0
        self._pending_run: int = 0
        self._batch_update = False

        root = QVBoxLayout(self)
        root.setContentsMargins(40, 30, 40, 20)
        root.setSpacing(16)

        # Header
        header = QHBoxLayout()
        back_btn = QPushButton("← Retour")
        back_btn.setFixedWidth(100)
        back_btn.setCursor(Qt.PointingHandCursor)
        back_btn.setStyleSheet("border: none; color: #4a90d9; font-size: 13px;")
        back_btn.clicked.connect(lambda: self.navigate.emit("browser"))
        header.addWidget(back_btn)
        header.addStretch()

        self._title = QLabel("Signal")
        font = QFont()
        font.setPointSize(16)
        font.setBold(True)
        self._title.setFont(font)
        header.addWidget(self._title)
        header.addStretch()
        root.addLayout(header)

        # Main splitter
        splitter = QSplitter(Qt.Horizontal)
        splitter.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Left panel — preprocessing + channel selector
        left = QWidget()
        left.setFixedWidth(185)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(6)

        left_layout.addWidget(self._build_preprocess_panel())

        channels_box = QGroupBox("Canaux")
        channels_layout = QVBoxLayout(channels_box)
        self._channel_list = QListWidget()
        self._channel_list.itemChanged.connect(self._on_channel_changed)
        channels_layout.addWidget(self._channel_list)

        btn_row = QHBoxLayout()
        all_btn = QPushButton("Tous")
        all_btn.setFixedHeight(26)
        all_btn.clicked.connect(self._select_all)
        none_btn = QPushButton("Aucun")
        none_btn.setFixedHeight(26)
        none_btn.clicked.connect(self._select_none)
        btn_row.addWidget(all_btn)
        btn_row.addWidget(none_btn)
        channels_layout.addLayout(btn_row)
        left_layout.addWidget(channels_box)
        splitter.addWidget(left)

        # Right panel — loading state + plot
        right_widget = QGroupBox()
        right_widget.setFlat(True)
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)

        self._stack = QStackedWidget()

        # Loading page
        loading_widget = QWidget()
        loading_page = QVBoxLayout(loading_widget)
        self._loading_label = QLabel("Chargement…")
        self._loading_label.setAlignment(Qt.AlignCenter)
        self._loading_label.setStyleSheet("color: #888; font-size: 14px;")
        self._loading_progress = QProgressBar()
        self._loading_progress.setRange(0, 0)
        self._loading_progress.setFixedHeight(10)
        loading_page.addStretch()
        loading_page.addWidget(self._loading_label)
        loading_page.addWidget(self._loading_progress)
        loading_page.addStretch()
        self._stack.addWidget(loading_widget)

        # Plot page
        self._plot_widget = pg.PlotWidget()
        self._plot_widget.setBackground("w")
        self._plot_widget.showGrid(x=True, y=False, alpha=0.3)
        self._plot_widget.getAxis("bottom").setLabel("Temps (s)")
        tick_font = QFont()
        tick_font.setPointSize(7)
        tick_font.setFixedPitch(True)
        self._plot_widget.getAxis("left").setStyle(tickFont=tick_font)
        self._stack.addWidget(self._plot_widget)

        right_layout.addWidget(self._stack)
        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        root.addWidget(splitter)

        self._status = QLabel("")
        self._status.setStyleSheet("color: #888; font-size: 11px;")
        root.addWidget(self._status)

    # ── Public API ────────────────────────────────────────────────────────────

    def prepare(self, path: str, subject: int, run: int):
        """Called by BrowserView before navigate.emit('signal')."""
        self._pending_path = path
        self._pending_subject = subject
        self._pending_run = run
        desc = RUN_DESCRIPTIONS.get(run, f"Run {run:02d}")
        self._title.setText(f"Signal — S{subject:03d} / R{run:02d} — {desc}")

    # ── Navigation lifecycle ──────────────────────────────────────────────────

    def on_navigate_to(self):
        if self._pending_path is None:
            return
        if self._pending_path == self._loaded_path:
            self._stack.setCurrentIndex(1)
            return
        self._start_loading()

    def _start_loading(self):
        from src.workers.signal_worker import SignalWorker
        self._stop_worker()
        self._stop_preprocess_worker()
        self._signal_data = None
        self._raw_signal = None
        self._processed_signal = None
        self._show_processed = False
        self._apply_btn.setEnabled(False)
        self._toggle_btn.setEnabled(False)
        self._toggle_btn.setText("→ Traité")
        self._channel_list.clear()
        self._stack.setCurrentIndex(0)
        self._status.setText("")

        self._worker = SignalWorker(self._pending_path)
        self._worker.data_ready.connect(self._on_data_ready)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    # ── Worker callbacks ──────────────────────────────────────────────────────

    def _on_data_ready(self, signal_data):
        self._raw_signal = signal_data
        self._processed_signal = None
        self._show_processed = False
        self._signal_data = signal_data
        self._loaded_path = self._pending_path
        self._worker = None

        # Populate reref combo with channel names
        self._reref_combo.clear()
        self._reref_combo.addItem("Moyenne")
        for ch in signal_data.ch_names:
            self._reref_combo.addItem(ch)

        self._apply_btn.setEnabled(True)
        self._toggle_btn.setEnabled(False)
        self._toggle_btn.setText("→ Traité")

        self._populate_channel_list()
        self._replot()
        self._stack.setCurrentIndex(1)
        n_ch = len(signal_data.ch_names)
        dur = signal_data.times[-1] if len(signal_data.times) else 0
        self._status.setText(
            f"{n_ch} canaux · {dur:.1f} s · {signal_data.sfreq:.0f} Hz · "
            f"{len(signal_data.annotations)} annotations"
        )

    def _on_error(self, message: str):
        self._loading_label.setText(f"Erreur : {message}")
        self._loading_progress.setVisible(False)
        self._worker = None

    # ── Channel list ──────────────────────────────────────────────────────────

    def _populate_channel_list(self):
        self._batch_update = True
        self._channel_list.clear()
        for i, name in enumerate(self._signal_data.ch_names):
            item = QListWidgetItem(name)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if i < _DEFAULT_N_CHANNELS else Qt.Unchecked)
            self._channel_list.addItem(item)
        self._batch_update = False

    def _on_channel_changed(self, item):
        if self._batch_update:
            return
        self._replot()

    def _select_all(self):
        self._batch_update = True
        for i in range(self._channel_list.count()):
            self._channel_list.item(i).setCheckState(Qt.Checked)
        self._batch_update = False
        self._replot()

    def _select_none(self):
        self._batch_update = True
        for i in range(self._channel_list.count()):
            self._channel_list.item(i).setCheckState(Qt.Unchecked)
        self._batch_update = False
        self._replot()

    # ── Plot ──────────────────────────────────────────────────────────────────

    def _replot(self):
        self._plot_widget.clear()

        if self._signal_data is None:
            return

        selected = []
        for i in range(self._channel_list.count()):
            item = self._channel_list.item(i)
            if item.checkState() == Qt.Checked:
                selected.append((i, item.text()))

        if not selected:
            text = pg.TextItem("Aucun canal sélectionné", color=(136, 136, 136), anchor=(0.5, 0.5))
            self._plot_widget.addItem(text)
            text.setPos(0.5, 0.5)
            return

        data = self._signal_data.data
        times = self._signal_data.times

        # Auto-scale: median peak-to-peak across selected channels
        ptp_values = [np.ptp(data[idx]) for idx, _ in selected]
        scale = float(np.median(ptp_values)) * 3 if ptp_values else 1e-4
        if scale == 0:
            scale = 1e-4

        pen = pg.mkPen("#2c7bb6", width=0.8)
        for i, (ch_idx, _ch_name) in enumerate(selected):
            self._plot_widget.plot(times, data[ch_idx] + i * scale, pen=pen)

        # Y-axis labels
        ticks = [(i * scale, name) for i, (_, name) in enumerate(selected)]
        self._plot_widget.getAxis("left").setTicks([ticks])

        # Annotations
        for onset, _dur, desc in self._signal_data.annotations:
            color = _ANNOTATION_COLORS.get(desc, (0, 0, 0, 128))
            line = pg.InfiniteLine(pos=onset, angle=90, pen=pg.mkPen(color, width=0.8))
            self._plot_widget.addItem(line)

        self._plot_widget.setXRange(times[0], times[-1], padding=0)
        self._plot_widget.setYRange(-scale, len(selected) * scale, padding=0.02)

    # ── Preprocessing panel ───────────────────────────────────────────────────

    def _build_preprocess_panel(self) -> QGroupBox:
        box = QGroupBox("Prétraitement")
        layout = QVBoxLayout(box)
        layout.setSpacing(4)

        # Bandpass
        self._bp_check = QCheckBox("Passe-bande")
        layout.addWidget(self._bp_check)

        bp_row = QHBoxLayout()
        bp_row.setContentsMargins(16, 0, 0, 0)
        self._bp_low = QDoubleSpinBox()
        self._bp_low.setRange(0.1, 500.0)
        self._bp_low.setValue(8.0)
        self._bp_low.setSuffix(" Hz")
        self._bp_low.setDecimals(1)
        self._bp_low.setEnabled(False)
        self._bp_high = QDoubleSpinBox()
        self._bp_high.setRange(0.1, 500.0)
        self._bp_high.setValue(30.0)
        self._bp_high.setSuffix(" Hz")
        self._bp_high.setDecimals(1)
        self._bp_high.setEnabled(False)
        bp_row.addWidget(self._bp_low)
        bp_row.addWidget(self._bp_high)
        layout.addLayout(bp_row)
        self._bp_check.toggled.connect(self._bp_low.setEnabled)
        self._bp_check.toggled.connect(self._bp_high.setEnabled)

        # Notch
        self._notch_check = QCheckBox("Notch")
        layout.addWidget(self._notch_check)

        notch_row = QHBoxLayout()
        notch_row.setContentsMargins(16, 0, 0, 0)
        self._notch_hz = QDoubleSpinBox()
        self._notch_hz.setRange(1.0, 500.0)
        self._notch_hz.setValue(50.0)
        self._notch_hz.setSuffix(" Hz")
        self._notch_hz.setDecimals(1)
        self._notch_hz.setEnabled(False)
        notch_row.addWidget(self._notch_hz)
        notch_row.addStretch()
        layout.addLayout(notch_row)
        self._notch_check.toggled.connect(self._notch_hz.setEnabled)

        # Rereferencing
        self._reref_check = QCheckBox("Référence")
        layout.addWidget(self._reref_check)

        reref_row = QHBoxLayout()
        reref_row.setContentsMargins(16, 0, 0, 0)
        self._reref_combo = QComboBox()
        self._reref_combo.addItem("Moyenne")
        self._reref_combo.setEnabled(False)
        reref_row.addWidget(self._reref_combo)
        layout.addLayout(reref_row)
        self._reref_check.toggled.connect(self._reref_combo.setEnabled)

        # Action buttons
        action_row = QHBoxLayout()
        self._apply_btn = QPushButton("Appliquer")
        self._apply_btn.setFixedHeight(26)
        self._apply_btn.setEnabled(False)
        self._apply_btn.clicked.connect(self._on_apply_preprocessing)
        self._toggle_btn = QPushButton("→ Traité")
        self._toggle_btn.setFixedHeight(26)
        self._toggle_btn.setEnabled(False)
        self._toggle_btn.clicked.connect(self._toggle_view_mode)
        action_row.addWidget(self._apply_btn)
        action_row.addWidget(self._toggle_btn)
        layout.addLayout(action_row)

        return box

    def _read_preprocess_config(self) -> PreprocessConfig:
        reref_text = self._reref_combo.currentText()
        reref_mode = "average" if reref_text == "Moyenne" else reref_text
        return PreprocessConfig(
            bandpass_enabled=self._bp_check.isChecked(),
            low_hz=self._bp_low.value(),
            high_hz=self._bp_high.value(),
            notch_enabled=self._notch_check.isChecked(),
            notch_hz=self._notch_hz.value(),
            reref_enabled=self._reref_check.isChecked(),
            reref_mode=reref_mode,
        )

    def _on_apply_preprocessing(self):
        if self._raw_signal is None:
            return
        from src.workers.preprocess_worker import PreprocessWorker
        config = self._read_preprocess_config()
        self._apply_btn.setEnabled(False)
        self._apply_btn.setText("Traitement…")
        self._stop_preprocess_worker()
        self._preprocess_worker = PreprocessWorker(self._raw_signal, config)
        self._preprocess_worker.result_ready.connect(self._on_preprocess_ready)
        self._preprocess_worker.error.connect(self._on_preprocess_error)
        self._preprocess_worker.start()

    def _on_preprocess_ready(self, signal_data):
        self._processed_signal = signal_data
        self._show_processed = True
        self._signal_data = signal_data
        self._preprocess_worker = None
        self._apply_btn.setEnabled(True)
        self._apply_btn.setText("Appliquer")
        self._toggle_btn.setEnabled(True)
        self._toggle_btn.setText("→ Brut")

        config = self._read_preprocess_config()
        parts = []
        if config.bandpass_enabled:
            parts.append(f"BP {config.low_hz:.0f}–{config.high_hz:.0f} Hz")
        if config.notch_enabled:
            parts.append(f"Notch {config.notch_hz:.0f} Hz")
        if config.reref_enabled:
            parts.append(f"Réf. {self._reref_combo.currentText()}")
        indicator = f"  [{', '.join(parts)}]" if parts else ""

        n_ch = len(self._raw_signal.ch_names)
        dur = self._raw_signal.times[-1] if len(self._raw_signal.times) else 0
        self._status.setText(
            f"{n_ch} canaux · {dur:.1f} s · {self._raw_signal.sfreq:.0f} Hz · "
            f"{len(self._raw_signal.annotations)} annotations{indicator}"
        )
        self._replot()

    def _on_preprocess_error(self, message: str):
        self._apply_btn.setEnabled(True)
        self._apply_btn.setText("Appliquer")
        self._preprocess_worker = None
        self._status.setText(f"Erreur prétraitement : {message}")

    def _toggle_view_mode(self):
        if self._processed_signal is None:
            return
        self._show_processed = not self._show_processed
        if self._show_processed:
            self._signal_data = self._processed_signal
            self._toggle_btn.setText("→ Brut")
        else:
            self._signal_data = self._raw_signal
            self._toggle_btn.setText("→ Traité")
        self._replot()

    # ── Worker cleanup ────────────────────────────────────────────────────────

    def _stop_worker(self):
        if self._worker is not None:
            try:
                self._worker.data_ready.disconnect()
                self._worker.error.disconnect()
            except (RuntimeError, TypeError):
                pass
            self._worker.quit()
            self._worker = None

    def _stop_preprocess_worker(self):
        if self._preprocess_worker is not None:
            try:
                self._preprocess_worker.result_ready.disconnect()
                self._preprocess_worker.error.disconnect()
            except (RuntimeError, TypeError):
                pass
            self._preprocess_worker.quit()
            self._preprocess_worker = None
