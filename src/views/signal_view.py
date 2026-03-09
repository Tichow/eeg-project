import numpy as np
import pyqtgraph as pg

from PyQt5.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QListWidget,
    QListWidgetItem, QGroupBox, QProgressBar, QSplitter, QSizePolicy,
    QStackedWidget, QWidget, QCheckBox, QDoubleSpinBox, QComboBox, QSpinBox,
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

from src.views.base_view import BaseView
from src.constants.eeg_constants import RUN_DESCRIPTIONS
from src.models.preprocess_config import PreprocessConfig
from src.workers.ica_worker import ICAWorker

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
        self._epoch_worker = None
        self._ica_worker = None
        self._frequency_worker = None
        self._erp_worker = None
        self._signal_data = None
        self._raw_signal = None
        self._processed_signal = None
        self._epoch_data = None
        self._epoch_data_base = None
        self._bad_epoch_indices: list[int] = []
        self._ica = None
        self._frequency_data = None
        self._erp_data = None
        self._freq_box = None
        self._erp_box = None
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
        left_layout.addWidget(self._build_epoch_panel())
        left_layout.addWidget(self._build_artifact_panel())
        left_layout.addWidget(self._build_frequency_panel())
        left_layout.addWidget(self._build_erp_panel())

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

        # Epoch plot page (index 2)
        self._epoch_plot_layout = pg.GraphicsLayoutWidget()
        self._epoch_plot_layout.setBackground("w")
        self._stack.addWidget(self._epoch_plot_layout)

        # Frequency plot page (index 3)
        self._freq_plot_widget = pg.PlotWidget()
        self._freq_plot_widget.setBackground("w")
        self._freq_plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self._stack.addWidget(self._freq_plot_widget)

        # ERP plot page (index 4)
        self._erp_plot_layout = pg.GraphicsLayoutWidget()
        self._erp_plot_layout.setBackground("w")
        self._stack.addWidget(self._erp_plot_layout)

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
        self._stop_epoch_worker()
        self._stop_ica_worker()
        self._stop_frequency_worker()
        self._stop_erp_worker()
        self._signal_data = None
        self._raw_signal = None
        self._processed_signal = None
        self._epoch_data = None
        self._epoch_data_base = None
        self._bad_epoch_indices = []
        self._ica = None
        self._frequency_data = None
        self._erp_data = None
        self._show_processed = False
        self._apply_btn.setEnabled(False)
        self._toggle_btn.setEnabled(False)
        self._toggle_btn.setText("→ Traité")
        self._epoch_extract_btn.setEnabled(False)
        self._epoch_back_btn.setVisible(False)
        self._artifact_box.setVisible(False)
        if self._freq_box is not None:
            self._freq_box.setVisible(False)
        if self._erp_box is not None:
            self._erp_box.setVisible(False)
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
        self._epoch_extract_btn.setEnabled(True)

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

    def _build_epoch_panel(self) -> QGroupBox:
        box = QGroupBox("Épocage")
        layout = QVBoxLayout(box)
        layout.setSpacing(4)

        # tmin / tmax
        time_row = QHBoxLayout()
        time_row.setContentsMargins(0, 0, 0, 0)
        self._tmin_spin = QDoubleSpinBox()
        self._tmin_spin.setRange(-10.0, 0.0)
        self._tmin_spin.setValue(-0.5)
        self._tmin_spin.setSuffix(" s")
        self._tmin_spin.setDecimals(1)
        self._tmin_spin.setSingleStep(0.1)
        self._tmax_spin = QDoubleSpinBox()
        self._tmax_spin.setRange(0.0, 10.0)
        self._tmax_spin.setValue(1.5)
        self._tmax_spin.setSuffix(" s")
        self._tmax_spin.setDecimals(1)
        self._tmax_spin.setSingleStep(0.1)
        time_row.addWidget(self._tmin_spin)
        time_row.addWidget(self._tmax_spin)
        layout.addLayout(time_row)

        btn_row = QHBoxLayout()
        self._epoch_extract_btn = QPushButton("Extraire")
        self._epoch_extract_btn.setFixedHeight(26)
        self._epoch_extract_btn.setEnabled(False)
        self._epoch_extract_btn.clicked.connect(self._on_extract_epochs)
        self._epoch_back_btn = QPushButton("← Signal")
        self._epoch_back_btn.setFixedHeight(26)
        self._epoch_back_btn.setVisible(False)
        self._epoch_back_btn.clicked.connect(self._on_back_to_signal)
        btn_row.addWidget(self._epoch_extract_btn)
        btn_row.addWidget(self._epoch_back_btn)
        layout.addLayout(btn_row)

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

    def _stop_epoch_worker(self):
        if self._epoch_worker is not None:
            try:
                self._epoch_worker.result_ready.disconnect()
                self._epoch_worker.error.disconnect()
            except (RuntimeError, TypeError):
                pass
            self._epoch_worker.quit()
            self._epoch_worker = None

    def _stop_ica_worker(self):
        if self._ica_worker is not None:
            try:
                self._ica_worker.result_ready.disconnect()
                self._ica_worker.error.disconnect()
            except (RuntimeError, TypeError):
                pass
            self._ica_worker.quit()
            self._ica_worker.wait()
            self._ica_worker = None

    def _stop_frequency_worker(self):
        if self._frequency_worker is not None:
            try:
                self._frequency_worker.result_ready.disconnect()
                self._frequency_worker.error.disconnect()
            except (RuntimeError, TypeError):
                pass
            self._frequency_worker.quit()
            self._frequency_worker.wait()
            self._frequency_worker = None

    # ── Epoching ──────────────────────────────────────────────────────────────

    def _on_extract_epochs(self):
        if self._signal_data is None:
            return
        tmin = self._tmin_spin.value()
        tmax = self._tmax_spin.value()
        if tmin >= tmax:
            self._status.setText("Erreur épocage : tmin doit être < tmax")
            return
        from src.workers.epoch_worker import EpochWorker
        self._epoch_extract_btn.setEnabled(False)
        self._epoch_extract_btn.setText("Extraction…")
        self._stop_epoch_worker()
        self._epoch_worker = EpochWorker(self._signal_data, tmin, tmax)
        self._epoch_worker.result_ready.connect(self._on_epoch_ready)
        self._epoch_worker.error.connect(self._on_epoch_error)
        self._epoch_worker.start()

    def _on_epoch_ready(self, epoch_data):
        self._epoch_data = epoch_data
        self._epoch_data_base = epoch_data
        self._bad_epoch_indices = []
        self._ica = None
        self._epoch_worker.wait()
        self._epoch_worker = None
        self._epoch_extract_btn.setEnabled(True)
        self._epoch_extract_btn.setText("Extraire")
        self._epoch_back_btn.setVisible(True)
        # Reset artefacts panel
        n_epochs = epoch_data.data.shape[0]
        self._threshold_status.setText(f"0 / {n_epochs} rejetées")
        self._apply_threshold_btn.setEnabled(False)
        self._ica_comp_list.setVisible(False)
        self._apply_ica_btn.setVisible(False)
        self._artifact_box.setVisible(True)
        # Frequency panel
        self._frequency_data = None
        self._erders_baseline_combo.clear()
        for lbl in sorted(set(epoch_data.labels)):
            self._erders_baseline_combo.addItem(lbl)
        if "T0" in set(epoch_data.labels):
            self._erders_baseline_combo.setCurrentText("T0")
        self._freq_box.setVisible(True)
        self._erp_box.setVisible(True)
        self._erp_back_btn.setVisible(False)
        self._replot_epochs()
        self._stack.setCurrentIndex(2)
        n_classes = len(set(epoch_data.labels))
        tmin = epoch_data.times[0]
        tmax = epoch_data.times[-1]
        self._status.setText(
            f"{n_epochs} époques · {n_classes} classe{'s' if n_classes > 1 else ''} · "
            f"{tmin:.2f}–{tmax:.2f} s"
        )

    def _on_epoch_error(self, message: str):
        self._epoch_worker.wait()
        self._epoch_worker = None
        self._epoch_extract_btn.setEnabled(True)
        self._epoch_extract_btn.setText("Extraire")
        self._status.setText(f"Erreur épocage : {message}")

    def _on_back_to_signal(self):
        self._stack.setCurrentIndex(1)
        self._epoch_back_btn.setVisible(False)
        n_ch = len(self._signal_data.ch_names)
        dur = self._signal_data.times[-1] if len(self._signal_data.times) else 0
        self._status.setText(
            f"{n_ch} canaux · {dur:.1f} s · {self._signal_data.sfreq:.0f} Hz · "
            f"{len(self._signal_data.annotations)} annotations"
        )

    # ── Artefacts panel ───────────────────────────────────────────────────────

    def _build_artifact_panel(self) -> QGroupBox:
        box = QGroupBox("Artefacts")
        box.setVisible(False)
        self._artifact_box = box
        layout = QVBoxLayout(box)
        layout.setSpacing(4)

        # ── Threshold section ─────────────────────────────────
        thresh_label = QLabel("Seuil pic-à-pic")
        thresh_label.setStyleSheet("font-weight: bold; font-size: 11px;")
        layout.addWidget(thresh_label)

        thresh_row = QHBoxLayout()
        self._threshold_spin = QDoubleSpinBox()
        self._threshold_spin.setRange(1.0, 5000.0)
        self._threshold_spin.setValue(100.0)
        self._threshold_spin.setSuffix(" µV")
        self._threshold_spin.setDecimals(0)
        thresh_row.addWidget(self._threshold_spin)
        layout.addLayout(thresh_row)

        thresh_btn_row = QHBoxLayout()
        self._detect_btn = QPushButton("Détecter")
        self._detect_btn.setFixedHeight(24)
        self._detect_btn.clicked.connect(self._on_detect_threshold)
        self._apply_threshold_btn = QPushButton("Appliquer")
        self._apply_threshold_btn.setFixedHeight(24)
        self._apply_threshold_btn.setEnabled(False)
        self._apply_threshold_btn.clicked.connect(self._on_apply_threshold)
        thresh_btn_row.addWidget(self._detect_btn)
        thresh_btn_row.addWidget(self._apply_threshold_btn)
        layout.addLayout(thresh_btn_row)

        self._threshold_status = QLabel("0 / 0 rejetées")
        self._threshold_status.setStyleSheet("color: #888; font-size: 10px;")
        layout.addWidget(self._threshold_status)

        # ── ICA section ───────────────────────────────────────
        ica_label = QLabel("ICA")
        ica_label.setStyleSheet("font-weight: bold; font-size: 11px; margin-top: 4px;")
        layout.addWidget(ica_label)

        ica_n_row = QHBoxLayout()
        ica_n_lbl = QLabel("Comp. :")
        self._ica_n_spin = QSpinBox()
        self._ica_n_spin.setRange(1, 64)
        self._ica_n_spin.setValue(20)
        ica_n_row.addWidget(ica_n_lbl)
        ica_n_row.addWidget(self._ica_n_spin)
        layout.addLayout(ica_n_row)

        self._fit_ica_btn = QPushButton("Ajuster ICA")
        self._fit_ica_btn.setFixedHeight(24)
        self._fit_ica_btn.clicked.connect(self._on_fit_ica)
        layout.addWidget(self._fit_ica_btn)

        self._ica_comp_list = QListWidget()
        self._ica_comp_list.setFixedHeight(80)
        self._ica_comp_list.setVisible(False)
        layout.addWidget(self._ica_comp_list)

        self._apply_ica_btn = QPushButton("Appliquer ICA")
        self._apply_ica_btn.setFixedHeight(24)
        self._apply_ica_btn.setVisible(False)
        self._apply_ica_btn.clicked.connect(self._on_apply_ica)
        layout.addWidget(self._apply_ica_btn)

        return box

    def _on_detect_threshold(self):
        if self._epoch_data is None:
            return
        from src.services.eeg_artifact_service import EEGArtifactService
        self._bad_epoch_indices = EEGArtifactService.detect_by_threshold(
            self._epoch_data, self._threshold_spin.value()
        )
        n_bad = len(self._bad_epoch_indices)
        n_total = self._epoch_data.data.shape[0]
        self._threshold_status.setText(f"{n_bad} / {n_total} rejetées")
        self._apply_threshold_btn.setEnabled(n_bad > 0)
        self._replot_epochs()

    def _on_apply_threshold(self):
        if self._epoch_data is None or not self._bad_epoch_indices:
            return
        from src.services.eeg_artifact_service import EEGArtifactService
        self._epoch_data = EEGArtifactService.apply_threshold_rejection(
            self._epoch_data, self._bad_epoch_indices
        )
        self._bad_epoch_indices = []
        self._apply_threshold_btn.setEnabled(False)
        n = self._epoch_data.data.shape[0]
        self._threshold_status.setText(f"0 / {n} rejetées")
        self._replot_epochs()
        n_classes = len(set(self._epoch_data.labels))
        tmin = self._epoch_data.times[0]
        tmax = self._epoch_data.times[-1]
        self._status.setText(
            f"{n} époques · {n_classes} classe{'s' if n_classes > 1 else ''} · "
            f"{tmin:.2f}–{tmax:.2f} s"
        )

    def _on_fit_ica(self):
        if self._epoch_data is None:
            return
        self._stop_ica_worker()
        self._fit_ica_btn.setEnabled(False)
        self._fit_ica_btn.setText("Ajustement…")
        self._ica_comp_list.setVisible(False)
        self._apply_ica_btn.setVisible(False)
        self._ica_worker = ICAWorker(self._epoch_data, self._ica_n_spin.value())
        self._ica_worker.result_ready.connect(self._on_ica_ready)
        self._ica_worker.error.connect(self._on_ica_error)
        self._ica_worker.start()

    def _on_ica_ready(self, ica, bad_components):
        self._ica = ica
        self._ica_worker.wait()
        self._ica_worker = None
        self._fit_ica_btn.setEnabled(True)
        self._fit_ica_btn.setText("Ajuster ICA")
        # Populate component list
        self._ica_comp_list.clear()
        n_comp = ica.n_components_
        for i in range(n_comp):
            item = QListWidgetItem(f"Comp. {i}")
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if i in bad_components else Qt.Unchecked)
            self._ica_comp_list.addItem(item)
        self._ica_comp_list.setVisible(True)
        self._apply_ica_btn.setVisible(True)
        n_auto = len(bad_components)
        self._status.setText(
            f"ICA ajustée — {n_comp} composantes — {n_auto} détectée{'s' if n_auto > 1 else ''} auto."
        )

    def _on_ica_error(self, message: str):
        self._ica_worker = None
        self._fit_ica_btn.setEnabled(True)
        self._fit_ica_btn.setText("Ajuster ICA")
        self._status.setText(f"Erreur ICA : {message}")

    def _on_apply_ica(self):
        if self._epoch_data is None or self._ica is None:
            return
        from src.services.eeg_artifact_service import EEGArtifactService
        exclude = [
            i for i in range(self._ica_comp_list.count())
            if self._ica_comp_list.item(i).checkState() == Qt.Checked
        ]
        self._epoch_data = EEGArtifactService.apply_ica(self._epoch_data, self._ica, exclude)
        self._ica = None
        self._ica_comp_list.setVisible(False)
        self._apply_ica_btn.setVisible(False)
        self._replot_epochs()
        n = self._epoch_data.data.shape[0]
        n_classes = len(set(self._epoch_data.labels))
        tmin = self._epoch_data.times[0]
        tmax = self._epoch_data.times[-1]
        self._status.setText(
            f"{n} époques · {n_classes} classe{'s' if n_classes > 1 else ''} · "
            f"{tmin:.2f}–{tmax:.2f} s · ICA appliquée ({len(exclude)} comp. exclues)"
        )

    # ── Epoch plot ────────────────────────────────────────────────────────────

    def _replot_epochs(self):
        self._epoch_plot_layout.clear()

        if self._epoch_data is None:
            return

        selected = []
        for i in range(self._channel_list.count()):
            item = self._channel_list.item(i)
            if item.checkState() == Qt.Checked:
                selected.append((i, item.text()))
        selected = selected[:6]  # cap at 6 subplots for performance

        if not selected:
            return

        times = self._epoch_data.times
        data = self._epoch_data.data       # (n_epochs, n_channels, n_times)
        labels = self._epoch_data.labels
        bad_set = set(self._bad_epoch_indices)

        for subplot_idx, (ch_idx, ch_name) in enumerate(selected):
            plot = self._epoch_plot_layout.addPlot(row=subplot_idx, col=0)
            plot.setLabel("left", ch_name)
            plot.showGrid(x=True, y=False, alpha=0.3)
            if subplot_idx == len(selected) - 1:
                plot.setLabel("bottom", "Temps (s)")

            for ep_idx, label in enumerate(labels):
                if ep_idx in bad_set:
                    pen = pg.mkPen((220, 50, 50, 160), width=0.8, style=Qt.DashLine)
                else:
                    color = _ANNOTATION_COLORS.get(label, (100, 100, 100, 80))
                    pen = pg.mkPen(color, width=0.6)
                plot.plot(times, data[ep_idx, ch_idx, :], pen=pen)

            # Vertical line at t=0 (event onset)
            plot.addItem(pg.InfiniteLine(
                pos=0.0, angle=90,
                pen=pg.mkPen((0, 0, 0, 60), width=0.8, style=Qt.DashLine),
            ))

    # ── Frequency panel ───────────────────────────────────────────────────────

    def _build_frequency_panel(self) -> QGroupBox:
        box = QGroupBox("Fréquences")
        box.setVisible(False)
        self._freq_box = box
        layout = QVBoxLayout(box)
        layout.setSpacing(4)

        self._freq_mode_combo = QComboBox()
        self._freq_mode_combo.addItems(["PSD", "ERD/ERS"])
        self._freq_mode_combo.currentIndexChanged.connect(self._on_freq_mode_changed)
        layout.addWidget(self._freq_mode_combo)

        # ── PSD controls ──────────────────────────────────────
        self._psd_controls = QWidget()
        psd_layout = QVBoxLayout(self._psd_controls)
        psd_layout.setContentsMargins(0, 0, 0, 0)
        psd_layout.setSpacing(3)

        win_row = QHBoxLayout()
        win_row.addWidget(QLabel("Fenêtre (s)"))
        self._psd_window_spin = QDoubleSpinBox()
        self._psd_window_spin.setRange(0.5, 10.0)
        self._psd_window_spin.setValue(2.0)
        self._psd_window_spin.setSingleStep(0.5)
        self._psd_window_spin.setDecimals(1)
        win_row.addWidget(self._psd_window_spin)
        psd_layout.addLayout(win_row)

        freq_row = QHBoxLayout()
        self._psd_fmin = QDoubleSpinBox()
        self._psd_fmin.setRange(0.5, 200.0)
        self._psd_fmin.setValue(1.0)
        self._psd_fmin.setSuffix(" Hz")
        self._psd_fmin.setDecimals(1)
        self._psd_fmax = QDoubleSpinBox()
        self._psd_fmax.setRange(1.0, 200.0)
        self._psd_fmax.setValue(40.0)
        self._psd_fmax.setSuffix(" Hz")
        self._psd_fmax.setDecimals(1)
        freq_row.addWidget(self._psd_fmin)
        freq_row.addWidget(self._psd_fmax)
        psd_layout.addLayout(freq_row)
        layout.addWidget(self._psd_controls)

        # ── ERD/ERS controls ──────────────────────────────────
        self._erders_controls = QWidget()
        self._erders_controls.setVisible(False)
        erd_layout = QVBoxLayout(self._erders_controls)
        erd_layout.setContentsMargins(0, 0, 0, 0)
        erd_layout.setSpacing(3)

        band_row = QHBoxLayout()
        self._erders_low = QDoubleSpinBox()
        self._erders_low.setRange(1.0, 200.0)
        self._erders_low.setValue(8.0)
        self._erders_low.setSuffix(" Hz")
        self._erders_low.setDecimals(1)
        self._erders_high = QDoubleSpinBox()
        self._erders_high.setRange(1.0, 200.0)
        self._erders_high.setValue(12.0)
        self._erders_high.setSuffix(" Hz")
        self._erders_high.setDecimals(1)
        band_row.addWidget(self._erders_low)
        band_row.addWidget(self._erders_high)
        erd_layout.addLayout(band_row)

        erd_layout.addWidget(QLabel("Référence"))
        self._erders_baseline_combo = QComboBox()
        erd_layout.addWidget(self._erders_baseline_combo)
        layout.addWidget(self._erders_controls)

        self._freq_calc_btn = QPushButton("Calculer")
        self._freq_calc_btn.setFixedHeight(26)
        self._freq_calc_btn.clicked.connect(self._on_calc_frequency)
        layout.addWidget(self._freq_calc_btn)

        return box

    def _on_freq_mode_changed(self, index: int):
        self._psd_controls.setVisible(index == 0)
        self._erders_controls.setVisible(index == 1)

    def _on_calc_frequency(self):
        if self._epoch_data is None:
            return

        selected_ch = []
        for i in range(self._channel_list.count()):
            item = self._channel_list.item(i)
            if item.checkState() == Qt.Checked:
                selected_ch.append(i)
        if not selected_ch:
            self._status.setText("Sélectionnez au moins un canal")
            return

        from src.workers.frequency_worker import FrequencyWorker

        mode = "psd" if self._freq_mode_combo.currentIndex() == 0 else "erders"
        epoch_len = self._epoch_data.data.shape[2]
        nperseg = min(int(self._psd_window_spin.value() * self._epoch_data.sfreq), epoch_len)

        self._freq_calc_btn.setEnabled(False)
        self._freq_calc_btn.setText("Calcul…")
        self._stop_frequency_worker()

        self._frequency_worker = FrequencyWorker(
            epoch_data=self._epoch_data,
            mode=mode,
            selected_ch_indices=selected_ch,
            fmin=self._psd_fmin.value(),
            fmax=self._psd_fmax.value(),
            nperseg=nperseg,
            band_low=self._erders_low.value(),
            band_high=self._erders_high.value(),
            baseline_label=self._erders_baseline_combo.currentText(),
        )
        self._frequency_worker.result_ready.connect(self._on_frequency_ready)
        self._frequency_worker.error.connect(self._on_frequency_error)
        self._frequency_worker.start()

    def _on_frequency_ready(self, freq_data):
        self._frequency_data = freq_data
        self._frequency_worker.wait()
        self._frequency_worker = None
        self._freq_calc_btn.setEnabled(True)
        self._freq_calc_btn.setText("Calculer")
        self._replot_frequency()
        self._stack.setCurrentIndex(3)
        n_ch = len(freq_data.ch_names)
        mode_lbl = "PSD" if freq_data.mode == "psd" else "ERD/ERS"
        self._status.setText(
            f"{mode_lbl} calculé — {n_ch} canal{'x' if n_ch > 1 else ''}"
        )

    def _on_frequency_error(self, message: str):
        self._frequency_worker.wait()
        self._frequency_worker = None
        self._freq_calc_btn.setEnabled(True)
        self._freq_calc_btn.setText("Calculer")
        self._status.setText(f"Erreur fréquences : {message}")

    def _replot_frequency(self):
        self._freq_plot_widget.clear()
        if self._frequency_data is None:
            return

        fd = self._frequency_data

        _CLASS_COLORS = {
            "T0": (150, 150, 150),
            "T1": (70, 130, 180),
            "T2": (255, 99, 71),
        }
        _FALLBACK = [(100, 180, 100), (180, 100, 180), (180, 160, 60)]

        if fd.mode == "psd":
            self._freq_plot_widget.setLabel("bottom", "Fréquence (Hz)")
            self._freq_plot_widget.setLabel("left", "PSD (µV²/Hz)")
            x = fd.freqs
            data_by_class = fd.psd_by_class
        else:
            self._freq_plot_widget.setLabel("bottom", "Temps (s)")
            self._freq_plot_widget.setLabel("left", "ERD/ERS (%)")
            x = fd.times
            data_by_class = fd.erders_by_class

        legend = self._freq_plot_widget.addLegend()  # noqa: F841
        fallback_idx = 0
        for label, matrix in data_by_class.items():
            mean_curve = matrix.mean(axis=0)
            color = _CLASS_COLORS.get(label, _FALLBACK[fallback_idx % len(_FALLBACK)])
            fallback_idx += 1
            pen = pg.mkPen(color, width=1.5)
            self._freq_plot_widget.plot(x, mean_curve, pen=pen, name=label)

        if fd.mode == "erders":
            # Horizontal zero reference
            self._freq_plot_widget.addItem(
                pg.InfiniteLine(
                    pos=0.0, angle=0,
                    pen=pg.mkPen((0, 0, 0, 80), width=0.8, style=Qt.DashLine),
                )
            )
            # Vertical event onset
            self._freq_plot_widget.addItem(
                pg.InfiniteLine(
                    pos=0.0, angle=90,
                    pen=pg.mkPen((0, 0, 0, 60), width=0.8, style=Qt.DashLine),
                )
            )

    # ── ERP panel ─────────────────────────────────────────────────────────────

    def _build_erp_panel(self) -> QGroupBox:
        box = QGroupBox("ERP")
        box.setVisible(False)
        self._erp_box = box
        layout = QVBoxLayout(box)
        layout.setSpacing(4)

        self._erp_bl_check = QCheckBox("Correction baseline")
        layout.addWidget(self._erp_bl_check)

        bl_row = QHBoxLayout()
        bl_row.setContentsMargins(16, 0, 0, 0)
        self._erp_bl_tmin = QDoubleSpinBox()
        self._erp_bl_tmin.setRange(-10.0, 0.0)
        self._erp_bl_tmin.setValue(-0.2)
        self._erp_bl_tmin.setSuffix(" s")
        self._erp_bl_tmin.setDecimals(2)
        self._erp_bl_tmin.setSingleStep(0.05)
        self._erp_bl_tmin.setEnabled(False)
        self._erp_bl_tmax = QDoubleSpinBox()
        self._erp_bl_tmax.setRange(-10.0, 1.0)
        self._erp_bl_tmax.setValue(0.0)
        self._erp_bl_tmax.setSuffix(" s")
        self._erp_bl_tmax.setDecimals(2)
        self._erp_bl_tmax.setSingleStep(0.05)
        self._erp_bl_tmax.setEnabled(False)
        bl_row.addWidget(self._erp_bl_tmin)
        bl_row.addWidget(self._erp_bl_tmax)
        layout.addLayout(bl_row)
        self._erp_bl_check.toggled.connect(self._erp_bl_tmin.setEnabled)
        self._erp_bl_check.toggled.connect(self._erp_bl_tmax.setEnabled)

        btn_row = QHBoxLayout()
        self._erp_calc_btn = QPushButton("Calculer")
        self._erp_calc_btn.setFixedHeight(26)
        self._erp_calc_btn.clicked.connect(self._on_calc_erp)
        self._erp_back_btn = QPushButton("← Époques")
        self._erp_back_btn.setFixedHeight(26)
        self._erp_back_btn.setVisible(False)
        self._erp_back_btn.clicked.connect(self._on_back_to_epochs)
        btn_row.addWidget(self._erp_calc_btn)
        btn_row.addWidget(self._erp_back_btn)
        layout.addLayout(btn_row)

        return box

    def _stop_erp_worker(self):
        if self._erp_worker is not None:
            try:
                self._erp_worker.result_ready.disconnect()
                self._erp_worker.error.disconnect()
            except (RuntimeError, TypeError):
                pass
            self._erp_worker.quit()
            self._erp_worker.wait()
            self._erp_worker = None

    def _on_calc_erp(self):
        if self._epoch_data is None:
            return

        selected_ch = []
        for i in range(self._channel_list.count()):
            item = self._channel_list.item(i)
            if item.checkState() == Qt.Checked:
                selected_ch.append(i)
        if not selected_ch:
            self._status.setText("Sélectionnez au moins un canal")
            return

        from src.workers.erp_worker import ERPWorker

        bl_correction = self._erp_bl_check.isChecked()
        bl_tmin = self._erp_bl_tmin.value() if bl_correction else float(self._epoch_data.times[0])
        bl_tmax = self._erp_bl_tmax.value() if bl_correction else 0.0

        self._erp_calc_btn.setEnabled(False)
        self._erp_calc_btn.setText("Calcul…")
        self._stop_erp_worker()

        self._erp_worker = ERPWorker(
            epoch_data=self._epoch_data,
            selected_ch_indices=selected_ch,
            baseline_correction=bl_correction,
            baseline_tmin=bl_tmin,
            baseline_tmax=bl_tmax,
        )
        self._erp_worker.result_ready.connect(self._on_erp_ready)
        self._erp_worker.error.connect(self._on_erp_error)
        self._erp_worker.start()

    def _on_erp_ready(self, erp_data):
        self._erp_data = erp_data
        self._erp_worker.wait()
        self._erp_worker = None
        self._erp_calc_btn.setEnabled(True)
        self._erp_calc_btn.setText("Calculer")
        self._erp_back_btn.setVisible(True)
        self._replot_erp()
        self._stack.setCurrentIndex(4)
        n_ch = len(erp_data.ch_names)
        n_classes = len(erp_data.erp_by_class)
        bl_suffix = " (baseline corr.)" if erp_data.baseline_corrected else ""
        self._status.setText(
            f"ERP — {n_ch} canal{'x' if n_ch > 1 else ''} · "
            f"{n_classes} classe{'s' if n_classes > 1 else ''}{bl_suffix}"
        )

    def _on_erp_error(self, message: str):
        self._erp_worker.wait()
        self._erp_worker = None
        self._erp_calc_btn.setEnabled(True)
        self._erp_calc_btn.setText("Calculer")
        self._status.setText(f"Erreur ERP : {message}")

    def _on_back_to_epochs(self):
        self._stack.setCurrentIndex(2)
        self._erp_back_btn.setVisible(False)
        n_epochs = self._epoch_data.data.shape[0]
        n_classes = len(set(self._epoch_data.labels))
        tmin = self._epoch_data.times[0]
        tmax = self._epoch_data.times[-1]
        self._status.setText(
            f"{n_epochs} époques · {n_classes} classe{'s' if n_classes > 1 else ''} · "
            f"{tmin:.2f}–{tmax:.2f} s"
        )

    def _replot_erp(self):
        self._erp_plot_layout.clear()

        if self._erp_data is None:
            return

        ed = self._erp_data
        times = ed.times

        _CLASS_COLORS = {
            "T0": (150, 150, 150),
            "T1": (70, 130, 180),
            "T2": (255, 99, 71),
        }
        _FALLBACK = [(100, 180, 100), (180, 100, 180), (180, 160, 60)]

        n_ch = len(ed.ch_names)
        for ch_subplot_idx, ch_name in enumerate(ed.ch_names):
            plot = self._erp_plot_layout.addPlot(row=ch_subplot_idx, col=0)
            plot.setLabel("left", ch_name)
            plot.showGrid(x=True, y=False, alpha=0.3)
            if ch_subplot_idx == n_ch - 1:
                plot.setLabel("bottom", "Temps (s)")

            # Vertical line at t=0 (event onset)
            plot.addItem(pg.InfiniteLine(
                pos=0.0, angle=90,
                pen=pg.mkPen((0, 0, 0, 60), width=0.8, style=Qt.DashLine),
            ))

            # Baseline window shading (if correction applied)
            if ed.baseline_corrected:
                bl_region = pg.LinearRegionItem(
                    values=[ed.baseline_tmin, ed.baseline_tmax],
                    brush=pg.mkBrush(200, 200, 200, 40),
                    movable=False,
                )
                plot.addItem(bl_region)

            fallback_idx = 0
            for label, matrix in ed.erp_by_class.items():
                waveform = matrix[ch_subplot_idx]
                color = _CLASS_COLORS.get(label, _FALLBACK[fallback_idx % len(_FALLBACK)])
                fallback_idx += 1
                pen = pg.mkPen(color, width=1.5)
                plot.plot(times, waveform, pen=pen, name=label if ch_subplot_idx == 0 else None)
