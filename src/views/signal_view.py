import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure

from PyQt5.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QListWidget,
    QListWidgetItem, QGroupBox, QProgressBar, QSplitter, QSizePolicy,
    QStackedWidget,
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

from src.views.base_view import BaseView
from src.constants.eeg_constants import RUN_DESCRIPTIONS

_ANNOTATION_COLORS = {"T0": "gray", "T1": "steelblue", "T2": "tomato"}
_DEFAULT_N_CHANNELS = 10


class SignalView(BaseView):
    def setup_ui(self):
        self._worker = None
        self._signal_data = None
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

        # Left panel — channel selector
        left = QGroupBox("Canaux")
        left_layout = QVBoxLayout(left)
        self._channel_list = QListWidget()
        self._channel_list.setFixedWidth(160)
        self._channel_list.itemChanged.connect(self._on_channel_changed)
        left_layout.addWidget(self._channel_list)

        btn_row = QHBoxLayout()
        all_btn = QPushButton("Tous")
        all_btn.setFixedHeight(26)
        all_btn.clicked.connect(self._select_all)
        none_btn = QPushButton("Aucun")
        none_btn.setFixedHeight(26)
        none_btn.clicked.connect(self._select_none)
        btn_row.addWidget(all_btn)
        btn_row.addWidget(none_btn)
        left_layout.addLayout(btn_row)
        splitter.addWidget(left)

        # Right panel — loading state + plot
        right_widget = QGroupBox()
        right_widget.setFlat(True)
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)

        self._stack = QStackedWidget()

        # Loading page
        loading_page = QVBoxLayout()
        loading_container_widget = __import__('PyQt5.QtWidgets', fromlist=['QWidget']).QWidget()
        loading_container_widget.setLayout(loading_page)
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
        self._stack.addWidget(loading_container_widget)

        # Plot page
        plot_widget = __import__('PyQt5.QtWidgets', fromlist=['QWidget']).QWidget()
        plot_layout = QVBoxLayout(plot_widget)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        self._figure = Figure(tight_layout=True)
        self._canvas = FigureCanvasQTAgg(self._figure)
        self._toolbar = NavigationToolbar2QT(self._canvas, plot_widget)
        plot_layout.addWidget(self._toolbar)
        plot_layout.addWidget(self._canvas)
        self._stack.addWidget(plot_widget)

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
            # Same file: just show the plot, no reload
            self._stack.setCurrentIndex(1)
            return
        self._start_loading()

    def _start_loading(self):
        from src.workers.signal_worker import SignalWorker
        self._stop_worker()
        self._signal_data = None
        self._channel_list.clear()
        self._stack.setCurrentIndex(0)
        self._status.setText("")

        self._worker = SignalWorker(self._pending_path)
        self._worker.data_ready.connect(self._on_data_ready)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    # ── Worker callbacks ──────────────────────────────────────────────────────

    def _on_data_ready(self, signal_data):
        self._signal_data = signal_data
        self._loaded_path = self._pending_path
        self._worker = None
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
        if self._signal_data is None:
            return

        selected = []
        for i in range(self._channel_list.count()):
            item = self._channel_list.item(i)
            if item.checkState() == Qt.Checked:
                selected.append((i, item.text()))

        self._figure.clear()
        ax = self._figure.add_subplot(111)

        if not selected:
            ax.text(0.5, 0.5, "Aucun canal sélectionné",
                    ha="center", va="center", transform=ax.transAxes, color="#888")
            self._canvas.draw()
            return

        data = self._signal_data.data
        times = self._signal_data.times

        # Auto-scale: median peak-to-peak across selected channels
        ptp_values = [np.ptp(data[idx]) for idx, _ in selected]
        scale = float(np.median(ptp_values)) * 3 if ptp_values else 1e-4
        if scale == 0:
            scale = 1e-4

        for i, (ch_idx, ch_name) in enumerate(selected):
            ax.plot(times, data[ch_idx] + i * scale,
                    linewidth=0.5, color="#2c7bb6", rasterized=True)

        # Y-axis labels
        ax.set_yticks([i * scale for i in range(len(selected))])
        ax.set_yticklabels([name for _, name in selected], fontsize=7)
        ax.tick_params(axis="y", length=0)

        # Annotations
        for onset, _dur, desc in self._signal_data.annotations:
            color = _ANNOTATION_COLORS.get(desc, "black")
            ax.axvline(x=onset, color=color, alpha=0.5, linewidth=0.8, zorder=0)

        ax.set_xlabel("Temps (s)", fontsize=9)
        ax.set_xlim(times[0], times[-1])
        ax.set_ylim(-scale, len(selected) * scale)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)

        self._canvas.draw()

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
