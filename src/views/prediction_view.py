from __future__ import annotations

import os
from collections import deque

import numpy as np
from PyQt5.QtCore import QPointF, QRectF, Qt
from PyQt5.QtGui import QColor, QFont, QPainter, QPen, QRadialGradient
from PyQt5.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from src.views.base_view import BaseView

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
_BTN_GREEN = """
    QPushButton {
        background-color: #4caf50; color: white;
        border: none; border-radius: 4px; padding: 6px 12px;
    }
    QPushButton:hover { background-color: #43a047; }
    QPushButton:pressed { background-color: #388e3c; }
    QPushButton:disabled { background-color: #aaaaaa; }
"""

_MI_CHANNELS_8 = ["C3", "FC1", "C4", "O1", "Cz", "O2", "FC2", "Pz"]


# ======================================================================
# Fullscreen cue window (reused from acquisition pattern)
# ======================================================================

class _CueWindow(QWidget):
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

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if event.key() == Qt.Key_Escape:
            self.close()


# ======================================================================
# Mini-game widget
# ======================================================================

class _BciGameWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(200)
        self._ball_x: float = 0.5
        self._trail: deque[tuple[float, float]] = deque(maxlen=20)
        self._active_target: int | None = None
        self._label_t1 = "T1"
        self._label_t2 = "T2"

    def set_labels(self, t1: str, t2: str) -> None:
        self._label_t1 = t1
        self._label_t2 = t2
        self.update()

    def update_position(self, p_t2: float, is_confident: bool, label: int) -> None:
        self._trail.append((self._ball_x, 1.0))
        self._trail = deque(
            ((x, o * 0.85) for x, o in self._trail if o > 0.05),
            maxlen=20,
        )
        self._ball_x = p_t2
        self._active_target = label if is_confident else None
        self.update()

    def reset(self) -> None:
        self._ball_x = 0.5
        self._trail.clear()
        self._active_target = None
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        zone_w = 70

        p.fillRect(0, 0, w, h, QColor("#1a1a2e"))

        t1_color = QColor("#61afef") if self._active_target == 0 else QColor("#2a4a6e")
        t2_color = QColor("#e06c75") if self._active_target == 1 else QColor("#6e2a2a")
        p.fillRect(0, 0, zone_w, h, t1_color)
        p.fillRect(w - zone_w, 0, zone_w, h, t2_color)

        p.setPen(QColor("white"))
        font = QFont()
        font.setPointSize(13)
        font.setBold(True)
        p.setFont(font)
        p.drawText(QRectF(0, 0, zone_w, h), Qt.AlignCenter, self._label_t1)
        p.drawText(QRectF(w - zone_w, 0, zone_w, h), Qt.AlignCenter, self._label_t2)

        p.setPen(QPen(QColor("#444"), 1, Qt.DashLine))
        p.drawLine(w // 2, 0, w // 2, h)

        track_left = zone_w
        track_width = w - 2 * zone_w
        cy = h / 2.0

        for tx, opacity in self._trail:
            c = QColor(255, 255, 255, int(opacity * 60))
            p.setBrush(c)
            p.setPen(Qt.NoPen)
            p.drawEllipse(QPointF(track_left + tx * track_width, cy), 6, 6)

        ball_x = track_left + self._ball_x * track_width
        gradient = QRadialGradient(ball_x, cy, 25)
        gradient.setColorAt(0, QColor(255, 255, 255, 140))
        gradient.setColorAt(1, QColor(255, 255, 255, 0))
        p.setBrush(gradient)
        p.setPen(Qt.NoPen)
        p.drawEllipse(QPointF(ball_x, cy), 25, 25)
        p.setBrush(QColor("white"))
        p.drawEllipse(QPointF(ball_x, cy), 13, 13)
        p.end()


# ======================================================================
# Prediction view
# ======================================================================

class PredictionView(BaseView):
    """Real-time BCI prediction view with quick calibration and visual feedback."""

    def setup_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Header
        header = QWidget()
        header.setFixedHeight(48)
        header.setStyleSheet("background:#2c2c2c;")
        h_layout = QHBoxLayout(header)
        h_layout.setContentsMargins(12, 0, 12, 0)
        back_btn = QPushButton("← Retour")
        back_btn.setStyleSheet(_BTN_GRAY)
        back_btn.setCursor(Qt.PointingHandCursor)
        back_btn.clicked.connect(lambda: self.navigate.emit("home"))
        title = QLabel("Prediction BCI — Temps reel")
        title.setStyleSheet("color: white; font-size: 15px; font-weight: bold;")
        h_layout.addWidget(back_btn)
        h_layout.addSpacing(12)
        h_layout.addWidget(title)
        h_layout.addStretch()
        root.addWidget(header)

        # Splitter
        splitter = QSplitter(Qt.Horizontal)
        root.addWidget(splitter)
        splitter.addWidget(self._build_left_panel())
        splitter.addWidget(self._build_right_panel())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([300, 900])

        # State
        self._board = None
        self._pipeline = None
        self._is_fbcsp = False
        self._connected = False
        self._model_loaded = False
        self._pred_worker = None
        self._record_worker = None
        self._cue_window: _CueWindow | None = None
        self._smoothed_proba = np.array([0.5, 0.5])
        self._prediction_count = 0
        self._confident_count = 0

        self._set_connected(False)

    # ------------------------------------------------------------------
    # Left panel
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
        layout.addWidget(self._build_model_group())
        layout.addWidget(self._build_calibration_group())
        layout.addWidget(self._build_commands_group())
        layout.addWidget(self._build_settings_group())

        # Start / Stop prediction
        self._start_btn = QPushButton("Demarrer la prediction")
        self._start_btn.setStyleSheet(_BTN_BLUE)
        self._start_btn.setCursor(Qt.PointingHandCursor)
        self._start_btn.setEnabled(False)
        self._start_btn.clicked.connect(self._on_start)
        layout.addWidget(self._start_btn)

        self._stop_btn = QPushButton("Arreter")
        self._stop_btn.setStyleSheet(_BTN_RED)
        self._stop_btn.setCursor(Qt.PointingHandCursor)
        self._stop_btn.setEnabled(False)
        self._stop_btn.clicked.connect(self._on_stop)
        layout.addWidget(self._stop_btn)

        # Buffer progress
        self._buffer_bar = QProgressBar()
        self._buffer_bar.setMaximum(750)
        self._buffer_bar.setValue(0)
        self._buffer_bar.setFormat("Buffer : %v / %m")
        layout.addWidget(self._buffer_bar)

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

        self._status_label = QLabel("Non connecte")
        self._status_label.setStyleSheet("color: #888; font-size: 11px;")
        layout.addWidget(self._status_label)

        self._refresh_ports()
        return grp

    def _build_model_group(self) -> QGroupBox:
        grp = QGroupBox("Modele (optionnel)")
        layout = QVBoxLayout(grp)

        path_row = QHBoxLayout()
        self._model_path_edit = QLineEdit()
        self._model_path_edit.setPlaceholderText("Charger un .pkl existant")
        browse_btn = QPushButton("...")
        browse_btn.setFixedWidth(30)
        browse_btn.setCursor(Qt.PointingHandCursor)
        browse_btn.clicked.connect(self._browse_model)
        path_row.addWidget(self._model_path_edit, 1)
        path_row.addWidget(browse_btn)
        layout.addLayout(path_row)

        load_btn = QPushButton("Charger")
        load_btn.setStyleSheet(_BTN_GRAY)
        load_btn.setCursor(Qt.PointingHandCursor)
        load_btn.clicked.connect(self._on_load_model)
        layout.addWidget(load_btn)

        self._model_type_label = QLabel("Aucun modele")
        self._model_type_label.setStyleSheet("color: #888; font-size: 11px;")
        layout.addWidget(self._model_type_label)

        return grp

    def _build_calibration_group(self) -> QGroupBox:
        grp = QGroupBox("Calibration rapide")
        layout = QVBoxLayout(grp)

        layout.addWidget(QLabel("Entraine un modele frais sur cette session :"))

        row_trials = QHBoxLayout()
        row_trials.addWidget(QLabel("Essais / classe :"))
        self._calib_trials_spin = QSpinBox()
        self._calib_trials_spin.setRange(5, 30)
        self._calib_trials_spin.setValue(10)
        row_trials.addWidget(self._calib_trials_spin)
        layout.addLayout(row_trials)

        self._calib_btn = QPushButton("Calibrer (~2 min)")
        self._calib_btn.setStyleSheet(_BTN_GREEN)
        self._calib_btn.setCursor(Qt.PointingHandCursor)
        self._calib_btn.setEnabled(False)
        self._calib_btn.clicked.connect(self._on_calibrate)
        layout.addWidget(self._calib_btn)

        self._calib_status = QLabel("")
        self._calib_status.setStyleSheet("font-size: 11px; color: #aaa;")
        self._calib_status.setWordWrap(True)
        layout.addWidget(self._calib_status)

        return grp

    def _build_commands_group(self) -> QGroupBox:
        grp = QGroupBox("Commandes")
        layout = QVBoxLayout(grp)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("T1 :"))
        self._cmd1_edit = QLineEdit("Mains")
        row1.addWidget(self._cmd1_edit, 1)
        layout.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("T2 :"))
        self._cmd2_edit = QLineEdit("Pieds")
        row2.addWidget(self._cmd2_edit, 1)
        layout.addLayout(row2)

        self._cmd1_edit.textChanged.connect(self._update_game_labels)
        self._cmd2_edit.textChanged.connect(self._update_game_labels)
        return grp

    def _build_settings_group(self) -> QGroupBox:
        grp = QGroupBox("Reglages")
        layout = QVBoxLayout(grp)

        row_alpha = QHBoxLayout()
        row_alpha.addWidget(QLabel("Lissage (EMA) :"))
        self._ema_spin = QDoubleSpinBox()
        self._ema_spin.setRange(0.05, 1.0)
        self._ema_spin.setSingleStep(0.05)
        self._ema_spin.setValue(0.5)
        row_alpha.addWidget(self._ema_spin)
        layout.addLayout(row_alpha)

        row_thresh = QHBoxLayout()
        row_thresh.addWidget(QLabel("Seuil confiance :"))
        self._threshold_spin = QDoubleSpinBox()
        self._threshold_spin.setRange(0.50, 0.95)
        self._threshold_spin.setSingleStep(0.05)
        self._threshold_spin.setValue(0.60)
        row_thresh.addWidget(self._threshold_spin)
        layout.addLayout(row_thresh)
        return grp

    # ------------------------------------------------------------------
    # Right panel
    # ------------------------------------------------------------------

    def _build_right_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        self._game = _BciGameWidget()
        self._game.set_labels("Mains", "Pieds")
        self._game.setMinimumHeight(200)
        layout.addWidget(self._game, 3)

        # Confidence bars
        bars = QWidget()
        bl = QVBoxLayout(bars)
        bl.setContentsMargins(0, 0, 0, 0)
        bl.setSpacing(4)

        row_t1 = QHBoxLayout()
        self._bar_t1_label = QLabel("Mains")
        self._bar_t1_label.setFixedWidth(60)
        self._bar_t1_label.setStyleSheet("font-weight: bold; color: #61afef;")
        self._bar_t1 = QProgressBar()
        self._bar_t1.setMaximum(100)
        self._bar_t1.setValue(50)
        self._bar_t1.setStyleSheet(
            "QProgressBar{background:#2a2a3e;border-radius:4px}"
            "QProgressBar::chunk{background:#61afef;border-radius:4px}"
        )
        self._bar_t1_pct = QLabel("50%")
        self._bar_t1_pct.setFixedWidth(40)
        row_t1.addWidget(self._bar_t1_label)
        row_t1.addWidget(self._bar_t1, 1)
        row_t1.addWidget(self._bar_t1_pct)
        bl.addLayout(row_t1)

        row_t2 = QHBoxLayout()
        self._bar_t2_label = QLabel("Pieds")
        self._bar_t2_label.setFixedWidth(60)
        self._bar_t2_label.setStyleSheet("font-weight: bold; color: #e06c75;")
        self._bar_t2 = QProgressBar()
        self._bar_t2.setMaximum(100)
        self._bar_t2.setValue(50)
        self._bar_t2.setStyleSheet(
            "QProgressBar{background:#2a2a3e;border-radius:4px}"
            "QProgressBar::chunk{background:#e06c75;border-radius:4px}"
        )
        self._bar_t2_pct = QLabel("50%")
        self._bar_t2_pct.setFixedWidth(40)
        row_t2.addWidget(self._bar_t2_label)
        row_t2.addWidget(self._bar_t2, 1)
        row_t2.addWidget(self._bar_t2_pct)
        bl.addLayout(row_t2)
        layout.addWidget(bars)

        # Stats
        stats_row = QHBoxLayout()
        self._stats_count = QLabel("Predictions : 0")
        self._stats_confident = QLabel("Confiantes : 0/0")
        self._stats_last = QLabel("Derniere : —")
        for lbl in (self._stats_count, self._stats_confident, self._stats_last):
            lbl.setStyleSheet("font-size: 11px; color: #aaa;")
            stats_row.addWidget(lbl)
        stats_row.addStretch()
        layout.addLayout(stats_row)

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
    # Connection
    # ------------------------------------------------------------------

    def _refresh_ports(self) -> None:
        from src.services.acquisition_service import AcquisitionService
        self._port_combo.clear()
        try:
            ports = AcquisitionService.list_serial_ports()
        except Exception as exc:
            self._append_log(f"[ERREUR] Ports : {exc}")
            return
        if ports:
            self._port_combo.addItems(ports)
        else:
            self._port_combo.addItem("(aucun port)")

    def _on_connect_toggle(self) -> None:
        if self._connected:
            self._disconnect()
        else:
            self._connect()

    def _connect(self) -> None:
        from src.services.acquisition_service import AcquisitionService
        port = self._port_combo.currentText()
        if not port or port.startswith("("):
            return
        self._connect_btn.setEnabled(False)
        self._append_log(f"[INFO] Connexion {port}...")
        try:
            board, sfreq = AcquisitionService.connect(port)
        except Exception as exc:
            self._append_log(f"[ERREUR] {exc}")
            self._connect_btn.setEnabled(True)
            return
        self._board = board
        self._append_log(f"[INFO] Connecte — {sfreq:.0f} Hz")
        self._set_connected(True)

    def _disconnect(self) -> None:
        from src.services.acquisition_service import AcquisitionService
        self._on_stop()
        self._stop_calibration()
        if self._board is not None:
            try:
                AcquisitionService.disconnect(self._board)
            except Exception:
                pass
            self._board = None
        self._set_connected(False)
        self._append_log("[INFO] Deconnecte.")

    def _set_connected(self, connected: bool) -> None:
        self._connected = connected
        if connected:
            self._connect_btn.setText("Deconnecter")
            self._connect_btn.setStyleSheet(_BTN_RED)
            self._status_label.setText("Connecte")
            self._status_label.setStyleSheet("color: #98c379; font-size: 11px;")
        else:
            self._connect_btn.setText("Connecter")
            self._connect_btn.setStyleSheet(_BTN_BLUE)
            self._status_label.setText("Non connecte")
            self._status_label.setStyleSheet("color: #888; font-size: 11px;")
        self._connect_btn.setEnabled(True)
        self._calib_btn.setEnabled(connected)
        self._update_start_button()

    # ------------------------------------------------------------------
    # Model load (optional — user can also just calibrate)
    # ------------------------------------------------------------------

    def _browse_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Charger un modele", "models/", "Pickle (*.pkl)"
        )
        if path:
            self._model_path_edit.setText(path)
            self._on_load_model()

    def _on_load_model(self) -> None:
        from src.services.eeg_classification_service import EEGClassificationService
        path = self._model_path_edit.text().strip()
        if not path:
            return
        try:
            pipeline = EEGClassificationService.load_pipeline(path)
            rng = np.random.RandomState(0)
            pipeline.predict(rng.randn(1, 8, 750) * 1e-5)
        except Exception as exc:
            self._append_log(f"[ERREUR] {exc}")
            return
        self._pipeline = pipeline
        self._is_fbcsp = "fbcsp" in pipeline.named_steps
        self._model_loaded = True
        self._model_type_label.setText(f"Charge : {os.path.basename(path)}")
        self._model_type_label.setStyleSheet("color: #98c379; font-size: 11px;")
        self._append_log(f"[OK] Modele charge : {os.path.basename(path)}")
        self._update_start_button()

    # ------------------------------------------------------------------
    # Quick calibration: guided protocol → train CSP+LDA on the spot
    # ------------------------------------------------------------------

    def _on_calibrate(self) -> None:
        if not self._connected or self._board is None:
            return

        from src.models.acquisition_config import AcquisitionConfig

        n_trials = self._calib_trials_spin.value()
        cmd1 = self._cmd1_edit.text() or "T1"
        cmd2 = self._cmd2_edit.text() or "T2"

        config = AcquisitionConfig(
            serial_port="",
            ch_names=list(_MI_CHANNELS_8),
            subject_id="calib",
            run_label="calib",
            output_dir="",
            n_trials_per_class=n_trials,
            t_baseline_s=1.5,
            t_cue_s=4.0,
            t_rest_s=1.0,
            classes=[cmd1, cmd2],
            annotation_labels=["T1", "T2"],
        )

        total = n_trials * 2
        duration = total * (1.5 + 4.0 + 1.0)
        self._append_log(
            f"[CALIB] {total} essais ({n_trials}x{cmd1} + {n_trials}x{cmd2}) — ~{duration:.0f}s"
        )

        self._calib_btn.setEnabled(False)
        self._start_btn.setEnabled(False)
        self._connect_btn.setEnabled(False)

        # Fullscreen cue window
        self._cue_window = _CueWindow()
        self._cue_window.showFullScreen()

        # Progress bar
        self._buffer_bar.setMaximum(total)
        self._buffer_bar.setValue(0)
        self._buffer_bar.setFormat("Calibration : %v / %m essais")

        from src.workers.acquisition_record_worker import AcquisitionRecordWorker

        self._record_worker = AcquisitionRecordWorker(self._board, config)
        self._record_worker.trial_update.connect(self._on_calib_trial)
        self._record_worker.phase_update.connect(self._on_calib_phase)
        self._record_worker.finished.connect(self._on_calib_finished)
        self._record_worker.error.connect(self._on_calib_error)
        self._record_worker.start()

    def _on_calib_trial(self, i: int, total: int, cue: str) -> None:
        self._buffer_bar.setValue(i)
        if self._cue_window is not None:
            self._cue_window.set_text(cue)
        self._calib_status.setText(f"Essai {i}/{total} — {cue}")

    def _on_calib_phase(self, phase: str) -> None:
        if phase in ("baseline", "rest") and self._cue_window is not None:
            self._cue_window.set_text("✛")

    def _on_calib_finished(self, signal_data) -> None:
        if self._cue_window is not None:
            self._cue_window.close()
            self._cue_window = None
        self._stop_calibration_worker()

        self._append_log("[CALIB] Enregistrement termine — entrainement du modele...")
        self._calib_status.setText("Entrainement en cours...")

        # Train CSP+LDA on the recorded data
        try:
            self._train_from_signal(signal_data)
        except Exception as exc:
            self._append_log(f"[ERREUR] Entrainement echoue : {exc}")
            self._calib_status.setText("Echec")
            self._calib_btn.setEnabled(True)
            self._connect_btn.setEnabled(True)
            self._buffer_bar.setMaximum(750)
            self._buffer_bar.setValue(0)
            self._buffer_bar.setFormat("Buffer : %v / %m")
            return

        self._calib_btn.setEnabled(True)
        self._connect_btn.setEnabled(True)
        self._buffer_bar.setMaximum(750)
        self._buffer_bar.setValue(0)
        self._buffer_bar.setFormat("Buffer : %v / %m")
        self._update_start_button()

    def _on_calib_error(self, msg: str) -> None:
        if self._cue_window is not None:
            self._cue_window.close()
            self._cue_window = None
        self._stop_calibration_worker()
        self._append_log(f"[ERREUR] Calibration : {msg}")
        self._calib_status.setText("Erreur")
        self._calib_btn.setEnabled(True)
        self._connect_btn.setEnabled(True)

    def _stop_calibration_worker(self) -> None:
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

    def _stop_calibration(self) -> None:
        if self._cue_window is not None:
            self._cue_window.close()
            self._cue_window = None
        self._stop_calibration_worker()

    def _train_from_signal(self, signal_data) -> None:
        """Preprocess + epoch + train CSP+LDA from calibration data."""
        from src.models.classification_data import ClassificationConfig
        from src.services.eeg_classification_service import EEGClassificationService

        config = ClassificationConfig(
            task="calibration",
            channels=[],
            notch_hz=50.0,
        )

        svc = EEGClassificationService

        # Preprocess (bandpass 8-30 Hz + notch 50 Hz)
        signal = svc.preprocess(signal_data, config)

        # Extract T1/T2 epochs
        X, y, class_names, n_rejected = svc.extract_mi_epochs(signal, config)

        n_t1 = int(np.sum(y == 0))
        n_t2 = int(np.sum(y == 1))
        self._append_log(
            f"[CALIB] Epochs: T1={n_t1}, T2={n_t2} (rejetes: {n_rejected})"
        )

        if len(y) < 6:
            raise ValueError(f"Pas assez d'epochs ({len(y)})")

        # Build and train pipeline
        pipeline = svc.build_pipeline(config, sfreq=signal_data.sfreq)
        pipeline.fit(X, y)

        # Quick leave-one-out accuracy estimate
        from sklearn.model_selection import cross_val_score
        n_folds = min(5, min(np.bincount(y)))
        if n_folds >= 2:
            scores = cross_val_score(
                svc.build_pipeline(config, sfreq=signal_data.sfreq),
                X, y, cv=n_folds, scoring="accuracy",
            )
            acc = np.mean(scores)
            self._append_log(f"[CALIB] Accuracy {n_folds}-fold: {acc:.0%}")
            self._calib_status.setText(f"Modele calibre — {acc:.0%} accuracy")
        else:
            self._calib_status.setText("Modele calibre")

        self._pipeline = pipeline
        self._is_fbcsp = False
        self._model_loaded = True
        self._model_type_label.setText("Calibration rapide (CSP + LDA)")
        self._model_type_label.setStyleSheet("color: #98c379; font-size: 11px;")
        self._append_log("[OK] Modele calibre — pret pour la prediction")

    # ------------------------------------------------------------------
    # Prediction start / stop
    # ------------------------------------------------------------------

    def _update_start_button(self) -> None:
        can = self._connected and self._model_loaded and self._pred_worker is None
        self._start_btn.setEnabled(can)

    def _on_start(self) -> None:
        if not self._connected or not self._model_loaded or self._board is None:
            return

        self._smoothed_proba = np.array([0.5, 0.5])
        self._prediction_count = 0
        self._confident_count = 0
        self._buffer_bar.setValue(0)
        self._game.reset()

        from src.workers.prediction_stream_worker import PredictionStreamWorker

        self._pred_worker = PredictionStreamWorker(
            self._board, self._pipeline, sfreq=250.0, is_fbcsp=self._is_fbcsp,
        )
        self._pred_worker.prediction_ready.connect(self._on_prediction)
        self._pred_worker.buffer_status.connect(self._on_buffer_status)
        self._pred_worker.error.connect(self._on_pred_error)
        self._pred_worker.start()

        self._start_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        self._calib_btn.setEnabled(False)
        self._append_log("[INFO] Prediction demarree — remplissage du buffer...")

    def _on_stop(self) -> None:
        if self._pred_worker is not None:
            try:
                self._pred_worker.prediction_ready.disconnect()
                self._pred_worker.buffer_status.disconnect()
                self._pred_worker.error.disconnect()
            except (RuntimeError, TypeError):
                pass
            self._pred_worker.stop()
            self._pred_worker.quit()
            self._pred_worker.wait()
            self._pred_worker = None
            self._append_log("[INFO] Prediction arretee.")
        self._stop_btn.setEnabled(False)
        self._calib_btn.setEnabled(self._connected)
        self._update_start_button()

    # ------------------------------------------------------------------
    # Prediction handler
    # ------------------------------------------------------------------

    def _on_prediction(self, label: int, proba: np.ndarray) -> None:
        alpha = self._ema_spin.value()
        self._smoothed_proba = alpha * proba + (1 - alpha) * self._smoothed_proba

        threshold = self._threshold_spin.value()
        dominant = int(np.argmax(self._smoothed_proba))
        is_confident = float(self._smoothed_proba[dominant]) >= threshold

        self._prediction_count += 1
        if is_confident:
            self._confident_count += 1

        self._game.update_position(
            float(self._smoothed_proba[1]), is_confident, dominant,
        )

        p0 = float(self._smoothed_proba[0])
        p1 = float(self._smoothed_proba[1])
        self._bar_t1.setValue(int(p0 * 100))
        self._bar_t2.setValue(int(p1 * 100))
        self._bar_t1_pct.setText(f"{p0:.0%}")
        self._bar_t2_pct.setText(f"{p1:.0%}")

        cmd = self._cmd1_edit.text() if dominant == 0 else self._cmd2_edit.text()
        self._stats_count.setText(f"Predictions : {self._prediction_count}")
        self._stats_confident.setText(
            f"Confiantes : {self._confident_count}/{self._prediction_count}"
        )
        self._stats_last.setText(f"Derniere : {cmd} ({max(p0, p1):.0%})")

        if self._prediction_count % 5 == 1:
            self._append_log(f"[PRED] {cmd} (p={max(p0, p1):.2f})")

    def _on_buffer_status(self, filled: int) -> None:
        self._buffer_bar.setValue(filled)

    def _on_pred_error(self, msg: str) -> None:
        self._append_log(f"[ERREUR] {msg}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _update_game_labels(self) -> None:
        self._game.set_labels(self._cmd1_edit.text(), self._cmd2_edit.text())
        self._bar_t1_label.setText(self._cmd1_edit.text())
        self._bar_t2_label.setText(self._cmd2_edit.text())

    def _append_log(self, text: str) -> None:
        self._log.append(text)
        self._log.verticalScrollBar().setValue(self._log.verticalScrollBar().maximum())
