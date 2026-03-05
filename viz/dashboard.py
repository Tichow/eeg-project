"""
Dashboard EEG temps réel — QMainWindow qui orchestre panels et widgets.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime

import numpy as np
from PyQt5.QtCore import QEvent, Qt, QTimer
from PyQt5.QtWidgets import (
    QComboBox, QDialog, QDialogButtonBox, QFormLayout, QHBoxLayout,
    QHeaderView, QInputDialog, QLabel, QMainWindow, QPushButton,
    QScrollArea, QSplitter, QTableWidget, QTableWidgetItem, QTextEdit,
    QVBoxLayout, QWidget,
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
_RECORDINGS_DIR   = 'recordings'
_CHANNEL_MAP_FILE = 'channel_map.json'

_SUBJECTS = ['Mattéo', 'Fabien', 'Némo']
_TEST_TYPES = [
    ('eyes_closed',      'Yeux fermés'),
    ('blink',            'Clignements'),
    ('hand_movement',    'Mouvement des mains'),
    ('flashing_stimuli', 'Stimulis clignotant'),
]
_TEST_ACTIONS: dict[str, list[str]] = {
    'eyes_closed':      ['Fermer les yeux', 'Ouvrir les yeux'],
    'blink':            ['Cligner', 'Pause'],
    'hand_movement':    ['Mouvement gauche', 'Mouvement droit', 'Pause'],
    'flashing_stimuli': ['Changer fréquence', 'Changer couleur',
                         'Début stimuli', 'Fin stimuli'],
}


class _StartRecordingDialog(QDialog):
    """Dialog affiché au démarrage d'un enregistrement pour choisir sujet et test."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle('Nouvel enregistrement')
        self.setMinimumWidth(320)

        form = QFormLayout(self)
        form.setSpacing(12)
        form.setContentsMargins(16, 16, 16, 16)

        self._subject_box = QComboBox()
        self._subject_box.addItems(_SUBJECTS)
        self._subject_box.setEditable(True)
        form.addRow('Sujet :', self._subject_box)

        self._test_box = QComboBox()
        for _, label in _TEST_TYPES:
            self._test_box.addItem(label)
        self._test_box.setEditable(True)
        form.addRow('Type de test :', self._test_box)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    @property
    def subject(self) -> str:
        return self._subject_box.currentText().strip()

    @property
    def test_type_id(self) -> str:
        idx = self._test_box.currentIndex()
        if 0 <= idx < len(_TEST_TYPES):
            return _TEST_TYPES[idx][0]
        return re.sub(r'\s+', '_', self._test_box.currentText().strip().lower())

    @property
    def test_type_label(self) -> str:
        return self._test_box.currentText().strip()


class _StopRecordingDialog(QDialog):
    """Dialog affiché à l'arrêt : labellisation des marqueurs + notes."""

    def __init__(
        self,
        markers: list[dict],
        test_type: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle('Fin de l\'enregistrement')
        self.setMinimumWidth(460)

        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(16, 16, 16, 16)

        actions = _TEST_ACTIONS.get(test_type, [])
        # Marqueurs labellisables (tous sauf le "début" auto à t=0)
        self._label_rows: list[tuple[float, QComboBox]] = []
        self._fixed_markers: list[dict] = []

        labelable = [m for m in markers if not (m['time_sec'] == 0.0 and m['action'] == 'début')]
        fixed     = [m for m in markers if m not in labelable]
        self._fixed_markers = fixed

        if labelable:
            layout.addWidget(QLabel('Labelliser les marqueurs :'))
            table = QTableWidget(len(labelable), 2)
            table.setHorizontalHeaderLabels(['t (s)', 'Action'])
            table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
            table.verticalHeader().setVisible(False)
            table.setEditTriggers(QTableWidget.NoEditTriggers)
            for row, m in enumerate(labelable):
                table.setItem(row, 0, QTableWidgetItem(str(m['time_sec'])))
                combo = QComboBox()
                combo.addItems(actions)
                combo.setEditable(True)
                if m.get('action'):
                    combo.setCurrentText(m['action'])
                table.setCellWidget(row, 1, combo)
                self._label_rows.append((m['time_sec'], combo))
            table.setFixedHeight(min(240, 30 + 30 * len(labelable)))
            layout.addWidget(table)
        else:
            layout.addWidget(QLabel('Aucun marqueur posé.'))

        layout.addWidget(QLabel('Notes :'))
        self._notes = QTextEdit()
        self._notes.setFixedHeight(70)
        layout.addWidget(self._notes)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)

    @property
    def notes(self) -> str:
        return self._notes.toPlainText().strip()

    @property
    def labeled_markers(self) -> list[dict]:
        result = list(self._fixed_markers)
        for t_sec, combo in self._label_rows:
            result.append({'time_sec': t_sec, 'action': combo.currentText().strip()})
        result.sort(key=lambda m: m['time_sec'])
        return result


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
        self._rec_active     = False
        self._rec_buffer:  list[np.ndarray] = []
        self._rec_markers: list[dict]       = []
        self._rec_meta:    dict             = {}
        self._rec_start_time: float         = 0.0
        self._ssvep_proc:         subprocess.Popen | None = None
        self._ssvep_events_file:  str | None              = None
        self._ssvep_events_read:  int                     = 0
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

            # Récupère les événements SSVEP depuis le fichier partagé
            if self._ssvep_events_file and os.path.exists(self._ssvep_events_file):
                try:
                    with open(self._ssvep_events_file) as fh:
                        lines = fh.readlines()
                    for line in lines[self._ssvep_events_read:]:
                        try:
                            self._rec_markers.append(json.loads(line.strip()))
                        except json.JSONDecodeError:
                            pass
                    self._ssvep_events_read = len(lines)
                except OSError:
                    pass

            # Auto-stop quand le processus SSVEP se termine
            if self._ssvep_proc is not None and self._ssvep_proc.poll() is not None:
                self._ssvep_proc = None
                QTimer.singleShot(300, self._toggle_recording)  # 300ms pour finir de lire les derniers events

            elapsed  = int(time.monotonic() - self._rec_start_time)
            n_marks  = len(self._rec_markers) - 1  # exclut le marqueur "début"
            mark_str = f' | ●{n_marks}' if n_marks > 0 else ''
            self.setWindowTitle(
                f'OpenBCI Cyton — EEG Temps Réel  '
                f'[● REC {elapsed // 60:02d}:{elapsed % 60:02d}{mark_str}]'
            )

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
            dlg = _StartRecordingDialog(self)
            if dlg.exec_() != QDialog.Accepted:
                return
            self._rec_meta = {
                'subject':          dlg.subject,
                'test_type':        dlg.test_type_id,
                'test_type_label':  dlg.test_type_label,
            }
            self._rec_buffer     = []
            self._rec_markers    = [{'time_sec': 0.0, 'action': 'début'}]
            self._rec_start_time = time.monotonic()
            self._rec_active     = True

            if dlg.test_type_id == 'flashing_stimuli':
                self._ssvep_events_file = '.ssvep_live.jsonl'
                open(self._ssvep_events_file, 'w').close()
                self._ssvep_events_read = 0
                rec_start_wall = time.time()
                self._ssvep_proc = subprocess.Popen([
                    sys.executable, 'ssvep_stimulus.py',
                    '--sync-file', self._ssvep_events_file,
                    '--rec-start', str(rec_start_wall),
                ])
                print('  [SSVEP] Stimulus lancé — appuie ESPACE dans la fenêtre SSVEP pour démarrer')

            print(f'\n  [REC] Enregistrement démarré — {dlg.subject} / {dlg.test_type_label} — appuie sur R pour arrêter, M pour marquer')
        else:
            self._rec_active = False
            self.setWindowTitle('OpenBCI Cyton — EEG Temps Réel')

            if self._ssvep_proc is not None:
                self._ssvep_proc.terminate()
                self._ssvep_proc = None
            if self._ssvep_events_file and os.path.exists(self._ssvep_events_file):
                os.remove(self._ssvep_events_file)
            self._ssvep_events_file = None
            self._ssvep_events_read = 0

            if self._rec_buffer:
                recorded = np.concatenate(self._rec_buffer, axis=1)
                if self._rec_meta.get('test_type') == 'flashing_stimuli':
                    # Markers déjà labellisés automatiquement par le stimulus SSVEP
                    self._save_recording(recorded, self._rec_markers)
                else:
                    dlg = _StopRecordingDialog(
                        self._rec_markers,
                        self._rec_meta.get('test_type', ''),
                        self,
                    )
                    dlg.exec_()
                    self._save_recording(recorded, dlg.labeled_markers, dlg.notes)
            self._rec_buffer  = []
            self._rec_markers = []
            self._rec_meta    = {}

    def _mark_event(self) -> None:
        if not self._rec_active:
            return
        elapsed = round(time.monotonic() - self._rec_start_time, 2)
        self._rec_markers.append({'time_sec': elapsed, 'action': ''})
        print(f'  [REC] ● Marqueur posé : t={elapsed} s')

    def _save_recording(self, data: np.ndarray, markers: list[dict], notes: str = '') -> None:
        os.makedirs(_RECORDINGS_DIR, exist_ok=True)
        ts      = datetime.now().strftime('%Y%m%d_%H%M%S')
        subject = self._rec_meta.get('subject', '')
        tt_id   = self._rec_meta.get('test_type', '')

        def _safe(s: str) -> str:
            return re.sub(r'\s+', '_', re.sub(r'[^\w\s-]', '', s)).strip('_')

        parts = [ts, _safe(subject), tt_id]
        stem  = '_'.join(p for p in parts if p)

        npy_path  = os.path.join(_RECORDINGS_DIR, f'{stem}.npy')
        meta_path = os.path.join(_RECORDINGS_DIR, f'{stem}.json')
        np.save(npy_path, data)

        channel_map: dict | None = None
        if os.path.isfile(_CHANNEL_MAP_FILE):
            with open(_CHANNEL_MAP_FILE) as f:
                channel_map = json.load(f)

        meta: dict = {
            'timestamp':        ts,
            'subject':          subject,
            'test_type':        tt_id,
            'test_type_label':  self._rec_meta.get('test_type_label', ''),
            'sfreq':            self._sfreq,
            'channels':         self._ch_labels,
        }
        if channel_map is not None:
            meta['channel_map'] = channel_map
        meta['n_samples']    = data.shape[1]
        meta['duration_sec'] = round(data.shape[1] / self._sfreq, 2)
        meta['events']       = markers
        if notes:
            meta['notes'] = notes

        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        print(f'\n  [REC] Sauvegardé : {npy_path} ({meta["duration_sec"]} s)')

    # ------------------------------------------------------------------
    # Événements Qt
    # ------------------------------------------------------------------

    def eventFilter(self, obj, event) -> bool:
        """Intercepte R et M globalement, sauf quand un dialog modal est ouvert."""
        if event.type() == QEvent.KeyPress:
            from PyQt5.QtWidgets import QApplication
            if QApplication.activeModalWidget() is not None:
                return super().eventFilter(obj, event)
            key = event.text().lower()
            if key == 'r':
                self._toggle_recording()
                return True
            if key == 'm':
                self._mark_event()
                return True
        return super().eventFilter(obj, event)

    def closeEvent(self, event) -> None:
        self._timer.stop()
        super().closeEvent(event)
