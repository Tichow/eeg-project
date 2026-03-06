"""
Browser PyQt5 pour choisir un enregistrement dans recordings/.
Filtres : sujet, type de test. Tri : date, durée.
"""

from __future__ import annotations

import json
import os
from datetime import datetime

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QComboBox, QDialog, QDialogButtonBox, QFormLayout, QHBoxLayout,
    QHeaderView, QLabel, QTableWidget, QTableWidgetItem, QVBoxLayout,
    QWidget,
)

_RECORDINGS_DIR = 'recordings'

_TEST_TYPE_LABELS: dict[str, str] = {
    'eyes_closed':      'Yeux fermés',
    'blink':            'Clignements',
    'hand_movement':    'Mouvement des mains',
    'flashing_stimuli': 'Stimulis clignotant',
}


def _load_recordings() -> list[dict]:
    """Charge tous les enregistrements depuis recordings/*.json."""
    records = []
    if not os.path.isdir(_RECORDINGS_DIR):
        return records
    for fname in os.listdir(_RECORDINGS_DIR):
        if not fname.endswith('.json'):
            continue
        json_path = os.path.join(_RECORDINGS_DIR, fname)
        npy_path  = json_path.replace('.json', '.npy')
        edf_path  = json_path.replace('.json', '.edf')
        if not os.path.isfile(npy_path) and not os.path.isfile(edf_path):
            continue
        try:
            with open(json_path, encoding='utf-8') as f:
                meta = json.load(f)
        except Exception:
            continue
        # Préférer EDF si disponible
        meta['_npy_path'] = edf_path if os.path.isfile(edf_path) else npy_path
        records.append(meta)
    return records


class RecordingBrowserDialog(QDialog):
    """Dialog de sélection d'un enregistrement avec filtres et tri."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle('Choisir un enregistrement')
        self.setMinimumWidth(700)
        self.setMinimumHeight(450)

        self._all_records: list[dict] = _load_recordings()
        self._selected_meta: dict     = {}
        self._selected_npy:  str      = ''

        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(16, 16, 16, 16)

        # ── Filtres ────────────────────────────────────────────────
        filter_layout = QHBoxLayout()

        filter_layout.addWidget(QLabel('Sujet :'))
        self._subject_box = QComboBox()
        self._subject_box.setMinimumWidth(120)
        filter_layout.addWidget(self._subject_box)
        filter_layout.addSpacing(16)

        filter_layout.addWidget(QLabel('Test :'))
        self._test_box = QComboBox()
        self._test_box.setMinimumWidth(160)
        filter_layout.addWidget(self._test_box)
        filter_layout.addSpacing(16)

        filter_layout.addWidget(QLabel('Tri :'))
        self._sort_box = QComboBox()
        self._sort_box.addItems(['Date (récent)', 'Durée (longue)', 'Durée (courte)'])
        self._sort_box.setMinimumWidth(140)
        filter_layout.addWidget(self._sort_box)
        filter_layout.addStretch()

        layout.addLayout(filter_layout)

        # ── Table ──────────────────────────────────────────────────
        self._table = QTableWidget(0, 5)
        self._table.setHorizontalHeaderLabels(['Sujet', 'Test', 'Date', 'Durée (s)', '●'])
        self._table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self._table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self._table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self._table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self._table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeToContents)
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._table.setSelectionBehavior(QTableWidget.SelectRows)
        self._table.setSelectionMode(QTableWidget.SingleSelection)
        self._table.verticalHeader().setVisible(False)
        self._table.doubleClicked.connect(self._on_double_click)
        layout.addWidget(self._table)

        # ── Boutons ────────────────────────────────────────────────
        self._buttons = QDialogButtonBox()
        self._btn_ok = self._buttons.addButton('Analyser', QDialogButtonBox.AcceptRole)
        self._buttons.addButton(QDialogButtonBox.Cancel)
        self._buttons.accepted.connect(self._on_accept)
        self._buttons.rejected.connect(self.reject)
        layout.addWidget(self._buttons)

        # Remplir les filtres et la table
        self._populate_filters()
        self._refresh_table()

        self._subject_box.currentIndexChanged.connect(self._refresh_table)
        self._test_box.currentIndexChanged.connect(self._refresh_table)
        self._sort_box.currentIndexChanged.connect(self._refresh_table)

    # ── Filtres ────────────────────────────────────────────────────

    def _populate_filters(self) -> None:
        subjects  = sorted({r.get('subject', '') for r in self._all_records if r.get('subject')})
        test_ids  = sorted({r.get('test_type', '') for r in self._all_records if r.get('test_type')})

        self._subject_box.addItem('Tous')
        self._subject_box.addItems(subjects)

        self._test_box.addItem('Tous')
        for tid in test_ids:
            label = _TEST_TYPE_LABELS.get(tid, tid)
            self._test_box.addItem(label, userData=tid)

    # ── Table ──────────────────────────────────────────────────────

    def _filtered_sorted(self) -> list[dict]:
        subj_filter = self._subject_box.currentText()
        test_filter = self._test_box.currentData()   # None when "Tous"
        sort_key    = self._sort_box.currentIndex()  # 0=date, 1=dur↓, 2=dur↑

        records = list(self._all_records)

        if subj_filter != 'Tous':
            records = [r for r in records if r.get('subject') == subj_filter]
        if test_filter is not None:
            records = [r for r in records if r.get('test_type') == test_filter]

        if sort_key == 0:
            records.sort(key=lambda r: r.get('timestamp', ''), reverse=True)
        elif sort_key == 1:
            records.sort(key=lambda r: r.get('duration_sec', 0), reverse=True)
        else:
            records.sort(key=lambda r: r.get('duration_sec', 0))

        return records

    def _refresh_table(self) -> None:
        records = self._filtered_sorted()
        self._table.setRowCount(len(records))
        self._table.setProperty('_records', records)  # keep reference

        for row, r in enumerate(records):
            ts = r.get('timestamp', '')
            try:
                dt = datetime.strptime(ts, '%Y%m%d_%H%M%S')
                date_str = dt.strftime('%d/%m %H:%M')
            except ValueError:
                date_str = ts

            tt_id    = r.get('test_type', '')
            tt_label = _TEST_TYPE_LABELS.get(tt_id, r.get('test_type_label', tt_id))
            n_events = max(0, len(r.get('events', [])) - 1)  # exclude auto "début"

            items = [
                r.get('subject', ''),
                tt_label,
                date_str,
                str(r.get('duration_sec', '')),
                str(n_events) if n_events else '—',
            ]
            for col, text in enumerate(items):
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignCenter if col >= 2 else Qt.AlignLeft | Qt.AlignVCenter)
                self._table.setItem(row, col, item)

        if records:
            self._table.selectRow(0)

    def _current_record(self) -> dict | None:
        row = self._table.currentRow()
        records = self._table.property('_records')
        if records and 0 <= row < len(records):
            return records[row]
        return None

    # ── Actions ────────────────────────────────────────────────────

    def _on_accept(self) -> None:
        rec = self._current_record()
        if rec is None:
            return
        self._selected_meta = rec
        self._selected_npy  = rec['_npy_path']
        self.accept()

    def _on_double_click(self) -> None:
        self._on_accept()

    # ── Résultats ──────────────────────────────────────────────────

    @property
    def selected_npy_path(self) -> str:
        return self._selected_npy

    @property
    def selected_meta(self) -> dict:
        return self._selected_meta
