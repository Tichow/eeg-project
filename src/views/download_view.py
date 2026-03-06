import os
from PyQt5.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QSpinBox,
    QListWidget, QListWidgetItem, QLineEdit, QFileDialog,
    QProgressBar, QTextEdit, QGroupBox, QSizePolicy, QAbstractItemView
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

from src.views.base_view import BaseView
from src.constants.eeg_constants import (
    SUBJECT_MIN, SUBJECT_MAX, RUN_DESCRIPTIONS, DEFAULT_DATA_PATH
)
from src.models.download_config import DownloadConfig


class DownloadView(BaseView):
    def setup_ui(self):
        self._worker = None

        root = QVBoxLayout(self)
        root.setContentsMargins(40, 30, 40, 30)
        root.setSpacing(20)

        # Header
        header = QHBoxLayout()
        back_btn = QPushButton("← Retour")
        back_btn.setFixedWidth(100)
        back_btn.setCursor(Qt.PointingHandCursor)
        back_btn.setStyleSheet("border: none; color: #4a90d9; font-size: 13px;")
        back_btn.clicked.connect(lambda: self.navigate.emit("home"))
        header.addWidget(back_btn)
        header.addStretch()

        title = QLabel("Télécharger les données PhysioNet")
        font = QFont()
        font.setPointSize(18)
        font.setBold(True)
        title.setFont(font)
        header.addWidget(title)
        header.addStretch()
        root.addLayout(header)

        # Config row
        config_row = QHBoxLayout()
        config_row.setSpacing(20)
        config_row.addWidget(self._build_subject_group())
        config_row.addWidget(self._build_run_group())
        config_row.addWidget(self._build_path_group())
        root.addLayout(config_row)

        # Start button
        self._start_btn = QPushButton("Lancer le téléchargement")
        self._start_btn.setFixedHeight(44)
        self._start_btn.setCursor(Qt.PointingHandCursor)
        self._start_btn.setStyleSheet("""
            QPushButton {
                background-color: #4a90d9;
                color: white;
                border-radius: 6px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #357abd; }
            QPushButton:pressed { background-color: #2a6099; }
            QPushButton:disabled { background-color: #aaaaaa; }
        """)
        self._start_btn.clicked.connect(self._on_start)
        root.addWidget(self._start_btn)

        # Progress bar
        self._progress = QProgressBar()
        self._progress.setVisible(False)
        self._progress.setFixedHeight(20)
        root.addWidget(self._progress)

        # Log area
        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setVisible(False)
        self._log.setStyleSheet("font-family: monospace; font-size: 12px;")
        root.addWidget(self._log)

    def _build_subject_group(self) -> QGroupBox:
        group = QGroupBox("Sujets")
        layout = QVBoxLayout(group)

        row = QHBoxLayout()
        row.addWidget(QLabel("De :"))
        self._subject_from = QSpinBox()
        self._subject_from.setRange(SUBJECT_MIN, SUBJECT_MAX)
        self._subject_from.setValue(1)
        row.addWidget(self._subject_from)

        row.addWidget(QLabel("À :"))
        self._subject_to = QSpinBox()
        self._subject_to.setRange(SUBJECT_MIN, SUBJECT_MAX)
        self._subject_to.setValue(1)
        row.addWidget(self._subject_to)
        layout.addLayout(row)

        hint = QLabel(f"(S001 – S{SUBJECT_MAX:03d})")
        hint.setStyleSheet("color: #888; font-size: 11px;")
        layout.addWidget(hint)
        layout.addStretch()

        return group

    def _build_run_group(self) -> QGroupBox:
        group = QGroupBox("Runs")
        layout = QVBoxLayout(group)

        self._run_list = QListWidget()
        self._run_list.setSelectionMode(QAbstractItemView.MultiSelection)
        for run_num, desc in RUN_DESCRIPTIONS.items():
            item = QListWidgetItem(f"R{run_num:02d} — {desc}")
            item.setData(Qt.UserRole, run_num)
            self._run_list.addItem(item)
        self._run_list.item(0).setSelected(True)
        layout.addWidget(self._run_list)

        return group

    def _build_path_group(self) -> QGroupBox:
        group = QGroupBox("Dossier de destination")
        layout = QVBoxLayout(group)

        row = QHBoxLayout()
        self._path_edit = QLineEdit(DEFAULT_DATA_PATH)
        row.addWidget(self._path_edit)

        browse_btn = QPushButton("Parcourir…")
        browse_btn.setFixedWidth(90)
        browse_btn.clicked.connect(self._browse_path)
        row.addWidget(browse_btn)
        layout.addLayout(row)

        hint = QLabel("Les fichiers EDF seront organisés\npar sujet sous ce dossier.")
        hint.setStyleSheet("color: #888; font-size: 11px;")
        layout.addWidget(hint)
        layout.addStretch()

        return group

    def _browse_path(self):
        path = QFileDialog.getExistingDirectory(
            self, "Choisir le dossier de destination", self._path_edit.text()
        )
        if path:
            self._path_edit.setText(path)

    def _on_start(self):
        from src.workers.download_worker import DownloadWorker

        subject_from = self._subject_from.value()
        subject_to = self._subject_to.value()
        if subject_from > subject_to:
            self._append_log("Erreur : le sujet de départ doit être ≤ au sujet de fin.")
            return

        selected_runs = [
            item.data(Qt.UserRole)
            for item in self._run_list.selectedItems()
        ]
        if not selected_runs:
            self._append_log("Erreur : sélectionnez au moins un run.")
            return

        data_path = self._path_edit.text().strip() or DEFAULT_DATA_PATH

        config = DownloadConfig(
            subjects=list(range(subject_from, subject_to + 1)),
            runs=sorted(selected_runs),
            data_path=data_path,
        )

        self._start_btn.setEnabled(False)
        self._progress.setVisible(True)
        self._progress.setRange(0, len(config.subjects) * len(config.runs))
        self._progress.setValue(0)
        self._log.setVisible(True)
        self._log.clear()
        self._append_log(
            f"Démarrage : {len(config.subjects)} sujet(s), "
            f"{len(config.runs)} run(s), destination : {data_path}"
        )

        self._worker = DownloadWorker(config)
        self._worker.progress.connect(self._on_progress)
        self._worker.subject_done.connect(self._on_subject_done)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_progress(self, current: int, total: int, subject: int):
        self._progress.setValue(current)
        self._append_log(f"[{current}/{total}] S{subject:03d} — fichier téléchargé.")

    def _on_subject_done(self, subject: int, path: str):
        self._append_log(f"  → {path}")

    def _on_finished(self):
        self._progress.setValue(self._progress.maximum())
        self._append_log("Téléchargement terminé.")
        self._start_btn.setEnabled(True)
        self._worker = None

    def _on_error(self, message: str):
        self._append_log(f"Erreur : {message}")
        self._start_btn.setEnabled(True)
        self._worker = None

    def _append_log(self, text: str):
        self._log.append(text)
        self._log.verticalScrollBar().setValue(self._log.verticalScrollBar().maximum())
