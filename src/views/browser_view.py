from PyQt5.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QListWidget,
    QListWidgetItem, QTableWidget, QTableWidgetItem, QGroupBox,
    QLineEdit, QFileDialog, QProgressBar, QSplitter, QAbstractItemView,
    QHeaderView, QSizePolicy,
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

from src.views.base_view import BaseView
from src.constants.eeg_constants import DEFAULT_DATA_PATH


class BrowserView(BaseView):
    def setup_ui(self):
        self._worker = None
        self._run_row: dict[int, int] = {}

        root = QVBoxLayout(self)
        root.setContentsMargins(40, 30, 40, 20)
        root.setSpacing(16)

        # Header
        header = QHBoxLayout()
        back_btn = QPushButton("← Retour")
        back_btn.setFixedWidth(100)
        back_btn.setCursor(Qt.PointingHandCursor)
        back_btn.setStyleSheet("border: none; color: #4a90d9; font-size: 13px;")
        back_btn.clicked.connect(lambda: self.navigate.emit("home"))
        header.addWidget(back_btn)
        header.addStretch()

        title = QLabel("Explorateur de données")
        font = QFont()
        font.setPointSize(18)
        font.setBold(True)
        title.setFont(font)
        header.addWidget(title)
        header.addStretch()
        root.addLayout(header)

        # Path row
        path_group = QGroupBox("Dossier de données")
        path_layout = QHBoxLayout(path_group)
        self._path_edit = QLineEdit(DEFAULT_DATA_PATH)
        path_layout.addWidget(self._path_edit)

        browse_btn = QPushButton("Parcourir…")
        browse_btn.setFixedWidth(90)
        browse_btn.clicked.connect(self._browse_path)
        path_layout.addWidget(browse_btn)

        refresh_btn = QPushButton("Actualiser")
        refresh_btn.setFixedWidth(90)
        refresh_btn.clicked.connect(self._on_refresh)
        path_layout.addWidget(refresh_btn)
        root.addWidget(path_group)

        # Main splitter
        splitter = QSplitter(Qt.Horizontal)
        splitter.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Left panel — subjects
        left = QGroupBox("Sujets")
        left_layout = QVBoxLayout(left)
        self._subject_list = QListWidget()
        self._subject_list.setFixedWidth(150)
        self._subject_list.itemClicked.connect(self._on_subject_selected)
        left_layout.addWidget(self._subject_list)
        self._subject_count = QLabel("")
        self._subject_count.setStyleSheet("color: #888; font-size: 11px;")
        left_layout.addWidget(self._subject_count)
        splitter.addWidget(left)

        # Right panel — files
        self._files_group = QGroupBox("Fichiers")
        right_layout = QVBoxLayout(self._files_group)

        self._file_table = QTableWidget(0, 7)
        self._file_table.setHorizontalHeaderLabels(
            ["Run", "Description", "Durée", "Fréquence", "Canaux", "Annotations", "Taille"]
        )
        self._file_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._file_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._file_table.verticalHeader().setVisible(False)
        hh = self._file_table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.Fixed);        self._file_table.setColumnWidth(0, 50)
        hh.setSectionResizeMode(1, QHeaderView.Stretch)
        hh.setSectionResizeMode(2, QHeaderView.Fixed);        self._file_table.setColumnWidth(2, 70)
        hh.setSectionResizeMode(3, QHeaderView.Fixed);        self._file_table.setColumnWidth(3, 80)
        hh.setSectionResizeMode(4, QHeaderView.Fixed);        self._file_table.setColumnWidth(4, 65)
        hh.setSectionResizeMode(5, QHeaderView.Fixed);        self._file_table.setColumnWidth(5, 90)
        hh.setSectionResizeMode(6, QHeaderView.Fixed);        self._file_table.setColumnWidth(6, 70)
        right_layout.addWidget(self._file_table)

        self._header_progress = QProgressBar()
        self._header_progress.setVisible(False)
        self._header_progress.setFixedHeight(14)
        right_layout.addWidget(self._header_progress)

        splitter.addWidget(self._files_group)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        root.addWidget(splitter)

        # Status bar
        self._status = QLabel("Prêt")
        self._status.setStyleSheet("color: #888; font-size: 11px;")
        root.addWidget(self._status)

    def on_navigate_to(self):
        if self._subject_list.count() == 0:
            self._on_refresh()

    # ── Path ─────────────────────────────────────────────────────────────────

    def _browse_path(self):
        path = QFileDialog.getExistingDirectory(
            self, "Choisir le dossier de données", self._path_edit.text()
        )
        if path:
            self._path_edit.setText(path)
            self._on_refresh()

    # ── Subject scan ──────────────────────────────────────────────────────────

    def _on_refresh(self):
        from src.workers.browser_worker import BrowserWorker
        self._stop_worker()
        self._subject_list.clear()
        self._file_table.setRowCount(0)
        self._subject_count.setText("")
        self._files_group.setTitle("Fichiers")
        self._status.setText("Scan en cours…")

        self._worker = BrowserWorker(self._path_edit.text())
        self._worker.subjects_ready.connect(self._on_subjects_ready)
        self._worker.error.connect(self._on_error)
        self._worker.finished.connect(self._on_scan_finished)
        self._worker.start()

    def _on_subjects_ready(self, subjects: list):
        for s in subjects:
            item = QListWidgetItem(f"S{s:03d}")
            item.setData(Qt.UserRole, s)
            self._subject_list.addItem(item)
        n = len(subjects)
        self._subject_count.setText(f"{n} sujet{'s' if n > 1 else ''} trouvé{'s' if n > 1 else ''}")
        if n > 0:
            self._subject_list.setCurrentRow(0)
            self._on_subject_selected(self._subject_list.item(0))

    def _on_scan_finished(self):
        if self._subject_list.count() == 0:
            self._status.setText("Aucun sujet trouvé dans ce dossier.")

    # ── File listing ──────────────────────────────────────────────────────────

    def _on_subject_selected(self, item: QListWidgetItem):
        from src.workers.browser_worker import BrowserWorker
        subject = item.data(Qt.UserRole)
        self._stop_worker()
        self._file_table.setRowCount(0)
        self._run_row.clear()
        self._files_group.setTitle(f"Fichiers — S{subject:03d}")
        self._status.setText(f"Chargement S{subject:03d}…")

        self._worker = BrowserWorker(self._path_edit.text(), subject=subject)
        self._worker.files_skeleton.connect(self._on_files_skeleton)
        self._worker.file_header_done.connect(self._on_file_header_done)
        self._worker.finished.connect(self._on_files_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_files_skeleton(self, files: list):
        self._file_table.setRowCount(len(files))
        self._header_progress.setRange(0, len(files))
        self._header_progress.setValue(0)
        self._header_progress.setVisible(True)
        for row, info in enumerate(files):
            self._run_row[info.run] = row
            self._set_cell(row, 0, f"R{info.run:02d}", Qt.AlignCenter)
            self._set_cell(row, 1, info.description)
            self._set_cell(row, 2, "—", Qt.AlignRight)
            self._set_cell(row, 3, "—", Qt.AlignRight)
            self._set_cell(row, 4, "—", Qt.AlignRight)
            self._set_cell(row, 5, "—", Qt.AlignRight)
            self._set_cell(row, 6, self._fmt_size(info.size_bytes), Qt.AlignRight)

    def _on_file_header_done(self, run: int, duration_s: float, sfreq: float, nchan: int, n_ann: int):
        row = self._run_row.get(run)
        if row is None:
            return
        self._set_cell(row, 2, f"{duration_s:.0f} s", Qt.AlignRight)
        self._set_cell(row, 3, f"{sfreq:.0f} Hz", Qt.AlignRight)
        self._set_cell(row, 4, str(nchan), Qt.AlignRight)
        self._set_cell(row, 5, str(n_ann), Qt.AlignRight)
        self._header_progress.setValue(self._header_progress.value() + 1)

    def _on_files_finished(self):
        self._header_progress.setVisible(False)
        n = self._file_table.rowCount()
        subject_text = self._files_group.title().replace("Fichiers — ", "")
        self._status.setText(f"{n} fichier{'s' if n > 1 else ''} — {subject_text}")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _on_error(self, message: str):
        self._status.setText(f"Erreur : {message}")
        self._header_progress.setVisible(False)

    def _stop_worker(self):
        if self._worker is not None:
            try:
                self._worker.subjects_ready.disconnect()
                self._worker.files_skeleton.disconnect()
                self._worker.file_header_done.disconnect()
                self._worker.finished.disconnect()
                self._worker.error.disconnect()
            except (RuntimeError, TypeError):
                pass
            self._worker.quit()
            self._worker = None

    def _set_cell(self, row: int, col: int, text: str, alignment: Qt.AlignmentFlag = Qt.AlignLeft):
        item = QTableWidgetItem(text)
        item.setTextAlignment(alignment | Qt.AlignVCenter)
        self._file_table.setItem(row, col, item)

    @staticmethod
    def _fmt_size(n_bytes: int) -> str:
        if n_bytes >= 1_048_576:
            return f"{n_bytes / 1_048_576:.1f} MB"
        return f"{n_bytes / 1024:.0f} KB"
