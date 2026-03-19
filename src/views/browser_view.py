import pyqtgraph as pg
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QFont
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from src.constants.eeg_constants import DEFAULT_DATA_PATH
from src.views.base_view import BaseView


class BrowserView(BaseView):
    def setup_ui(self):
        from src.services.favorites_service import FavoritesService
        self._worker = None
        self._berger_worker = None
        self._run_row: dict[int, int] = {}
        self._run_info: dict[int, object] = {}   # run → EdfFileInfo
        self._signal_view = None
        self._favorites = FavoritesService()

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

        # Filter toolbar
        filter_row = QHBoxLayout()
        filter_row.setContentsMargins(0, 0, 0, 4)
        self._fav_filter_btn = QPushButton("☆ Favoris")
        self._fav_filter_btn.setCheckable(True)
        self._fav_filter_btn.setFixedWidth(90)
        self._fav_filter_btn.setCursor(Qt.PointingHandCursor)
        self._fav_filter_btn.toggled.connect(self._apply_filter)
        filter_row.addStretch()
        filter_row.addWidget(self._fav_filter_btn)
        right_layout.addLayout(filter_row)

        self._file_table = QTableWidget(0, 8)
        self._file_table.setHorizontalHeaderLabels(
            ["Run", "Description", "Durée", "Fréquence", "Canaux", "Annotations", "Taille", "★"]
        )
        self._file_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._file_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._file_table.verticalHeader().setVisible(False)
        self._file_table.cellDoubleClicked.connect(self._on_row_double_clicked)
        self._file_table.cellClicked.connect(self._on_cell_clicked)
        hh = self._file_table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.Fixed);        self._file_table.setColumnWidth(0, 50)
        hh.setSectionResizeMode(1, QHeaderView.Stretch)
        hh.setSectionResizeMode(2, QHeaderView.Fixed);        self._file_table.setColumnWidth(2, 70)
        hh.setSectionResizeMode(3, QHeaderView.Fixed);        self._file_table.setColumnWidth(3, 80)
        hh.setSectionResizeMode(4, QHeaderView.Fixed);        self._file_table.setColumnWidth(4, 65)
        hh.setSectionResizeMode(5, QHeaderView.Fixed);        self._file_table.setColumnWidth(5, 110)
        hh.setSectionResizeMode(6, QHeaderView.Fixed);        self._file_table.setColumnWidth(6, 70)
        hh.setSectionResizeMode(7, QHeaderView.Fixed);        self._file_table.setColumnWidth(7, 30)
        right_layout.addWidget(self._file_table)

        self._header_progress = QProgressBar()
        self._header_progress.setVisible(False)
        self._header_progress.setFixedHeight(14)
        right_layout.addWidget(self._header_progress)

        # Berger effect panel
        self._berger_panel = QWidget()
        berger_layout = QVBoxLayout(self._berger_panel)
        berger_layout.setContentsMargins(0, 4, 0, 0)
        berger_layout.setSpacing(4)

        badge_row = QHBoxLayout()
        badge_row.setSpacing(8)
        berger_title = QLabel("Effet Berger :")
        berger_title.setStyleSheet("color: #888; font-size: 11px;")
        badge_row.addWidget(berger_title)
        self._berger_label = QLabel("—")
        self._berger_label.setStyleSheet("font-size: 11px; padding: 2px 6px; border-radius: 3px;")
        badge_row.addWidget(self._berger_label)
        badge_row.addStretch()
        berger_layout.addLayout(badge_row)

        self._berger_plot = pg.PlotWidget(background="#1e1e1e")
        self._berger_plot.setFixedHeight(160)
        self._berger_plot.setLabel("bottom", "Fréquence (Hz)")
        self._berger_plot.setLabel("left", "µV²/Hz")
        self._berger_plot.setXRange(1, 30, padding=0)
        self._berger_plot.showGrid(x=True, y=True, alpha=0.2)

        alpha_region = pg.LinearRegionItem(
            [8, 13], movable=False, brush=pg.mkBrush(200, 200, 200, 30)
        )
        alpha_region.setZValue(-10)
        self._berger_plot.addItem(alpha_region)

        alpha_label = pg.TextItem("α", color=(180, 180, 180), anchor=(0.5, 1))
        alpha_label.setPos(10.5, 0)
        self._berger_plot.addItem(alpha_label)

        self._curve_open = self._berger_plot.plot(
            pen=pg.mkPen("#4a90d9", width=1.5, style=Qt.DashLine),
            name="R01 yeux ouverts",
        )
        self._curve_closed = self._berger_plot.plot(
            pen=pg.mkPen("#e74c3c", width=2),
            name="R02 yeux fermés",
        )

        legend = self._berger_plot.addLegend(offset=(10, 10))
        legend.addItem(self._curve_open, "R01 — yeux ouverts")
        legend.addItem(self._curve_closed, "R02 — yeux fermés")

        berger_layout.addWidget(self._berger_plot)
        self._berger_panel.setVisible(False)
        right_layout.addWidget(self._berger_panel)

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
            label = f"S{s:03d}" if isinstance(s, int) else str(s)
            item = QListWidgetItem(label)
            item.setData(Qt.UserRole, s)
            self._subject_list.addItem(item)
        self._refresh_subject_stars()
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
        label = f"S{subject:03d}" if isinstance(subject, int) else str(subject)
        self._stop_worker()
        self._stop_berger_worker()
        self._berger_panel.setVisible(False)
        self._file_table.setRowCount(0)
        self._run_row.clear()
        self._files_group.setTitle(f"Fichiers — {label}")
        self._status.setText(f"Chargement {label}…")

        self._worker = BrowserWorker(self._path_edit.text(), subject=subject)
        self._worker.files_skeleton.connect(self._on_files_skeleton)
        self._worker.file_header_done.connect(self._on_file_header_done)
        self._worker.finished.connect(self._on_files_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_files_skeleton(self, files: list):
        self._run_info.clear()
        self._file_table.setRowCount(len(files))
        self._header_progress.setRange(0, len(files))
        self._header_progress.setValue(0)
        self._header_progress.setVisible(True)
        for row, info in enumerate(files):
            self._run_row[info.run] = row
            self._run_info[info.run] = info
            self._set_cell(row, 0, f"R{info.run:02d}", Qt.AlignCenter)
            self._set_cell(row, 1, info.description)
            self._set_cell(row, 2, "—", Qt.AlignRight)
            self._set_cell(row, 3, "—", Qt.AlignRight)
            self._set_cell(row, 4, "—", Qt.AlignRight)
            self._set_cell(row, 5, "—", Qt.AlignRight)
            self._set_cell(row, 6, self._fmt_size(info.size_bytes), Qt.AlignRight)
            self._set_star_cell(row, info.path)
        if self._fav_filter_btn.isChecked():
            self._apply_filter(True)

    def _on_file_header_done(self, run: int, duration_s: float, sfreq: float, nchan: int, ann_counts: dict):
        row = self._run_row.get(run)
        if row is None:
            return
        self._set_cell(row, 2, f"{duration_s:.0f} s", Qt.AlignRight)
        self._set_cell(row, 3, f"{sfreq:.0f} Hz", Qt.AlignRight)
        self._set_cell(row, 4, str(nchan), Qt.AlignRight)
        ann_str = " ".join(f"{k}:{v}" for k, v in sorted(ann_counts.items())) or "0"
        self._set_cell(row, 5, ann_str, Qt.AlignRight)
        self._header_progress.setValue(self._header_progress.value() + 1)

    def _on_files_finished(self):
        self._header_progress.setVisible(False)
        n = self._file_table.rowCount()
        subject_text = self._files_group.title().replace("Fichiers — ", "")
        self._status.setText(f"{n} fichier{'s' if n > 1 else ''} — {subject_text}")
        self._launch_berger()

    # ── Berger effect check ───────────────────────────────────────────────────

    def _launch_berger(self):
        from src.workers.berger_worker import BergerWorker
        self._stop_berger_worker()
        info_open = self._run_info.get(1)
        info_closed = self._run_info.get(2)
        if info_open is None or info_closed is None:
            self._berger_panel.setVisible(False)
            return

        self._berger_panel.setVisible(True)
        self._berger_label.setText("Calcul en cours…")
        self._berger_label.setStyleSheet("font-size: 11px; padding: 2px 6px; color: #888;")
        self._curve_open.setData([], [])
        self._curve_closed.setData([], [])

        self._berger_worker = BergerWorker(info_open.path, info_closed.path)
        self._berger_worker.result_ready.connect(self._on_berger_result)
        self._berger_worker.error.connect(self._on_berger_error)
        self._berger_worker.start()

    def _on_berger_result(self, result):
        if self._berger_worker is not None:
            self._berger_worker.wait()
            self._berger_worker = None
        ch_str = "/".join(result.channels_used[:3])
        snr_sign = "+" if result.snr_closed >= 0 else ""
        self._berger_label.setText(
            f"{result.quality} — ×{result.ratio:.1f}  ({ch_str})"
            f"  |  SNR fermé: {snr_sign}{result.snr_closed:.1f} dB"
        )
        self._berger_label.setStyleSheet(
            f"font-size: 11px; padding: 2px 8px; border-radius: 3px; "
            f"background-color: {result.color}; color: white;"
        )
        self._curve_open.setData(result.freqs, result.psd_open)
        self._curve_closed.setData(result.freqs, result.psd_closed)
        alpha_mask = (result.freqs >= 8.0) & (result.freqs <= 13.0)
        y_max = max(result.psd_closed[alpha_mask].max(), result.psd_open[alpha_mask].max()) * 2.0
        self._berger_plot.setYRange(0, y_max, padding=0)

    def _on_berger_error(self, message: str):
        if self._berger_worker is not None:
            self._berger_worker.wait()
            self._berger_worker = None
        self._berger_label.setText("Indisponible")
        self._berger_label.setStyleSheet("font-size: 11px; padding: 2px 6px; color: #888;")
        self._curve_open.setData([], [])
        self._curve_closed.setData([], [])

    def _stop_berger_worker(self):
        if self._berger_worker is not None:
            try:
                self._berger_worker.result_ready.disconnect()
                self._berger_worker.error.disconnect()
            except (RuntimeError, TypeError):
                pass
            self._berger_worker.quit()
            self._berger_worker = None

    # ── Signal view wiring ────────────────────────────────────────────────────

    def set_signal_view(self, signal_view):
        self._signal_view = signal_view

    def _on_row_double_clicked(self, row: int, col: int):
        if col == 7 or self._signal_view is None:
            return
        run_item = self._file_table.item(row, 0)
        if run_item is None:
            return
        try:
            run = int(run_item.text()[1:])
        except (ValueError, IndexError):
            return
        info = self._run_info.get(run)
        if info is None:
            return
        self._signal_view.prepare(info.path, info.subject, info.run)
        self.navigate.emit("signal")

    # ── Favorites ─────────────────────────────────────────────────────────────

    def _set_star_cell(self, row: int, path: str):
        is_fav = self._favorites.is_favorite(path)
        item = QTableWidgetItem("★" if is_fav else "☆")
        item.setTextAlignment(Qt.AlignCenter | Qt.AlignVCenter)
        item.setForeground(QColor("#f39c12") if is_fav else QColor("#888888"))
        item.setData(Qt.UserRole, path)
        self._file_table.setItem(row, 7, item)

    def _on_cell_clicked(self, row: int, col: int):
        if col != 7:
            return
        item = self._file_table.item(row, 7)
        if item is None:
            return
        path = item.data(Qt.UserRole)
        if path:
            self._favorites.toggle(path)
            self._set_star_cell(row, path)
            self._refresh_subject_stars()
            if self._fav_filter_btn.isChecked():
                self._apply_filter(True)

    def _refresh_subject_stars(self):
        fav_subjects = self._favorites.favorite_subjects()
        for i in range(self._subject_list.count()):
            item = self._subject_list.item(i)
            subject = item.data(Qt.UserRole)
            label = f"S{subject:03d}" if isinstance(subject, int) else str(subject)
            has_fav = label in fav_subjects
            item.setText(("★ " if has_fav else "") + label)
            if has_fav:
                item.setForeground(QColor("#f39c12"))
            else:
                item.setData(Qt.ForegroundRole, None)

    def _apply_filter(self, active: bool):
        self._fav_filter_btn.setText("★ Favoris" if active else "☆ Favoris")
        for row in range(self._file_table.rowCount()):
            star_item = self._file_table.item(row, 7)
            if active and star_item:
                path = star_item.data(Qt.UserRole)
                self._file_table.setRowHidden(row, not self._favorites.is_favorite(path))
            else:
                self._file_table.setRowHidden(row, False)

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
