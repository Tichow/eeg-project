from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from src.views.base_view import BaseView
from src.views.widgets.head_map_widget import HeadMapWidget

_BTN_BLUE = """
    QPushButton {
        background-color: #4a90d9;
        color: white;
        border: none;
        border-radius: 6px;
        padding: 6px 14px;
        font-size: 13px;
    }
    QPushButton:hover { background-color: #357abd; }
    QPushButton:pressed { background-color: #2a6099; }
    QPushButton:disabled { background-color: #aaaaaa; }
"""

_BTN_GRAY = """
    QPushButton {
        background-color: #555555;
        color: white;
        border: none;
        border-radius: 6px;
        padding: 6px 14px;
        font-size: 13px;
    }
    QPushButton:hover { background-color: #444444; }
    QPushButton:pressed { background-color: #333333; }
"""

_BTN_RED = """
    QPushButton {
        background-color: #c0392b;
        color: white;
        border: none;
        border-radius: 6px;
        padding: 6px 14px;
        font-size: 13px;
    }
    QPushButton:hover { background-color: #a93226; }
    QPushButton:pressed { background-color: #922b21; }
    QPushButton:disabled { background-color: #aaaaaa; }
"""


class PresetView(BaseView):
    """Dedicated page for creating and managing electrode channel presets."""

    def setup_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        root.addWidget(self._build_header())

        body = QHBoxLayout()
        body.setContentsMargins(16, 16, 16, 16)
        body.setSpacing(16)
        body.addWidget(self._build_head_map_panel(), stretch=2)
        body.addWidget(self._build_sidebar(), stretch=0)

        body_widget = QWidget()
        body_widget.setLayout(body)
        root.addWidget(body_widget, stretch=1)

        self._refresh_preset_list()

    def _build_header(self) -> QWidget:
        header = QWidget()
        header.setFixedHeight(50)
        header.setStyleSheet("background: #2c2c2c;")
        layout = QHBoxLayout(header)
        layout.setContentsMargins(14, 0, 14, 0)

        back_btn = QPushButton("← Retour")
        back_btn.setStyleSheet(_BTN_GRAY)
        back_btn.setCursor(Qt.PointingHandCursor)
        back_btn.setFixedWidth(90)
        back_btn.clicked.connect(lambda: self.navigate.emit("home"))

        title = QLabel("Préréglages d'électrodes")
        title.setStyleSheet("color: white; font-size: 15px; font-weight: bold;")

        layout.addWidget(back_btn)
        layout.addSpacing(16)
        layout.addWidget(title)
        layout.addStretch()
        return header

    def _build_head_map_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        instr = QLabel(
            "Cliquez sur une électrode pour la sélectionner, "
            "puis appuyez sur 1–8 pour l'assigner à un canal. "
            "Appuyez sur 0 ou Suppr pour effacer."
        )
        instr.setWordWrap(True)
        instr.setStyleSheet("color: #666666; font-size: 12px;")
        layout.addWidget(instr)

        self._head_map = HeadMapWidget()
        self._head_map.assignment_changed.connect(self._on_assignment_changed)
        layout.addWidget(self._head_map, stretch=1)
        return panel

    def _build_sidebar(self) -> QWidget:
        sidebar = QWidget()
        sidebar.setFixedWidth(260)
        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        layout.addWidget(self._build_assignment_group())
        layout.addWidget(self._build_preset_group())
        layout.addStretch()
        return sidebar

    def _build_assignment_group(self) -> QGroupBox:
        grp = QGroupBox("Assignation actuelle")
        layout = QVBoxLayout(grp)
        layout.setSpacing(4)
        self._ch_labels: list[QLabel] = []
        for i in range(1, 9):
            row = QHBoxLayout()
            ch_lbl = QLabel(f"CH{i}")
            ch_lbl.setFixedWidth(32)
            ch_lbl.setStyleSheet("color: #888888; font-size: 12px;")
            val_lbl = QLabel("—")
            val_lbl.setStyleSheet("font-weight: bold; font-size: 13px;")
            row.addWidget(ch_lbl)
            row.addWidget(val_lbl, 1)
            layout.addLayout(row)
            self._ch_labels.append(val_lbl)
        return grp

    def _build_preset_group(self) -> QGroupBox:
        grp = QGroupBox("Préréglages sauvegardés")
        layout = QVBoxLayout(grp)
        layout.setSpacing(6)

        self._preset_combo = QComboBox()
        self._preset_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        layout.addWidget(self._preset_combo)

        btn_load = QPushButton("Charger")
        btn_load.setStyleSheet(_BTN_BLUE)
        btn_load.setCursor(Qt.PointingHandCursor)
        btn_load.clicked.connect(self._on_load_preset)
        layout.addWidget(btn_load)

        btn_save = QPushButton("Sauvegarder sous…")
        btn_save.setStyleSheet(_BTN_BLUE)
        btn_save.setCursor(Qt.PointingHandCursor)
        btn_save.clicked.connect(self._on_save_preset)
        layout.addWidget(btn_save)

        self._btn_delete = QPushButton("Supprimer")
        self._btn_delete.setStyleSheet(_BTN_RED)
        self._btn_delete.setCursor(Qt.PointingHandCursor)
        self._btn_delete.clicked.connect(self._on_delete_preset)
        layout.addWidget(self._btn_delete)

        btn_clear = QPushButton("Effacer tout")
        btn_clear.setStyleSheet(_BTN_GRAY)
        btn_clear.setCursor(Qt.PointingHandCursor)
        btn_clear.clicked.connect(self._on_clear_all)
        layout.addWidget(btn_clear)

        return grp

    # ── Slots ─────────────────────────────────────────────────────────────────

    def _on_assignment_changed(self, assignment: dict) -> None:
        for i, lbl in enumerate(self._ch_labels, start=1):
            name = assignment.get(i, "")
            lbl.setText(name if name else "—")

    def _on_load_preset(self) -> None:
        from src.services.preset_service import PresetService
        name = self._preset_combo.currentText()
        if not name:
            return
        preset = PresetService.get_preset(name)
        if preset:
            self._head_map.set_assignment(preset.channels)

    def _on_save_preset(self) -> None:
        from src.services.preset_service import PresetService
        from src.models.electrode_preset import ElectrodePreset

        assignment = self._head_map.get_assignment()
        if not assignment:
            QMessageBox.warning(self, "Aucune assignation", "Assignez au moins un canal avant de sauvegarder.")
            return

        name, ok = QInputDialog.getText(self, "Sauvegarder le préréglage", "Nom du préréglage :")
        if not ok or not name.strip():
            return

        preset = ElectrodePreset(name=name.strip(), channels=assignment)
        PresetService.save_preset(preset)
        self._refresh_preset_list()

        idx = self._preset_combo.findText(name.strip())
        if idx >= 0:
            self._preset_combo.setCurrentIndex(idx)

    def _on_delete_preset(self) -> None:
        from src.services.preset_service import PresetService
        name = self._preset_combo.currentText()
        if not name:
            return
        reply = QMessageBox.question(
            self,
            "Confirmer la suppression",
            f"Supprimer le préréglage « {name} » ?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            PresetService.delete_preset(name)
            self._refresh_preset_list()

    def _on_clear_all(self) -> None:
        self._head_map.clear_all()

    def _refresh_preset_list(self) -> None:
        from src.services.preset_service import PresetService
        current = self._preset_combo.currentText()
        self._preset_combo.blockSignals(True)
        self._preset_combo.clear()
        for p in PresetService.load_all():
            self._preset_combo.addItem(p.name)
        idx = self._preset_combo.findText(current)
        if idx >= 0:
            self._preset_combo.setCurrentIndex(idx)
        self._preset_combo.blockSignals(False)
        has_presets = self._preset_combo.count() > 0
        self._btn_delete.setEnabled(has_presets)

    def on_navigate_to(self) -> None:
        self._refresh_preset_list()
