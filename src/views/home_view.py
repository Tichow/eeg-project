from PyQt5.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFrame, QSizePolicy
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

from src.views.base_view import BaseView


class HomeView(BaseView):
    def setup_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(60, 60, 60, 60)
        root.setSpacing(40)

        # Title
        title = QLabel("EEG Analysis")
        title.setAlignment(Qt.AlignCenter)
        font = QFont()
        font.setPointSize(28)
        font.setBold(True)
        title.setFont(font)
        root.addWidget(title)

        subtitle = QLabel("PhysioNet EEGMMIDB — 109 sujets · 64 canaux · 160 Hz")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet("color: #888888; font-size: 13px;")
        root.addWidget(subtitle)

        # Separator
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet("color: #dddddd;")
        root.addWidget(line)

        # Feature buttons grid
        grid = QHBoxLayout()
        grid.setSpacing(20)
        grid.addStretch()
        grid.addWidget(self._make_card(
            title="Télécharger les données",
            description="Récupère les fichiers EDF\ndepuis PhysioNet via MNE",
            target="download",
        ))
        grid.addWidget(self._make_card(
            title="Explorer les données",
            description="Navigue parmi les fichiers\nEDF téléchargés localement",
            target="browser",
        ))
        grid.addWidget(self._make_card(
            title="Acquisition",
            description="Enregistrer une session\navec le Cyton",
            target="acquisition",
        ))
        grid.addStretch()
        root.addLayout(grid)

        root.addStretch()

    def _make_card(self, title: str, description: str, target: str) -> QPushButton:
        btn = QPushButton()
        btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        btn.setFixedSize(200, 160)
        btn.setCursor(Qt.PointingHandCursor)
        btn.setStyleSheet("""
            QPushButton {
                background-color: #f5f5f5;
                border: 1px solid #e0e0e0;
                border-radius: 10px;
                padding: 20px;
                text-align: center;
                font-size: 14px;
                font-weight: bold;
                color: #333333;
            }
            QPushButton:hover {
                background-color: #e8f0fe;
                border-color: #4a90d9;
                color: #1a5fa8;
            }
            QPushButton:pressed {
                background-color: #d0e3f7;
            }
        """)
        btn.setText(f"{title}\n\n{description}")
        btn.clicked.connect(lambda: self.navigate.emit(target))
        return btn
