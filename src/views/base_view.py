from abc import abstractmethod
from PyQt5.QtWidgets import QWidget
from PyQt5.QtCore import pyqtSignal


class BaseView(QWidget):
    navigate = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()

    @abstractmethod
    def setup_ui(self):
        """Build and arrange all widgets for this view."""

    def on_navigate_to(self):
        """Called each time this view becomes the active page. Override if needed."""
