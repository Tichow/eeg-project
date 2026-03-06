from PyQt5.QtWidgets import QMainWindow, QStackedWidget
from PyQt5.QtCore import pyqtSlot


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EEG Analysis")
        self.setMinimumSize(900, 600)

        self._stack = QStackedWidget()
        self.setCentralWidget(self._stack)

        self._views: dict = {}
        self._setup_views()
        self._navigate("home")

    def _setup_views(self):
        from src.views.home_view import HomeView
        from src.views.download_view import DownloadView

        self._register_view("home", HomeView())
        self._register_view("download", DownloadView())

    def _register_view(self, name: str, view):
        view.navigate.connect(self._navigate)
        self._stack.addWidget(view)
        self._views[name] = view

    @pyqtSlot(str)
    def _navigate(self, name: str):
        view = self._views.get(name)
        if view is not None:
            self._stack.setCurrentWidget(view)
            if hasattr(view, "on_navigate_to"):
                view.on_navigate_to()
