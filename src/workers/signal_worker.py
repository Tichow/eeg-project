from PyQt5.QtCore import QThread, pyqtSignal

from src.services.eeg_signal_service import EEGSignalService


class SignalWorker(QThread):
    data_ready = pyqtSignal(object)   # SignalData
    error      = pyqtSignal(str)

    def __init__(self, path: str, parent=None):
        super().__init__(parent)
        self._path = path

    def run(self):
        try:
            signal_data = EEGSignalService.load_signal(self._path)
            self.data_ready.emit(signal_data)
        except Exception as exc:
            self.error.emit(str(exc))
