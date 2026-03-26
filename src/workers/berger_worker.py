from PyQt5.QtCore import QThread, pyqtSignal


class BergerWorker(QThread):
    result_ready = pyqtSignal(object)   # BergerResult
    error = pyqtSignal(str)

    def __init__(self, path_open: str, path_closed: str):
        super().__init__()
        self._path_open = path_open
        self._path_closed = path_closed

    def run(self):
        from src.services.eeg_berger_service import EEGBergerService
        try:
            result = EEGBergerService.compute(self._path_open, self._path_closed)
            self.result_ready.emit(result)
        except Exception as exc:
            self.error.emit(str(exc))
