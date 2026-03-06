from PyQt5.QtCore import QThread, pyqtSignal

from src.models.signal_data import SignalData


class EpochWorker(QThread):
    result_ready = pyqtSignal(object)   # EpochData
    error        = pyqtSignal(str)

    def __init__(self, signal_data: SignalData, tmin: float, tmax: float, parent=None):
        super().__init__(parent)
        self._signal_data = signal_data
        self._tmin = tmin
        self._tmax = tmax

    def run(self):
        try:
            from src.services.eeg_epoch_service import EEGEpochService
            result = EEGEpochService.extract(self._signal_data, self._tmin, self._tmax)
            self.result_ready.emit(result)
        except Exception as exc:
            self.error.emit(str(exc))
