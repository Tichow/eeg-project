from PyQt5.QtCore import QThread, pyqtSignal

from src.models.signal_data import SignalData
from src.models.preprocess_config import PreprocessConfig
from src.services.eeg_preprocess_service import EEGPreprocessService


class PreprocessWorker(QThread):
    result_ready = pyqtSignal(object)  # SignalData
    error = pyqtSignal(str)

    def __init__(self, signal_data: SignalData, config: PreprocessConfig, parent=None):
        super().__init__(parent)
        self._signal_data = signal_data
        self._config = config

    def run(self):
        try:
            result = EEGPreprocessService.apply(self._signal_data, self._config)
            self.result_ready.emit(result)
        except Exception as exc:
            self.error.emit(str(exc))
