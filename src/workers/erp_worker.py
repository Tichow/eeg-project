from PyQt5.QtCore import QThread, pyqtSignal

from src.models.epoch_data import EpochData


class ERPWorker(QThread):
    result_ready = pyqtSignal(object)   # ERPData
    error = pyqtSignal(str)

    def __init__(
        self,
        epoch_data: EpochData,
        selected_ch_indices: list[int],
        baseline_correction: bool,
        baseline_tmin: float,
        baseline_tmax: float,
        parent=None,
    ):
        super().__init__(parent)
        self._epoch_data = epoch_data
        self._selected_ch_indices = selected_ch_indices
        self._baseline_correction = baseline_correction
        self._baseline_tmin = baseline_tmin
        self._baseline_tmax = baseline_tmax

    def run(self):
        from src.services.eeg_erp_service import EEGERPService
        try:
            result = EEGERPService.compute(
                self._epoch_data,
                self._selected_ch_indices,
                self._baseline_correction,
                self._baseline_tmin,
                self._baseline_tmax,
            )
            self.result_ready.emit(result)
        except Exception as e:
            self.error.emit(str(e))
