from PyQt5.QtCore import QThread, pyqtSignal

from src.models.epoch_data import EpochData


class FrequencyWorker(QThread):
    result_ready = pyqtSignal(object)  # FrequencyData
    error = pyqtSignal(str)

    def __init__(
        self,
        epoch_data: EpochData,
        mode: str,
        selected_ch_indices: list[int],
        fmin: float,
        fmax: float,
        nperseg: int,
        band_low: float,
        band_high: float,
        baseline_label: str,
    ):
        super().__init__()
        self._epoch_data = epoch_data
        self._mode = mode
        self._selected_ch_indices = selected_ch_indices
        self._fmin = fmin
        self._fmax = fmax
        self._nperseg = nperseg
        self._band_low = band_low
        self._band_high = band_high
        self._baseline_label = baseline_label

    def run(self):
        from src.services.eeg_frequency_service import EEGFrequencyService
        try:
            if self._mode == "psd":
                result = EEGFrequencyService.compute_psd(
                    self._epoch_data,
                    self._selected_ch_indices,
                    self._fmin,
                    self._fmax,
                    self._nperseg,
                )
            else:
                result = EEGFrequencyService.compute_erders(
                    self._epoch_data,
                    self._selected_ch_indices,
                    self._band_low,
                    self._band_high,
                    self._baseline_label,
                    self._nperseg,
                )
            self.result_ready.emit(result)
        except Exception as e:
            self.error.emit(str(e))
