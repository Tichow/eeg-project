from __future__ import annotations

from PyQt5.QtCore import QThread, pyqtSignal

from src.models.erp_data import ERPData
from src.models.frequency_data import FrequencyData


class TopoMapWorker(QThread):
    result_ready = pyqtSignal(object)  # TopoMapData
    error = pyqtSignal(str)

    def __init__(
        self,
        mode: str,
        erp_data: ERPData | None = None,
        freq_data: FrequencyData | None = None,
        tmin: float = 0.0,
        tmax: float = 0.5,
        fmin: float = 8.0,
        fmax: float = 13.0,
    ):
        super().__init__()
        self._mode = mode
        self._erp_data = erp_data
        self._freq_data = freq_data
        self._tmin = tmin
        self._tmax = tmax
        self._fmin = fmin
        self._fmax = fmax

    def run(self):
        from src.services.eeg_topomap_service import EEGTopoMapService
        try:
            if self._mode == "amplitude":
                result = EEGTopoMapService.compute_amplitude(
                    self._erp_data, self._tmin, self._tmax
                )
            else:
                result = EEGTopoMapService.compute_power(
                    self._freq_data, self._fmin, self._fmax
                )
            self.result_ready.emit(result)
        except Exception as exc:  # noqa: BLE001
            self.error.emit(str(exc))
