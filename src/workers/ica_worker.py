from PyQt5.QtCore import QThread, pyqtSignal

from src.models.epoch_data import EpochData


class ICAWorker(QThread):
    result_ready = pyqtSignal(object, list)  # (ica, bad_components: list[int])
    error = pyqtSignal(str)

    def __init__(self, epoch_data: EpochData, n_components: int):
        super().__init__()
        self._epoch_data = epoch_data
        self._n_components = n_components

    def run(self):
        from src.services.eeg_artifact_service import EEGArtifactService
        try:
            ica, bad_components = EEGArtifactService.fit_ica(self._epoch_data, self._n_components)
            self.result_ready.emit(ica, bad_components)
        except Exception as e:
            self.error.emit(str(e))
