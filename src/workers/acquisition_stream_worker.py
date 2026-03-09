from __future__ import annotations

from PyQt5.QtCore import QThread, pyqtSignal


class AcquisitionStreamWorker(QThread):
    """QThread that continuously reads chunks from a BrainFlow board.

    Signals
    -------
    chunk_ready(object)
        Emitted every ~100 ms with a np.ndarray of shape (8, n_samples) in Volts.
    error(str)
        Emitted on any exception in the acquisition loop.
    """

    chunk_ready = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, board, parent=None):
        super().__init__(parent)
        self._board = board
        self._running = False

    def run(self) -> None:
        from src.services.acquisition_service import AcquisitionService

        self._running = True
        try:
            while self._running:
                self.msleep(100)
                chunk = AcquisitionService.get_chunk(self._board)
                if chunk.shape[1] > 0:
                    self.chunk_ready.emit(chunk)
        except Exception as exc:  # noqa: BLE001
            self.error.emit(str(exc))

    def stop(self) -> None:
        """Request the loop to stop. Caller must call wait() before setting to None."""
        self._running = False
