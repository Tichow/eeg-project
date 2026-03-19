from __future__ import annotations

import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal
from scipy.signal import butter, iirnotch, filtfilt, sosfiltfilt


class PredictionStreamWorker(QThread):
    """QThread that streams EEG from a BrainFlow board and runs predictions.

    Accumulates a 3-second ring buffer (750 samples at 250 Hz), applies causal
    bandpass 8-30 Hz + notch 50 Hz, then calls the loaded CSP+LDA pipeline
    every ~0.5 seconds.

    Signals
    -------
    prediction_ready(int, object)
        (label, probabilities_ndarray) after each prediction.
    buffer_status(int)
        Current fill level of the ring buffer (0 to ``window_samples``).
    error(str)
        Emitted on any exception in the acquisition/prediction loop.
    """

    prediction_ready = pyqtSignal(int, object)
    buffer_status = pyqtSignal(int)
    error = pyqtSignal(str)

    def __init__(
        self,
        board,
        pipeline,
        sfreq: float = 250.0,
        is_fbcsp: bool = False,
        parent=None,
    ):
        super().__init__(parent)
        self._board = board
        self._pipeline = pipeline
        self._sfreq = sfreq
        self._is_fbcsp = is_fbcsp
        self._running = False

        # Ring buffer: 3 seconds of 8-channel data
        self._window_samples = int(3.0 * sfreq)  # 750
        self._n_channels = 8
        self._ring = np.zeros((self._n_channels, self._window_samples), dtype=np.float64)
        self._buf_pos = 0
        self._total_samples = 0

        # Prediction every 0.5 s
        self._pred_interval = int(0.5 * sfreq)  # 125
        self._samples_since_pred = 0

        # Causal bandpass 8-30 Hz (Butterworth 4th order, SOS)
        nyq = sfreq / 2.0
        self._sos_bp = butter(4, [8.0 / nyq, 30.0 / nyq], btype="bandpass", output="sos")

        # Notch 50 Hz
        self._b_notch, self._a_notch = iirnotch(50.0 / nyq, Q=30)

    def run(self) -> None:
        from src.services.acquisition_service import AcquisitionService

        self._running = True
        try:
            while self._running:
                self.msleep(100)
                chunk = AcquisitionService.get_chunk(self._board)
                n = chunk.shape[1]
                if n == 0:
                    continue

                self._append(chunk)
                self._samples_since_pred += n
                filled = min(self._total_samples, self._window_samples)
                self.buffer_status.emit(filled)

                if filled < self._window_samples:
                    continue
                if self._samples_since_pred < self._pred_interval:
                    continue

                self._samples_since_pred = 0
                window = self._ordered_buffer()
                filtered = self._filter(window)

                try:
                    X = filtered[np.newaxis, ...]  # (1, 8, 750)
                    label = int(self._pipeline.predict(X)[0])
                    proba = self._pipeline.predict_proba(X)[0]
                    self.prediction_ready.emit(label, proba)
                except Exception as exc:  # noqa: BLE001
                    self.error.emit(f"Prediction: {exc}")

        except Exception as exc:  # noqa: BLE001
            self.error.emit(str(exc))

    def stop(self) -> None:
        self._running = False

    # ------------------------------------------------------------------
    # Ring buffer
    # ------------------------------------------------------------------

    def _append(self, chunk: np.ndarray) -> None:
        n = chunk.shape[1]
        buf_len = self._window_samples

        if n >= buf_len:
            self._ring[:] = chunk[:, -buf_len:]
            self._buf_pos = 0
        else:
            end = self._buf_pos + n
            if end <= buf_len:
                self._ring[:, self._buf_pos:end] = chunk
            else:
                first = buf_len - self._buf_pos
                self._ring[:, self._buf_pos:] = chunk[:, :first]
                self._ring[:, : n - first] = chunk[:, first:]
            self._buf_pos = end % buf_len

        self._total_samples += n

    def _ordered_buffer(self) -> np.ndarray:
        return np.roll(self._ring, -self._buf_pos, axis=1).copy()

    # ------------------------------------------------------------------
    # Causal filtering
    # ------------------------------------------------------------------

    def _filter(self, window: np.ndarray) -> np.ndarray:
        # Match training preprocessing exactly: zero-phase filtering (sosfiltfilt).
        # We have the full 3-second window, so sosfiltfilt works fine —
        # it pads the edges internally to handle the short signal.
        if not self._is_fbcsp:
            window = sosfiltfilt(self._sos_bp, window, axis=1)
        window = filtfilt(self._b_notch, self._a_notch, window, axis=1)
        return window
