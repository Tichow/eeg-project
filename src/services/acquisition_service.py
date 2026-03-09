from __future__ import annotations

import numpy as np


class AcquisitionService:
    """Wraps BrainFlow for OpenBCI Cyton acquisition. No Qt imports."""

    @staticmethod
    def list_serial_ports() -> list[str]:
        """Return available serial port names via pyserial."""
        from serial.tools import list_ports  # local import — optional dep

        return [p.device for p in list_ports.comports()]

    @staticmethod
    def connect(serial_port: str):
        """Connect to an OpenBCI Cyton board and start streaming.

        Returns
        -------
        tuple[BoardShim, float]
            (board, sfreq) — sfreq is always 250 Hz for the Cyton.
        """
        from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds

        BoardShim.disable_board_logger()
        params = BrainFlowInputParams()
        params.serial_port = serial_port
        board = BoardShim(BoardIds.CYTON_BOARD.value, params)
        board.prepare_session()
        board.start_stream()
        sfreq = BoardShim.get_sampling_rate(BoardIds.CYTON_BOARD.value)
        return board, float(sfreq)

    @staticmethod
    def get_chunk(board) -> np.ndarray:
        """Retrieve all samples currently available in the BrainFlow ring buffer.

        Returns
        -------
        np.ndarray
            Shape (8, n_samples), data in Volts (converted from µV).
            Returns an empty array of shape (8, 0) if no data is available.
        """
        from brainflow.board_shim import BoardShim, BoardIds

        eeg_channels = BoardShim.get_eeg_channels(BoardIds.CYTON_BOARD.value)  # 8 channels
        raw = board.get_board_data()  # (n_all_channels, n_samples)
        if raw.shape[1] == 0:
            return np.zeros((8, 0), dtype=np.float64)
        eeg = raw[eeg_channels, :]  # (8, n_samples) in µV
        return eeg / 1e6  # convert to Volts

    @staticmethod
    def disconnect(board) -> None:
        """Stop streaming and release the BrainFlow session."""
        try:
            board.stop_stream()
        except Exception:
            pass
        try:
            board.release_session()
        except Exception:
            pass
