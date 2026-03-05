"""
Visualisation EEG temps réel — OpenBCI Cyton via BrainFlow.

Usage :
    python realtime_eeg.py
    python realtime_eeg.py --port /dev/cu.usbserial-XXXXX
    python realtime_eeg.py --port /dev/cu.usbserial-XXXXX --channels 1 2 5 6
"""

import argparse
import sys
import time

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from PyQt5.QtWidgets import QApplication

from viz import Dashboard

DEFAULT_PORT = "/dev/cu.usbserial-DM03H2DU"
BOARD_ID     = BoardIds.CYTON_BOARD.value
WINDOW_SEC   = 5
UPDATE_MS    = 100
GAIN_CMD     = "x{ch}030110X"  # gain 6x (commande OpenBCI)


def setup_board(port: str) -> tuple:
    """Initialise et démarre la session BrainFlow."""
    params = BrainFlowInputParams()
    params.serial_port = port

    board        = BoardShim(BOARD_ID, params)
    sfreq        = BoardShim.get_sampling_rate(BOARD_ID)
    eeg_channels = BoardShim.get_eeg_channels(BOARD_ID)

    board.prepare_session()

    print("Configuration Gain 6x...")
    for i in range(1, 9):
        board.config_board(GAIN_CMD.format(ch=i))
        time.sleep(0.02)

    board.start_stream()
    print(f"Stream démarré — {sfreq} Hz, {len(eeg_channels)} canaux EEG")
    return board, sfreq, eeg_channels


def run_realtime(port: str, ch_indices: list = None) -> None:
    """Lance le dashboard EEG temps réel."""
    board, sfreq, all_eeg = setup_board(port)
    n_samples = int(WINDOW_SEC * sfreq)

    if ch_indices is None:
        ch_indices = list(range(len(all_eeg)))
    channels  = [all_eeg[i] for i in ch_indices if i < len(all_eeg)]
    ch_labels = [f"CH{i+1}" for i in ch_indices]

    app = QApplication.instance() or QApplication(sys.argv)

    dashboard = Dashboard(
        board=board,
        channels=channels,
        ch_labels=ch_labels,
        sfreq=sfreq,
        n_samples=n_samples,
        window_sec=WINDOW_SEC,
        update_ms=UPDATE_MS,
    )
    dashboard.resize(1400, 1000)
    dashboard.show()
    print("  Astuce : appuie sur R dans la fenêtre pour démarrer/arrêter l'enregistrement")

    try:
        app.exec_()
    finally:
        board.stop_stream()
        board.release_session()
        print("Session fermée.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualisation EEG temps réel — OpenBCI Cyton"
    )
    parser.add_argument("--port", default=DEFAULT_PORT,
                        help=f"Port série du Cyton (défaut: {DEFAULT_PORT})")
    parser.add_argument("--channels", type=int, nargs="+", default=None,
                        metavar="N",
                        help="Indices 0-based des canaux (défaut: tous). "
                             "Ex: --channels 0 1 4 5")
    args = parser.parse_args()
    run_realtime(port=args.port, ch_indices=args.channels)
