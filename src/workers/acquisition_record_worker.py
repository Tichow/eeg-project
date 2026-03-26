from __future__ import annotations

import random
import time

import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal


class AcquisitionRecordWorker(QThread):
    """QThread that runs the guided acquisition protocol and records EEG data.

    Protocol per trial:
        baseline (T0) → cue (T1 or T2) → rest

    Signals
    -------
    trial_update(int, int, str)
        (trial_index, total_trials, cue_label) emitted at the start of each cue phase.
    phase_update(str)
        "baseline" | "cue" | "rest" — current phase name.
    finished(object)
        Emitted with the resulting SignalData when recording is complete.
    error(str)
        Emitted on any exception.
    """

    trial_update = pyqtSignal(int, int, str)
    phase_update = pyqtSignal(str)
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, board, config, parent=None):
        super().__init__(parent)
        self._board = board
        self._config = config

    def run(self) -> None:
        from src.services.acquisition_service import AcquisitionService
        from src.models.signal_data import SignalData
        from brainflow.board_shim import BoardShim, BoardIds

        cfg = self._config
        try:
            # Flush the board buffer before starting
            self._board.get_board_data()

            eeg_ch = BoardShim.get_eeg_channels(BoardIds.CYTON_BOARD.value)
            sfreq = float(BoardShim.get_sampling_rate(BoardIds.CYTON_BOARD.value))

            t_baseline_ms = int(cfg.t_baseline_s * 1000)
            t_cue_ms = int(cfg.t_cue_s * 1000)
            t_rest_ms = int(cfg.t_rest_s * 1000)

            # Build balanced, shuffled trial order using protocol annotation_labels
            ann_labels = cfg.annotation_labels
            if len(cfg.classes) == 1:
                labels = [ann_labels[0]] * cfg.n_trials_per_class
            else:
                lbl0 = ann_labels[0] if len(ann_labels) > 0 else "T1"
                lbl1 = ann_labels[1] if len(ann_labels) > 1 else "T2"
                labels = [lbl0] * cfg.n_trials_per_class + [lbl1] * cfg.n_trials_per_class
                random.shuffle(labels)
            total = len(labels)

            annotations: list[tuple[float, float, str]] = []
            chunks: list[np.ndarray] = []
            t_start = time.monotonic()

            for i, label in enumerate(labels):
                # --- Baseline phase ---
                self.phase_update.emit("baseline")
                onset_t0 = time.monotonic() - t_start
                self._sleep_and_collect(t_baseline_ms, chunks, eeg_ch)
                if cfg.t_baseline_s > 0:
                    annotations.append((onset_t0, cfg.t_baseline_s, "T0"))

                # --- Cue phase ---
                try:
                    class_idx = cfg.annotation_labels.index(label)
                except ValueError:
                    class_idx = 0
                cue_text = cfg.classes[class_idx] if class_idx < len(cfg.classes) else cfg.classes[0]
                onset_cue = time.monotonic() - t_start
                self.trial_update.emit(i + 1, total, cue_text)
                self.phase_update.emit("cue")
                self._sleep_and_collect(t_cue_ms, chunks, eeg_ch)
                annotations.append((onset_cue, cfg.t_cue_s, label))

                # --- Rest phase ---
                self.phase_update.emit("rest")
                self._sleep_and_collect(t_rest_ms, chunks, eeg_ch)

            # Final drain
            raw = self._board.get_board_data()
            if raw.shape[1] > 0:
                chunks.append(raw[eeg_ch, :] / 1e6)

            if not chunks:
                self.error.emit(
                    "Aucune donnée reçue — la board s'est-elle éteinte pendant l'enregistrement ?"
                )
                return

            eeg_data = np.concatenate(chunks, axis=1)  # (8, n_samples) in Volts
            n_samples = eeg_data.shape[1]
            times = np.arange(n_samples) / sfreq

            signal_data = SignalData(
                data=eeg_data,
                times=times,
                ch_names=cfg.ch_names,
                sfreq=sfreq,
                annotations=annotations,
            )
            self.finished.emit(signal_data)

        except Exception as exc:  # noqa: BLE001
            self.error.emit(str(exc))

    def _sleep_and_collect(
        self, duration_ms: int, chunks: list, eeg_ch: list
    ) -> None:
        """Sleep in 100ms increments, draining the board buffer each step."""
        elapsed = 0
        while elapsed < duration_ms:
            step = min(100, duration_ms - elapsed)
            self.msleep(step)
            elapsed += step
            raw = self._board.get_board_data()
            if raw.shape[1] > 0:
                chunks.append(raw[eeg_ch, :] / 1e6)
