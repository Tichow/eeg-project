import numpy as np
from scipy.signal import butter, sosfiltfilt, iirnotch, filtfilt

from src.models.signal_data import SignalData
from src.models.preprocess_config import PreprocessConfig


class EEGPreprocessService:
    @staticmethod
    def apply(signal_data: SignalData, config: PreprocessConfig) -> SignalData:
        data = signal_data.data.copy()
        sfreq = signal_data.sfreq
        nyquist = sfreq / 2.0

        if config.bandpass_enabled:
            low = config.low_hz
            high = config.high_hz
            if not (0 < low < high < nyquist):
                raise ValueError(
                    f"Passe-bande invalide : {low}–{high} Hz (Nyquist = {nyquist} Hz)"
                )
            sos = butter(4, [low / nyquist, high / nyquist], btype="bandpass", output="sos")
            data = sosfiltfilt(sos, data, axis=1)

        if config.notch_enabled:
            notch_hz = config.notch_hz
            if not (0 < notch_hz < nyquist):
                raise ValueError(
                    f"Fréquence notch invalide : {notch_hz} Hz (Nyquist = {nyquist} Hz)"
                )
            b, a = iirnotch(notch_hz / nyquist, Q=30)
            data = filtfilt(b, a, data, axis=1)

        if config.reref_enabled:
            if config.reref_mode == "average":
                data = data - data.mean(axis=0)
            else:
                ch_names = signal_data.ch_names
                if config.reref_mode not in ch_names:
                    raise ValueError(f"Canal de référence introuvable : {config.reref_mode!r}")
                ref_idx = ch_names.index(config.reref_mode)
                data = data - data[ref_idx]

        return SignalData(
            data=data,
            times=signal_data.times,
            ch_names=signal_data.ch_names,
            sfreq=signal_data.sfreq,
            annotations=signal_data.annotations,
        )
