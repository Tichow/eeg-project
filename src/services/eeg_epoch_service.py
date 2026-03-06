import numpy as np

from src.models.signal_data import SignalData
from src.models.epoch_data import EpochData


class EEGEpochService:
    @staticmethod
    def extract(signal_data: SignalData, tmin: float, tmax: float) -> EpochData:
        """Cut signal around each annotation onset. Returns EpochData.

        Epochs that fall outside the recording boundaries are silently skipped.
        Raises ValueError if no valid epochs can be extracted.
        """
        if tmin >= tmax:
            raise ValueError(f"tmin ({tmin:.2f}) doit être < tmax ({tmax:.2f})")

        sfreq = signal_data.sfreq
        n_samples = signal_data.data.shape[1]
        n_before = int(round(-tmin * sfreq))
        n_after = int(round(tmax * sfreq))
        epoch_len = n_before + n_after

        times = np.arange(epoch_len) / sfreq + tmin

        epochs: list[np.ndarray] = []
        labels: list[str] = []
        onsets: list[float] = []

        for onset_s, _dur, desc in signal_data.annotations:
            center = int(round(onset_s * sfreq))
            start = center - n_before
            end = center + n_after
            if start < 0 or end > n_samples:
                continue
            epochs.append(signal_data.data[:, start:end])
            labels.append(desc)
            onsets.append(onset_s)

        if not epochs:
            raise ValueError(
                "Aucune epoch valide — vérifiez que tmin/tmax restent dans les limites du signal."
            )

        data = np.stack(epochs, axis=0)  # (n_epochs, n_channels, n_times)

        return EpochData(
            data=data,
            times=times,
            ch_names=signal_data.ch_names,
            sfreq=sfreq,
            labels=labels,
            onsets=onsets,
        )
