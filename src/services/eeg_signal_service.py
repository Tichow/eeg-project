import mne

from src.models.signal_data import SignalData


class EEGSignalService:
    @staticmethod
    def load_signal(path: str) -> SignalData:
        """Load a full EDF file and return structured signal data."""
        raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
        mne.datasets.eegbci.standardize(raw)  # strip trailing dots from ch_names (e.g. "Fc5." → "Fc5")
        annotations = [
            (float(ann["onset"]), float(ann["duration"]), str(ann["description"]))
            for ann in raw.annotations
        ]
        return SignalData(
            data=raw.get_data(),
            times=raw.times,
            ch_names=raw.ch_names,
            sfreq=float(raw.info["sfreq"]),
            annotations=annotations,
        )
