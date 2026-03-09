import numpy as np

from src.models.epoch_data import EpochData


class EEGArtifactService:

    @staticmethod
    def detect_by_threshold(epoch_data: EpochData, threshold_uv: float) -> list[int]:
        """Retourne les indices des époques dont le pic-à-pic max dépasse threshold_uv µV."""
        threshold_v = threshold_uv * 1e-6
        bad = []
        for i, epoch in enumerate(epoch_data.data):  # epoch: (n_channels, n_times)
            if np.ptp(epoch, axis=1).max() > threshold_v:
                bad.append(i)
        return bad

    @staticmethod
    def apply_threshold_rejection(epoch_data: EpochData, bad_indices: list[int]) -> EpochData:
        """Retourne un EpochData sans les époques marquées mauvaises."""
        good = np.ones(len(epoch_data.data), dtype=bool)
        good[bad_indices] = False
        return EpochData(
            data=epoch_data.data[good],
            times=epoch_data.times,
            ch_names=epoch_data.ch_names,
            sfreq=epoch_data.sfreq,
            labels=[l for i, l in enumerate(epoch_data.labels) if good[i]],
            onsets=[o for i, o in enumerate(epoch_data.onsets) if good[i]],
        )

    @staticmethod
    def fit_ica(epoch_data: EpochData, n_components: int) -> tuple:
        """Ajuste une ICA sur les époques et auto-détecte les composantes EOG (Fp1/Fp2).
        Retourne (ica, bad_components: list[int])."""
        import mne
        n_components = min(n_components, epoch_data.data.shape[0] - 1, epoch_data.data.shape[1])
        n_components = max(n_components, 1)
        info = mne.create_info(epoch_data.ch_names, epoch_data.sfreq, ch_types="eeg")
        epochs_mne = mne.EpochsArray(
            epoch_data.data, info, tmin=float(epoch_data.times[0]), verbose=False
        )
        ica = mne.preprocessing.ICA(n_components=n_components, random_state=42, max_iter="auto", verbose=False)
        ica.fit(epochs_mne, verbose=False)
        bad_components: list[int] = []
        eog_chs = [ch for ch in ["Fp1", "Fp2"] if ch in epoch_data.ch_names]
        if eog_chs:
            try:
                inds, _ = ica.find_bads_eog(epochs_mne, ch_name=eog_chs[0], verbose=False)
                bad_components = list(set(int(i) for i in inds))
            except Exception:
                pass
        return ica, bad_components

    @staticmethod
    def apply_ica(epoch_data: EpochData, ica, exclude: list[int]) -> EpochData:
        """Reconstruit les données en supprimant les composantes exclues."""
        import mne
        info = mne.create_info(epoch_data.ch_names, epoch_data.sfreq, ch_types="eeg")
        epochs_mne = mne.EpochsArray(
            epoch_data.data, info, tmin=float(epoch_data.times[0]), verbose=False
        )
        ica.exclude = exclude
        clean = ica.apply(epochs_mne.copy(), verbose=False)
        return EpochData(
            data=clean.get_data(),
            times=epoch_data.times,
            ch_names=epoch_data.ch_names,
            sfreq=epoch_data.sfreq,
            labels=epoch_data.labels,
            onsets=epoch_data.onsets,
        )
