from __future__ import annotations

import numpy as np
from scipy.signal import welch, spectrogram

from src.models.epoch_data import EpochData
from src.models.frequency_data import FrequencyData


class EEGFrequencyService:

    @staticmethod
    def compute_psd(
        epoch_data: EpochData,
        selected_ch_indices: list[int],
        fmin: float,
        fmax: float,
        nperseg: int,
    ) -> FrequencyData:
        """Calcule la PSD (Welch) par classe sur les canaux sélectionnés.

        Concatène les epochs d'une même classe le long du temps pour obtenir
        une estimation stable même avec des epochs courtes.

        Returns
        -------
        FrequencyData avec psd_by_class[label] shape (n_ch, n_freqs) en µV²/Hz.
        """
        sfreq = epoch_data.sfreq
        epoch_len = epoch_data.data.shape[2]
        nperseg = min(nperseg, epoch_len)

        unique_labels = sorted(set(epoch_data.labels))
        ch_names = [epoch_data.ch_names[i] for i in selected_ch_indices]

        # Calculer les fréquences du vecteur de sortie (même pour tous les labels)
        _freqs_ref, _ = welch(
            np.zeros(max(epoch_len, nperseg)),
            fs=sfreq,
            nperseg=nperseg,
        )
        freq_mask = (_freqs_ref >= fmin) & (_freqs_ref <= fmax)
        freqs_out = _freqs_ref[freq_mask]

        psd_by_class: dict[str, np.ndarray] = {}

        labels_arr = np.array(epoch_data.labels)
        for label in unique_labels:
            mask = labels_arr == label
            subset = epoch_data.data[mask]  # (n_epochs, n_channels, n_times)

            psd_ch = np.zeros((len(selected_ch_indices), freqs_out.size))
            for k, ch_idx in enumerate(selected_ch_indices):
                # Concatenate along time axis
                concat = subset[:, ch_idx, :].ravel()
                freqs, psd = welch(concat, fs=sfreq, nperseg=nperseg, scaling="density")
                psd_uv = psd * 1e12  # V²/Hz → µV²/Hz
                psd_ch[k] = psd_uv[freq_mask]

            psd_by_class[label] = psd_ch

        return FrequencyData(
            mode="psd",
            freqs=freqs_out,
            ch_names=ch_names,
            psd_by_class=psd_by_class,
        )

    @staticmethod
    def compute_erders(
        epoch_data: EpochData,
        selected_ch_indices: list[int],
        band_low: float,
        band_high: float,
        baseline_label: str,
        nperseg: int,
    ) -> FrequencyData:
        """Calcule l'ERD/ERS (%) par classe non-baseline.

        Algorithme :
        1. Baseline power : Welch sur concat des epochs baseline → moyenne sur la bande.
        2. Event power : spectrogram sur l'epoch moyenne par classe → sélection bande → moyenne.
        3. ERD/ERS = (event_power - baseline_power) / baseline_power * 100.

        Returns
        -------
        FrequencyData avec erders_by_class[label] shape (n_ch, n_times) en %.
        """
        sfreq = epoch_data.sfreq
        epoch_len = epoch_data.data.shape[2]
        nperseg = min(nperseg, epoch_len)
        noverlap = nperseg // 2

        nyquist = sfreq / 2.0
        if not (0 < band_low < band_high < nyquist):
            raise ValueError(
                f"Bande invalide : {band_low}–{band_high} Hz (Nyquist = {nyquist:.1f} Hz)"
            )

        labels_arr = np.array(epoch_data.labels)
        ch_names = [epoch_data.ch_names[i] for i in selected_ch_indices]

        # ── Étape 1 : baseline power ──────────────────────────────────────────
        bl_mask = labels_arr == baseline_label
        if not bl_mask.any():
            raise ValueError(
                f"Aucune epoch avec le label baseline '{baseline_label}' trouvée."
            )
        bl_subset = epoch_data.data[bl_mask]  # (n_bl, n_ch, n_times)

        baseline_power = np.zeros(len(selected_ch_indices))
        for k, ch_idx in enumerate(selected_ch_indices):
            concat_bl = bl_subset[:, ch_idx, :].ravel()
            freqs_w, psd_w = welch(concat_bl, fs=sfreq, nperseg=nperseg, scaling="density")
            band_mask = (freqs_w >= band_low) & (freqs_w <= band_high)
            baseline_power[k] = psd_w[band_mask].mean() if band_mask.any() else 0.0

        # ── Étape 2 : event power via spectrogram ────────────────────────────
        unique_labels = sorted(set(epoch_data.labels))
        event_labels = [lbl for lbl in unique_labels if lbl != baseline_label]

        erders_by_class: dict[str, np.ndarray] = {}
        times_out = epoch_data.times  # référence temporelle commune

        for label in event_labels:
            ev_mask = labels_arr == label
            ev_subset = epoch_data.data[ev_mask]  # (n_ev, n_ch, n_times)
            mean_epoch = ev_subset.mean(axis=0)  # (n_ch, n_times)

            # Initialise avec la taille de times_out
            event_power = np.zeros((len(selected_ch_indices), times_out.size))

            for k, ch_idx in enumerate(selected_ch_indices):
                f_sg, t_sg, Sxx = spectrogram(
                    mean_epoch[ch_idx],
                    fs=sfreq,
                    nperseg=nperseg,
                    noverlap=noverlap,
                )
                band_mask = (f_sg >= band_low) & (f_sg <= band_high)
                if band_mask.any():
                    band_power = Sxx[band_mask, :].mean(axis=0)  # (n_t_sg,)
                else:
                    band_power = np.zeros(t_sg.size)

                # Aligne t_sg (relatif au début de l'epoch) sur epoch_data.times
                t_epoch_start = times_out[0]
                t_sg_aligned = t_sg + t_epoch_start
                event_power[k] = np.interp(times_out, t_sg_aligned, band_power)

            # ── Étape 3 : ERD/ERS % ───────────────────────────────────────────
            bp = baseline_power[:, np.newaxis]  # (n_ch, 1)
            erders = np.where(
                bp == 0,
                np.nan,
                (event_power - bp) / bp * 100.0,
            )
            erders_by_class[label] = erders

        freqs_dummy = np.array([band_low, band_high])

        return FrequencyData(
            mode="erders",
            freqs=freqs_dummy,
            ch_names=ch_names,
            times=times_out,
            erders_by_class=erders_by_class,
            baseline_label=baseline_label,
        )
