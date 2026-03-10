from __future__ import annotations

import numpy as np
from mne.filter import filter_data
from scipy.signal import welch

from src.models.berger_result import BergerResult
from src.services.eeg_signal_service import EEGSignalService


class EEGBergerService:
    ALPHA_LOW = 8.0
    ALPHA_HIGH = 13.0
    PREFERRED_CHANNELS = ["O1", "Oz", "O2", "Pz", "P3", "P4"]
    PSD_FMIN = 2.0
    PSD_FMAX = 30.0

    @staticmethod
    def compute(path_open: str, path_closed: str) -> BergerResult:
        sig_open = EEGSignalService.load_signal(path_open)
        sig_closed = EEGSignalService.load_signal(path_closed)

        # Sélection canaux occipito-pariétaux (case-insensitive), fallback tous canaux
        preferred_upper = {c.upper() for c in EEGBergerService.PREFERRED_CHANNELS}
        channels_used = [
            ch for ch in sig_open.ch_names
            if ch.upper() in preferred_upper
        ]
        if not channels_used:
            channels_used = list(sig_open.ch_names)

        sfreq = sig_open.sfreq
        n_samples = sig_open.data.shape[1]
        nperseg = min(int(4 * sfreq), n_samples)

        data_open = sig_open.data.copy()
        data_open -= data_open.mean(axis=1, keepdims=True)          # DC detrend par canal
        data_open = filter_data(data_open, sfreq, l_freq=2.0, h_freq=30.0, verbose=False)
        data_open -= data_open.mean(axis=0, keepdims=True)          # average reference

        data_closed = sig_closed.data.copy()
        data_closed -= data_closed.mean(axis=1, keepdims=True)      # DC detrend par canal
        data_closed = filter_data(data_closed, sfreq, l_freq=2.0, h_freq=30.0, verbose=False)
        data_closed -= data_closed.mean(axis=0, keepdims=True)      # average reference

        freqs_open, psd_open_mean = EEGBergerService._mean_psd(data_open, sig_open.ch_names, channels_used, sfreq, nperseg)
        _, psd_closed_mean = EEGBergerService._mean_psd(data_closed, sig_closed.ch_names, channels_used, sfreq, nperseg)

        # Restreindre la plage d'affichage à [PSD_FMIN, PSD_FMAX]
        display_mask = (freqs_open >= EEGBergerService.PSD_FMIN) & (freqs_open <= EEGBergerService.PSD_FMAX)
        freqs_out = freqs_open[display_mask]
        psd_open_out = psd_open_mean[display_mask]
        psd_closed_out = psd_closed_mean[display_mask]

        # Score alpha
        alpha_mask = (freqs_open >= EEGBergerService.ALPHA_LOW) & (freqs_open <= EEGBergerService.ALPHA_HIGH)
        alpha_open = float(psd_open_mean[alpha_mask].mean()) if alpha_mask.any() else 0.0
        alpha_closed = float(psd_closed_mean[alpha_mask].mean()) if alpha_mask.any() else 0.0

        ratio = alpha_closed / alpha_open if alpha_open > 0 else 0.0
        quality, color = EEGBergerService._classify(ratio)

        snr_open = EEGBergerService._snr(freqs_open, psd_open_mean)
        snr_closed = EEGBergerService._snr(freqs_open, psd_closed_mean)

        return BergerResult(
            alpha_open=alpha_open,
            alpha_closed=alpha_closed,
            ratio=ratio,
            quality=quality,
            color=color,
            snr_open=snr_open,
            snr_closed=snr_closed,
            channels_used=channels_used,
            freqs=freqs_out,
            psd_open=psd_open_out,
            psd_closed=psd_closed_out,
        )

    @staticmethod
    def _snr(freqs: np.ndarray, psd: np.ndarray) -> float:
        """SNR alpha en dB : puissance 8–13 Hz / puissance noise (1–8 Hz + 13–30 Hz)."""
        alpha_mask = (freqs >= 8.0) & (freqs <= 13.0)
        noise_mask = ((freqs >= 1.0) & (freqs < 8.0)) | ((freqs > 13.0) & (freqs <= 30.0))
        alpha_pow = psd[alpha_mask].mean() if alpha_mask.any() else 0.0
        noise_pow = psd[noise_mask].mean() if noise_mask.any() else 0.0
        if noise_pow <= 0 or alpha_pow <= 0:
            return 0.0
        return float(10.0 * np.log10(alpha_pow / noise_pow))

    @staticmethod
    def _mean_psd(data: np.ndarray, ch_names: list[str], channels: list[str], sfreq: float, nperseg: int):
        """Calcule la PSD Welch moyennée cross-canaux. Retourne (freqs, psd_mean)."""
        ch_upper = {ch.upper(): i for i, ch in enumerate(ch_names)}
        psds = []
        freqs_ref = None
        for ch in channels:
            idx = ch_upper.get(ch.upper())
            if idx is None:
                continue
            freqs, psd = welch(
                data[idx],
                fs=sfreq,
                nperseg=nperseg,
                scaling="density",
            )
            psd_uv = psd * 1e12   # V²/Hz → µV²/Hz
            psds.append(psd_uv)
            freqs_ref = freqs
        if not psds:
            freqs_ref = np.array([0.0])
            return freqs_ref, np.zeros(1)
        return freqs_ref, np.mean(psds, axis=0)

    @staticmethod
    def _classify(ratio: float) -> tuple[str, str]:
        if ratio > 2.0:
            return "Excellent", "#4caf50"
        if ratio >= 1.5:
            return "Bon", "#8bc34a"
        if ratio >= 1.2:
            return "Moyen", "#ff9800"
        return "Faible", "#f44336"
