"""
Calcul du SNR (Signal-to-Noise Ratio) pour un signal EEG.

Définition utilisée :
    SNR = P_bande / P_hors_bande

où P_bande est la puissance dans la bande d'intérêt et P_hors_bande
est la puissance dans le reste du spectre mesuré (fmin–fmax).

Retourne le SNR en dB : SNR_dB = 10 * log10(SNR_linéaire)
"""

import numpy as np

from .constants import FREQ_BANDS
from .psd import compute_psd_welch


def compute_snr(
    signal: np.ndarray,
    sfreq: float,
    band_name: str,
    fmin: float = 0.5,
    fmax: float = 50.0,
) -> float:
    """
    Calcule le SNR d'un signal EEG pour une bande fréquentielle donnée.

    Args:
        signal    : signal 1D en Volts (n_samples,)
        sfreq     : fréquence d'échantillonnage en Hz
        band_name : bande d'intérêt — clé de FREQ_BANDS ('Delta', 'Theta', 'Alpha', 'Beta')
        fmin      : borne basse du spectre de référence (Hz)
        fmax      : borne haute du spectre de référence (Hz)

    Returns:
        snr_db : SNR en dB (peut être négatif si la bande est moins puissante que le bruit)
    """
    f_low, f_high = FREQ_BANDS[band_name]

    freqs, psd = compute_psd_welch(signal, sfreq, fmin=fmin, fmax=fmax)

    signal_mask = (freqs >= f_low) & (freqs <= f_high)
    noise_mask  = ~signal_mask

    p_signal = np.mean(psd[signal_mask]) if signal_mask.any() else 0.0
    p_noise  = np.mean(psd[noise_mask])  if noise_mask.any()  else 1.0

    if p_noise == 0 or p_signal == 0:
        return 0.0

    return 10.0 * np.log10(p_signal / p_noise)
