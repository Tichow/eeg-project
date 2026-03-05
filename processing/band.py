"""
Calcul de la puissance par bande fréquentielle EEG.
"""

import numpy as np

from .constants import FREQ_BANDS
from .psd import compute_psd_welch


def band_power(signal: np.ndarray, sfreq: float, band_name: str) -> float:
    """
    Calcule la puissance moyenne d'un signal dans une bande fréquentielle.

    Utile pour le feedback temps réel (ex: afficher la puissance alpha en live).

    Args:
        signal    : signal 1D en Volts (n_samples,)
        sfreq     : fréquence d'échantillonnage
        band_name : clé de FREQ_BANDS ('Delta', 'Theta', 'Alpha', 'Beta')

    Returns:
        puissance moyenne en µV²/Hz sur la bande demandée
    """
    f_low, f_high = FREQ_BANDS[band_name]
    freqs, psd_uv2 = compute_psd_welch(signal, sfreq, fmin=f_low, fmax=f_high)
    return float(np.mean(psd_uv2))
