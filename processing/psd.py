"""
Calcul de la densité spectrale de puissance (PSD) via la méthode de Welch.
"""

import numpy as np
from scipy.signal import welch


def compute_psd_welch(
    signal: np.ndarray,
    sfreq: float,
    fmin: float = 0.5,
    fmax: float = 60.0,
) -> tuple:
    """
    Calcule la PSD via la méthode de Welch.

    Args:
        signal : signal 1D d'un seul canal, en Volts (n_samples,)
        sfreq  : fréquence d'échantillonnage en Hz
        fmin   : fréquence minimale à retourner (Hz)
        fmax   : fréquence maximale à retourner (Hz)

    Returns:
        freqs   : array des fréquences (Hz)
        psd_uv2 : puissance en µV²/Hz
    """
    freqs, psd = welch(signal, fs=sfreq, nperseg=int(sfreq * 4))
    psd_uv2 = psd * 1e12  # V²/Hz → µV²/Hz
    mask = (freqs >= fmin) & (freqs <= fmax)
    return freqs[mask], psd_uv2[mask]
