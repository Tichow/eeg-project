"""
Filtrage du signal EEG : passe-bande Butterworth + filtre notch.

Supporte deux modes :
  - causal=False (défaut) : sosfiltfilt — zéro déphasage, pour analyse offline
  - causal=True           : sosfilt     — causal, pour traitement temps réel
"""

import numpy as np
from scipy.signal import butter, sosfilt, sosfiltfilt, iirnotch, lfilter, lfilter_zi


def filter_signal(
    data: np.ndarray,
    sfreq: float,
    l_freq: float = 1.0,
    h_freq: float = 50.0,
    notch_freq: float = 60.0,
    causal: bool = False,
    apply_bandpass: bool = True,
    apply_notch: bool = True,
) -> np.ndarray:
    """
    Applique un filtre passe-bande Butterworth ordre 4 + filtre notch.

    Args:
        data           : signal EEG, shape (n_channels, n_samples) ou (n_samples,)
        sfreq          : fréquence d'échantillonnage en Hz
        l_freq         : borne basse du passe-bande (Hz)
        h_freq         : borne haute du passe-bande (Hz)
        notch_freq     : fréquence du filtre coupe-bande (60 Hz USA, 50 Hz Europe)
        causal         : False → filtfilt (offline), True → causal (temps réel)
        apply_bandpass : appliquer le filtre passe-bande
        apply_notch    : appliquer le filtre coupe-bande

    Returns:
        data_filtered : même shape que data
    """
    is_1d = data.ndim == 1
    if is_1d:
        data = data[np.newaxis, :]

    if not apply_bandpass and not apply_notch:
        return data[0].copy() if is_1d else data.copy()

    fn = sfreq / 2.0

    if apply_bandpass:
        sos_bp = butter(4, [l_freq / fn, h_freq / fn], btype="bandpass", output="sos")
    if apply_notch:
        b_notch, a_notch = iirnotch(notch_freq / fn, Q=30)

    filtered = np.empty_like(data)
    for i, channel in enumerate(data):
        ch = channel.copy()
        if apply_bandpass:
            ch = sosfiltfilt(sos_bp, ch) if not causal else sosfilt(sos_bp, ch)
        if apply_notch:
            ch = lfilter(b_notch, a_notch, ch)
        filtered[i] = ch

    return filtered[0] if is_1d else filtered


def make_filter_state(
    sfreq: float,
    l_freq: float = 1.0,
    h_freq: float = 50.0,
    notch_freq: float = 60.0,
) -> dict:
    """
    Crée les états initiaux pour le filtrage causal sample-par-sample en temps réel.

    Args:
        sfreq      : fréquence d'échantillonnage
        l_freq     : borne basse du passe-bande
        h_freq     : borne haute du passe-bande
        notch_freq : fréquence du notch

    Returns:
        state : dict avec les coefficients et états (zi_bp, zi_notch)
    """
    fn = sfreq / 2.0
    sos_bp = butter(4, [l_freq / fn, h_freq / fn], btype="bandpass", output="sos")
    b_notch, a_notch = iirnotch(notch_freq / fn, Q=30)
    return {
        "sos_bp": sos_bp,
        "b_notch": b_notch,
        "a_notch": a_notch,
        "zi_bp": sosfilt(sos_bp, np.zeros(1)),
        "zi_notch": lfilter_zi(b_notch, a_notch),
    }
