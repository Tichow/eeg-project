"""
Fonctions de traitement du signal EEG — couche partagée numpy.

Ce module est indépendant de la source de données :
  - Offline  : appelé depuis eeg_analysis.py (données MNE converties en numpy)
  - Temps réel : appelé depuis realtime_eeg.py (données BrainFlow, déjà numpy)

Dépendances : numpy, scipy uniquement.
"""

import numpy as np
from scipy.signal import butter, sosfilt, sosfiltfilt, iirnotch, lfilter, lfilter_zi


# ---------------------------------------------------------------------------
# Bandes fréquentielles standard EEG
# ---------------------------------------------------------------------------

FREQ_BANDS = {
    "Delta": (0.5, 4),
    "Theta": (4, 8),
    "Alpha": (8, 13),
    "Beta":  (13, 30),
}

BAND_COLORS = {
    "Delta": "#4e79a7",
    "Theta": "#76b7b2",
    "Alpha": "#f28e2b",
    "Beta":  "#e15759",
}


# ---------------------------------------------------------------------------
# Filtrage
# ---------------------------------------------------------------------------

def filter_signal(
    data: np.ndarray,
    sfreq: float,
    l_freq: float = 1.0,
    h_freq: float = 50.0,
    notch_freq: float = 60.0,
    causal: bool = False,
) -> np.ndarray:
    """
    Applique un filtre passe-bande Butterworth ordre 4 + filtre notch.

    Supporte deux modes :
      - causal=False (défaut) : sosfiltfilt — zéro déphasage, pour analyse offline
      - causal=True           : sosfilt     — causal, pour traitement temps réel

    Args:
        data       : signal EEG, shape (n_channels, n_samples) ou (n_samples,)
        sfreq      : fréquence d'échantillonnage en Hz
        l_freq     : borne basse du passe-bande (Hz)
        h_freq     : borne haute du passe-bande (Hz)
        notch_freq : fréquence du filtre coupe-bande (60 Hz USA, 50 Hz Europe)
        causal     : False → filtfilt (offline), True → causal (temps réel)

    Returns:
        data_filtered : même shape que data
    """
    is_1d = data.ndim == 1
    if is_1d:
        data = data[np.newaxis, :]  # (1, n_samples)

    fn = sfreq / 2.0  # fréquence de Nyquist

    # -- Bandpass Butterworth ordre 4 --
    sos_bp = butter(4, [l_freq / fn, h_freq / fn], btype="bandpass", output="sos")

    # -- Notch --
    b_notch, a_notch = iirnotch(notch_freq / fn, Q=30)

    filtered = np.empty_like(data)
    for i, channel in enumerate(data):
        if causal:
            filtered[i] = sosfilt(sos_bp, channel)
            filtered[i] = lfilter(b_notch, a_notch, filtered[i])
        else:
            filtered[i] = sosfiltfilt(sos_bp, channel)
            filtered[i] = lfilter(b_notch, a_notch, filtered[i])

    return filtered[0] if is_1d else filtered


def make_filter_state(sfreq: float, l_freq: float = 1.0, h_freq: float = 50.0,
                      notch_freq: float = 60.0):
    """
    Crée les états initiaux pour le filtrage causal sample-par-sample en temps réel.

    Retourne un dict à passer à filter_sample() à chaque nouveau sample.

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
        "zi_bp": sosfilt(sos_bp, np.zeros(1)),      # init à zéro
        "zi_notch": lfilter_zi(b_notch, a_notch),   # init à zéro
    }


# ---------------------------------------------------------------------------
# Calcul de la PSD
# ---------------------------------------------------------------------------

def compute_psd_welch(
    signal: np.ndarray,
    sfreq: float,
    fmin: float = 0.5,
    fmax: float = 60.0,
) -> tuple:
    """
    Calcule la densité spectrale de puissance (PSD) via la méthode de Welch.

    Args:
        signal : signal 1D d'un seul canal, en Volts (n_samples,)
        sfreq  : fréquence d'échantillonnage en Hz
        fmin   : fréquence minimale à retourner (Hz)
        fmax   : fréquence maximale à retourner (Hz)

    Returns:
        freqs   : array des fréquences (Hz)
        psd_uv2 : puissance en µV²/Hz
    """
    from scipy.signal import welch

    freqs, psd = welch(signal, fs=sfreq, nperseg=int(sfreq * 4))
    psd_uv2 = psd * 1e12  # V²/Hz → µV²/Hz
    mask = (freqs >= fmin) & (freqs <= fmax)
    return freqs[mask], psd_uv2[mask]


# ---------------------------------------------------------------------------
# Puissance par bande
# ---------------------------------------------------------------------------

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
