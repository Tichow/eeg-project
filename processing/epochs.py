"""
Extraction et moyennage d'epochs EEG autour d'événements.
"""

from __future__ import annotations

import numpy as np

from .car import apply_car
from .constants import FREQ_BANDS
from .filter import filter_signal
from .psd import compute_psd_welch


def extract_epochs(
    data_uv: np.ndarray,
    sfreq: float,
    annotations: list[dict],
    event_label: str,
    t_before: float,
    t_after: float,
) -> list[np.ndarray]:
    """
    Extrait les epochs centrées sur chaque événement d'un label donné.

    Args:
        data_uv    : (n_ch, n_samples) en µV
        sfreq      : fréquence d'échantillonnage
        annotations: liste de {'time_sec': float, 'label': str}
        event_label: label à sélectionner (ex: 'T1')
        t_before   : secondes avant l'événement
        t_after    : secondes après l'événement

    Returns:
        liste d'arrays (n_ch, n_epoch_samples), une par événement trouvé
    """
    n_before = int(round(t_before * sfreq))
    n_after  = int(round(t_after  * sfreq))
    n_total  = data_uv.shape[1]
    epochs   = []

    for ann in annotations:
        if ann.get('label', '') != event_label:
            continue
        t0     = ann['time_sec']
        center = int(round(t0 * sfreq))
        start  = center - n_before
        end    = center + n_after
        if start < 0 or end > n_total:
            continue
        epochs.append(data_uv[:, start:end].copy())

    return epochs


def average_epochs(
    epochs: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Moyenne une liste d'epochs.

    Args:
        epochs : liste de (n_ch, n_samples)

    Returns:
        mean  : (n_ch, n_samples) — moyenne
        std   : (n_ch, n_samples) — écart-type
        times : (n_samples,)      — axe temporel en secondes (None si epochs vide)
    """
    if not epochs:
        return (
            np.zeros((1, 0)),
            np.zeros((1, 0)),
            np.array([]),
        )
    stack = np.stack(epochs, axis=0)  # (n_epochs, n_ch, n_samples)
    return np.mean(stack, axis=0), np.std(stack, axis=0)


def group_by_label(
    data_uv: np.ndarray,
    sfreq: float,
    annotations: list[dict],
    t_before: float,
    t_after: float,
    labels: list[str] | None = None,
    notch_freq: float = 50.0,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, int]]:
    """
    Extrait et moyenne les epochs pour chaque label d'événement.

    Applique un filtrage 1–50 Hz + notch + CAR avant l'epoching.
    Applique une correction baseline (soustrait la moyenne de [-t_before, 0]).

    Args:
        data_uv   : (n_ch, n_samples) en µV
        sfreq     : fréquence d'échantillonnage
        annotations: liste de {'time_sec': float, 'label': str}
        t_before  : secondes avant l'événement (fenêtre baseline)
        t_after   : secondes après l'événement
        labels    : labels à conserver (None = auto-détection depuis annotations)
        notch_freq: fréquence du filtre coupe-bande (50 Hz FR, 60 Hz PhysioNet US)

    Returns:
        dict[label] = (mean, std, times, n_epochs)
          mean/std : (n_ch, n_samples)
          times    : (n_samples,) axe temporel en secondes, 0 = onset
          n_epochs : nombre d'epochs utilisées
    """
    # Filtrage global
    data_filt = filter_signal(
        data_uv, sfreq,
        l_freq=1.0, h_freq=50.0, notch_freq=notch_freq,
        causal=False,
    )
    data_filt = apply_car(data_filt)

    if labels is None:
        seen = set()
        labels = []
        for ann in annotations:
            lb = ann.get('label', '')
            if lb and lb not in seen:
                seen.add(lb)
                labels.append(lb)

    n_before = int(round(t_before * sfreq))
    n_after  = int(round(t_after  * sfreq))
    n_epoch  = n_before + n_after
    times    = np.linspace(-t_before, t_after, n_epoch, endpoint=False)

    result: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, int]] = {}

    for label in labels:
        epochs = extract_epochs(data_filt, sfreq, annotations, label, t_before, t_after)
        if not epochs:
            continue

        # Baseline correction : soustrait la moyenne de [-t_before, 0] par canal
        n_bl = n_before
        corrected = []
        for ep in epochs:
            baseline = ep[:, :n_bl].mean(axis=1, keepdims=True)
            corrected.append(ep - baseline)

        stack = np.stack(corrected, axis=0)  # (n_epochs, n_ch, n_samples)
        mean  = np.mean(stack, axis=0)
        std   = np.std(stack,  axis=0)
        result[label] = (mean, std, times, len(epochs))

    return result


def epoch_band_power(
    data_uv: np.ndarray,
    sfreq: float,
    annotations: list[dict],
    t_before: float,
    t_after: float,
    labels: list[str] | None = None,
    notch_freq: float = 50.0,
) -> dict[str, dict[str, np.ndarray]]:
    """
    Calcule la puissance moyenne par bande et par canal pour chaque condition.

    Returns:
        dict[label][band_name] = (n_ch,) array de puissance moyenne (µV²/Hz)
    """
    data_filt = filter_signal(
        data_uv, sfreq,
        l_freq=1.0, h_freq=50.0, notch_freq=notch_freq,
        causal=False,
    )
    data_filt = apply_car(data_filt)

    if labels is None:
        seen = set()
        labels = []
        for ann in annotations:
            lb = ann.get('label', '')
            if lb and lb not in seen:
                seen.add(lb)
                labels.append(lb)

    result: dict[str, dict[str, np.ndarray]] = {}
    n_ch = data_filt.shape[0]

    for label in labels:
        epochs = extract_epochs(data_filt, sfreq, annotations, label, t_before, t_after)
        if not epochs:
            continue

        band_powers: dict[str, np.ndarray] = {}
        for band_name, (f_low, f_high) in FREQ_BANDS.items():
            per_epoch_ch = []
            for ep in epochs:
                ch_powers = []
                for ch in range(n_ch):
                    sig_v = ep[ch] * 1e-6
                    freqs, psd = compute_psd_welch(sig_v, sfreq, fmin=f_low, fmax=f_high)
                    ch_powers.append(float(np.mean(psd)))
                per_epoch_ch.append(ch_powers)
            band_powers[band_name] = np.mean(per_epoch_ch, axis=0)  # (n_ch,)
        result[label] = band_powers

    return result
