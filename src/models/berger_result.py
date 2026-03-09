from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class BergerResult:
    alpha_open: float           # pouvoir alpha yeux ouverts (µV²/Hz)
    alpha_closed: float         # pouvoir alpha yeux fermés (µV²/Hz)
    ratio: float                # alpha_closed / alpha_open
    quality: str                # "Excellent" | "Bon" | "Moyen" | "Faible"
    color: str                  # couleur CSS hex
    channels_used: list[str]    # canaux moyennés
    freqs: np.ndarray           # vecteur fréquences (plage 1–30 Hz)
    psd_open: np.ndarray        # PSD moyennée R01 (µV²/Hz), shape (n_freqs,)
    psd_closed: np.ndarray      # PSD moyennée R02 (µV²/Hz), shape (n_freqs,)
