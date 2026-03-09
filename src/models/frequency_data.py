from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class FrequencyData:
    mode: str  # "psd" | "erders"
    freqs: np.ndarray  # (n_freqs,) Hz
    ch_names: list[str]  # canaux sélectionnés
    psd_by_class: dict[str, np.ndarray] = field(default_factory=dict)
    # label → (n_ch, n_freqs) µV²/Hz
    times: np.ndarray | None = None  # (n_times,) s relatifs — ERD/ERS seulement
    erders_by_class: dict[str, np.ndarray] = field(default_factory=dict)
    # label → (n_ch, n_times) %
    baseline_label: str = "T0"
