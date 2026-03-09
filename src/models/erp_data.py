from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ERPData:
    times: np.ndarray                            # (n_times,) s relatifs à l'onset
    ch_names: list[str]                          # canaux sélectionnés
    erp_by_class: dict[str, np.ndarray] = field(default_factory=dict)
    # label → (n_ch, n_times) en µV
    baseline_corrected: bool = False
    baseline_tmin: float = 0.0
    baseline_tmax: float = 0.0
