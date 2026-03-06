from dataclasses import dataclass, field

import numpy as np


@dataclass
class EpochData:
    data: np.ndarray       # (n_epochs, n_channels, n_times), Volts
    times: np.ndarray      # (n_times,) secondes relatives à l'onset
    ch_names: list[str]
    sfreq: float
    labels: list[str]      # longueur n_epochs — ex. ["T0", "T1", "T2"]
    onsets: list[float]    # longueur n_epochs — onset absolu en secondes
