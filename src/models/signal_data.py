from dataclasses import dataclass, field

import numpy as np


@dataclass
class SignalData:
    data: np.ndarray                          # (n_channels, n_samples), Volts
    times: np.ndarray                         # (n_samples,), seconds
    ch_names: list[str]                       # channel name strings
    sfreq: float                              # sampling frequency
    annotations: list[tuple[float, float, str]] = field(default_factory=list)
    # each tuple: (onset_s, duration_s, description)
