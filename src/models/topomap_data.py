from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class TopoMapData:
    ch_names: list[str]
    by_class: dict[str, np.ndarray] = field(default_factory=dict)
    # label → (n_ch,) valeurs scalaires
    unit: str = "µV"
    clim: tuple[float, float] = (0.0, 1.0)
    mode: str = "amplitude"           # "amplitude" | "power"
    window_label: str = ""            # ex: "0.1–0.5 s" ou "8–13 Hz"
