from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AcquisitionConfig:
    serial_port: str
    ch_names: list[str] = field(
        default_factory=lambda: ["Fz", "C3", "Cz", "C4", "Pz", "PO3", "Oz", "PO4"]
    )
    subject_id: str = "S001"
    run_label: str = "run01"
    output_dir: str = "data/custom/"
    n_trials_per_class: int = 20
    t_baseline_s: float = 2.0
    t_cue_s: float = 4.0
    t_rest_s: float = 1.5
    classes: list[str] = field(default_factory=lambda: ["left", "right"])
