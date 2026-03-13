from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ProtocolPreset:
    name: str
    run_label: str
    n_trials_per_class: int
    t_baseline_s: float
    t_cue_s: float
    t_rest_s: float
    classes: list[str]         # identifiants internes envoyés au worker
    class_labels: list[str]    # textes affichés dans les checkboxes
    cue_display_map: dict[str, str] = field(default_factory=dict)
    # class_name → texte affiché dans le cue label pendant l'enregistrement
    annotation_labels: list[str] = field(default_factory=lambda: ["T1", "T2"])
    # labels EDF écrits dans les annotations (un par classe, dans le même ordre que classes)
