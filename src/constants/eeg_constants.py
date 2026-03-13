from src.models.protocol_preset import ProtocolPreset

# Electrodes disponibles sur le casque Ultracortex Mark IV (35 positions officielles)
# T3/T4/T5/T6 = ancienne notation 10-20 (= T7/T8/P7/P8 dans la nomenclature moderne)
PRESET_ELECTRODES: list[str] = [
    "Fp1", "Fpz", "Fp2",
    "AF3", "AFz", "AF4",
    "F7", "F3", "Fz", "F4", "F8",
    "FC5", "FC1", "FC2", "FC6",
    "T3", "C3", "Cz", "C4", "T4",
    "CP5", "CP1", "CP2", "CP6",
    "T5", "P3", "Pz", "P4", "T6",
    "PO3", "POz", "PO4",
    "O1", "Oz", "O2",
]

SUBJECT_MIN = 1
SUBJECT_MAX = 109

DEFAULT_DATA_PATH = "data/"

RUN_DESCRIPTIONS: dict[int, str] = {
    1:  "Baseline — yeux ouverts",
    2:  "Baseline — yeux fermés",
    3:  "Imagerie motrice — main G/D (cible)",
    4:  "Imagerie motrice — main G/D (alternance)",
    5:  "Imagerie motrice — 2 mains / 2 pieds (cible)",
    6:  "Imagerie motrice — 2 mains / 2 pieds (alternance)",
    7:  "Imagerie motrice — main G/D (cible)",
    8:  "Imagerie motrice — main G/D (alternance)",
    9:  "Imagerie motrice — 2 mains / 2 pieds (cible)",
    10: "Imagerie motrice — 2 mains / 2 pieds (alternance)",
    11: "Imagerie motrice — main G/D (cible)",
    12: "Imagerie motrice — main G/D (alternance)",
    13: "Imagerie motrice — 2 mains / 2 pieds (cible)",
    14: "Imagerie motrice — 2 mains / 2 pieds (alternance)",
}

ACQUISITION_PROTOCOLS: list[ProtocolPreset] = [
    ProtocolPreset(
        name="Personnalisé",
        run_label="run01",
        n_trials_per_class=20,
        t_baseline_s=2.0,
        t_cue_s=4.0,
        t_rest_s=1.5,
        classes=["left", "right"],
        class_labels=["Gauche (T1)", "Droite (T2)"],
        cue_display_map={"left": "← Gauche", "right": "→ Droite"},
    ),
    ProtocolPreset(
        name="Baseline — yeux ouverts (R01)",
        run_label="R01",
        n_trials_per_class=1,
        t_baseline_s=0.0,
        t_cue_s=60.0,
        t_rest_s=0.0,
        classes=["repos"],
        class_labels=["Repos (T0)"],
        cue_display_map={"repos": "✛"},
        annotation_labels=["T0"],
    ),
    ProtocolPreset(
        name="Baseline — yeux fermés (R02)",
        run_label="R02",
        n_trials_per_class=1,
        t_baseline_s=0.0,
        t_cue_s=60.0,
        t_rest_s=0.0,
        classes=["repos"],
        class_labels=["Repos (T0)"],
        cue_display_map={"repos": "✛"},
        annotation_labels=["T0"],
    ),
    ProtocolPreset(
        name="Tâche 1 — Mouvement main G/D (R03)",
        run_label="R03",
        n_trials_per_class=15,
        t_baseline_s=2.0,
        t_cue_s=4.0,
        t_rest_s=2.0,
        classes=["left_fist", "right_fist"],
        class_labels=["Poing gauche (T1)", "Poing droit (T2)"],
        cue_display_map={"left_fist": "← Poing G", "right_fist": "→ Poing D"},
    ),
    ProtocolPreset(
        name="Tâche 2 — Imagerie main G/D (R04)",
        run_label="R04",
        n_trials_per_class=15,
        t_baseline_s=2.0,
        t_cue_s=4.0,
        t_rest_s=2.0,
        classes=["left_fist_img", "right_fist_img"],
        class_labels=["Imagerie poing G (T1)", "Imagerie poing D (T2)"],
        cue_display_map={"left_fist_img": "← Imagerie G", "right_fist_img": "→ Imagerie D"},
    ),
    ProtocolPreset(
        name="Tâche 3 — Mouvement 2 mains / 2 pieds (R05)",
        run_label="R05",
        n_trials_per_class=15,
        t_baseline_s=2.0,
        t_cue_s=4.0,
        t_rest_s=2.0,
        classes=["both_fists", "both_feet"],
        class_labels=["2 poings (T1)", "2 pieds (T2)"],
        cue_display_map={"both_fists": "↑↑ 2 Poings", "both_feet": "↓↓ 2 Pieds"},
    ),
    ProtocolPreset(
        name="Tâche 4 — Imagerie 2 mains / 2 pieds (R06)",
        run_label="R06",
        n_trials_per_class=15,
        t_baseline_s=2.0,
        t_cue_s=4.0,
        t_rest_s=2.0,
        classes=["both_fists_img", "both_feet_img"],
        class_labels=["Imagerie 2 poings (T1)", "Imagerie 2 pieds (T2)"],
        cue_display_map={"both_fists_img": "↑↑ Imagerie mains", "both_feet_img": "↓↓ Imagerie pieds"},
    ),
]
