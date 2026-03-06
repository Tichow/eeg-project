"""
Chargement et visualisation interactive des données PhysioNet EEGMMIDB
dans le dashboard PyQt.

Dataset : EEG Motor Movement/Imagery Dataset
URL     : https://physionet.org/content/eegmmidb/1.0.0/
Format  : EDF+, 64 canaux, 160 Hz, annotations T0/T1/T2

Usage direct :
    python physionet_eeg.py              # menu interactif
    python physionet_eeg.py 1 2          # sujet 1, run 2 (yeux fermés)
    python physionet_eeg.py 1 1 2        # comparaison run 1 vs run 2

Voir doc/physionet_eegmmidb.md pour la description complète du dataset.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

# Notch US (PhysioNet est enregistré aux USA → 60 Hz secteur)
_NOTCH_FREQ = 60.0
_DATA_DIR   = 'data'

# Canaux affichés par défaut au démarrage (preset Occipital)
_PRESET_DEFAULT = {'O1', 'OZ', 'O2', 'P3', 'PZ', 'P4', 'C3', 'C4'}

# Description des runs pour affichage
_RUN_LABELS: dict[int, str] = {
    1:  'Baseline yeux ouverts',
    2:  'Baseline yeux fermés',
    3:  'Imagerie motrice — main G/D (cible)',
    4:  'Imagerie motrice — main G/D (alternance)',
    5:  'Imagerie motrice — 2 mains / 2 pieds (cible)',
    6:  'Imagerie motrice — 2 mains / 2 pieds (alternance)',
    7:  'Imagerie motrice — main G/D (cible)',
    8:  'Imagerie motrice — main G/D (alternance)',
    9:  'Imagerie motrice — 2 mains / 2 pieds (cible)',
    10: 'Imagerie motrice — 2 mains / 2 pieds (alternance)',
    11: 'Imagerie motrice — main G/D (cible)',
    12: 'Imagerie motrice — main G/D (alternance)',
    13: 'Imagerie motrice — 2 mains / 2 pieds (cible)',
    14: 'Imagerie motrice — 2 mains / 2 pieds (alternance)',
}


def _edf_path(subject_id: int, run_id: int, data_dir: str = _DATA_DIR) -> str:
    """Retourne le chemin vers le fichier EDF+ d'un sujet/run."""
    subj = f'S{subject_id:03d}'
    return os.path.join(data_dir, subj, f'{subj}R{run_id:02d}.edf')


def _check_file(subject_id: int, run_id: int, data_dir: str = _DATA_DIR) -> str:
    """Vérifie que le fichier EDF existe et retourne son chemin."""
    path = _edf_path(subject_id, run_id, data_dir)
    if not os.path.exists(path):
        print(f'[ERREUR] Fichier manquant : {path}')
        print('  → Lancer d\'abord : python download_data.py')
        sys.exit(1)
    return path


def load_physionet(
    subject_id: int,
    run_id: int,
    data_dir: str = _DATA_DIR,
) -> tuple[np.ndarray, float, list[str], list[dict]]:
    """
    Charge un fichier EDF+ PhysioNet et retourne les données brutes.

    Args:
        subject_id : numéro du sujet (1-109)
        run_id     : numéro du run (1-14)
        data_dir   : répertoire racine des données (défaut: 'data/')

    Returns:
        data_uv    : (n_ch, n_samples) en µV
        sfreq      : fréquence d'échantillonnage (160.0 Hz)
        ch_labels  : liste des noms de canaux (ex: ['FP1', 'FP2', ..., 'O2'])
        annotations: liste d'événements [{'time_sec': float, 'label': str}, ...]
                     Vide pour R01 et R02 (baselines sans tâche).
    """
    import mne
    from eeg_analysis import load_raw  # réutilise le nettoyage des noms de canaux

    path = _check_file(subject_id, run_id, data_dir)
    print(f'  Chargement : {path}')

    raw = load_raw(path)
    raw.load_data()

    sfreq     = raw.info['sfreq']
    ch_labels = raw.ch_names
    data_v    = raw.get_data()   # (n_ch, n_samples) en Volts
    data_uv   = data_v * 1e6    # → µV

    # Extraction des annotations TAL (T0/T1/T2)
    annotations: list[dict] = []
    if raw.annotations is not None and len(raw.annotations) > 0:
        for ann in raw.annotations:
            annotations.append({
                'time_sec': float(ann['onset']),
                'label':    ann['description'],
            })

    print(f'  {len(ch_labels)} canaux  |  {sfreq:.0f} Hz  |  '
          f'{data_uv.shape[1] / sfreq:.1f} s  |  '
          f'{len(annotations)} annotations')

    return data_uv, sfreq, ch_labels, annotations


def run_physionet_dashboard(
    subject_id: int,
    run_id: int,
    data_dir: str = _DATA_DIR,
) -> None:
    """
    Charge un run PhysioNet et l'ouvre dans le dashboard PyQt interactif.

    Les annotations T0/T1/T2 sont affichées sous forme de lignes verticales
    dans le panel Time Series.

    Args:
        subject_id : numéro du sujet (1-109)
        run_id     : numéro du run (1-14)
        data_dir   : répertoire des données
    """
    from offline_eeg import run_dashboard_from_array

    run_label  = _RUN_LABELS.get(run_id, f'Run {run_id:02d}')
    title      = f'PhysioNet S{subject_id:03d}R{run_id:02d} — {run_label}'

    print(f'\n=== {title} ===')
    data_uv, sfreq, ch_labels, annotations = load_physionet(
        subject_id, run_id, data_dir
    )

    # Visibilité initiale : preset Occipital (8 canaux), fallback 8 premiers
    initial_visible = [ch.upper() in _PRESET_DEFAULT for ch in ch_labels]
    if not any(initial_visible):
        initial_visible = [i < 8 for i in range(len(ch_labels))]

    run_dashboard_from_array(
        data_uv,
        sfreq,
        ch_labels,
        title=title,
        notch_freq=_NOTCH_FREQ,
        annotations=annotations if annotations else None,
        initial_visible=initial_visible,
    )


def run_physionet_erp(
    subject_id: int,
    run_id: int,
    data_dir: str = _DATA_DIR,
    t_before: float = 0.5,
    t_after: float = 4.0,
) -> None:
    """
    Charge un run PhysioNet et ouvre la vue ERP (moyennes par condition).

    Recommandé pour les runs d'imagerie motrice (3-14) qui contiennent T0/T1/T2.
    Les runs R01/R02 (baselines) n'ont pas d'annotations T1/T2 — la fenêtre
    affichera un message informatif dans ce cas.

    Args:
        subject_id : numéro du sujet (1-109)
        run_id     : numéro du run (1-14)
        t_before   : secondes avant l'événement (fenêtre baseline)
        t_after    : secondes après l'événement
    """
    from offline_eeg import run_erp_analysis

    run_label = _RUN_LABELS.get(run_id, f'Run {run_id:02d}')
    title     = f'ERP — PhysioNet S{subject_id:03d}R{run_id:02d} — {run_label}'

    print(f'\n=== {title} ===')
    data_uv, sfreq, ch_labels, annotations = load_physionet(
        subject_id, run_id, data_dir
    )

    if not annotations:
        print("  [INFO] Ce run ne contient pas d'annotations T0/T1/T2.")
        print("         Utilisez un run d'imagerie motrice (ex: run 3, 4, 5…).")

    run_erp_analysis(
        data_uv, sfreq, ch_labels, annotations,
        title=title, notch_freq=_NOTCH_FREQ,
        t_before=t_before, t_after=t_after,
    )


def run_physionet_comparison(
    subject_id: int,
    run_a: int,
    run_b: int,
    data_dir: str = _DATA_DIR,
) -> None:
    """
    Compare deux runs PhysioNet dans une fenêtre PSD statique (2 onglets :
    PSD absolue + Différence A−B).

    Usage principal : R01 (yeux ouverts) vs R02 (yeux fermés) pour observer
    l'effet Berger (pic alpha sur O1/Oz/O2 yeux fermés).

    Args:
        subject_id : numéro du sujet (1-109)
        run_a      : premier run
        run_b      : deuxième run
        data_dir   : répertoire des données
    """
    from offline_eeg import run_static_psd

    label_a = _RUN_LABELS.get(run_a, f'Run {run_a:02d}')
    label_b = _RUN_LABELS.get(run_b, f'Run {run_b:02d}')
    title   = f'PhysioNet S{subject_id:03d} — PSD R{run_a:02d} vs R{run_b:02d}'

    print(f'\n=== {title} ===')
    print(f'  Run A : R{run_a:02d} — {label_a}')
    print(f'  Run B : R{run_b:02d} — {label_b}')

    data_a, sfreq, ch_labels, _ = load_physionet(subject_id, run_a, data_dir)
    data_b, _,     _,          _ = load_physionet(subject_id, run_b, data_dir)

    segments = [
        (f'R{run_a:02d} — {label_a}', data_a),
        (f'R{run_b:02d} — {label_b}', data_b),
    ]
    run_static_psd(segments, sfreq, ch_labels, title=title)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def _prompt_subject_run() -> tuple[int, int]:
    """Demande sujet et run à l'utilisateur."""
    try:
        subject = int(input('  Numéro de sujet [1-109] : ').strip())
        run     = int(input('  Numéro de run   [1-14]  : ').strip())
    except (ValueError, EOFError):
        print('[ERREUR] Entrée invalide.')
        sys.exit(1)
    return subject, run


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Visualisation PhysioNet EEGMMIDB dans le dashboard PyQt'
    )
    parser.add_argument('subject', nargs='?', type=int, help='Numéro de sujet (1-109)')
    parser.add_argument('run_a',   nargs='?', type=int, help='Numéro de run principal (1-14)')
    parser.add_argument('run_b',   nargs='?', type=int,
                        help='Run de référence pour comparaison (optionnel)')
    parser.add_argument('--data-dir', default=_DATA_DIR, help='Répertoire des données EDF+')
    args = parser.parse_args()

    if args.subject and args.run_a and args.run_b:
        run_physionet_comparison(args.subject, args.run_a, args.run_b, args.data_dir)
    elif args.subject and args.run_a:
        run_physionet_dashboard(args.subject, args.run_a, args.data_dir)
    else:
        # Menu interactif
        print('\n=== PhysioNet EEGMMIDB — Dashboard PyQt ===')
        print('  [1] Explorer un run')
        print('  [2] Comparer deux runs (ex: yeux ouverts vs fermés)')
        choice = input('\n  Choix : ').strip()

        if choice == '1':
            subject, run = _prompt_subject_run()
            run_physionet_dashboard(subject, run, args.data_dir)
        elif choice == '2':
            subject = int(input('  Numéro de sujet [1-109]    : ').strip())
            run_a   = int(input('  Run principal   [1-14]     : ').strip())
            run_b   = int(input('  Run référence   [1-14]     : ').strip())
            run_physionet_comparison(subject, run_a, run_b, args.data_dir)
        else:
            print('Choix invalide.')
            sys.exit(1)


if __name__ == '__main__':
    main()
