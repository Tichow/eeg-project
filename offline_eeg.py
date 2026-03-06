"""
Dashboard EEG offline — rejoue un enregistrement .npy dans le même dashboard
que le mode temps réel (4 panels : TimeSeries, PSD, SNRBar, Spectrogram).

Usage :
    python offline_eeg.py
    python offline_eeg.py recordings/20260303_105500.npy
"""

import argparse
import json
import os
import sys

import numpy as np
from PyQt5.QtWidgets import QApplication, QFileDialog

from viz import Dashboard
from viz.offline_player import OfflinePlayer
from viz.widgets.playback import PlaybackWidget

WINDOW_SEC = 5
UPDATE_MS  = 100


def run_offline(npy_path: str = None) -> None:
    """Lance le dashboard EEG en mode offline."""
    app = QApplication.instance() or QApplication(sys.argv)

    # Sélection du fichier si non fourni
    if npy_path is None:
        rec_dir = 'recordings' if os.path.isdir('recordings') else '.'
        npy_path, _ = QFileDialog.getOpenFileName(
            None,
            'Ouvrir un enregistrement EEG',
            rec_dir,
            'Fichiers EEG (*.npy)',
        )
        if not npy_path:
            print('Annulé.')
            return

    # Chargement
    meta_path = npy_path.replace('.npy', '.json')
    if not os.path.exists(meta_path):
        print(f'[ERREUR] Métadonnées manquantes : {meta_path}')
        sys.exit(1)

    data_uv = np.load(npy_path)          # (n_ch, n_total) en µV
    with open(meta_path) as f:
        meta = json.load(f)

    sfreq     = float(meta['sfreq'])
    ch_labels = meta['channels']
    n_ch      = data_uv.shape[0]
    duration  = data_uv.shape[1] / sfreq

    print(f'Enregistrement : {os.path.basename(npy_path)}')
    print(f'  {n_ch} canaux  |  {sfreq:.0f} Hz  |  {duration:.1f} s')

    # Canaux 0-indexés — OfflinePlayer renvoie row[i] = canal i
    channels  = list(range(n_ch))
    n_samples = int(WINDOW_SEC * sfreq)

    player    = OfflinePlayer(data_uv, sfreq, update_ms=UPDATE_MS)
    pb_widget = PlaybackWidget(player)

    dashboard = Dashboard(
        board=player,
        channels=channels,
        ch_labels=ch_labels,
        sfreq=sfreq,
        n_samples=n_samples,
        window_sec=WINDOW_SEC,
        update_ms=UPDATE_MS,
        sidebar_extra=pb_widget,
    )
    dashboard.setWindowTitle(f'EEG Offline — {os.path.basename(npy_path)}')
    dashboard.showMaximized()
    print('  Astuce : barre d\'espace ou bouton ⏸ pour pause / ▶ pour reprendre')

    app.exec_()


def run_dashboard_from_array(
    data_uv: np.ndarray,
    sfreq: float,
    ch_labels: list,
    title: str = "EEG Dashboard",
    notch_freq: float = 50.0,
    annotations: list[dict] | None = None,
    initial_visible: list[bool] | None = None,
) -> None:
    """
    Lance le dashboard à partir d'un tableau numpy (n_ch, n_samples) en µV.

    Args:
        data_uv     : données EEG en µV, shape (n_ch, n_samples)
        sfreq       : fréquence d'échantillonnage en Hz
        ch_labels   : noms des canaux
        title       : titre de la fenêtre
        notch_freq  : fréquence du filtre coupe-bande (50 Hz FR, 60 Hz US/PhysioNet)
        annotations : liste d'événements [{time_sec, label}] pour les event markers
    """
    app = QApplication.instance() or QApplication(sys.argv)

    n_ch      = data_uv.shape[0]
    channels  = list(range(n_ch))
    n_samples = int(WINDOW_SEC * sfreq)

    player    = OfflinePlayer(data_uv, sfreq, update_ms=UPDATE_MS)
    pb_widget = PlaybackWidget(player)

    dashboard = Dashboard(
        board=player,
        channels=channels,
        ch_labels=ch_labels,
        sfreq=sfreq,
        n_samples=n_samples,
        window_sec=WINDOW_SEC,
        update_ms=UPDATE_MS,
        sidebar_extra=pb_widget,
        notch_freq=notch_freq,
        annotations=annotations,
        title=title,
        initial_visible=initial_visible,
    )
    dashboard.showMaximized()
    print('  Astuce : barre d\'espace ou bouton ⏸ pour pause / ▶ pour reprendre')

    app.exec_()


def run_comparison_overlay(
    data_main: np.ndarray,
    data_ref: np.ndarray,
    sfreq: float,
    ch_labels: list,
    label_main: str = "Segment principal",
    label_ref: str = "Référence",
    notch_freq: float = 50.0,
) -> None:
    """
    Lance le dashboard avec data_main en lecture et la PSD moyenne de
    data_ref affichée en pointillés sur le panel PSD.

    Args:
        data_main : segment joué en direct, (n_ch, n_samples) en µV
        data_ref  : segment de référence, (n_ch, n_samples) en µV
        sfreq     : fréquence d'échantillonnage en Hz
        ch_labels : noms des canaux
        label_main: label du segment principal (titre fenêtre)
        label_ref : label affiché dans la légende PSD pointillée
    """
    from processing import compute_psd_welch
    from processing.filter import filter_signal

    app = QApplication.instance() or QApplication(sys.argv)

    # Pré-calcul de la PSD moyenne du segment de référence
    # Filtrer d'abord pour retirer le DC offset et la dérive lente
    data_ref_filt = filter_signal(data_ref, sfreq, l_freq=1.0, h_freq=50.0,
                                  notch_freq=notch_freq, causal=False)
    ref_freqs = None
    ref_psd_per_ch = []
    for ch_data in data_ref_filt:
        freqs, psd_uv2 = compute_psd_welch(ch_data * 1e-6, sfreq)
        if ref_freqs is None:
            ref_freqs = freqs
        ref_psd_per_ch.append(psd_uv2)

    ref_psd = {
        'label':      label_ref,
        'freqs':      ref_freqs,
        'psd_per_ch': ref_psd_per_ch,
    }

    n_ch      = data_main.shape[0]
    channels  = list(range(n_ch))
    n_samples = int(WINDOW_SEC * sfreq)

    player    = OfflinePlayer(data_main, sfreq, update_ms=UPDATE_MS)
    pb_widget = PlaybackWidget(player)

    dashboard = Dashboard(
        board=player,
        channels=channels,
        ch_labels=ch_labels,
        sfreq=sfreq,
        n_samples=n_samples,
        window_sec=WINDOW_SEC,
        update_ms=UPDATE_MS,
        sidebar_extra=pb_widget,
        ref_psd=ref_psd,
        notch_freq=notch_freq,
        title=f'{label_main}  ·  PSD réf : {label_ref}',
    )
    dashboard.showMaximized()
    print(f'  Live : {label_main}  |  PSD pointillée : {label_ref}')
    print('  Astuce : barre d\'espace ou bouton ⏸ pour pause / ▶ pour reprendre')

    app.exec_()


def run_erp_analysis(
    data_uv: np.ndarray,
    sfreq: float,
    ch_labels: list,
    annotations: list[dict],
    title: str = "ERP — Moyennes",
    notch_freq: float = 50.0,
    t_before: float = 0.5,
    t_after: float = 4.0,
) -> None:
    """Lance ERPWindow à partir d'un array numpy + liste d'annotations."""
    from viz.erp_window import ERPWindow

    app = QApplication.instance() or QApplication(sys.argv)
    win = ERPWindow(
        data_uv, sfreq, ch_labels, annotations,
        t_before=t_before, t_after=t_after,
        title=title, notch_freq=notch_freq,
    )
    win.showMaximized()
    app.exec_()


def run_static_psd(
    segments: list[tuple[str, np.ndarray]],
    sfreq: float,
    ch_labels: list,
    title: str = "PSD par section",
) -> None:
    """Lance la fenêtre PSD statique (une courbe par section)."""
    from viz.static_psd_window import StaticPSDWindow

    app = QApplication.instance() or QApplication(sys.argv)
    win = StaticPSDWindow(segments, sfreq, ch_labels, title=title)
    win.show()
    app.exec_()


# ──────────────────────────────────────────────────────────────────────────────
# Analyse adaptée par type de test
# ──────────────────────────────────────────────────────────────────────────────

def _extract_segments(
    data: np.ndarray,
    sfreq: float,
    events: list[dict],
    start_actions: list[str],
    end_actions: list[str],
) -> np.ndarray:
    """
    Extrait et concatène les plages de data comprises entre un event
    start_actions et le prochain end_actions.
    Retourne data entier si aucun segment trouvé.
    """
    segments = []
    in_seg   = False
    start_s  = 0

    for ev in events:
        action = ev.get('action', '')
        t      = int(ev.get('time_sec', 0) * sfreq)
        if not in_seg and action in start_actions:
            start_s = t
            in_seg  = True
        elif in_seg and action in end_actions:
            end_s = min(t, data.shape[1])
            if end_s > start_s:
                segments.append(data[:, start_s:end_s])
            in_seg = False

    if in_seg:  # segment jusqu'à la fin de l'enregistrement
        if data.shape[1] > start_s:
            segments.append(data[:, start_s:])

    if not segments:
        return data
    return np.concatenate(segments, axis=1)


def _extract_individual_segments(
    data: np.ndarray,
    sfreq: float,
    events: list[dict],
    known_actions: dict[str, str],
    initial_label: str | None = None,
) -> list[tuple[str, np.ndarray]]:
    """
    Retourne une liste de (label, segment) — une entrée par section reconnue.
    Chaque event marque le début d'une section qui dure jusqu'au suivant event.
    Seuls les events dont l'action est dans known_actions sont retenus.

    Si `initial_label` est fourni et que le premier event reconnu démarre après t=0,
    la période initiale est ajoutée en tête avec ce label.
    """
    segments = []
    n = data.shape[1]
    filtered = [ev for ev in events if ev.get('action', '') in known_actions]

    if not filtered:
        return segments

    # Période avant le premier marqueur reconnu
    if initial_label is not None:
        first_s = int(filtered[0].get('time_sec', 0) * sfreq)
        if first_s > 0:
            segments.append((initial_label, data[:, :first_s]))

    for i, ev in enumerate(filtered):
        action  = ev['action']
        start_s = int(ev.get('time_sec', 0) * sfreq)
        end_s   = int(filtered[i + 1].get('time_sec', 0) * sfreq) if i + 1 < len(filtered) else n
        end_s   = min(end_s, n)
        if end_s > start_s:
            segments.append((known_actions[action], data[:, start_s:end_s]))

    return segments


def _complement_segments(
    data: np.ndarray,
    sfreq: float,
    events: list[dict],
    start_actions: list[str],
    end_actions: list[str],
) -> np.ndarray:
    """
    Extrait les plages qui ne correspondent PAS aux segments définis par
    start/end_actions (complément de _extract_segments).
    """
    mask = np.zeros(data.shape[1], dtype=bool)
    in_seg  = False
    start_s = 0

    for ev in events:
        action = ev.get('action', '')
        t      = int(ev.get('time_sec', 0) * sfreq)
        if not in_seg and action in start_actions:
            start_s = t
            in_seg  = True
        elif in_seg and action in end_actions:
            end_s = min(t, data.shape[1])
            mask[start_s:end_s] = True
            in_seg = False

    if in_seg:
        mask[start_s:] = True

    complement = data[:, ~mask]
    return complement if complement.shape[1] > 0 else data


def run_eyes_closed_analysis(
    data: np.ndarray,
    meta: dict,
    sfreq: float,
    ch_labels: list[str],
) -> None:
    """PSD statique par section : yeux fermés / yeux ouverts."""
    events = meta.get('events', [])
    known  = {'Fermer les yeux': 'Yeux fermés', 'Ouvrir les yeux': 'Yeux ouverts'}
    segments = _extract_individual_segments(data, sfreq, events, known,
                                            initial_label='Yeux ouverts')

    # Fallback si aucun marqueur reconnu : split au milieu
    if not segments:
        mid = data.shape[1] // 2
        segments = [('Yeux ouverts', data[:, :mid]), ('Yeux fermés', data[:, mid:])]

    subject = meta.get('subject', '')
    run_static_psd(
        segments, sfreq, ch_labels,
        title=f"PSD yeux ouverts / fermés — {subject}",
    )


def run_blink_analysis(
    data: np.ndarray,
    meta: dict,
    sfreq: float,
    ch_labels: list[str],
) -> None:
    """Dashboard standard, canaux frontaux (CH1/CH2) en avant."""
    run_dashboard_from_array(
        data, sfreq, ch_labels,
        title=f"Clignements — {meta.get('subject', '')}",
    )


def run_hand_movement_analysis(
    data: np.ndarray,
    meta: dict,
    sfreq: float,
    ch_labels: list[str],
) -> None:
    """PSD statique par section : mouvement gauche / droit / repos."""
    events = meta.get('events', [])
    known  = {
        'Mouvement gauche': 'Mouvement gauche',
        'Mouvement droit':  'Mouvement droit',
        'Pause':            'Repos',
    }
    segments = _extract_individual_segments(data, sfreq, events, known,
                                            initial_label='Repos')

    # Fallback si aucun marqueur reconnu : split au milieu
    if not segments:
        mid = data.shape[1] // 2
        segments = [('Mouvement', data[:, :mid]), ('Repos', data[:, mid:])]

    subject = meta.get('subject', '')
    run_static_psd(
        segments, sfreq, ch_labels,
        title=f"PSD mouvement des mains — {subject}",
    )


def run_flashing_analysis(
    data: np.ndarray,
    meta: dict,
    sfreq: float,
    ch_labels: list[str],
) -> None:
    """Dashboard standard, canaux occipitaux (CH7/CH8) en avant."""
    run_dashboard_from_array(
        data, sfreq, ch_labels,
        title=f"Stimulis clignotant — {meta.get('subject', '')}",
    )


def run_recording_analysis(npy_path: str, meta: dict) -> None:
    """Dispatche vers la vue adaptée selon le test_type du meta."""
    data      = np.load(npy_path)
    sfreq     = float(meta.get('sfreq', 250))
    ch_labels = meta.get('channels', [f'CH{i+1}' for i in range(data.shape[0])])
    tt        = meta.get('test_type', '')

    subject = meta.get('subject', '')
    print(f'\n  [{tt}] {subject}  —  {meta.get("duration_sec", "?")} s')

    if tt == 'eyes_closed':
        run_eyes_closed_analysis(data, meta, sfreq, ch_labels)
    elif tt == 'blink':
        run_blink_analysis(data, meta, sfreq, ch_labels)
    elif tt == 'hand_movement':
        run_hand_movement_analysis(data, meta, sfreq, ch_labels)
    else:
        run_flashing_analysis(data, meta, sfreq, ch_labels)


# ──────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Dashboard EEG offline — rejoue un enregistrement .npy'
    )
    parser.add_argument(
        'npy_path', nargs='?', default=None,
        help='Chemin vers le fichier .npy (optionnel, sinon file picker Qt)'
    )
    args = parser.parse_args()
    run_offline(npy_path=args.npy_path)
