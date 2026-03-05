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
    dashboard.resize(1400, 1000)
    dashboard.show()
    print('  Astuce : barre d\'espace ou bouton ⏸ pour pause / ▶ pour reprendre')

    app.exec_()


def run_dashboard_from_array(
    data_uv: np.ndarray,
    sfreq: float,
    ch_labels: list,
    title: str = "EEG Dashboard",
) -> None:
    """Lance le dashboard à partir d'un tableau numpy (n_ch, n_samples) en µV."""
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
    )
    dashboard.setWindowTitle(title)
    dashboard.resize(1400, 1000)
    dashboard.show()
    print('  Astuce : barre d\'espace ou bouton ⏸ pour pause / ▶ pour reprendre')

    app.exec_()


def run_comparison_overlay(
    data_main: np.ndarray,
    data_ref: np.ndarray,
    sfreq: float,
    ch_labels: list,
    label_main: str = "Segment principal",
    label_ref: str = "Référence",
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

    app = QApplication.instance() or QApplication(sys.argv)

    # Pré-calcul de la PSD moyenne du segment de référence
    ref_freqs = None
    ref_psd_per_ch = []
    for ch_data in data_ref:
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
    )
    dashboard.setWindowTitle(f'{label_main}  ·  PSD réf : {label_ref}')
    dashboard.resize(1400, 1000)
    dashboard.show()
    print(f'  Live : {label_main}  |  PSD pointillée : {label_ref}')
    print('  Astuce : barre d\'espace ou bouton ⏸ pour pause / ▶ pour reprendre')

    app.exec_()


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
