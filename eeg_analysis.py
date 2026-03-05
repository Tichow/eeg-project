"""
Pipeline d'analyse EEG — Reproduction de "EEG 101 using OpenBCI Ultracortex"
(Towards Data Science) sur le dataset PhysioNet EEGMMIDB.

Dataset    : PhysioNet EEGMMIDB — 64 canaux, 160 Hz, format EDF+
Objectifs  :
  1. Chargement des fichiers EDF+
  2. Visualisation du signal brut (time series)
  3. Filtrage : bandpass 1-50 Hz (Butterworth ordre 4) + notch 60 Hz
  4. PSD (Welch) avec visualisation des bandes fréquentielles
  5. Comparaison yeux ouverts vs yeux fermés → pic alpha (~10 Hz) sur O1/Oz/O2
  6. Imagerie motrice → rythme mu (8-12 Hz) sur C3/C4

Adaptabilité OpenBCI Cyton :
  - Toutes les fonctions acceptent sfreq et channels en paramètres
  - Changer sfreq=160 → 250, et adapter la liste de canaux
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import mne

from processing import FREQ_BANDS, BAND_COLORS, filter_signal, compute_psd_welch

mne.set_log_level("WARNING")  # Évite le verbeux MNE


# ---------------------------------------------------------------------------
# 1. Chargement
# ---------------------------------------------------------------------------

def load_raw(filepath: str) -> mne.io.BaseRaw:
    """
    Charge un fichier EDF+ avec MNE et normalise les noms de canaux.

    Le dataset PhysioNet préfixe parfois les noms avec '.' (ex: '.O1').
    On les retire pour un accès uniforme.

    Args:
        filepath : chemin vers le fichier .edf

    Returns:
        raw : objet MNE Raw (non chargé en mémoire → lazy loading)
    """
    raw = mne.io.read_raw_edf(filepath, preload=False, verbose=False)

    # Nettoyage : retire les '.' en début et fin (ex: 'C3..' → 'C3', '.O1' → 'O1')
    rename_map = {}
    for ch in raw.ch_names:
        cleaned = ch.strip(".").strip().upper()
        if cleaned != ch:
            rename_map[ch] = cleaned
    if rename_map:
        raw.rename_channels(rename_map)

    return raw


def load_recording(npy_path: str) -> mne.io.BaseRaw:
    """
    Charge un enregistrement sauvegardé par realtime_eeg.py.

    Lit le fichier .npy et son .json de métadonnées associé,
    puis retourne un objet MNE RawArray compatible avec toutes
    les fonctions d'analyse (apply_filters, compute_psd, etc.).

    Args:
        npy_path : chemin vers le fichier .npy (ex: recordings/20260303_105500.npy)

    Returns:
        raw : objet MNE RawArray prêt à l'analyse
    """
    meta_path = npy_path.replace(".npy", ".json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Métadonnées manquantes : {meta_path}")

    data = np.load(npy_path) * 1e-6  # BrainFlow stocke en µV → conversion V pour MNE

    with open(meta_path) as f:
        meta = json.load(f)

    sfreq = meta["sfreq"]
    ch_names = meta["channels"]

    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    raw = mne.io.RawArray(data, info, verbose=False)
    return raw


# ---------------------------------------------------------------------------
# 2. Visualisation du signal brut
# ---------------------------------------------------------------------------

def plot_raw_signal(
    raw: mne.io.BaseRaw,
    channels: list,
    duration: float = 10.0,
    title: str = "Signal EEG brut",
) -> None:
    """
    Affiche le signal brut (time series) pour une liste de canaux.

    Args:
        raw      : objet MNE Raw
        channels : liste de noms de canaux (ex: ['O1', 'Oz', 'O2'])
        duration : fenêtre temporelle à afficher en secondes
        title    : titre de la figure
    """
    raw.load_data()
    sfreq = raw.info["sfreq"]
    n_samples = int(duration * sfreq)

    # Sélection des canaux demandés (intersection avec ce qui est disponible)
    available = [ch for ch in channels if ch in raw.ch_names]
    if not available:
        print(f"[WARNING] Aucun canal trouvé parmi {channels}")
        return

    data, times = raw.get_data(picks=available, start=0, stop=n_samples, return_times=True)

    # Échelle Y commune basée sur les données réelles (±3σ, min ±30 µV)
    data_uv = data * 1e6
    global_std = np.std(data_uv)
    ylim = max(30.0, 3.0 * global_std)

    fig, axes = plt.subplots(len(available), 1, figsize=(12, 2 * len(available)), sharex=True)
    if len(available) == 1:
        axes = [axes]

    fig.suptitle(title, fontsize=13)

    colors = plt.cm.tab10.colors
    for i, (ax, ch_name) in enumerate(zip(axes, available)):
        signal_uv = data_uv[i]
        ax.plot(times, signal_uv, lw=0.7, color=colors[i % len(colors)])
        ax.set_ylim(-ylim, ylim)
        ax.set_ylabel(f"{ch_name}\n(µV)", fontsize=9)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Temps (s)")
    plt.tight_layout()
    plt.show(block=False)


# ---------------------------------------------------------------------------
# 3. Filtrage
# ---------------------------------------------------------------------------

def apply_filters(
    raw: mne.io.BaseRaw,
    l_freq: float = 1.0,
    h_freq: float = 50.0,
    notch_freq: float = 60.0,
) -> mne.io.BaseRaw:
    """
    Applique un filtre passe-bande Butterworth + filtre notch sur le signal.

    Le filtrage est fait in-place sur une copie pour ne pas modifier l'original.

    Args:
        raw        : objet MNE Raw (sera copié)
        l_freq     : borne basse du passe-bande (Hz)
        h_freq     : borne haute du passe-bande (Hz)
        notch_freq : fréquence du filtre coupe-bande (60 Hz pour données US)

    Returns:
        raw_filtered : copie filtrée
    """
    raw_filtered = raw.copy().load_data()

    # Délègue à eeg_processing.filter_signal (scipy, zéro-phase en mode offline)
    raw_filtered._data = filter_signal(
        raw_filtered.get_data(), raw_filtered.info["sfreq"],
        l_freq=l_freq, h_freq=h_freq, notch_freq=notch_freq, causal=False,
    )

    return raw_filtered


# ---------------------------------------------------------------------------
# 4. PSD avec bandes colorées
# ---------------------------------------------------------------------------

def compute_psd(
    raw: mne.io.BaseRaw,
    channels: list,
    fmin: float = 0.5,
    fmax: float = 60.0,
    title: str = "Densité Spectrale de Puissance (PSD)",
) -> None:
    """
    Calcule et affiche la PSD via la méthode de Welch.
    Les bandes delta/theta/alpha/beta sont mises en évidence.

    Args:
        raw      : objet MNE Raw (filtré de préférence)
        channels : liste de canaux à afficher
        fmin     : fréquence minimale affichée (Hz)
        fmax     : fréquence maximale affichée (Hz)
        title    : titre de la figure
    """
    raw.load_data()
    sfreq = raw.info["sfreq"]

    available = [ch for ch in channels if ch in raw.ch_names]
    if not available:
        print(f"[WARNING] Aucun canal trouvé parmi {channels}")
        return

    data, _ = raw.get_data(picks=available, return_times=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle(title, fontsize=13)

    # Fond coloré par bande
    for band_name, (f_low, f_high) in FREQ_BANDS.items():
        ax.axvspan(f_low, f_high, alpha=0.12, color=BAND_COLORS[band_name])

    colors = plt.cm.tab10.colors
    for i, (ch_name, signal) in enumerate(zip(available, data)):
        freqs, psd_uv2 = compute_psd_welch(signal, sfreq, fmin=fmin, fmax=fmax)
        ax.semilogy(freqs, psd_uv2, lw=1.5,
                    label=ch_name, color=colors[i % len(colors)])

    ax.set_xlabel("Fréquence (Hz)")
    ax.set_ylabel("Puissance (µV²/Hz)")
    ax.legend(loc="upper right")
    ax.grid(True, which="both", alpha=0.3)

    # Légende des bandes
    patches = [
        mpatches.Patch(color=BAND_COLORS[b], alpha=0.4, label=f"{b} ({f[0]}-{f[1]} Hz)")
        for b, f in FREQ_BANDS.items()
    ]
    ax.legend(handles=patches + ax.get_lines(), loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.show(block=False)


# ---------------------------------------------------------------------------
# 5. Comparaison yeux ouverts vs yeux fermés (effet Berger)
# ---------------------------------------------------------------------------

def compare_alpha(
    raw_eo: mne.io.BaseRaw,
    raw_ec: mne.io.BaseRaw,
    occ_channels: list = None,
) -> None:
    """
    Superpose les PSD yeux ouverts (EO) et yeux fermés (EC) sur les canaux
    occipitaux pour observer l'effet Berger (pic alpha à ~10 Hz yeux fermés).

    Args:
        raw_eo       : Raw filtré — baseline yeux ouverts (R01)
        raw_ec       : Raw filtré — baseline yeux fermés  (R02)
        occ_channels : canaux occipitaux (défaut: O1, Oz, O2)
    """
    if occ_channels is None:
        occ_channels = ["O1", "OZ", "O2"]

    for raw in (raw_eo, raw_ec):
        raw.load_data()

    sfreq = raw_eo.info["sfreq"]

    # Intersection avec les canaux disponibles
    available = [ch for ch in occ_channels if ch in raw_eo.ch_names and ch in raw_ec.ch_names]
    if not available:
        print(f"[WARNING] Canaux occipitaux non trouvés : {occ_channels}")
        print(f"  Canaux disponibles : {raw_eo.ch_names[:10]} ...")
        return

    n_cols = len(available)
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5), sharey=True)
    if n_cols == 1:
        axes = [axes]

    fig.suptitle("Effet Berger — Comparaison yeux ouverts vs fermés\n(bande alpha 8-13 Hz)", fontsize=13)

    for ax, ch_name in zip(axes, available):
        for raw, label, color, ls in [
            (raw_eo, "Yeux ouverts", "#4e79a7", "-"),
            (raw_ec, "Yeux fermés",  "#e15759", "--"),
        ]:
            signal = raw.get_data(picks=[ch_name])[0]
            freqs, psd_uv2 = compute_psd_welch(signal, sfreq, fmin=1, fmax=40)
            ax.semilogy(freqs, psd_uv2, lw=1.8,
                        label=label, color=color, linestyle=ls)

        # Highlight bande alpha
        ax.axvspan(8, 13, alpha=0.15, color=BAND_COLORS["Alpha"], label="Alpha (8-13 Hz)")
        ax.set_title(ch_name)
        ax.set_xlabel("Fréquence (Hz)")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)

    axes[0].set_ylabel("Puissance (µV²/Hz)")
    plt.tight_layout()
    plt.show(block=False)


# ---------------------------------------------------------------------------
# 6. Rythme mu — imagerie motrice
# ---------------------------------------------------------------------------

def plot_mu_rhythm(
    raw_motor: mne.io.BaseRaw,
    channels: list = None,
) -> None:
    """
    Affiche la PSD sur les canaux moteurs centraux (C3/C4) et met en évidence
    le rythme mu (8-12 Hz) lié à l'imagerie motrice.

    Args:
        raw_motor : Raw filtré — run d'imagerie motrice (R04/R08/R12)
        channels  : canaux moteurs (défaut: C3, CZ, C4)
    """
    if channels is None:
        channels = ["C3", "CZ", "C4"]

    raw_motor.load_data()
    sfreq = raw_motor.info["sfreq"]

    available = [ch for ch in channels if ch in raw_motor.ch_names]
    if not available:
        print(f"[WARNING] Canaux moteurs non trouvés : {channels}")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle("Rythme mu (imagerie motrice) — Canaux C3/C4", fontsize=13)

    # Zones de référence
    ax.axvspan(8, 12, alpha=0.2, color="#f28e2b", label="Rythme mu (8-12 Hz)")
    ax.axvspan(13, 30, alpha=0.1, color="#e15759", label="Beta (13-30 Hz)")

    colors = plt.cm.Set1.colors
    for i, ch_name in enumerate(available):
        signal = raw_motor.get_data(picks=[ch_name])[0]
        freqs, psd_uv2 = compute_psd_welch(signal, sfreq, fmin=1, fmax=50)
        ax.semilogy(freqs, psd_uv2, lw=1.8,
                    label=ch_name, color=colors[i % len(colors)])

    ax.set_xlabel("Fréquence (Hz)")
    ax.set_ylabel("Puissance (µV²/Hz)")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.show(block=False)


# ---------------------------------------------------------------------------
# Script principal
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    DATA_DIR = "data"
    SUBJECT = "S001"

    # Chemins des fichiers
    path_eo    = os.path.join(DATA_DIR, SUBJECT, f"{SUBJECT}R01.edf")  # yeux ouverts
    path_ec    = os.path.join(DATA_DIR, SUBJECT, f"{SUBJECT}R02.edf")  # yeux fermés
    path_motor = os.path.join(DATA_DIR, SUBJECT, f"{SUBJECT}R04.edf")  # imagerie motrice

    for p in (path_eo, path_ec, path_motor):
        if not os.path.exists(p):
            print(f"[ERREUR] Fichier manquant : {p}")
            print("  → Lancer d'abord : python download_data.py")
            exit(1)

    # -- Étape 1 : Chargement --------------------------------------------------
    print("Chargement des fichiers EDF...")
    raw_eo    = load_raw(path_eo)
    raw_ec    = load_raw(path_ec)
    raw_motor = load_raw(path_motor)

    print(f"  Canaux disponibles ({len(raw_eo.ch_names)}) : {raw_eo.ch_names[:8]} ...")
    print(f"  Fréquence d'échantillonnage : {raw_eo.info['sfreq']} Hz")

    # Canaux d'intérêt (10-20 system, nommage PhysioNet en majuscules)
    OCC_CHANNELS   = ["O1", "OZ", "O2"]
    MOTOR_CHANNELS = ["C3", "CZ", "C4"]
    DISPLAY_CH     = ["FP1", "C3", "O1", "C4", "O2"]

    # -- Étape 2 : Signal brut -------------------------------------------------
    print("\nAffichage du signal brut...")
    plot_raw_signal(
        raw_eo, channels=DISPLAY_CH, duration=10,
        title=f"{SUBJECT} R01 — Signal brut (yeux ouverts, 10 s)"
    )

    # -- Étape 3 : Filtrage ----------------------------------------------------
    print("\nFiltrage bandpass 1-50 Hz + notch 60 Hz...")
    raw_eo_filt    = apply_filters(raw_eo,    l_freq=1, h_freq=50, notch_freq=60)
    raw_ec_filt    = apply_filters(raw_ec,    l_freq=1, h_freq=50, notch_freq=60)
    raw_motor_filt = apply_filters(raw_motor, l_freq=1, h_freq=50, notch_freq=60)
    print("  Filtrage terminé.")

    # -- Étape 4 : PSD globale -------------------------------------------------
    print("\nCalcul de la PSD (signal filtré, yeux ouverts)...")
    compute_psd(
        raw_eo_filt,
        channels=DISPLAY_CH,
        fmin=0.5, fmax=55,
        title=f"{SUBJECT} R01 — PSD filtrée (yeux ouverts)",
    )

    # -- Étape 5 : Comparaison yeux ouverts vs yeux fermés (effet Berger) ------
    print("\nComparaison alpha : yeux ouverts vs yeux fermés...")
    compare_alpha(raw_eo_filt, raw_ec_filt, occ_channels=OCC_CHANNELS)

    # -- Étape 6 : Rythme mu — imagerie motrice --------------------------------
    print("\nVisualisation du rythme mu (imagerie motrice)...")
    plot_mu_rhythm(raw_motor_filt, channels=MOTOR_CHANNELS)

    print("\nAnalyse terminée. Ferme les fenêtres pour quitter.")
    plt.show()
