"""
Visualisation EEG temps réel — OpenBCI Cyton via BrainFlow.

Affiche deux panneaux mis à jour en continu :
  - Gauche  : time series (fenêtre glissante de WINDOW_SEC secondes)
  - Droite  : PSD en temps réel (Welch sur la même fenêtre)

Utilise eeg_processing.py pour le filtrage et le calcul de PSD,
exactement comme eeg_analysis.py — seule la source de données change.

Usage :
    python realtime_eeg.py
    python realtime_eeg.py --port /dev/cu.usbserial-XXXXX
    python realtime_eeg.py --port /dev/cu.usbserial-XXXXX --channels 1 2 5 6
"""

import argparse
import json
import os
import time
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds

from eeg_processing import FREQ_BANDS, BAND_COLORS, filter_signal, compute_psd_welch, band_power


# ---------------------------------------------------------------------------
# Configuration par défaut
# ---------------------------------------------------------------------------

DEFAULT_PORT    = "/dev/cu.usbserial-DM03H2DU"
BOARD_ID        = BoardIds.CYTON_BOARD.value
WINDOW_SEC      = 5       # secondes de signal affichées
UPDATE_MS       = 100     # intervalle de rafraîchissement (ms) → 10 FPS
GAIN_CMD        = "x{ch}030110X"  # gain 6x (commande OpenBCI)

# Filtre (même réglage que eeg_analysis.py, mais causal=False sur fenêtre glissante)
L_FREQ      = 1.0
H_FREQ      = 50.0
NOTCH_FREQ  = 50.0   # 50 Hz Europe (secteur français)

COLORS = ['#e6194b', '#3cb44b', '#4363d8', '#f58231',
          '#911eb4', '#42d4f4', '#f032e6', '#bfef45']

RECORDINGS_DIR = "recordings"


# ---------------------------------------------------------------------------
# Sauvegarde
# ---------------------------------------------------------------------------

def save_recording(data: np.ndarray, sfreq: float, ch_labels: list) -> str:
    """
    Sauvegarde un enregistrement EEG brut.

    Crée deux fichiers dans recordings/ :
      - <timestamp>.npy  : données brutes (n_channels, n_samples) en Volts
      - <timestamp>.json : métadonnées (sfreq, canaux, durée, date)

    Args:
        data      : array (n_channels, n_samples) en Volts
        sfreq     : fréquence d'échantillonnage
        ch_labels : noms des canaux (ex: ['CH1', 'CH2', ...])

    Returns:
        Chemin du fichier .npy sauvegardé
    """
    os.makedirs(RECORDINGS_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    npy_path  = os.path.join(RECORDINGS_DIR, f"{ts}.npy")
    meta_path = os.path.join(RECORDINGS_DIR, f"{ts}.json")

    np.save(npy_path, data)

    meta = {
        "timestamp": ts,
        "sfreq": sfreq,
        "channels": ch_labels,
        "n_samples": data.shape[1],
        "duration_sec": round(data.shape[1] / sfreq, 2),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n  [REC] Sauvegardé : {npy_path} ({meta['duration_sec']} s)")
    return npy_path


# ---------------------------------------------------------------------------
# Setup BrainFlow
# ---------------------------------------------------------------------------

def setup_board(port: str) -> tuple:
    """
    Initialise et démarre la session BrainFlow.

    Returns:
        (board, sfreq, eeg_channels)
    """
    params = BrainFlowInputParams()
    params.serial_port = port

    board = BoardShim(BOARD_ID, params)
    sfreq = BoardShim.get_sampling_rate(BOARD_ID)
    eeg_channels = BoardShim.get_eeg_channels(BOARD_ID)

    board.prepare_session()

    # Gain 6x sur les 8 canaux
    print("Configuration Gain 6x...")
    for i in range(1, 9):
        board.config_board(GAIN_CMD.format(ch=i))
        time.sleep(0.02)

    board.start_stream()
    print(f"Stream démarré — {sfreq} Hz, {len(eeg_channels)} canaux EEG")
    return board, sfreq, eeg_channels


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def run_realtime(port: str, ch_indices: list = None) -> None:
    """
    Lance la visualisation temps réel.

    Args:
        port       : port série du Cyton (ex: /dev/cu.usbserial-XXXXX)
        ch_indices : indices 0-based des canaux à afficher (défaut: tous)
    """
    board, sfreq, all_eeg = setup_board(port)
    n_samples = int(WINDOW_SEC * sfreq)

    # Sélection des canaux à afficher
    if ch_indices is None:
        ch_indices = list(range(len(all_eeg)))
    channels = [all_eeg[i] for i in ch_indices if i < len(all_eeg)]
    ch_labels = [f"CH{i+1}" for i in ch_indices]
    n_ch = len(channels)

    x_axis = np.linspace(0, WINDOW_SEC, n_samples)

    # -- Layout : time series (gauche) + PSD (droite) --
    fig = plt.figure(figsize=(14, max(6, 2 * n_ch)))
    fig.canvas.manager.set_window_title("OpenBCI Cyton — EEG Temps Réel")

    gs = fig.add_gridspec(n_ch, 2, width_ratios=[2, 1], hspace=0.4, wspace=0.3)

    # Axes time series
    ax_ts = [fig.add_subplot(gs[i, 0]) for i in range(n_ch)]
    lines_ts = []
    for i, ax in enumerate(ax_ts):
        line, = ax.plot(x_axis, np.zeros(n_samples), lw=0.8, color=COLORS[i % len(COLORS)])
        lines_ts.append(line)
        ax.set_ylabel(f"{ch_labels[i]}\n(µV)", fontsize=8)
        ax.set_ylim(-150, 150)
        ax.grid(True, alpha=0.25)
        if i < n_ch - 1:
            ax.set_xticklabels([])
    ax_ts[-1].set_xlabel("Temps (s)")
    ax_ts[0].set_title("Signal brut filtré", fontsize=10)

    # Axe PSD (partagé pour tous les canaux)
    ax_psd = fig.add_subplot(gs[:, 1])
    ax_psd.set_title("PSD (Welch)", fontsize=10)
    ax_psd.set_xlabel("Fréquence (Hz)")
    ax_psd.set_ylabel("Puissance (µV²/Hz)")
    ax_psd.set_xlim(0, 50)
    ax_psd.grid(True, which="both", alpha=0.25)

    # Fond coloré par bande sur la PSD
    for band_name, (f_low, f_high) in FREQ_BANDS.items():
        ax_psd.axvspan(f_low, f_high, alpha=0.10, color=BAND_COLORS[band_name])
    band_patches = [
        mpatches.Patch(color=BAND_COLORS[b], alpha=0.5, label=f"{b}")
        for b in FREQ_BANDS
    ]

    lines_psd = []
    for i in range(n_ch):
        line, = ax_psd.semilogy([], [], lw=1.2, color=COLORS[i % len(COLORS)],
                                label=ch_labels[i])
        lines_psd.append(line)

    ax_psd.legend(handles=band_patches + lines_psd, fontsize=7, loc="upper right")

    # Texte puissance alpha (coin bas droit)
    alpha_text = ax_psd.text(
        0.98, 0.05, "", transform=ax_psd.transAxes,
        ha="right", va="bottom", fontsize=9,
        bbox=dict(boxstyle="round", fc="wheat", alpha=0.7),
    )

    # Indicateur REC (coin haut gauche du premier axe)
    rec_text = ax_ts[0].text(
        0.01, 0.95, "", transform=ax_ts[0].transAxes,
        ha="left", va="top", fontsize=10, fontweight="bold",
        color="red", animated=True,
    )

    # État d'enregistrement (dict pour mutabilité dans la closure)
    rec = {"active": False, "buffer": []}

    def on_key(event):
        if event.key != "r":
            return
        if not rec["active"]:
            rec["active"] = True
            rec["buffer"] = []
            print("\n  [REC] Enregistrement démarré — appuie sur R pour arrêter")
        else:
            rec["active"] = False
            if rec["buffer"]:
                recorded = np.concatenate(rec["buffer"], axis=1)
                save_recording(recorded, sfreq, ch_labels)
            rec["buffer"] = []

    fig.canvas.mpl_connect("key_press_event", on_key)
    print("  Astuce : appuie sur R dans la fenêtre pour démarrer/arrêter l'enregistrement")

    plt.tight_layout()

    # -- Boucle d'animation --
    def update(_frame):
        raw = board.get_current_board_data(n_samples)
        if raw.shape[1] < n_samples:
            return lines_ts + lines_psd

        # Extraction : BrainFlow Cyton renvoie en µV
        data_uv = np.array([raw[ch] for ch in channels])   # (n_ch, n_samples) en µV
        data_v  = data_uv * 1e-6                            # V pour le pipeline

        # Enregistrement en µV brut (load_recording applique *1e-6 au chargement)
        if rec["active"]:
            chunk_size = int(UPDATE_MS / 1000 * sfreq)
            rec["buffer"].append(data_uv[:, -chunk_size:].copy())
            rec_text.set_text("● REC")
        else:
            rec_text.set_text("")

        data_filt = filter_signal(data_v, sfreq,
                                  l_freq=L_FREQ, h_freq=H_FREQ,
                                  notch_freq=NOTCH_FREQ, causal=False)

        # -- Mise à jour time series --
        for i, line in enumerate(lines_ts):
            signal_uv = data_filt[i] * 1e6  # V → µV pour affichage
            line.set_ydata(signal_uv)
            # Autoscale doux : ±3 std, min ±30 µV
            std = np.std(signal_uv)
            lim = max(30, 3 * std)
            ax_ts[i].set_ylim(-lim, lim)

        # -- Mise à jour PSD --
        alpha_powers = []
        for i, line in enumerate(lines_psd):
            freqs, psd_uv2 = compute_psd_welch(
                data_filt[i], sfreq, fmin=0.5, fmax=50
            )
            line.set_data(freqs, psd_uv2)
            alpha_powers.append(band_power(data_filt[i], sfreq, "Alpha"))

        # Rescale PSD Y
        all_psd = np.concatenate([l.get_ydata() for l in lines_psd
                                  if len(l.get_ydata()) > 0], axis=0)
        if len(all_psd) > 0:
            ax_psd.set_ylim(max(1e-4, all_psd.min() * 0.5), all_psd.max() * 2)

        # Affichage puissance alpha moyenne
        mean_alpha = np.mean(alpha_powers)
        alpha_text.set_text(f"Alpha: {mean_alpha:.1f} µV²/Hz")

        return lines_ts + lines_psd + [alpha_text, rec_text]

    ani = FuncAnimation(fig, update, interval=UPDATE_MS, blit=True, cache_frame_data=False)

    try:
        plt.show()
    finally:
        board.stop_stream()
        board.release_session()
        print("Session fermée.")


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualisation EEG temps réel — OpenBCI Cyton")
    parser.add_argument("--port", default=DEFAULT_PORT,
                        help=f"Port série du Cyton (défaut: {DEFAULT_PORT})")
    parser.add_argument("--channels", type=int, nargs="+", default=None,
                        metavar="N",
                        help="Indices 0-based des canaux à afficher (défaut: tous). "
                             "Ex: --channels 0 1 4 5 pour CH1 CH2 CH5 CH6")
    args = parser.parse_args()

    run_realtime(port=args.port, ch_indices=args.channels)
