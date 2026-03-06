"""
Diagnostic du pipeline EEG — stats à chaque étape du traitement.

Usage :
    python debug_pipeline.py recordings/20260303_105500_nemo_eyes_closed.npy
    python debug_pipeline.py  # file picker
"""

import argparse
import json
import os
import sys

import numpy as np

from processing.filter import filter_signal
from processing.car import apply_car
from processing.psd import compute_psd_welch
from processing.constants import FREQ_BANDS


def _stats(data: np.ndarray, label: str, per_channel: bool = False) -> None:
    """Affiche mean / std / min / max d'un array (n_ch, n_samples) ou 1D."""
    if data.ndim == 1:
        print(f"  {label:30s}  mean={data.mean():.4g}  std={data.std():.4g}"
              f"  min={data.min():.4g}  max={data.max():.4g}")
    else:
        # Stats globales
        print(f"  {label:30s}  mean={data.mean():.4g}  std={data.std():.4g}"
              f"  min={data.min():.4g}  max={data.max():.4g}")
        if per_channel:
            for i, ch in enumerate(data):
                print(f"    CH{i+1:>2}  mean={ch.mean():+.4g}  std={ch.std():.4g}"
                      f"  min={ch.min():.4g}  max={ch.max():.4g}")


def _band_stats(freqs: np.ndarray, psd: np.ndarray) -> None:
    """Affiche la puissance moyenne par bande."""
    for band, (flo, fhi) in FREQ_BANDS.items():
        mask = (freqs >= flo) & (freqs <= fhi)
        if mask.any():
            print(f"    {band:6s} ({flo:4.1f}–{fhi:4.1f} Hz) : {psd[mask].mean():.4g} µV²/Hz")


def run_diagnostic(npy_path: str, ch_idx: int = 0) -> None:
    # ── Chargement ────────────────────────────────────────────────────
    meta_path = npy_path.replace('.npy', '.json')
    data_uv = np.load(npy_path)
    with open(meta_path) as f:
        meta = json.load(f)

    sfreq     = float(meta['sfreq'])
    ch_labels = meta.get('channels', [f'CH{i+1}' for i in range(data_uv.shape[0])])
    n_ch, n_samples = data_uv.shape
    duration = n_samples / sfreq

    print(f"\n{'='*60}")
    print(f"Fichier  : {os.path.basename(npy_path)}")
    print(f"Sujet    : {meta.get('subject', '?')}  |  {meta.get('test_type', '?')}")
    print(f"Shape    : {n_ch} canaux × {n_samples} samples  ({duration:.1f} s @ {sfreq:.0f} Hz)")
    print(f"Canal analysé en détail : {ch_labels[ch_idx]} (index {ch_idx})")
    print(f"{'='*60}\n")

    # ── Étape 1 : données brutes ───────────────────────────────────────
    print("ÉTAPE 1 — Données brutes (.npy), unités : µV")
    _stats(data_uv, "Tous canaux", per_channel=True)

    # ── Étape 2 : filtre passe-bande ──────────────────────────────────
    print("\nÉTAPE 2 — Après filter_signal (1–50 Hz + notch 50 Hz), unités : µV")
    data_filt = filter_signal(data_uv, sfreq, l_freq=1.0, h_freq=50.0,
                              notch_freq=50.0, causal=False)
    _stats(data_filt, "Tous canaux", per_channel=True)

    # ── Étape 3 : CAR ─────────────────────────────────────────────────
    print("\nÉTAPE 3 — Après apply_car, unités : µV")
    data_car = apply_car(data_filt)
    _stats(data_car, "Tous canaux", per_channel=True)

    # ── Étape 4 : conversion µV → V ───────────────────────────────────
    print(f"\nÉTAPE 4 — Après × 1e-6 (µV → V), canal {ch_labels[ch_idx]}")
    sig_v = data_car[ch_idx] * 1e-6
    _stats(sig_v, f"{ch_labels[ch_idx]} en V")

    # ── Étape 5 : PSD Welch ───────────────────────────────────────────
    print(f"\nÉTAPE 5 — PSD Welch, canal {ch_labels[ch_idx]}, unités : µV²/Hz")
    freqs, psd_uv2 = compute_psd_welch(sig_v, sfreq)
    print(f"  {'PSD globale':30s}  min={psd_uv2.min():.4g}  max={psd_uv2.max():.4g}"
          f"  median={np.median(psd_uv2):.4g}")
    print("  Puissance par bande :")
    _band_stats(freqs, psd_uv2)

    # ── Comparaison sans filtrage (avant fix) ─────────────────────────
    print(f"\n--- Comparaison : PSD SANS filtrage (comme avant le fix) ---")
    sig_v_raw = data_uv[ch_idx] * 1e-6
    freqs_raw, psd_raw = compute_psd_welch(sig_v_raw, sfreq)
    print(f"  {'PSD brute':30s}  min={psd_raw.min():.4g}  max={psd_raw.max():.4g}"
          f"  median={np.median(psd_raw):.4g}")
    print("  Puissance par bande (brut) :")
    _band_stats(freqs_raw, psd_raw)

    ratio_alpha_delta_filt = _band_mean(freqs, psd_uv2, 'Alpha') / max(_band_mean(freqs, psd_uv2, 'Delta'), 1e-12)
    ratio_alpha_delta_raw  = _band_mean(freqs_raw, psd_raw, 'Alpha') / max(_band_mean(freqs_raw, psd_raw, 'Delta'), 1e-12)
    print(f"\n  Ratio Alpha/Delta — filtré : {ratio_alpha_delta_filt:.3f}"
          f"  |  brut : {ratio_alpha_delta_raw:.3f}")
    print(f"  (> 1 = alpha domine, < 1 = delta domine)")
    print()


def _band_mean(freqs, psd, band_name):
    flo, fhi = FREQ_BANDS[band_name]
    mask = (freqs >= flo) & (freqs <= fhi)
    return psd[mask].mean() if mask.any() else 0.0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Diagnostic pipeline EEG')
    parser.add_argument('npy_path', nargs='?', default=None)
    parser.add_argument('--ch', type=int, default=6,
                        help='Index canal à analyser en détail (défaut: 6 = CH7)')
    args = parser.parse_args()

    if args.npy_path is None:
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk(); root.withdraw()
            args.npy_path = filedialog.askopenfilename(
                title='Choisir un enregistrement .npy',
                initialdir='recordings',
                filetypes=[('EEG numpy', '*.npy')],
            )
        except Exception:
            print("Usage : python debug_pipeline.py <fichier.npy>")
            sys.exit(1)

    if not args.npy_path:
        sys.exit(0)

    run_diagnostic(args.npy_path, ch_idx=args.ch)
