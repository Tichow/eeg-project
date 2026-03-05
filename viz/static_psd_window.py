"""
Fenêtre statique affichant une courbe PSD par section EEG.

Usage :
    win = StaticPSDWindow(segments, sfreq, ch_labels, title="PSD — eyes_closed")
    win.show()
    app.exec_()

`segments` est une liste de tuples (label, data_uv) où :
  - label   : nom de la section (ex. "Yeux fermés", "Yeux ouverts")
  - data_uv : numpy array (n_ch, n_samples) en µV
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (
    QComboBox, QHBoxLayout, QLabel, QMainWindow, QVBoxLayout, QWidget,
)

from processing import FREQ_BANDS, BAND_COLORS, compute_psd_welch

# Palettes de couleurs par type de section (jusqu'à 4 instances)
_ACTION_PALETTES: dict[str, list[str]] = {
    'Yeux fermés':      ['#2255cc', '#4477ee', '#6699ff', '#88bbff'],
    'Yeux ouverts':     ['#cc3333', '#ee5555', '#ff8888', '#ffaaaa'],
    'Mouvement gauche': ['#228833', '#44aa55', '#66cc77', '#88ee99'],
    'Mouvement droit':  ['#cc7700', '#ee9922', '#ffbb44', '#ffcc77'],
    'Repos':            ['#555555', '#777777', '#999999', '#bbbbbb'],
    'Mouvement':        ['#116644', '#338866', '#55aa88', '#77ccaa'],
}
_FALLBACK_COLORS = [
    '#e6194b', '#3cb44b', '#4363d8', '#f58231',
    '#911eb4', '#42d4f4', '#f032e6', '#bfef45',
]


def _color_for(label: str, idx_in_group: int) -> str:
    palette = _ACTION_PALETTES.get(label, None)
    if palette:
        return palette[idx_in_group % len(palette)]
    return _FALLBACK_COLORS[idx_in_group % len(_FALLBACK_COLORS)]


class StaticPSDWindow(QMainWindow):
    """Fenêtre statique : une courbe PSD par section, pas de lecture."""

    def __init__(
        self,
        segments: list[tuple[str, np.ndarray]],
        sfreq: float,
        ch_labels: list[str],
        title: str = "PSD par section",
        fmin: float = 0.5,
        fmax: float = 50.0,
    ) -> None:
        super().__init__()
        self.setWindowTitle(title)
        self.resize(1100, 700)

        self._segments  = segments
        self._sfreq     = sfreq
        self._ch_labels = ch_labels
        self._fmin      = fmin
        self._fmax      = fmax

        # Pré-calcul des PSD : dict[canal_idx_ou_"avg"] → list[(freqs, psd)]
        self._psd_cache: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
        self._precompute_psd()

        # ── Widget central ─────────────────────────────────────────────
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(8)

        # ── Sélecteur de canal ─────────────────────────────────────────
        top = QHBoxLayout()
        top.addWidget(QLabel('Canal :'))
        self._ch_box = QComboBox()
        self._ch_box.addItem('Moyenne')
        self._ch_box.addItems(ch_labels)
        self._ch_box.setMinimumWidth(100)
        self._ch_box.currentIndexChanged.connect(self._on_channel_changed)
        top.addWidget(self._ch_box)
        top.addStretch()
        main_layout.addLayout(top)

        # ── PlotWidget ─────────────────────────────────────────────────
        self._pw = pg.PlotWidget()
        self._pw.setBackground('#1a1a2e')
        self._pw.setLogMode(x=False, y=True)
        self._pw.setLabel('left',   'Puissance', units='µV²/Hz')
        self._pw.setLabel('bottom', 'Fréquence', units='Hz')
        self._pw.setXRange(fmin, fmax, padding=0)
        self._pw.showGrid(x=True, y=True, alpha=0.25)
        self._pw.addLegend(offset=(10, 10))
        main_layout.addWidget(self._pw)

        # Bandes fréquentielles en arrière-plan
        for band_name, (f_low, f_high) in FREQ_BANDS.items():
            rgba = QColor(BAND_COLORS[band_name])
            rgba.setAlpha(25)
            region = pg.LinearRegionItem(
                values=(f_low, f_high),
                brush=pg.mkBrush(rgba),
                movable=False,
            )
            region.setZValue(-10)
            self._pw.addItem(region)

        # Labels de bandes (Alpha surtout utile)
        for band_name, (f_low, f_high) in FREQ_BANDS.items():
            label = pg.TextItem(
                text=band_name,
                color=BAND_COLORS[band_name],
                anchor=(0.5, 1.0),
            )
            label.setPos((f_low + f_high) / 2, 0)
            label.setZValue(-5)
            self._pw.addItem(label)

        # ── Courbes ────────────────────────────────────────────────────
        self._curves: list[pg.PlotDataItem] = []
        self._draw_curves('avg')

    # ── Pré-calcul ─────────────────────────────────────────────────────

    def _precompute_psd(self) -> None:
        n_ch = len(self._ch_labels)
        # Pour chaque clé canal ("avg", "0", "1", ...)
        keys = ['avg'] + [str(i) for i in range(n_ch)]
        for key in keys:
            psds: list[tuple[np.ndarray, np.ndarray]] = []
            for _label, data_uv in self._segments:
                if key == 'avg':
                    sig_v = data_uv.mean(axis=0) * 1e-6
                else:
                    ch_i = int(key)
                    if ch_i >= data_uv.shape[0]:
                        sig_v = data_uv[0] * 1e-6
                    else:
                        sig_v = data_uv[ch_i] * 1e-6
                freqs, psd_uv2 = compute_psd_welch(sig_v, self._sfreq,
                                                    fmin=self._fmin, fmax=self._fmax)
                psds.append((freqs, psd_uv2))
            self._psd_cache[key] = psds

    # ── Affichage ──────────────────────────────────────────────────────

    def _draw_curves(self, ch_key: str) -> None:
        for c in self._curves:
            self._pw.removeItem(c)
        self._curves.clear()

        # Compte combien de fois chaque label apparaît
        label_counts: dict[str, int] = {}
        for label, _ in self._segments:
            label_counts[label] = label_counts.get(label, 0) + 1

        label_idx: dict[str, int] = {}
        all_psd: list[np.ndarray] = []

        psds = self._psd_cache[ch_key]
        for seg_i, (label, _) in enumerate(self._segments):
            idx = label_idx.get(label, 0)
            label_idx[label] = idx + 1

            total = label_counts[label]
            display = f'{label} {idx + 1}/{total}' if total > 1 else label

            color = _color_for(label, idx)
            freqs, psd_uv2 = psds[seg_i]

            curve = self._pw.plot(
                freqs, psd_uv2,
                pen=pg.mkPen(color=color, width=1.8),
                name=display,
            )
            self._curves.append(curve)
            all_psd.append(psd_uv2)

        # Auto-scale Y
        if all_psd:
            combined = np.concatenate(all_psd)
            combined = combined[combined > 0]
            if len(combined):
                ymin = max(1e-4, combined.min() * 0.5)
                ymax = combined.max() * 2.0
                self._pw.setYRange(np.log10(ymin), np.log10(ymax), padding=0)

    # ── Slot ───────────────────────────────────────────────────────────

    def _on_channel_changed(self, index: int) -> None:
        ch_key = 'avg' if index == 0 else str(index - 1)
        self._draw_curves(ch_key)
