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
    QComboBox, QHBoxLayout, QLabel, QMainWindow,
    QTabWidget, QVBoxLayout, QWidget,
)

from processing import FREQ_BANDS, BAND_COLORS, compute_psd_welch, apply_car
from processing.filter import filter_signal

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


def _segment_display_names(segments: list[tuple[str, ...]]) -> list[str]:
    """Retourne une liste de noms affichables pour chaque segment."""
    counts: dict[str, int] = {}
    for label, *_ in segments:
        counts[label] = counts.get(label, 0) + 1
    idx: dict[str, int] = {}
    names = []
    for label, *_ in segments:
        i = idx.get(label, 0)
        idx[label] = i + 1
        names.append(f'{label} {i + 1}/{counts[label]}' if counts[label] > 1 else label)
    return names


class StaticPSDWindow(QMainWindow):
    """Fenêtre statique : une courbe PSD par section + onglet Différence."""

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

        self._segments      = segments
        self._sfreq         = sfreq
        self._ch_labels     = ch_labels
        self._fmin          = fmin
        self._fmax          = fmax
        self._seg_names     = _segment_display_names(segments)

        # Pré-calcul des PSD : dict[canal_idx_ou_"avg"] → list[(freqs, psd)]
        self._psd_cache: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
        self._precompute_psd()

        # ── Widget central ─────────────────────────────────────────────
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)

        tabs = QTabWidget()
        layout.addWidget(tabs)
        tabs.addTab(self._build_abs_tab(),  "PSD absolue")
        tabs.addTab(self._build_diff_tab(), "Différence")

    # ── Pré-calcul ─────────────────────────────────────────────────────

    def _precompute_psd(self) -> None:
        n_ch = len(self._ch_labels)
        keys = ['avg'] + [str(i) for i in range(n_ch)]

        # Filtrer chaque segment une seule fois (passe-bande 1–50 Hz + notch 50 Hz)
        # pour retirer le DC offset et la dérive lente qui noient le pic alpha.
        filtered_segments: list[tuple[str, np.ndarray]] = []
        for label, data_uv in self._segments:
            data_filt = filter_signal(
                data_uv, self._sfreq,
                l_freq=1.0, h_freq=50.0, notch_freq=50.0,
                causal=False,
            )
            data_filt = apply_car(data_filt)
            filtered_segments.append((label, data_filt))

        for key in keys:
            psds: list[tuple[np.ndarray, np.ndarray]] = []
            for _label, data_filt in filtered_segments:
                if key == 'avg':
                    # Moyenne des PSD par canal — PAS la PSD du signal moyenné.
                    # Après CAR, data_filt.mean(axis=0) ≈ 0 par définition,
                    # ce qui donnerait une PSD quasi-nulle (~1e-30 µV²/Hz).
                    per_ch: list[np.ndarray] = []
                    for ch in range(data_filt.shape[0]):
                        sig_v = data_filt[ch] * 1e-6
                        freqs, psd_ch = compute_psd_welch(sig_v, self._sfreq,
                                                          fmin=self._fmin, fmax=self._fmax)
                        per_ch.append(psd_ch)
                    psd_uv2 = np.mean(per_ch, axis=0)
                else:
                    ch_i = int(key)
                    if ch_i >= data_filt.shape[0]:
                        sig_v = data_filt[0] * 1e-6
                    else:
                        sig_v = data_filt[ch_i] * 1e-6
                    freqs, psd_uv2 = compute_psd_welch(sig_v, self._sfreq,
                                                        fmin=self._fmin, fmax=self._fmax)
                psds.append((freqs, psd_uv2))
            self._psd_cache[key] = psds

    # ── Onglet 1 : PSD absolue ─────────────────────────────────────────

    def _build_abs_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        top = QHBoxLayout()
        top.addWidget(QLabel('Canal :'))
        self._ch_box = QComboBox()
        self._ch_box.addItem('Moyenne')
        self._ch_box.addItems(self._ch_labels)
        self._ch_box.setMinimumWidth(100)
        self._ch_box.currentIndexChanged.connect(self._on_channel_changed)
        top.addWidget(self._ch_box)
        top.addStretch()
        layout.addLayout(top)

        self._pw = pg.PlotWidget()
        self._pw.setBackground('#1a1a2e')
        self._pw.setLogMode(x=False, y=True)
        self._pw.setLabel('left',   'Puissance', units='µV²/Hz')
        self._pw.setLabel('bottom', 'Fréquence', units='Hz')
        self._pw.setXRange(self._fmin, self._fmax, padding=0)
        self._pw.showGrid(x=True, y=True, alpha=0.25)
        self._pw.addLegend(offset=(10, 10))
        self._add_band_regions(self._pw)
        layout.addWidget(self._pw)

        self._curves: list[pg.PlotDataItem] = []
        self._draw_curves('avg')
        return w

    def _add_band_regions(self, pw: pg.PlotWidget) -> None:
        for band_name, (f_low, f_high) in FREQ_BANDS.items():
            rgba = QColor(BAND_COLORS[band_name])
            rgba.setAlpha(25)
            region = pg.LinearRegionItem(
                values=(f_low, f_high), brush=pg.mkBrush(rgba), movable=False,
            )
            region.setZValue(-10)
            pw.addItem(region)
            lbl = pg.TextItem(text=band_name, color=BAND_COLORS[band_name], anchor=(0.5, 1.0))
            lbl.setPos((f_low + f_high) / 2, 0)
            lbl.setZValue(-5)
            pw.addItem(lbl)

    # ── Onglet 2 : Différence ──────────────────────────────────────────

    def _build_diff_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        if len(self._segments) < 2:
            layout.addWidget(QLabel(
                "  Au moins 2 segments requis pour calculer une différence.",
                alignment=Qt.AlignCenter,
            ))
            return w

        # Sélecteurs
        top = QHBoxLayout()
        top.addWidget(QLabel('Segment A :'))
        self._diff_a_box = QComboBox()
        self._diff_a_box.addItems(self._seg_names)
        self._diff_a_box.setMinimumWidth(130)
        top.addWidget(self._diff_a_box)

        top.addWidget(QLabel('  −  Segment B :'))
        self._diff_b_box = QComboBox()
        self._diff_b_box.addItems(self._seg_names)
        self._diff_b_box.setCurrentIndex(1)
        self._diff_b_box.setMinimumWidth(130)
        top.addWidget(self._diff_b_box)

        top.addSpacing(16)
        top.addWidget(QLabel('Canal :'))
        self._diff_ch_box = QComboBox()
        self._diff_ch_box.addItem('Moyenne')
        self._diff_ch_box.addItems(self._ch_labels)
        self._diff_ch_box.setMinimumWidth(100)
        top.addWidget(self._diff_ch_box)
        top.addStretch()
        layout.addLayout(top)

        # Plot
        self._diff_pw = pg.PlotWidget()
        self._diff_pw.setBackground('#1a1a2e')
        self._diff_pw.setLabel('left',   'A − B', units='µV²/Hz')
        self._diff_pw.setLabel('bottom', 'Fréquence', units='Hz')
        self._diff_pw.setXRange(self._fmin, self._fmax, padding=0)
        self._diff_pw.showGrid(x=True, y=True, alpha=0.25)
        self._diff_pw.addLegend(offset=(10, 10))

        # Ligne zéro
        zero_line = pg.InfiniteLine(
            pos=0, angle=0, movable=False,
            pen=pg.mkPen(color='#aaaaaa', width=1, style=2),
        )
        self._diff_pw.addItem(zero_line)

        self._add_band_regions(self._diff_pw)
        layout.addWidget(self._diff_pw)

        self._diff_curves: list[pg.PlotDataItem] = []
        self._draw_diff()

        self._diff_a_box.currentIndexChanged.connect(lambda _: self._draw_diff())
        self._diff_b_box.currentIndexChanged.connect(lambda _: self._draw_diff())
        self._diff_ch_box.currentIndexChanged.connect(lambda _: self._draw_diff())

        return w

    def _draw_diff(self) -> None:
        for c in self._diff_curves:
            self._diff_pw.removeItem(c)
        self._diff_curves.clear()

        idx_a  = self._diff_a_box.currentIndex()
        idx_b  = self._diff_b_box.currentIndex()
        ch_idx = self._diff_ch_box.currentIndex()
        ch_key = 'avg' if ch_idx == 0 else str(ch_idx - 1)

        if idx_a == idx_b:
            return

        psds   = self._psd_cache[ch_key]
        freqs_a, psd_a = psds[idx_a]
        freqs_b, psd_b = psds[idx_b]

        # Interpoler sur la même grille si nécessaire
        if not np.array_equal(freqs_a, freqs_b):
            psd_b = np.interp(freqs_a, freqs_b, psd_b)
        freqs = freqs_a

        diff = psd_a - psd_b

        name_a = self._seg_names[idx_a]
        name_b = self._seg_names[idx_b]

        # Courbe différence
        c = self._diff_pw.plot(
            freqs, diff,
            pen=pg.mkPen(color='#f0c040', width=2),
            name=f'{name_a} − {name_b}',
        )
        self._diff_curves.append(c)

        # Zones positives (A > B) en vert clair, négatives en rouge clair
        pos = np.clip(diff, 0, None)
        neg = np.clip(diff, None, 0)

        c_zero_pos = self._diff_pw.plot(freqs, np.zeros_like(diff), pen=pg.mkPen(None))
        c_pos      = self._diff_pw.plot(freqs, pos, pen=pg.mkPen(None))
        fill_pos   = pg.FillBetweenItem(c_pos, c_zero_pos, brush=pg.mkBrush(QColor(100, 200, 100, 60)))
        self._diff_pw.addItem(fill_pos)

        c_zero_neg = self._diff_pw.plot(freqs, np.zeros_like(diff), pen=pg.mkPen(None))
        c_neg      = self._diff_pw.plot(freqs, neg, pen=pg.mkPen(None))
        fill_neg   = pg.FillBetweenItem(c_neg, c_zero_neg, brush=pg.mkBrush(QColor(220, 80, 80, 60)))
        self._diff_pw.addItem(fill_neg)

        self._diff_curves.extend([c_zero_pos, c_pos, c_zero_neg, c_neg])
        # fills ne sont pas dans removeItem list → les garder séparément
        if not hasattr(self, '_diff_fills'):
            self._diff_fills: list = []
        else:
            for f in self._diff_fills:
                self._diff_pw.removeItem(f)
        self._diff_fills = [fill_pos, fill_neg]

        # Auto-scale Y
        margin = max(abs(diff).max() * 0.1, 1e-4)
        self._diff_pw.setYRange(diff.min() - margin, diff.max() + margin, padding=0)

    # ── Affichage ──────────────────────────────────────────────────────

    def _draw_curves(self, ch_key: str) -> None:
        for c in self._curves:
            self._pw.removeItem(c)
        self._curves.clear()

        label_counts: dict[str, int] = {}
        for label, _ in self._segments:
            label_counts[label] = label_counts.get(label, 0) + 1

        label_idx: dict[str, int] = {}
        all_psd: list[np.ndarray] = []

        psds = self._psd_cache[ch_key]
        for seg_i, (label, _) in enumerate(self._segments):
            idx = label_idx.get(label, 0)
            label_idx[label] = idx + 1

            total   = label_counts[label]
            display = f'{label} {idx + 1}/{total}' if total > 1 else label
            color   = _color_for(label, idx)
            freqs, psd_uv2 = psds[seg_i]

            curve = self._pw.plot(
                freqs, psd_uv2,
                pen=pg.mkPen(color=color, width=1.8),
                name=display,
            )
            self._curves.append(curve)
            all_psd.append(psd_uv2)

        if all_psd:
            combined = np.concatenate(all_psd)
            combined = combined[combined > 0]
            if len(combined):
                ymin = max(1e-4, combined.min() * 0.5)
                ymax = combined.max() * 2.0
                self._pw.setYRange(np.log10(ymin), np.log10(ymax), padding=0)

    # ── Slots ──────────────────────────────────────────────────────────

    def _on_channel_changed(self, index: int) -> None:
        ch_key = 'avg' if index == 0 else str(index - 1)
        self._draw_curves(ch_key)
