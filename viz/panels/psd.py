"""
Panel de densité spectrale de puissance (PSD) en temps réel.

Utilise pyqtgraph PlotWidget avec axe Y logarithmique.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QWidget

from .base import BasePanel, DashboardState
from processing import (
    FREQ_BANDS, BAND_COLORS,
    compute_psd_welch, band_power, compute_snr,
)

_COLORS = [
    '#e6194b', '#3cb44b', '#4363d8', '#f58231',
    '#911eb4', '#42d4f4', '#f032e6', '#bfef45',
]


class PSDPanel(BasePanel):
    """
    Panel PSD temps réel : une courbe par canal, axe Y log, bandes colorées.

    ⚠️  Avec setLogMode(y=True), setYRange() attend des valeurs log10.
    """

    def __init__(
        self,
        ch_labels: list[str],
        fmin: float = 0.5,
        fmax: float = 50.0,
        parent: QWidget | None = None,
    ) -> None:
        self._ch_labels = ch_labels
        self._n_ch      = len(ch_labels)
        self._fmin      = fmin
        self._fmax      = fmax

        self._pw = pg.PlotWidget(parent=parent)
        self._pw.setBackground('#1a1a2e')
        self._pw.setLogMode(x=False, y=True)
        self._pw.setLabel('left',   'Puissance', units='µV²/Hz')
        self._pw.setLabel('bottom', 'Fréquence', units='Hz')
        self._pw.setTitle('<span style="color:#cccccc">PSD (Welch)</span>')
        self._pw.setXRange(fmin, fmax, padding=0)
        self._pw.showGrid(x=True, y=True, alpha=0.25)

        # Bandes fréquentielles
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

        # Courbes par canal (live)
        self._curves: list[pg.PlotDataItem] = []
        for i, label in enumerate(ch_labels):
            curve = self._pw.plot(
                pen=pg.mkPen(color=_COLORS[i % len(_COLORS)], width=1.2),
                name=label,
            )
            self._curves.append(curve)

        # Courbes de référence (pointillées, cachées par défaut)
        self._ref_curves: list[pg.PlotDataItem] = []
        for i in range(len(ch_labels)):
            ref_curve = self._pw.plot(
                pen=pg.mkPen(color=_COLORS[i % len(_COLORS)], width=1.2,
                             style=Qt.DashLine),
            )
            ref_curve.setVisible(False)
            self._ref_curves.append(ref_curve)

        # Overlay texte alpha / SNR
        self._overlay = pg.TextItem(
            text='',
            color='#222222',
            fill=pg.mkBrush('#f5deb3cc'),
            anchor=(1.0, 1.0),
        )
        self._pw.addItem(self._overlay)
        # Repositionné à chaque frame en coordonnées data space
        self._pw.getViewBox().sigRangeChanged.connect(self._reposition_overlay)

    # ------------------------------------------------------------------
    # BasePanel
    # ------------------------------------------------------------------

    @property
    def widget(self) -> QWidget:
        return self._pw

    def update(
        self,
        data_filt: np.ndarray,
        sfreq: float,
        state: DashboardState,
    ) -> None:
        alpha_powers: list[float] = []
        snr_parts:    list[str]   = []
        all_psd: list[np.ndarray] = []

        for i, curve in enumerate(self._curves):
            if not state.ch_visible[i]:
                curve.setVisible(False)
                continue
            curve.setVisible(True)

            freqs, psd_uv2 = compute_psd_welch(
                data_filt[i], sfreq, fmin=self._fmin, fmax=self._fmax
            )
            curve.setData(freqs, psd_uv2)
            all_psd.append(psd_uv2)
            alpha_powers.append(band_power(data_filt[i], sfreq, 'Alpha'))

            if state.show_snr:
                snr = compute_snr(data_filt[i], sfreq, 'Alpha')
                snr_parts.append(f'{self._ch_labels[i]}: {snr:+.1f} dB')

        # Courbes de référence (pointillées)
        ref = state.ref_psd
        if ref is not None:
            for i, ref_curve in enumerate(self._ref_curves):
                if state.ch_visible[i] and i < len(ref['psd_per_ch']):
                    ref_curve.setData(ref['freqs'], ref['psd_per_ch'][i])
                    ref_curve.setVisible(True)
                    all_psd.append(ref['psd_per_ch'][i])
                else:
                    ref_curve.setVisible(False)
            title_html = (
                f'<span style="color:#cccccc">PSD — </span>'
                f'<span style="color:#aaaaff">— live</span>'
                f'<span style="color:#cccccc"> · </span>'
                f'<span style="color:#aaaaff">- - {ref["label"]}</span>'
            )
            self._pw.setTitle(title_html)
        else:
            for ref_curve in self._ref_curves:
                ref_curve.setVisible(False)
            self._pw.setTitle('<span style="color:#cccccc">PSD (Welch)</span>')

        # Auto-scale Y (setYRange avec logMode=True attend log10 des valeurs)
        if all_psd:
            combined = np.concatenate(all_psd)
            combined = combined[combined > 0]
            if len(combined):
                ymin = max(1e-4, combined.min() * 0.5)
                ymax = combined.max() * 2.0
                self._pw.setYRange(np.log10(ymin), np.log10(ymax), padding=0)

        # Overlay
        if alpha_powers:
            text = f'Alpha: {np.mean(alpha_powers):.1f} µV²/Hz'
            if state.show_snr and snr_parts:
                text += '\nSNR:\n' + '\n'.join(snr_parts)
            self._overlay.setText(text)
        else:
            self._overlay.setText('')

        self._reposition_overlay()

    # ------------------------------------------------------------------
    # Privé
    # ------------------------------------------------------------------

    def _reposition_overlay(self, *_) -> None:
        """Ancre l'overlay en bas à droite de la vue courante."""
        vb = self._pw.getViewBox()
        xmin, xmax = vb.viewRange()[0]
        ymin, ymax = vb.viewRange()[1]
        self._overlay.setPos(xmax, 10 ** ymin)
