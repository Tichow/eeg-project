"""
Panel SNR temps réel — bar chart par bande fréquentielle.

Pour chaque frame, le PSD est calculé une seule fois par canal visible,
puis le SNR (en dB) est dérivé pour chaque bande Delta/Theta/Alpha/Beta.
La valeur affichée est la moyenne sur tous les canaux visibles.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt5.QtWidgets import QWidget

from .base import BasePanel, DashboardState
from processing import FREQ_BANDS, BAND_COLORS, compute_psd_welch

_BAND_NAMES = list(FREQ_BANDS.keys())  # ordre stable : Delta, Theta, Alpha, Beta
_X          = list(range(len(_BAND_NAMES)))
_BRUSHES    = [BAND_COLORS[b] for b in _BAND_NAMES]


def _snr_from_psd(freqs: np.ndarray, psd: np.ndarray) -> dict[str, float]:
    """SNR par bande depuis un PSD déjà calculé (µV²/Hz)."""
    snrs = {}
    for band_name, (f_low, f_high) in FREQ_BANDS.items():
        sig_mask   = (freqs >= f_low) & (freqs <= f_high)
        noise_mask = ~sig_mask
        p_sig  = float(np.mean(psd[sig_mask]))  if sig_mask.any()   else 0.0
        p_noi  = float(np.mean(psd[noise_mask])) if noise_mask.any() else 1.0
        if p_sig > 0 and p_noi > 0:
            snrs[band_name] = 10.0 * np.log10(p_sig / p_noi)
        else:
            snrs[band_name] = 0.0
    return snrs


class SNRBarPanel(BasePanel):
    """
    Panel SNR : 4 barres (Delta / Theta / Alpha / Beta) en dB.

    Le SNR est calculé par bande à partir d'un seul appel Welch par canal
    visible, puis moyenné sur ces canaux.
    """

    _YMIN = -20.0
    _YMAX =  30.0

    def __init__(
        self,
        ch_labels: list[str],
        parent: QWidget | None = None,
    ) -> None:
        self._n_ch = len(ch_labels)

        self._pw = pg.PlotWidget(parent=parent)
        self._pw.setBackground('#1a1a2e')
        self._pw.setTitle('<span style="color:#cccccc">SNR par bande (dB)</span>')
        self._pw.setLabel('left', 'SNR', units='dB')
        self._pw.setYRange(self._YMIN, self._YMAX, padding=0)
        self._pw.showGrid(x=False, y=True, alpha=0.25)

        # Axe X personnalisé avec noms des bandes
        ax = self._pw.getAxis('bottom')
        ax.setTicks([list(zip(_X, _BAND_NAMES))])

        # Ligne de référence 0 dB
        ref = pg.InfiniteLine(
            pos=0,
            angle=0,
            pen=pg.mkPen('#ffffff', width=1, style=pg.QtCore.Qt.DashLine),
        )
        self._pw.addItem(ref)

        # Barres
        self._bars = pg.BarGraphItem(
            x=_X,
            height=[0.0] * len(_X),
            width=0.6,
            brushes=_BRUSHES,
            pens=[pg.mkPen('#ffffff', width=0.5)] * len(_X),
        )
        self._pw.addItem(self._bars)

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
        visible = [i for i in range(self._n_ch) if state.ch_visible[i]]
        if not visible:
            self._bars.setOpts(height=[0.0] * len(_X))
            return

        # Un seul PSD par canal visible → SNR pour toutes les bandes
        band_snrs: dict[str, list[float]] = {b: [] for b in _BAND_NAMES}
        for i in visible:
            freqs, psd_uv2 = compute_psd_welch(data_filt[i], sfreq)
            snrs = _snr_from_psd(freqs, psd_uv2)
            for b, v in snrs.items():
                band_snrs[b].append(v)

        heights = [
            float(np.clip(np.mean(band_snrs[b]), self._YMIN, self._YMAX))
            for b in _BAND_NAMES
        ]
        self._bars.setOpts(height=heights)
