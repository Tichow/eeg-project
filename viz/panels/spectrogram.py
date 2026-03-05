"""
Panel spectrogramme temps réel — vue temps-fréquence glissante.

À chaque frame, le PSD (Welch) est moyenné sur les canaux visibles et
inséré dans un buffer circulaire (n_history × n_freq). L'image est
affichée en log10(µV²/Hz) avec la colormap inferno.

Initialisation du buffer paresseuse : effectuée au premier appel de
update(), quand la dimension fréquentielle est connue.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import QRectF
from PyQt5.QtWidgets import QWidget

from .base import BasePanel, DashboardState
from processing import compute_psd_welch

_FMIN = 0.5
_FMAX = 50.0


class SpectrogramPanel(BasePanel):
    """
    Panel spectrogramme : image 2D (temps × fréquence) glissante.

    X = temps (secondes, 0 = le plus ancien, n_history_sec = le plus récent)
    Y = fréquence (Hz, de _FMIN à _FMAX)
    Couleur = puissance en dB (log10 µV²/Hz), colormap inferno
    """

    def __init__(
        self,
        ch_labels: list[str],
        update_ms: int       = 100,
        n_history_sec: float = 60.0,
        parent: QWidget | None = None,
    ) -> None:
        self._n_ch          = len(ch_labels)
        self._n_history     = max(1, int(n_history_sec * 1000 / update_ms))
        self._n_history_sec = n_history_sec

        # Buffer initialisé à la première frame (dimension fréquence inconnue ici)
        self._buf:    np.ndarray | None = None
        self._n_freq: int | None        = None

        # Widget
        self._pw = pg.PlotWidget(parent=parent)
        self._pw.setBackground('#1a1a2e')
        self._pw.setTitle('<span style="color:#cccccc">Spectrogramme (moy. canaux visibles)</span>')
        self._pw.setLabel('left',   'Fréquence', units='Hz')
        self._pw.setLabel('bottom', 'Temps',     units='s')
        self._pw.setYRange(_FMIN, _FMAX, padding=0)
        self._pw.setXRange(0, n_history_sec, padding=0)

        # Image
        self._img = pg.ImageItem()
        self._img.setColorMap(pg.colormap.get('inferno'))
        self._pw.addItem(self._img)

        # Overlay texte niveaux
        self._overlay = pg.TextItem(text='', color='#cccccc', anchor=(0.0, 1.0))
        self._pw.addItem(self._overlay)

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
            return

        # PSD moyen sur les canaux visibles
        psds = []
        for i in visible:
            freqs, psd_uv2 = compute_psd_welch(
                data_filt[i], sfreq, fmin=_FMIN, fmax=_FMAX
            )
            psds.append(psd_uv2)
        mean_psd = np.mean(psds, axis=0)

        # Init paresseuse
        if self._buf is None:
            self._n_freq = len(freqs)
            self._buf    = np.zeros((self._n_history, self._n_freq))
            self._img.setRect(
                QRectF(0.0, _FMIN, float(self._n_history_sec), _FMAX - _FMIN)
            )

        # Insertion de la nouvelle frame (buffer circulaire)
        self._buf = np.roll(self._buf, -1, axis=0)
        self._buf[-1, :] = mean_psd

        # Conversion en dB
        log_buf = 10.0 * np.log10(np.maximum(self._buf, 1e-10))

        # Niveaux adaptatifs (percentiles sur les données non-vides)
        filled = log_buf[log_buf > -90.0]
        if len(filled) > 20:
            vmin = float(np.percentile(filled, 2))
            vmax = float(np.percentile(filled, 99))
        else:
            vmin, vmax = -10.0, 30.0

        self._img.setImage(log_buf, autoLevels=False, levels=(vmin, vmax))

        # Overlay niveaux
        self._overlay.setText(f'{vmin:.0f} – {vmax:.0f} dB')
        vb = self._pw.getViewBox()
        xr, yr = vb.viewRange()
        self._overlay.setPos(xr[0], yr[0])
