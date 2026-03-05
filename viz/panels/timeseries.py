"""
Panel de visualisation des séries temporelles EEG.

Un PlotItem par canal, empilés verticalement dans un GraphicsLayoutWidget.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt5.QtWidgets import QWidget

from .base import BasePanel, DashboardState

_COLORS = [
    '#e6194b', '#3cb44b', '#4363d8', '#f58231',
    '#911eb4', '#42d4f4', '#f032e6', '#bfef45',
]


class TimeSeriesPanel(BasePanel):
    """
    Panel de séries temporelles : un graphe par canal EEG, empilés verticalement.

    Axes X liés (tous les canaux défilent ensemble).
    Auto-scaling Y par canal : centre ± 3σ, minimum ±30 µV.
    """

    def __init__(
        self,
        ch_labels: list[str],
        n_samples: int,
        window_sec: float,
        parent: QWidget | None = None,
    ) -> None:
        self._ch_labels = ch_labels
        self._n_ch      = len(ch_labels)
        self._n_samples = n_samples
        self._x_axis    = np.linspace(0.0, window_sec, n_samples)

        self._glw = pg.GraphicsLayoutWidget(parent=parent)
        self._glw.setBackground('#1a1a2e')
        self._glw.setMinimumHeight(self._n_ch * 80)

        self._plots:  list[pg.PlotItem]     = []
        self._curves: list[pg.PlotDataItem] = []

        prev_plot = None
        for i, label in enumerate(ch_labels):
            is_last = (i == self._n_ch - 1)
            plot = self._glw.addPlot(row=i, col=0)
            plot.setLabel('left', label, units='µV')
            plot.setYRange(-150, 150, padding=0)
            plot.showGrid(x=False, y=True, alpha=0.25)
            if not is_last:
                plot.hideAxis('bottom')
            else:
                plot.setLabel('bottom', 'Temps', units='s')

            if prev_plot is not None:
                plot.setXLink(prev_plot)
            prev_plot = plot

            color = _COLORS[i % len(_COLORS)]
            curve = plot.plot(
                self._x_axis,
                np.zeros(n_samples),
                pen=pg.mkPen(color=color, width=1),
            )
            self._plots.append(plot)
            self._curves.append(curve)

        # Titre (ligne supplémentaire sous les plots)
        self._title_item = self._glw.addLabel(
            "Signal filtré",
            row=self._n_ch, col=0,
            color='#cccccc', size='9pt',
        )

    # ------------------------------------------------------------------
    # BasePanel
    # ------------------------------------------------------------------

    @property
    def widget(self) -> QWidget:
        return self._glw

    def update(
        self,
        data_filt: np.ndarray,
        sfreq: float,
        state: DashboardState,
    ) -> None:
        for i, (curve, plot) in enumerate(zip(self._curves, self._plots)):
            if not state.ch_visible[i]:
                continue
            signal_uv = data_filt[i] * 1e6
            curve.setData(self._x_axis, signal_uv)
            mean = float(np.mean(signal_uv))
            half = max(30.0, 3.0 * float(np.std(signal_uv)))
            plot.setYRange(mean - half, mean + half, padding=0)

    def on_channels_changed(self, ch_visible: list[bool]) -> None:
        for i, (curve, plot) in enumerate(zip(self._curves, self._plots)):
            visible = ch_visible[i]
            plot.setVisible(visible)
            curve.setVisible(visible)

    # ------------------------------------------------------------------
    # API publique
    # ------------------------------------------------------------------

    def set_title(self, text: str) -> None:
        self._title_item.setText(text, color='#cccccc', size='9pt')
