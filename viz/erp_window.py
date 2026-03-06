"""
Fenêtre d'analyse par epochs — 3 onglets :
  1. Formes d'onde (ERP) : signal moyenné par condition, ±1 std
  2. PSD moyenne          : densité spectrale par condition
  3. Carte topo           : puissance par bande et par canal (vue du dessus)
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QLinearGradient, QPainter
from PyQt5.QtWidgets import (
    QComboBox, QHBoxLayout, QLabel, QMainWindow,
    QTabWidget, QVBoxLayout, QWidget,
)

from processing import FREQ_BANDS, BAND_COLORS, compute_psd_welch, apply_car
from processing.filter import filter_signal
from processing.epochs import group_by_label, epoch_band_power

# Couleurs par condition
_COND_COLORS: dict[str, str] = {
    'T0': '#888888',
    'T1': '#e15759',
    'T2': '#4e79a7',
}
_FALLBACK_COLORS = ['#f0c040', '#3cb44b', '#f58231', '#911eb4', '#42d4f4', '#f032e6']


def _cond_color(label: str, idx: int = 0) -> str:
    return _COND_COLORS.get(label, _FALLBACK_COLORS[idx % len(_FALLBACK_COLORS)])


class ERPWindow(QMainWindow):
    """Fenêtre d'analyse statique par epochs."""

    def __init__(
        self,
        data_uv: np.ndarray,
        sfreq: float,
        ch_labels: list[str],
        annotations: list[dict],
        t_before: float = 0.5,
        t_after: float = 4.0,
        title: str = "ERP — Moyennes",
        notch_freq: float = 50.0,
    ) -> None:
        super().__init__()
        self.setWindowTitle(title)
        self.resize(1200, 800)

        self._ch_labels  = list(ch_labels)
        self._sfreq      = sfreq
        self._notch_freq = notch_freq

        # ── Pré-calcul des epochs ──────────────────────────────────────────
        self._groups = group_by_label(
            data_uv, sfreq, annotations,
            t_before, t_after,
            notch_freq=notch_freq,
        )

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)

        if not self._groups:
            layout.addWidget(QLabel(
                "  Aucun événement reconnu dans ces données.\n"
                "  Utilisez un run avec annotations T0/T1/T2 (ex: PhysioNet run 3-14).",
                alignment=Qt.AlignCenter,
            ))
            return

        # Données pour PSD : moyenne par condition comme « segment »
        self._psd_segments = [
            (label, self._groups[label][0])   # (label, mean_data)
            for label in self._groups
        ]

        # PSD par condition pré-calculée
        self._psd_cache = self._precompute_psd()

        # Puissance par bande pour la carte topo
        self._band_powers = epoch_band_power(
            data_uv, sfreq, annotations,
            t_before, t_after,
            notch_freq=notch_freq,
        )

        tabs = QTabWidget()
        layout.addWidget(tabs)
        tabs.addTab(self._build_erp_tab(),            "Formes d'onde")
        tabs.addTab(self._build_psd_tab(),            "PSD moyenne")
        tabs.addTab(self._build_topo_tab(),           "Carte topo")

    # ── Onglet 1 : ERP ────────────────────────────────────────────────────

    def _build_erp_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(6)

        # Sélecteur de canal
        top = QHBoxLayout()
        top.addWidget(QLabel("Canal :"))
        self._erp_ch_box = QComboBox()
        self._erp_ch_box.addItem("Moyenne")
        self._erp_ch_box.addItems(self._ch_labels)
        self._erp_ch_box.setMinimumWidth(120)
        top.addWidget(self._erp_ch_box)
        top.addStretch()
        layout.addLayout(top)

        # Plot
        self._erp_pw = pg.PlotWidget()
        self._erp_pw.setBackground('#1a1a2e')
        self._erp_pw.setLabel('left',   'Amplitude', units='µV')
        self._erp_pw.setLabel('bottom', 'Temps',     units='s')
        self._erp_pw.showGrid(x=True, y=True, alpha=0.25)
        self._erp_pw.addLegend(offset=(10, 10))

        # Ligne verticale à t = 0 (onset)
        onset_line = pg.InfiniteLine(
            pos=0, angle=90, movable=False,
            pen=pg.mkPen(color='#ffffff', width=1, style=2),
            label='onset', labelOpts={'position': 0.9, 'color': '#cccccc'},
        )
        self._erp_pw.addItem(onset_line)

        layout.addWidget(self._erp_pw)

        # Première trace
        self._erp_curves:  list[pg.PlotDataItem]  = []
        self._erp_fills:   list[pg.FillBetweenItem] = []
        self._draw_erp(0)
        self._erp_ch_box.currentIndexChanged.connect(self._draw_erp)

        return w

    def _draw_erp(self, ch_index: int) -> None:
        for c in self._erp_curves:
            self._erp_pw.removeItem(c)
        for f in self._erp_fills:
            self._erp_pw.removeItem(f)
        self._erp_curves.clear()
        self._erp_fills.clear()

        all_vals = []
        for i, (label, (mean, std, times, n)) in enumerate(self._groups.items()):
            if ch_index == 0:
                # Moyenne sur tous les canaux
                m = mean.mean(axis=0)
                s = std.mean(axis=0)
            else:
                ch_i = ch_index - 1
                m = mean[ch_i]
                s = std[ch_i]

            color = _cond_color(label, i)

            # Bande ±1 std (semi-transparente)
            rgba = QColor(color)
            rgba.setAlpha(50)
            c_upper = self._erp_pw.plot(times, m + s, pen=pg.mkPen(None))
            c_lower = self._erp_pw.plot(times, m - s, pen=pg.mkPen(None))
            fill = pg.FillBetweenItem(c_upper, c_lower, brush=pg.mkBrush(rgba))
            self._erp_pw.addItem(fill)
            self._erp_fills.extend([fill])
            self._erp_curves.extend([c_upper, c_lower])

            # Courbe moyenne
            c_mean = self._erp_pw.plot(
                times, m,
                pen=pg.mkPen(color=color, width=2),
                name=f'{label}  (n={n})',
            )
            self._erp_curves.append(c_mean)
            all_vals.extend([m + s, m - s])

        if all_vals:
            combined = np.concatenate(all_vals)
            margin   = (combined.max() - combined.min()) * 0.05 or 1.0
            self._erp_pw.setYRange(combined.min() - margin, combined.max() + margin, padding=0)

    # ── Onglet 2 : PSD ────────────────────────────────────────────────────

    def _precompute_psd(self) -> dict[str, list[tuple[np.ndarray, np.ndarray]]]:
        """Calcule la PSD de la moyenne par condition, pour chaque canal."""
        n_ch  = len(self._ch_labels)
        keys  = ['avg'] + [str(i) for i in range(n_ch)]
        cache: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {k: [] for k in keys}

        for label, (mean_data, _, _, _) in self._groups.items():
            for key in keys:
                if key == 'avg':
                    per_ch = []
                    for ch in range(mean_data.shape[0]):
                        sig_v = mean_data[ch] * 1e-6
                        freqs, psd = compute_psd_welch(sig_v, self._sfreq)
                        per_ch.append(psd)
                    cache['avg'].append((freqs, np.mean(per_ch, axis=0)))
                else:
                    ch_i  = int(key)
                    sig_v = mean_data[ch_i if ch_i < mean_data.shape[0] else 0] * 1e-6
                    freqs, psd = compute_psd_welch(sig_v, self._sfreq)
                    cache[key].append((freqs, psd))

        return cache

    def _build_psd_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(6)

        top = QHBoxLayout()
        top.addWidget(QLabel("Canal :"))
        self._psd_ch_box = QComboBox()
        self._psd_ch_box.addItem("Moyenne")
        self._psd_ch_box.addItems(self._ch_labels)
        self._psd_ch_box.setMinimumWidth(120)
        top.addWidget(self._psd_ch_box)
        top.addStretch()
        layout.addLayout(top)

        self._psd_pw = pg.PlotWidget()
        self._psd_pw.setBackground('#1a1a2e')
        self._psd_pw.setLogMode(x=False, y=True)
        self._psd_pw.setLabel('left',   'Puissance', units='µV²/Hz')
        self._psd_pw.setLabel('bottom', 'Fréquence', units='Hz')
        self._psd_pw.setXRange(0.5, 50.0, padding=0)
        self._psd_pw.showGrid(x=True, y=True, alpha=0.25)
        self._psd_pw.addLegend(offset=(10, 10))

        for band_name, (f_low, f_high) in FREQ_BANDS.items():
            rgba = QColor(BAND_COLORS[band_name])
            rgba.setAlpha(25)
            region = pg.LinearRegionItem(
                values=(f_low, f_high), brush=pg.mkBrush(rgba), movable=False,
            )
            region.setZValue(-10)
            self._psd_pw.addItem(region)
            lbl = pg.TextItem(text=band_name, color=BAND_COLORS[band_name], anchor=(0.5, 1.0))
            lbl.setPos((f_low + f_high) / 2, 0)
            lbl.setZValue(-5)
            self._psd_pw.addItem(lbl)

        layout.addWidget(self._psd_pw)

        self._psd_curves: list[pg.PlotDataItem] = []
        self._draw_psd('avg')
        self._psd_ch_box.currentIndexChanged.connect(
            lambda idx: self._draw_psd('avg' if idx == 0 else str(idx - 1))
        )

        return w

    def _draw_psd(self, ch_key: str) -> None:
        for c in self._psd_curves:
            self._psd_pw.removeItem(c)
        self._psd_curves.clear()

        psds_for_key = self._psd_cache.get(ch_key, [])
        all_psd = []

        for i, (label, _) in enumerate(self._psd_segments):
            if i >= len(psds_for_key):
                break
            freqs, psd = psds_for_key[i]
            color = _cond_color(label, i)
            n = self._groups[label][3]
            c = self._psd_pw.plot(
                freqs, psd,
                pen=pg.mkPen(color=color, width=2),
                name=f'{label}  (n={n})',
            )
            self._psd_curves.append(c)
            all_psd.append(psd)

        if all_psd:
            combined = np.concatenate(all_psd)
            combined = combined[combined > 0]
            if len(combined):
                ymin = max(1e-4, combined.min() * 0.5)
                ymax = combined.max() * 2.0
                self._psd_pw.setYRange(np.log10(ymin), np.log10(ymax), padding=0)

    # ── Onglet 3 : Carte topo ─────────────────────────────────────────────

    def _build_topo_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(6)

        # Récupération des positions 10-20
        pos2d = self._get_electrode_positions()

        if not pos2d:
            layout.addWidget(QLabel(
                "  Les noms de canaux ne correspondent pas au montage 10-20 standard.\n"
                "  La carte topo nécessite des canaux nommés selon la convention 10-20\n"
                "  (ex: C3, Cz, O1, Fz…).",
                alignment=Qt.AlignCenter,
            ))
            return w

        # Sélecteurs condition + bande
        top = QHBoxLayout()
        top.addWidget(QLabel("Condition :"))
        self._topo_cond_box = QComboBox()
        self._topo_cond_box.addItems(list(self._band_powers.keys()))
        top.addWidget(self._topo_cond_box)
        top.addSpacing(20)
        top.addWidget(QLabel("Bande :"))
        self._topo_band_box = QComboBox()
        self._topo_band_box.addItems(list(FREQ_BANDS.keys()))
        self._topo_band_box.setCurrentText('Alpha')
        top.addWidget(self._topo_band_box)
        top.addStretch()
        layout.addLayout(top)

        # Zone plot + colorbar côte à côte
        hbox = QHBoxLayout()
        layout.addLayout(hbox)

        self._topo_vb = pg.PlotWidget(aspectLocked=True)
        self._topo_vb.setBackground('#1a1a2e')
        self._topo_vb.hideAxis('left')
        self._topo_vb.hideAxis('bottom')
        hbox.addWidget(self._topo_vb, stretch=9)

        self._colorbar_widget = _ColorBarWidget()
        hbox.addWidget(self._colorbar_widget, stretch=1)

        self._topo_pos2d    = pos2d
        self._topo_scatter  = pg.ScatterPlotItem(size=22, pxMode=True)
        self._topo_vb.addItem(self._topo_scatter)
        self._topo_labels:  list[pg.TextItem] = []

        self._draw_topo()
        self._topo_cond_box.currentTextChanged.connect(lambda _: self._draw_topo())
        self._topo_band_box.currentTextChanged.connect(lambda _: self._draw_topo())

        return w

    def _get_electrode_positions(self) -> dict[str, tuple[float, float]]:
        """
        Retourne un dict {ch_name: (x, y)} pour les canaux reconnus
        dans le montage standard_1020 de MNE.
        """
        try:
            import mne
            montage = mne.channels.make_standard_montage('standard_1020')
            mne_pos = montage.get_positions()['ch_pos']
        except Exception:
            return {}

        pos2d: dict[str, tuple[float, float]] = {}
        ch_upper = {ch.upper(): ch for ch in self._ch_labels}

        for mne_name, xyz in mne_pos.items():
            key = mne_name.upper()
            if key in ch_upper:
                # Projection 2D : x = droite, y = avant (haut sur écran)
                pos2d[ch_upper[key]] = (float(xyz[0]), float(xyz[1]))

        return pos2d

    def _draw_topo(self) -> None:
        cond_label = self._topo_cond_box.currentText()
        band_name  = self._topo_band_box.currentText()

        bp = self._band_powers.get(cond_label, {}).get(band_name)
        if bp is None:
            return

        # Normalisation [0, 1]
        ch_indices = [
            i for i, ch in enumerate(self._ch_labels)
            if ch in self._topo_pos2d
        ]
        if not ch_indices:
            return

        powers = bp[ch_indices]
        p_min, p_max = powers.min(), powers.max()
        if p_max <= p_min:
            norm = np.zeros_like(powers)
        else:
            norm = (powers - p_min) / (p_max - p_min)

        # Supprimer anciens labels
        for lbl in self._topo_labels:
            self._topo_vb.removeItem(lbl)
        self._topo_labels.clear()

        spots = []
        for k, ch_i in enumerate(ch_indices):
            ch_name = self._ch_labels[ch_i]
            x, y    = self._topo_pos2d[ch_name]
            t       = float(norm[k])
            # Colormap froid → chaud : bleu → rouge
            r = int(t * 255)
            b = int((1 - t) * 255)
            spots.append({
                'pos': (x, y),
                'brush': pg.mkBrush(QColor(r, 0, b, 220)),
                'pen':   pg.mkPen(QColor(180, 180, 180), width=0.5),
            })
            lbl = pg.TextItem(ch_name, color='#cccccc', anchor=(0.5, -0.3))
            lbl.setPos(x, y)
            lbl.setFont(pg.QtGui.QFont("Arial", 6))
            self._topo_vb.addItem(lbl)
            self._topo_labels.append(lbl)

        self._topo_scatter.setData(spots)

        # Mise à jour de la colorbar
        self._colorbar_widget.set_range(p_min, p_max, band_name)


class _ColorBarWidget(QWidget):
    """Barre de couleur verticale (bleu → rouge) avec valeurs min/max."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(60)
        self._p_min  = 0.0
        self._p_max  = 1.0
        self._label  = ""

    def set_range(self, p_min: float, p_max: float, label: str = "") -> None:
        self._p_min = p_min
        self._p_max = p_max
        self._label = label
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        rect = self.rect()
        bar_x      = 20
        bar_width  = 20
        bar_top    = 40
        bar_bottom = rect.height() - 40
        bar_height = bar_bottom - bar_top

        grad = QLinearGradient(bar_x, bar_bottom, bar_x, bar_top)
        grad.setColorAt(0.0, QColor(0, 0, 255))
        grad.setColorAt(1.0, QColor(255, 0, 0))
        painter.fillRect(bar_x, bar_top, bar_width, bar_height, grad)

        painter.setPen(QColor(200, 200, 200))
        painter.setFont(pg.QtGui.QFont("Arial", 8))
        painter.drawText(bar_x, bar_top - 5, f"{self._p_max:.2g}")
        painter.drawText(bar_x, bar_bottom + 14, f"{self._p_min:.2g}")

        if self._label:
            painter.drawText(2, rect.height() // 2, self._label)
        painter.end()
