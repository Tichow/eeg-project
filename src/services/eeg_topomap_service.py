from __future__ import annotations

import numpy as np

from src.models.erp_data import ERPData
from src.models.frequency_data import FrequencyData
from src.models.topomap_data import TopoMapData


class EEGTopoMapService:

    @staticmethod
    def compute_amplitude(
        erp_data: ERPData,
        tmin: float,
        tmax: float,
    ) -> TopoMapData:
        """Moyenne l'amplitude ERP sur la fenêtre [tmin, tmax] par canal et classe.

        Parameters
        ----------
        erp_data
            Résultat du calcul ERP, shape par classe (n_ch, n_times) en µV.
        tmin, tmax
            Bornes de la fenêtre de temps (secondes relatifs à l'onset).

        Returns
        -------
        TopoMapData avec by_class[label] de shape (n_ch,) en µV.
        """
        times = erp_data.times
        t_mask = (times >= tmin) & (times <= tmax)
        if not t_mask.any():
            raise ValueError(f"Aucun échantillon dans la fenêtre [{tmin}, {tmax}] s")

        by_class: dict[str, np.ndarray] = {}
        for label, data in erp_data.erp_by_class.items():
            by_class[label] = data[:, t_mask].mean(axis=1)  # (n_ch,)

        all_vals = np.concatenate(list(by_class.values()))
        max_abs = float(np.nanmax(np.abs(all_vals)))
        if max_abs == 0:
            max_abs = 1.0

        window_label = f"{tmin:.2f}–{tmax:.2f} s"
        return TopoMapData(
            ch_names=erp_data.ch_names,
            by_class=by_class,
            unit="µV",
            clim=(-max_abs, max_abs),
            mode="amplitude",
            window_label=window_label,
        )

    @staticmethod
    def compute_power(
        freq_data: FrequencyData,
        fmin: float,
        fmax: float,
    ) -> TopoMapData:
        """Moyenne la puissance PSD sur la bande [fmin, fmax] par canal et classe.

        Parameters
        ----------
        freq_data
            Résultat du calcul PSD, shape par classe (n_ch, n_freqs) en µV²/Hz.
        fmin, fmax
            Bornes de la bande de fréquences (Hz).

        Returns
        -------
        TopoMapData avec by_class[label] de shape (n_ch,) en µV²/Hz.
        """
        if freq_data.mode != "psd":
            raise ValueError("compute_power requiert un FrequencyData en mode 'psd'")

        freqs = freq_data.freqs
        f_mask = (freqs >= fmin) & (freqs <= fmax)
        if not f_mask.any():
            raise ValueError(f"Aucune fréquence dans la bande [{fmin}, {fmax}] Hz")

        by_class: dict[str, np.ndarray] = {}
        for label, data in freq_data.psd_by_class.items():
            by_class[label] = data[:, f_mask].mean(axis=1)  # (n_ch,)

        all_vals = np.concatenate(list(by_class.values()))
        max_val = float(np.nanmax(all_vals))
        if max_val == 0:
            max_val = 1.0

        window_label = f"{fmin:.1f}–{fmax:.1f} Hz"
        return TopoMapData(
            ch_names=freq_data.ch_names,
            by_class=by_class,
            unit="µV²/Hz",
            clim=(0.0, max_val),
            mode="power",
            window_label=window_label,
        )
