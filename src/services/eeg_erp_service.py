from __future__ import annotations

import numpy as np

from src.models.epoch_data import EpochData
from src.models.erp_data import ERPData


class EEGERPService:

    @staticmethod
    def compute(
        epoch_data: EpochData,
        selected_ch_indices: list[int],
        baseline_correction: bool = False,
        baseline_tmin: float | None = None,
        baseline_tmax: float = 0.0,
    ) -> ERPData:
        """Moyenne les époques par classe avec correction baseline optionnelle.

        Parameters
        ----------
        epoch_data
            Époques d'entrée, shape (n_epochs, n_channels, n_times).
        selected_ch_indices
            Indices des canaux à inclure dans le résultat.
        baseline_correction
            Si True, soustrait la moyenne de la fenêtre [baseline_tmin, baseline_tmax].
        baseline_tmin
            Début de la fenêtre baseline en secondes. Défaut : times[0].
        baseline_tmax
            Fin de la fenêtre baseline en secondes. Défaut : 0.0 (onset stimulus).

        Returns
        -------
        ERPData avec erp_by_class[label] de shape (n_ch, n_times) en µV.
        """
        times = epoch_data.times
        labels_arr = np.array(epoch_data.labels)
        unique_labels = sorted(set(epoch_data.labels))
        ch_names = [epoch_data.ch_names[i] for i in selected_ch_indices]

        if baseline_tmin is None:
            baseline_tmin = float(times[0])

        erp_by_class: dict[str, np.ndarray] = {}

        for label in unique_labels:
            mask = labels_arr == label
            subset = epoch_data.data[mask]                          # (n_label, n_channels, n_times)
            mean_uv = subset[:, selected_ch_indices, :].mean(axis=0) * 1e6  # (n_ch, n_times) µV

            if baseline_correction:
                bl_mask = (times >= baseline_tmin) & (times <= baseline_tmax)
                if bl_mask.any():
                    baseline_mean = mean_uv[:, bl_mask].mean(axis=1, keepdims=True)
                    mean_uv = mean_uv - baseline_mean

            erp_by_class[label] = mean_uv

        return ERPData(
            times=times,
            ch_names=ch_names,
            erp_by_class=erp_by_class,
            baseline_corrected=baseline_correction,
            baseline_tmin=baseline_tmin,
            baseline_tmax=baseline_tmax,
        )
