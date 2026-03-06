"""
Interface commune pour tous les panels de visualisation EEG.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from PyQt5.QtWidgets import QWidget


@dataclass
class DashboardState:
    """État global du dashboard transmis à chaque panel à chaque frame."""
    ch_visible:      list[bool]
    show_snr:        bool
    ref_psd:         dict | None       = None
    """
    Optionnel — PSD de référence pour la comparaison de segments.
    Structure : {
        "label":      str,
        "freqs":      np.ndarray,       # (n_freqs,) Hz
        "psd_per_ch": list[np.ndarray]  # un (n_freqs,) µV²/Hz par canal
    }
    """
    annotations:     list[dict] | None = None
    """
    Optionnel — annotations temporelles (PhysioNet T0/T1/T2 ou events JSON).
    Structure : [{"time_sec": float, "label": str}, ...]
    """
    current_pos_sec: float             = 0.0
    """Position de lecture courante en secondes (pour les event markers)."""


class BasePanel(ABC):
    """
    Contrat que tout panel doit respecter.

    Un panel possède un QWidget (self.widget) que le Dashboard
    intègre dans son layout. update() est appelé à chaque tick du timer.
    """

    @property
    @abstractmethod
    def widget(self) -> QWidget:
        """Retourne le widget Qt racine de ce panel."""
        ...

    @abstractmethod
    def update(
        self,
        data_filt: np.ndarray,   # (n_ch, n_samples) en Volts
        sfreq: float,
        state: DashboardState,
    ) -> None:
        """
        Met à jour le panel avec les nouvelles données filtrées.

        Args:
            data_filt : données EEG filtrées en Volts, shape (n_ch, n_samples).
                        Chaque panel convertit en µV en interne si besoin.
            sfreq     : fréquence d'échantillonnage en Hz.
            state     : état global (canaux visibles, options d'affichage).
        """
        ...

    def on_channels_changed(self, ch_visible: list[bool]) -> None:
        """
        Hook appelé immédiatement quand la visibilité des canaux change
        (sans attendre le prochain tick du timer). No-op par défaut.
        """
