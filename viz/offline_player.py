"""
Adaptateur OfflinePlayer — rejoue un enregistrement EEG avec la même interface
que BrainFlow (get_current_board_data), permettant à Dashboard de fonctionner
en mode offline sans aucune modification de sa logique d'acquisition.

Données attendues : tableau numpy (n_ch, n_total_samples) en µV, exactement
comme BrainFlow les stocke. Le Dashboard indexe les données par [channel_index],
donc OfflinePlayer renvoie un tableau où row i = canal i.
"""

from __future__ import annotations

import numpy as np


class OfflinePlayer:
    """
    Adaptateur drop-in pour Dashboard.board en mode offline.

    Expose get_current_board_data(n_samples) identique à BrainFlow,
    plus des méthodes de contrôle de lecture (pause/resume/seek/speed).
    """

    _SPEED_OPTIONS = (0.5, 1.0, 2.0, 4.0)

    def __init__(
        self,
        data_uv: np.ndarray,
        sfreq: float,
        update_ms: int = 100,
    ) -> None:
        """
        Args:
            data_uv   : données en µV, shape (n_ch, n_total_samples)
            sfreq     : fréquence d'échantillonnage en Hz
            update_ms : intervalle du timer d'acquisition (pour calculer le pas)
        """
        if data_uv.ndim != 2:
            raise ValueError(f"data_uv doit être 2D (n_ch, n_samples), got {data_uv.shape}")

        self._data      = data_uv
        self._n_ch      = data_uv.shape[0]
        self._n_total   = data_uv.shape[1]
        self._sfreq     = float(sfreq)
        self._update_ms = update_ms

        # Pas de base = nombre de samples pour couvrir update_ms à vitesse 1×
        self._base_step  = max(1, int(sfreq * update_ms / 1000))
        self._speed      = 1.0
        self._paused     = False
        # Démarre assez loin pour avoir une fenêtre complète dès la 1ère frame
        self._pos        = self._base_step

    # ------------------------------------------------------------------
    # Interface BrainFlow (seule méthode utilisée par Dashboard._update)
    # ------------------------------------------------------------------

    def get_current_board_data(self, n_samples: int) -> np.ndarray:
        """
        Renvoie une fenêtre glissante de n_samples terminant à _pos.

        Si en pause, _pos n'avance pas et la même fenêtre est renvoyée.
        En fin d'enregistrement, boucle automatiquement au début.

        Returns:
            array shape (n_ch, n_samples) en µV, indexable par row = canal
        """
        end   = int(self._pos)
        start = end - n_samples

        if start < 0:
            # Pad de zéros en début d'enregistrement
            pad   = np.zeros((self._n_ch, -start))
            chunk = np.concatenate([pad, self._data[:, :end]], axis=1)
        else:
            chunk = self._data[:, start:end].copy()

        # Avancer la position (seulement si en lecture)
        if not self._paused:
            step = max(1, int(self._base_step * self._speed))
            self._pos += step
            if self._pos >= self._n_total:
                self._pos = self._base_step  # boucle

        return chunk

    # ------------------------------------------------------------------
    # Contrôles de lecture
    # ------------------------------------------------------------------

    def pause(self) -> None:
        self._paused = True

    def resume(self) -> None:
        self._paused = False

    def seek(self, sample: int) -> None:
        """Sauter à la position sample (en indices samples)."""
        self._pos = max(self._base_step, min(int(sample), self._n_total - 1))

    def set_speed(self, factor: float) -> None:
        """Vitesse de lecture : 0.5, 1.0, 2.0 ou 4.0."""
        self._speed = float(factor)

    # ------------------------------------------------------------------
    # Propriétés publiques (lues par PlaybackWidget)
    # ------------------------------------------------------------------

    @property
    def is_paused(self) -> bool:
        return self._paused

    @property
    def current_pos(self) -> int:
        return int(self._pos)

    @property
    def n_total(self) -> int:
        return self._n_total

    @property
    def sfreq(self) -> float:
        return self._sfreq
