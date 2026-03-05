import numpy as np


def apply_car(
    data: np.ndarray,
    exclude: list[int] | None = None,
) -> np.ndarray:
    """
    Applique le Common Average Reference (CAR) sur les données EEG.

    Soustrait la moyenne de tous les canaux (sauf exclus) à chaque instant.
    Élimine le bruit commun à toutes les électrodes (interférences 50/60 Hz,
    artefacts globaux de mouvement, etc.).

    Args:
        data: (n_channels, n_samples) — signal EEG en Volts
        exclude: indices des canaux à exclure du calcul de la moyenne
                 (ex: canaux trop bruités). Si None, tous les canaux sont utilisés.

    Returns:
        (n_channels, n_samples) — signal re-référencé en Volts
    """
    mask = list(range(data.shape[0]))
    if exclude:
        mask = [i for i in mask if i not in exclude]
    mean = data[mask].mean(axis=0, keepdims=True)  # (1, n_samples)
    return data - mean
