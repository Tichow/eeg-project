from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ClassificationConfig:
    """Configuration for the CSP+LDA motor imagery pipeline."""

    task: str = "left_vs_right"           # "left_vs_right" | "hands_vs_feet"
    runs: list[int] = field(default_factory=list)
    channels: list[str] = field(default_factory=list)
    # Preprocessing
    bandpass_low: float = 8.0             # Hz
    bandpass_high: float = 30.0           # Hz
    notch_hz: float | None = None         # Hz (None = disabled)
    # Epoching
    tmin: float = 0.5                     # s after onset
    tmax: float = 3.5                     # s after onset
    # Artifact rejection
    reject_threshold_uv: float = 500.0    # µV peak-to-peak (PhysioNet ~200-400 µV typical)
    # CSP
    n_csp_components: int = 6
    # Filter Bank CSP
    use_fbcsp: bool = False
    fbcsp_bands: list[tuple[float, float]] = field(default_factory=lambda: [
        (8, 13), (13, 30),
    ])
    n_fbcsp_components: int = 3   # CSP components per sub-band
    # Cross-validation
    n_folds: int = 5
    # Model persistence
    model_dir: str = "models/"


@dataclass
class ClassificationResult:
    """Result of a single subject classification run."""

    subject: int | str
    task: str
    n_epochs: dict[str, int] = field(default_factory=dict)   # class_label -> count
    n_epochs_rejected: int = 0
    cv_accuracies: np.ndarray = field(default_factory=lambda: np.array([]))
    mean_accuracy: float = 0.0
    std_accuracy: float = 0.0
    confusion_matrix: np.ndarray = field(default_factory=lambda: np.array([]))
    class_names: list[str] = field(default_factory=list)
    csp_patterns: np.ndarray | None = None
    model_path: str | None = None
