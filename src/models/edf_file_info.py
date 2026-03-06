from dataclasses import dataclass


@dataclass
class EdfFileInfo:
    path: str
    subject: int
    run: int
    description: str
    size_bytes: int
    # Filled in after EDF header read; -1 = not yet read
    duration_s: float = -1.0
    sfreq: float = -1.0
    nchan: int = -1
    n_annotations: int = -1
