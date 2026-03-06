from dataclasses import dataclass


@dataclass
class PreprocessConfig:
    bandpass_enabled: bool = False
    low_hz: float = 8.0
    high_hz: float = 30.0
    notch_enabled: bool = False
    notch_hz: float = 50.0
    reref_enabled: bool = False
    reref_mode: str = "average"  # "average" | nom du canal de référence
