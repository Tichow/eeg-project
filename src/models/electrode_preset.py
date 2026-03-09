from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ElectrodePreset:
    name: str
    channels: dict[int, str] = field(default_factory=dict)  # {1: "Cz", 2: "C3", ...}

    def get_ordered(self, n: int = 8) -> list[str]:
        """Return electrode names ordered by channel 1..n.

        Channels without assignment return empty string.
        """
        return [self.channels.get(i, "") for i in range(1, n + 1)]
