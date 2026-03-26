from __future__ import annotations

import json
import os

from src.models.electrode_preset import ElectrodePreset

_DEFAULT_PATH = "data/presets/electrode_presets.json"


class PresetService:
    """Pure business logic — no Qt. CRUD for electrode presets on disk."""

    @staticmethod
    def load_all(path: str = _DEFAULT_PATH) -> list[ElectrodePreset]:
        """Read JSON → list of ElectrodePreset. Returns [] if file absent."""
        if not os.path.isfile(path):
            return []
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return [
            ElectrodePreset(
                name=entry["name"],
                channels={int(k): v for k, v in entry["channels"].items()},
            )
            for entry in raw
        ]

    @staticmethod
    def save_all(presets: list[ElectrodePreset], path: str = _DEFAULT_PATH) -> None:
        """Write full list of presets to JSON (overwrite)."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = [
            {"name": p.name, "channels": {str(k): v for k, v in p.channels.items()}}
            for p in presets
        ]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @staticmethod
    def save_preset(preset: ElectrodePreset, path: str = _DEFAULT_PATH) -> None:
        """Upsert a preset by name (add new or replace existing)."""
        presets = PresetService.load_all(path)
        for i, p in enumerate(presets):
            if p.name == preset.name:
                presets[i] = preset
                break
        else:
            presets.append(preset)
        PresetService.save_all(presets, path)

    @staticmethod
    def delete_preset(name: str, path: str = _DEFAULT_PATH) -> None:
        """Remove preset with given name."""
        presets = [p for p in PresetService.load_all(path) if p.name != name]
        PresetService.save_all(presets, path)

    @staticmethod
    def get_preset(name: str, path: str = _DEFAULT_PATH) -> ElectrodePreset | None:
        """Retrieve a single preset by name, or None if not found."""
        for p in PresetService.load_all(path):
            if p.name == name:
                return p
        return None
