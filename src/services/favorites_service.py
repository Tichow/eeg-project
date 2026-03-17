from __future__ import annotations

import json
import os

_DEFAULT_PATH = "data/presets/favorites.json"


class FavoritesService:
    """Pure business logic — no Qt. Persists a set of favorite EDF file paths."""

    def __init__(self, path: str = _DEFAULT_PATH):
        self._path = path
        self._favorites: set[str] = set(self._load())

    def is_favorite(self, path: str) -> bool:
        return path in self._favorites

    def toggle(self, path: str) -> bool:
        """Toggle favorite status. Returns new state (True = now a favorite)."""
        if path in self._favorites:
            self._favorites.discard(path)
        else:
            self._favorites.add(path)
        self._save()
        return path in self._favorites

    def _load(self) -> list[str]:
        if not os.path.isfile(self._path):
            return []
        with open(self._path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("favorites", [])

    def favorite_subjects(self) -> set[str]:
        """Return the set of subject folder names that have at least one favorite.

        Works by extracting the parent directory name from each stored path,
        e.g. "data/custom/NEMO3/NEMO3R01.edf" → "NEMO3",
             ".../S001/S001R01.edf"            → "S001".
        """
        subjects: set[str] = set()
        for path in self._favorites:
            parts = path.replace("\\", "/").split("/")
            if len(parts) >= 2:
                subjects.add(parts[-2])
        return subjects

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self._path), exist_ok=True)
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump({"favorites": sorted(self._favorites)}, f, indent=2, ensure_ascii=False)
