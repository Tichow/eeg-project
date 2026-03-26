from __future__ import annotations

import os
import re

import mne

from src.constants.eeg_constants import RUN_DESCRIPTIONS
from src.models.edf_file_info import EdfFileInfo

_MNE_SUBPATH = os.path.join("MNE-eegbci-data", "files", "eegmmidb", "1.0.0")
_SUBJECT_RE = re.compile(r"^S(\d{3})$")
_CUSTOM_SUBJECT_RE = re.compile(r"^[A-Za-z]+\d+$")  # MATTEO1, NEMO3, AUTRE2…
_FILE_RE = re.compile(r"^S(\d{3})R(\d{2})\.edf$", re.IGNORECASE)
_RUN_LABELED_FILE_RE = re.compile(r"^.+R(\d{2})\.edf$", re.IGNORECASE)  # {ANY}R{NN}.edf
_CUSTOM_FILE_RE = re.compile(r"\.edf$", re.IGNORECASE)


def _resolve_data_root(data_path: str) -> str:
    """Return the actual directory containing subject folders."""
    mne_path = os.path.join(data_path, _MNE_SUBPATH)
    if os.path.isdir(mne_path):
        return mne_path
    return data_path


class EEGDataService:
    @staticmethod
    def scan_subjects(data_path: str) -> list[int | str]:
        """Return sorted list of subject IDs found in the data directory.

        PhysioNet subjects (S001…S109) are returned as int.
        Custom subjects (MATTEO1, NEMO2…) are returned as str.
        """
        root = _resolve_data_root(data_path)
        if not os.path.isdir(root):
            return []
        int_subjects: list[int] = []
        str_subjects: list[str] = []
        for name in os.listdir(root):
            if not os.path.isdir(os.path.join(root, name)):
                continue
            m = _SUBJECT_RE.match(name)
            if m:
                int_subjects.append(int(m.group(1)))
            elif _CUSTOM_SUBJECT_RE.match(name):
                str_subjects.append(name)
        return sorted(int_subjects) + sorted(str_subjects)

    @staticmethod
    def list_edf_files(data_path: str, subject: int | str) -> list["EdfFileInfo"]:
        """List EDF files for a subject with basic filesystem metadata (no MNE read)."""
        root = _resolve_data_root(data_path)
        folder = f"S{subject:03d}" if isinstance(subject, int) else subject
        subject_dir = os.path.join(root, folder)
        if not os.path.isdir(subject_dir):
            return []
        files = []
        custom_run = 100  # PhysioNet uses 1-14; custom files start at 100
        for name in sorted(os.listdir(subject_dir)):
            path = os.path.join(subject_dir, name)
            m = _FILE_RE.match(name)
            if m:
                run = int(m.group(2))
                description = RUN_DESCRIPTIONS.get(run, "Run inconnu")
            else:
                m_run = _RUN_LABELED_FILE_RE.match(name)
                if m_run:
                    run = int(m_run.group(1))
                    description = RUN_DESCRIPTIONS.get(run, f"Run {run:02d}")
                elif _CUSTOM_FILE_RE.search(name):
                    run = custom_run
                    custom_run += 1
                    description = os.path.splitext(name)[0]
                else:
                    continue
            files.append(EdfFileInfo(
                path=path,
                subject=subject,
                run=run,
                description=description,
                size_bytes=os.path.getsize(path),
            ))
        return sorted(files, key=lambda f: f.run)

    @staticmethod
    def read_edf_header(path: str) -> tuple[float, float, int, dict[str, int]]:
        """Read EDF header metadata. Returns (duration_s, sfreq, nchan, ann_counts)."""
        raw = mne.io.read_raw_edf(path, preload=False, verbose=False)
        duration_s = float(raw.times[-1]) if len(raw.times) > 0 else 0.0
        ann_counts: dict[str, int] = {}
        for ann in raw.annotations:
            desc = str(ann["description"])
            ann_counts[desc] = ann_counts.get(desc, 0) + 1
        return duration_s, float(raw.info["sfreq"]), int(raw.info["nchan"]), ann_counts
