import os
import re

import mne

from src.constants.eeg_constants import RUN_DESCRIPTIONS
from src.models.edf_file_info import EdfFileInfo

_MNE_SUBPATH = os.path.join("MNE-eegbci-data", "files", "eegmmidb", "1.0.0")
_SUBJECT_RE = re.compile(r"^S(\d{3})$")
_FILE_RE = re.compile(r"^S(\d{3})R(\d{2})\.edf$", re.IGNORECASE)


def _resolve_data_root(data_path: str) -> str:
    """Return the actual directory containing subject folders."""
    mne_path = os.path.join(data_path, _MNE_SUBPATH)
    if os.path.isdir(mne_path):
        return mne_path
    return data_path


class EEGDataService:
    @staticmethod
    def scan_subjects(data_path: str) -> list[int]:
        """Return sorted list of subject numbers found in the data directory."""
        root = _resolve_data_root(data_path)
        if not os.path.isdir(root):
            return []
        subjects = []
        for name in os.listdir(root):
            m = _SUBJECT_RE.match(name)
            if m and os.path.isdir(os.path.join(root, name)):
                subjects.append(int(m.group(1)))
        return sorted(subjects)

    @staticmethod
    def list_edf_files(data_path: str, subject: int) -> list["EdfFileInfo"]:
        """List EDF files for a subject with basic filesystem metadata (no MNE read)."""
        root = _resolve_data_root(data_path)
        subject_dir = os.path.join(root, f"S{subject:03d}")
        if not os.path.isdir(subject_dir):
            return []
        files = []
        for name in os.listdir(subject_dir):
            m = _FILE_RE.match(name)
            if not m:
                continue
            run = int(m.group(2))
            path = os.path.join(subject_dir, name)
            files.append(EdfFileInfo(
                path=path,
                subject=subject,
                run=run,
                description=RUN_DESCRIPTIONS.get(run, "Run inconnu"),
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
