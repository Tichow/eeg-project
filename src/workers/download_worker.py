from PyQt5.QtCore import QThread, pyqtSignal

from src.models.download_config import DownloadConfig
from src.services.eeg_download_service import EEGDownloadService


class DownloadWorker(QThread):
    progress = pyqtSignal(int, int, int)   # (current_idx, total, subject_num)
    subject_done = pyqtSignal(int, str)    # (subject_num, path)
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, config: DownloadConfig, parent=None):
        super().__init__(parent)
        self._config = config

    def run(self):
        subjects = self._config.subjects
        runs = self._config.runs
        total = len(subjects) * len(runs)
        current = 0
        try:
            for subject in subjects:
                for run in runs:
                    paths = EEGDownloadService.download_subject(
                        subject=subject,
                        runs=[run],
                        path=self._config.data_path,
                    )
                    current += 1
                    self.progress.emit(current, total, subject)
                    for p in paths:
                        self.subject_done.emit(subject, p)
            self.finished.emit()
        except Exception as exc:
            self.error.emit(str(exc))
