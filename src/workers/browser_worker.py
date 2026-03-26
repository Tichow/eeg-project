from __future__ import annotations

from PyQt5.QtCore import QThread, pyqtSignal

from src.services.eeg_data_service import EEGDataService


class BrowserWorker(QThread):
    subjects_ready   = pyqtSignal(list)              # list[int | str]
    files_skeleton   = pyqtSignal(list)              # list[EdfFileInfo] with metadata=-1
    file_header_done = pyqtSignal(int, float, float, int, object)  # (run, duration_s, sfreq, nchan, ann_counts)
    finished         = pyqtSignal()
    error            = pyqtSignal(str)

    def __init__(self, data_path: str, subject: int | str | None = None, parent=None):
        super().__init__(parent)
        self._data_path = data_path
        self._subject = subject

    def run(self):
        try:
            if self._subject is None:
                subjects = EEGDataService.scan_subjects(self._data_path)
                self.subjects_ready.emit(subjects)
            else:
                files = EEGDataService.list_edf_files(self._data_path, self._subject)
                self.files_skeleton.emit(files)
                for info in files:
                    duration_s, sfreq, nchan, ann_counts = EEGDataService.read_edf_header(info.path)
                    self.file_header_done.emit(info.run, duration_s, sfreq, nchan, ann_counts)
            self.finished.emit()
        except Exception as exc:
            self.error.emit(str(exc))
