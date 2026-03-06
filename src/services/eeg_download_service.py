import mne


class EEGDownloadService:
    @staticmethod
    def download_subject(subject: int, runs: list[int], path: str) -> list[str]:
        """Download EDF files for one subject and return their local paths."""
        file_paths = mne.datasets.eegbci.load_data(
            subjects=subject,
            runs=runs,
            path=path,
            update_path=False,
            verbose=False,
        )
        return [str(p) for p in file_paths]
