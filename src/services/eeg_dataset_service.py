import numpy as np

from src.models.signal_data import SignalData
from src.models.epoch_data import EpochData
from src.models.preprocess_config import PreprocessConfig
from src.services.eeg_signal_service import EEGSignalService
from src.services.eeg_preprocess_service import EEGPreprocessService
from src.services.eeg_epoch_service import EEGEpochService
from src.services.eeg_artifact_service import EEGArtifactService
from src.services.eeg_alignment_service import EEGAlignmentService
from src.services.eeg_download_service import EEGDownloadService

EXCLUDED_SUBJECTS = {88, 89, 92, 100, 104, 106}

MI_RUNS_HANDS = [4, 8, 12]
MI_RUNS_FEET = [6, 10, 14]
MI_RUNS_ALL = MI_RUNS_HANDS + MI_RUNS_FEET

# Label remapping: T1/T2 meaning depends on run type
_LABEL_MAP_HANDS = {"T1": "left_hand", "T2": "right_hand"}
_LABEL_MAP_FEET = {"T1": "both_fists", "T2": "both_feet"}

PHYSIONET_PREPROCESS = PreprocessConfig(
    bandpass_enabled=True,
    low_hz=4.0,
    high_hz=40.0,
    notch_enabled=True,
    notch_hz=60.0,
    reref_enabled=True,
    reref_mode="average",
)

OPENBCI_PREPROCESS = PreprocessConfig(
    bandpass_enabled=True,
    low_hz=4.0,
    high_hz=40.0,
    notch_enabled=True,
    notch_hz=50.0,
    reref_enabled=True,
    reref_mode="average",
)


class EEGDatasetService:

    @staticmethod
    def get_valid_subjects(n_subjects: int = 109) -> list[int]:
        """Return first n valid subject IDs (excluding known bad subjects)."""
        valid = [s for s in range(1, 110) if s not in EXCLUDED_SUBJECTS]
        return valid[:n_subjects]

    @staticmethod
    def download_subjects(
        subjects: list[int],
        runs: list[int],
        data_path: str = "data/",
    ) -> dict[int, list[str]]:
        """Download EDF files for multiple subjects. Returns {subject: [paths]}."""
        result = {}
        for subj in subjects:
            paths = EEGDownloadService.download_subject(subj, runs, data_path)
            result[subj] = paths
        return result

    @staticmethod
    def process_single_run(
        edf_path: str,
        run: int,
        config: PreprocessConfig,
        epoch_tmin: float = 0.5,
        epoch_tmax: float = 4.0,
        artifact_threshold_uv: float = 200.0,
    ) -> tuple[np.ndarray, list[str]] | None:
        """Process one EDF file into classification-ready epochs.

        Parameters
        ----------
        edf_path : path to EDF file
        run : run number (determines T1/T2 label meaning)
        config : preprocessing config
        epoch_tmin : start of epoch relative to annotation onset (seconds)
        epoch_tmax : end of epoch relative to annotation onset (seconds)
        artifact_threshold_uv : peak-to-peak rejection threshold in µV
            (200 µV for 64 channels — some peripheral channels have high amplitude)

        Returns
        -------
        (X, labels) or None if no valid epochs.
            X : (n_trials, n_channels, n_times) float32 in Volts
            labels : list of semantic labels (e.g. "left_hand", "right_hand")
        """
        signal = EEGSignalService.load_signal(edf_path)
        signal = EEGPreprocessService.apply(signal, config)

        try:
            epochs = EEGEpochService.extract(signal, tmin=epoch_tmin, tmax=epoch_tmax)
        except ValueError:
            return None

        bad = EEGArtifactService.detect_by_threshold(epochs, artifact_threshold_uv)
        if bad:
            epochs = EEGArtifactService.apply_threshold_rejection(epochs, bad)

        if run in MI_RUNS_HANDS:
            label_map = _LABEL_MAP_HANDS
        elif run in MI_RUNS_FEET:
            label_map = _LABEL_MAP_FEET
        else:
            return None

        task_mask = [l in ("T1", "T2") for l in epochs.labels]
        if not any(task_mask):
            return None

        X = epochs.data[task_mask].astype(np.float32)
        labels = [label_map[l] for l, m in zip(epochs.labels, task_mask) if m]

        return X, labels

    @staticmethod
    def build_dataset(
        subjects: list[int],
        runs: list[int],
        data_path: str = "data/",
        config: PreprocessConfig | None = None,
        epoch_tmin: float = 0.5,
        epoch_tmax: float = 4.0,
        artifact_threshold_uv: float = 200.0,
        apply_alignment: bool = True,
        normalize: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build a complete dataset from PhysioNet EDF files.

        Parameters
        ----------
        subjects : list of subject IDs
        runs : list of run numbers
        data_path : root data directory
        config : preprocessing config (defaults to PHYSIONET_PREPROCESS)
        epoch_tmin, epoch_tmax : epoch window relative to onset
        artifact_threshold_uv : rejection threshold
        apply_alignment : apply Euclidean Alignment per subject
        normalize : apply Z-score normalization per channel per epoch

        Returns
        -------
        X : (n_total_trials, n_channels, n_times) float32
        y : (n_total_trials,) int labels
        subject_ids : (n_total_trials,) subject ID for each trial
        """
        if config is None:
            config = PHYSIONET_PREPROCESS

        all_X = []
        all_y = []
        all_subj = []

        label_to_int: dict[str, int] = {}

        for subj in subjects:
            subj_X_list = []
            subj_y_list = []

            paths = EEGDownloadService.download_subject(subj, runs, data_path)
            run_for_path = dict(zip(paths, runs))

            for path in paths:
                run = run_for_path[path]
                result = EEGDatasetService.process_single_run(
                    path, run, config, epoch_tmin, epoch_tmax, artifact_threshold_uv,
                )
                if result is None:
                    continue
                X_run, labels_run = result

                for label in labels_run:
                    if label not in label_to_int:
                        label_to_int[label] = len(label_to_int)

                subj_X_list.append(X_run)
                subj_y_list.extend([label_to_int[l] for l in labels_run])

            if not subj_X_list:
                continue

            X_subj = np.concatenate(subj_X_list, axis=0)

            # EA first (needs raw covariance structure), then normalize
            if apply_alignment and X_subj.shape[0] > 1:
                X_subj, _ = EEGAlignmentService.euclidean_alignment(X_subj)

            if normalize:
                X_subj = EEGDatasetService._normalize(X_subj)

            all_X.append(X_subj)
            all_y.extend(subj_y_list)
            all_subj.extend([subj] * X_subj.shape[0])

            print(f"  S{subj:03d}: {X_subj.shape[0]} trials")

        X = np.concatenate(all_X, axis=0).astype(np.float32)
        y = np.array(all_y, dtype=np.int64)
        subject_ids = np.array(all_subj, dtype=np.int64)

        print(f"\nDataset: {X.shape[0]} trials, {X.shape[1]} channels, {X.shape[2]} samples")
        print(f"Classes: {label_to_int}")
        return X, y, subject_ids

    @staticmethod
    def _normalize(X: np.ndarray) -> np.ndarray:
        """Z-score normalization per channel per epoch."""
        X_norm = X.copy()
        for i in range(X_norm.shape[0]):
            for ch in range(X_norm.shape[1]):
                std = X_norm[i, ch].std()
                if std > 0:
                    X_norm[i, ch] = (X_norm[i, ch] - X_norm[i, ch].mean()) / std
        return X_norm
