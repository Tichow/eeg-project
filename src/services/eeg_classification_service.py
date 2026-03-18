from __future__ import annotations

import os
import pickle

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from src.models.classification_data import ClassificationConfig, ClassificationResult
from src.models.preprocess_config import PreprocessConfig
from src.models.signal_data import SignalData
from src.services.eeg_artifact_service import EEGArtifactService
from src.services.eeg_epoch_service import EEGEpochService
from src.services.eeg_preprocess_service import EEGPreprocessService
from src.services.eeg_signal_service import EEGSignalService


class EEGClassificationService:
    """CSP + LDA classification pipeline for Motor Imagery EEG."""

    RUNS_LEFT_VS_RIGHT_MI = [4, 8, 12]
    RUNS_HANDS_VS_FEET_MI = [6, 10, 14]
    RUNS_LEFT_VS_RIGHT_MO = [3, 7, 11]
    RUNS_HANDS_VS_FEET_MO = [5, 9, 13]

    MI_8CH = ["C3", "FC1", "C4", "CP1", "Cz", "CP2", "FC2", "Pz"]

    @staticmethod
    def resolve_runs(config: ClassificationConfig) -> list[int]:
        """Return PhysioNet run numbers for the given task."""
        if config.runs:
            return config.runs
        mapping = {
            "left_vs_right": EEGClassificationService.RUNS_LEFT_VS_RIGHT_MI,
            "hands_vs_feet": EEGClassificationService.RUNS_HANDS_VS_FEET_MI,
            "left_vs_right_mo": EEGClassificationService.RUNS_LEFT_VS_RIGHT_MO,
            "hands_vs_feet_mo": EEGClassificationService.RUNS_HANDS_VS_FEET_MO,
        }
        return mapping.get(config.task, EEGClassificationService.RUNS_LEFT_VS_RIGHT_MI)

    @staticmethod
    def load_and_merge_runs(
        data_path: str,
        subject: int | str,
        runs: list[int],
        channels: list[str] | None = None,
    ) -> SignalData:
        """Load multiple EDF runs, optionally pick channels, concatenate."""
        all_data: list[np.ndarray] = []
        all_annotations: list[tuple[float, float, str]] = []
        ch_names: list[str] = []
        sfreq: float = 0.0
        time_offset: float = 0.0

        for run in runs:
            path = EEGClassificationService._build_path(data_path, subject, run)
            sig = EEGSignalService.load_signal(path)

            # Pick channels if requested
            if channels:
                sig = EEGClassificationService._pick_channels(sig, channels)

            if not ch_names:
                ch_names = sig.ch_names
                sfreq = sig.sfreq

            all_data.append(sig.data)
            for onset, dur, desc in sig.annotations:
                all_annotations.append((onset + time_offset, dur, desc))
            time_offset += sig.data.shape[1] / sig.sfreq

        merged = np.concatenate(all_data, axis=1)
        times = np.arange(merged.shape[1]) / sfreq

        return SignalData(
            data=merged,
            times=times,
            ch_names=ch_names,
            sfreq=sfreq,
            annotations=all_annotations,
        )

    @staticmethod
    def preprocess(signal_data: SignalData, config: ClassificationConfig) -> SignalData:
        """Apply bandpass + optional notch via existing EEGPreprocessService."""
        preprocess_config = PreprocessConfig(
            bandpass_enabled=True,
            low_hz=config.bandpass_low,
            high_hz=config.bandpass_high,
            notch_enabled=config.notch_hz is not None,
            notch_hz=config.notch_hz or 50.0,
            reref_enabled=False,
        )
        return EEGPreprocessService.apply(signal_data, preprocess_config)

    @staticmethod
    def extract_mi_epochs(
        signal_data: SignalData,
        config: ClassificationConfig,
    ) -> tuple[np.ndarray, np.ndarray, list[str], int]:
        """Extract MI epochs, reject artifacts, encode labels.

        Returns (X, y, class_names, n_rejected) where:
        - X: (n_epochs, n_channels, n_times)
        - y: (n_epochs,) integer labels
        - class_names: ["T1", "T2"]
        - n_rejected: count of rejected epochs
        """
        epoch_data = EEGEpochService.extract(signal_data, config.tmin, config.tmax)

        # Keep only T1 and T2 (drop T0 rest)
        mi_mask = [i for i, lbl in enumerate(epoch_data.labels) if lbl in ("T1", "T2")]
        if not mi_mask:
            raise ValueError("Aucune epoch T1/T2 trouvee dans les annotations.")

        from src.models.epoch_data import EpochData

        mi_epochs = EpochData(
            data=epoch_data.data[mi_mask],
            times=epoch_data.times,
            ch_names=epoch_data.ch_names,
            sfreq=epoch_data.sfreq,
            labels=[epoch_data.labels[i] for i in mi_mask],
            onsets=[epoch_data.onsets[i] for i in mi_mask],
        )

        # Artifact rejection
        bad = EEGArtifactService.detect_by_threshold(mi_epochs, config.reject_threshold_uv)
        n_rejected = len(bad)
        if bad:
            mi_epochs = EEGArtifactService.apply_threshold_rejection(mi_epochs, bad)

        if len(mi_epochs.labels) < 4:
            raise ValueError(
                f"Trop peu d'epochs apres rejet ({len(mi_epochs.labels)}). "
                "Verifiez la qualite du signal."
            )

        # Encode labels: T1 -> 0, T2 -> 1
        class_names = ["T1", "T2"]
        label_map = {"T1": 0, "T2": 1}
        y = np.array([label_map[lbl] for lbl in mi_epochs.labels])

        return mi_epochs.data, y, class_names, n_rejected

    @staticmethod
    def build_pipeline(n_csp_components: int = 6) -> Pipeline:
        """Create sklearn Pipeline: CSP -> LDA."""
        from mne.decoding import CSP

        csp = CSP(
            n_components=n_csp_components,
            reg=None,
            log=True,
            norm_trace=False,
        )
        lda = LinearDiscriminantAnalysis()
        return Pipeline([("csp", csp), ("lda", lda)])

    @staticmethod
    def cross_validate(
        X: np.ndarray,
        y: np.ndarray,
        config: ClassificationConfig,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Stratified k-fold CV. Returns (cv_accuracies, confusion_matrix)."""
        n_folds = min(config.n_folds, min(np.bincount(y)))
        if n_folds < 2:
            n_folds = 2

        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        accuracies = []
        cm_total = np.zeros((2, 2), dtype=int)

        for train_idx, test_idx in skf.split(X, y):
            pipe = EEGClassificationService.build_pipeline(config.n_csp_components)
            pipe.fit(X[train_idx], y[train_idx])
            y_pred = pipe.predict(X[test_idx])
            accuracies.append(accuracy_score(y[test_idx], y_pred))
            cm_total += confusion_matrix(y[test_idx], y_pred, labels=[0, 1])

        return np.array(accuracies), cm_total

    @staticmethod
    def train_final(
        X: np.ndarray,
        y: np.ndarray,
        config: ClassificationConfig,
        subject: int | str,
    ) -> tuple[Pipeline, str]:
        """Train on all data, save pipeline. Returns (pipeline, model_path)."""
        pipe = EEGClassificationService.build_pipeline(config.n_csp_components)
        pipe.fit(X, y)

        os.makedirs(config.model_dir, exist_ok=True)
        subj_str = f"S{subject:03d}" if isinstance(subject, int) else str(subject)
        filename = f"{subj_str}_{config.task}_csp_lda.pkl"
        model_path = os.path.join(config.model_dir, filename)

        with open(model_path, "wb") as f:
            pickle.dump(pipe, f)

        return pipe, model_path

    @staticmethod
    def load_pipeline(path: str) -> Pipeline:
        """Load a saved CSP+LDA pipeline."""
        with open(path, "rb") as f:
            return pickle.load(f)

    @staticmethod
    def predict_epoch(pipeline: Pipeline, epoch: np.ndarray) -> tuple[int, np.ndarray]:
        """Classify a single epoch for online BCI.

        Parameters
        ----------
        epoch : (n_channels, n_times)

        Returns
        -------
        label : int (0 or 1)
        probabilities : (n_classes,)
        """
        X = epoch[np.newaxis, ...]  # (1, n_ch, n_times)
        label = int(pipeline.predict(X)[0])
        proba = pipeline.predict_proba(X)[0]
        return label, proba

    @staticmethod
    def run_subject(
        data_path: str,
        subject: int | str,
        config: ClassificationConfig,
        save_model: bool = True,
    ) -> ClassificationResult:
        """Full pipeline: load -> preprocess -> epoch -> CV -> save."""
        svc = EEGClassificationService

        runs = svc.resolve_runs(config)
        channels = config.channels or None

        # 1. Load and merge runs
        signal = svc.load_and_merge_runs(data_path, subject, runs, channels)

        # 2. Preprocess
        signal = svc.preprocess(signal, config)

        # 3. Extract MI epochs
        X, y, class_names, n_rejected = svc.extract_mi_epochs(signal, config)

        # 4. Cross-validate
        cv_accs, cm = svc.cross_validate(X, y, config)

        # 5. Build result
        unique, counts = np.unique(y, return_counts=True)
        n_epochs = {class_names[u]: int(c) for u, c in zip(unique, counts)}

        result = ClassificationResult(
            subject=subject,
            task=config.task,
            n_epochs=n_epochs,
            n_epochs_rejected=n_rejected,
            cv_accuracies=cv_accs,
            mean_accuracy=float(np.mean(cv_accs)),
            std_accuracy=float(np.std(cv_accs)),
            confusion_matrix=cm,
            class_names=class_names,
        )

        # 6. Train final model and save
        if save_model:
            pipe, model_path = svc.train_final(X, y, config, subject)
            result.model_path = model_path
            result.csp_patterns = pipe.named_steps["csp"].patterns_.copy()

        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_path(data_path: str, subject: int | str, run: int) -> str:
        """Build EDF file path for PhysioNet or custom subject."""
        if isinstance(subject, int):
            return os.path.join(
                data_path,
                "MNE-eegbci-data", "files", "eegmmidb", "1.0.0",
                f"S{subject:03d}",
                f"S{subject:03d}R{run:02d}.edf",
            )
        # Custom subject
        return os.path.join(
            data_path, "custom", str(subject),
            f"{subject}R{run:02d}.edf",
        )

    @staticmethod
    def _pick_channels(signal: SignalData, channels: list[str]) -> SignalData:
        """Select channels by name (case-insensitive)."""
        sig_upper = {ch.upper(): i for i, ch in enumerate(signal.ch_names)}
        indices = []
        matched_names = []
        for ch in channels:
            idx = sig_upper.get(ch.upper())
            if idx is not None:
                indices.append(idx)
                matched_names.append(signal.ch_names[idx])

        if not indices:
            available = ", ".join(signal.ch_names[:10])
            raise ValueError(
                f"Aucun canal correspondant. Demandes: {channels}. "
                f"Disponibles: {available}..."
            )

        if len(indices) != len(channels):
            found = set(ch.upper() for ch in matched_names)
            missing = [ch for ch in channels if ch.upper() not in found]
            print(f"  [WARN] Canaux manquants: {missing}")

        return SignalData(
            data=signal.data[indices],
            times=signal.times,
            ch_names=matched_names,
            sfreq=signal.sfreq,
            annotations=signal.annotations,
        )
