"""Classify user's own 8-channel data recorded via the app's acquisition view.

Usage:
    python -m src.scripts.classify_custom --subject MATTEO2 --runs 3 --task left_vs_right
    python -m src.scripts.classify_custom --subject MATTEO2 --runs 3,4,5 --task left_vs_right
    python -m src.scripts.classify_custom --subject MATTEO2 --runs 3 --transfer models/S001_left_vs_right_csp_lda.pkl
"""
from __future__ import annotations

import argparse

import numpy as np

from src.models.classification_data import ClassificationConfig, ClassificationResult
from src.services.eeg_classification_service import EEGClassificationService


def parse_runs(spec: str) -> list[int]:
    """Parse '3' or '3,4,5' or '3-5' into a list of ints."""
    if "-" in spec:
        parts = spec.split("-")
        return list(range(int(parts[0]), int(parts[1]) + 1))
    return [int(x.strip()) for x in spec.split(",")]


def main() -> None:
    parser = argparse.ArgumentParser(description="MI classification on custom Cyton data")
    parser.add_argument("--subject", type=str, required=True, help="Subject ID (e.g., MATTEO2)")
    parser.add_argument("--runs", type=str, required=True, help="Run(s): '3' or '3,4,5'")
    parser.add_argument(
        "--task", choices=["left_vs_right", "hands_vs_feet"],
        default="left_vs_right",
    )
    parser.add_argument("--data-path", type=str, default="data/")
    parser.add_argument("--save-model", action="store_true", default=False)
    parser.add_argument(
        "--transfer", type=str, default=None,
        help="Path to a pre-trained model (e.g., from PhysioNet) to test without retraining",
    )
    args = parser.parse_args()

    runs = parse_runs(args.runs)

    # Custom data: already 8 channels, no channel picking needed
    # Add notch 50 Hz for European line noise
    config = ClassificationConfig(
        task=args.task,
        runs=runs,
        channels=[],
        notch_hz=50.0,
    )

    print(f"\nSujet: {args.subject}")
    print(f"Runs: {runs}")
    print(f"Task: {args.task}")
    print(f"Bandpass: {config.bandpass_low}-{config.bandpass_high} Hz + notch 50 Hz")

    if args.transfer:
        # Transfer mode: load pre-trained model, just evaluate
        print(f"\n--- Mode transfert: {args.transfer} ---")
        svc = EEGClassificationService

        signal = svc.load_and_merge_runs(args.data_path, args.subject, runs)

        # Resample to 160 Hz if needed (PhysioNet models expect 160 Hz)
        if signal.sfreq != 160.0:
            from scipy.signal import resample

            n_target = int(signal.data.shape[1] * 160.0 / signal.sfreq)
            resampled = resample(signal.data, n_target, axis=1)
            from src.models.signal_data import SignalData

            scale = 160.0 / signal.sfreq
            signal = SignalData(
                data=resampled,
                times=np.arange(n_target) / 160.0,
                ch_names=signal.ch_names,
                sfreq=160.0,
                annotations=[(o * scale, d * scale, desc) for o, d, desc in signal.annotations],
            )

        signal = svc.preprocess(signal, config)
        X, y, class_names, n_rejected = svc.extract_mi_epochs(signal, config)

        pipeline = svc.load_pipeline(args.transfer)
        y_pred = pipeline.predict(X)

        from sklearn.metrics import accuracy_score, confusion_matrix

        acc = accuracy_score(y, y_pred)
        cm = confusion_matrix(y, y_pred, labels=[0, 1])

        print(f"  Epochs: T1={np.sum(y == 0)}, T2={np.sum(y == 1)} (rejetes: {n_rejected})")
        print(f"  Accuracy (transfert): {acc:.1%}")
        print(f"  Confusion:")
        print(f"               pred T1  pred T2")
        print(f"    vrai T1    {cm[0, 0]:5d}    {cm[0, 1]:5d}")
        print(f"    vrai T2    {cm[1, 0]:5d}    {cm[1, 1]:5d}")
    else:
        # Standard mode: cross-validate on custom data
        print(f"\n--- Mode cross-validation ---")

        try:
            result = EEGClassificationService.run_subject(
                data_path=args.data_path,
                subject=args.subject,
                config=config,
                save_model=args.save_model,
            )
            print(f"  Epochs     : {result.n_epochs} (rejetes: {result.n_epochs_rejected})")
            print(f"  CV accuracy: {result.mean_accuracy:.1%} +/- {result.std_accuracy:.1%}")
            print(f"  Par fold   : {[f'{a:.1%}' for a in result.cv_accuracies]}")
            if result.model_path:
                print(f"  Modele     : {result.model_path}")
        except Exception as exc:
            print(f"  [ERREUR] {exc}")


if __name__ == "__main__":
    main()
