"""Validate CSP+LDA on PhysioNet subjects (within-subject cross-validation).

Usage:
    python -m src.scripts.classify_physionet --subjects 1 --task left_vs_right
    python -m src.scripts.classify_physionet --subjects 1-10 --task left_vs_right
    python -m src.scripts.classify_physionet --subjects 1-5 --task hands_vs_feet --save-model
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

from src.models.classification_data import ClassificationConfig, ClassificationResult
from src.services.eeg_classification_service import EEGClassificationService


def parse_subjects(spec: str) -> list[int]:
    """Parse '1-10' or '1' or '1,3,5' into a list of ints."""
    if "-" in spec and not spec.startswith("-"):
        parts = spec.split("-")
        return list(range(int(parts[0]), int(parts[1]) + 1))
    if "," in spec:
        return [int(x.strip()) for x in spec.split(",")]
    return [int(spec)]


def print_result(result: ClassificationResult) -> None:
    print(f"  Epochs     : {result.n_epochs} (rejetes: {result.n_epochs_rejected})")
    print(f"  CV accuracy: {result.mean_accuracy:.1%} +/- {result.std_accuracy:.1%}")
    print(f"  Par fold   : {[f'{a:.1%}' for a in result.cv_accuracies]}")
    print(f"  Confusion  :")
    print(f"               pred T1  pred T2")
    print(f"    vrai T1    {result.confusion_matrix[0, 0]:5d}    {result.confusion_matrix[0, 1]:5d}")
    print(f"    vrai T2    {result.confusion_matrix[1, 0]:5d}    {result.confusion_matrix[1, 1]:5d}")
    if result.model_path:
        print(f"  Modele     : {result.model_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="MI classification on PhysioNet data")
    parser.add_argument(
        "--subjects", type=str, default="1",
        help="Subject(s): '1', '1-10', or '1,3,5'",
    )
    parser.add_argument(
        "--task", choices=["left_vs_right", "hands_vs_feet"],
        default="left_vs_right",
    )
    parser.add_argument(
        "--channels", type=int, choices=[8, 64], default=8,
        help="8 = MI motor strip subset, 64 = all PhysioNet channels",
    )
    parser.add_argument("--data-path", type=str, default="data/")
    parser.add_argument("--save-model", action="store_true", default=False)
    parser.add_argument(
        "--fbcsp", action="store_true", default=False,
        help="Enable Filter Bank CSP (default: single-band regularized CSP+LDA)",
    )
    args = parser.parse_args()

    channels = EEGClassificationService.MI_8CH if args.channels == 8 else []
    config = ClassificationConfig(
        task=args.task,
        channels=channels,
        use_fbcsp=args.fbcsp,
    )

    subjects = parse_subjects(args.subjects)
    results: list[ClassificationResult] = []

    mode = "FBCSP (7 sub-bands) + SelectKBest + LDA" if config.use_fbcsp else f"CSP({config.n_csp_components}) + LDA"
    print(f"\nPipeline: {mode}  [reg=ledoit_wolf]")
    if not config.use_fbcsp:
        print(f"Bandpass: {config.bandpass_low}-{config.bandpass_high} Hz")
    else:
        print(f"Sub-bands: {config.fbcsp_bands}")
    print(f"Epoch: [{config.tmin}, {config.tmax}]s")
    print(f"Canaux: {len(channels)} ({', '.join(channels) if channels else 'tous'})")
    print(f"Task: {args.task}")

    for subject in subjects:
        print(f"\n{'=' * 60}")
        print(f"Sujet S{subject:03d} -- {args.task}")
        print(f"{'=' * 60}")

        try:
            result = EEGClassificationService.run_subject(
                data_path=args.data_path,
                subject=subject,
                config=config,
                save_model=args.save_model,
            )
            results.append(result)
            print_result(result)
        except Exception as exc:
            print(f"  [ERREUR] {exc}")

    if len(results) > 1:
        accs = np.array([r.mean_accuracy for r in results])
        print(f"\n{'=' * 60}")
        print("RESUME")
        print(f"{'=' * 60}")
        print(f"  Sujets testes   : {len(results)}")
        print(f"  Accuracy moyenne: {np.mean(accs):.1%} +/- {np.std(accs):.1%}")
        best_idx = int(np.argmax(accs))
        worst_idx = int(np.argmin(accs))
        print(f"  Meilleur        : S{results[best_idx].subject:03d} ({accs[best_idx]:.1%})")
        print(f"  Pire            : S{results[worst_idx].subject:03d} ({accs[worst_idx]:.1%})")
        print(f"  Mediane         : {np.median(accs):.1%}")


if __name__ == "__main__":
    main()
