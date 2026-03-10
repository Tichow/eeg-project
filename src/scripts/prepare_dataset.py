"""Download PhysioNet data and build the classification dataset.

Usage:
    python -m src.scripts.prepare_dataset --n-subjects 5
    python -m src.scripts.prepare_dataset --n-subjects 50 --runs hands
    python -m src.scripts.prepare_dataset --n-subjects 50 --runs all
"""

import argparse
import os
import numpy as np

from src.services.eeg_dataset_service import (
    EEGDatasetService,
    MI_RUNS_HANDS,
    MI_RUNS_ALL,
)


def main():
    parser = argparse.ArgumentParser(description="Prepare PhysioNet MI dataset")
    parser.add_argument("--n-subjects", type=int, default=5, help="Number of subjects")
    parser.add_argument(
        "--runs",
        choices=["hands", "all"],
        default="all",
        help="'hands' for R04/R08/R12 only, 'all' for hands+feet",
    )
    parser.add_argument("--data-path", default="data/", help="Download directory")
    parser.add_argument("--output-dir", default="models/", help="Output directory for .npz")
    parser.add_argument("--no-alignment", action="store_true", help="Skip Euclidean Alignment")
    parser.add_argument("--no-normalize", action="store_true", help="Skip Z-score normalization")
    args = parser.parse_args()

    runs = MI_RUNS_HANDS if args.runs == "hands" else MI_RUNS_ALL
    subjects = EEGDatasetService.get_valid_subjects(args.n_subjects)

    print(f"Building dataset: {len(subjects)} subjects, runs={runs}")
    print(f"Downloading to {args.data_path}...\n")

    X, y, subject_ids = EEGDatasetService.build_dataset(
        subjects=subjects,
        runs=runs,
        data_path=args.data_path,
        apply_alignment=not args.no_alignment,
        normalize=not args.no_normalize,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "physionet_dataset.npz")
    np.savez_compressed(out_path, X=X, y=y, subject_ids=subject_ids)
    print(f"\nSaved to {out_path}")
    print(f"  X: {X.shape} ({X.dtype})")
    print(f"  y: {y.shape} — classes: {np.unique(y, return_counts=True)}")
    print(f"  subjects: {len(np.unique(subject_ids))}")


if __name__ == "__main__":
    main()
