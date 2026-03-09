from __future__ import annotations

import os

import numpy as np


class EdfExportService:
    """Export a SignalData to an EDF+ file using MNE. No Qt imports."""

    @staticmethod
    def export(signal_data, config) -> str:
        """Write signal_data to an EDF+ file and return the output path.

        Parameters
        ----------
        signal_data : SignalData
            The recorded EEG data (data in Volts, annotations as list of
            (onset_s, duration_s, description) tuples).
        config : AcquisitionConfig
            Provides subject_id, run_label, output_dir, and ch_names.

        Returns
        -------
        str
            Absolute path to the created EDF file.
        """
        import mne

        # Build output path
        subject_dir = os.path.join(config.output_dir, config.subject_id)
        os.makedirs(subject_dir, exist_ok=True)
        filename = f"{config.subject_id}{config.run_label}.edf"
        output_path = os.path.join(subject_dir, filename)

        # Create MNE info
        info = mne.create_info(
            ch_names=signal_data.ch_names,
            sfreq=signal_data.sfreq,
            ch_types=["eeg"] * len(signal_data.ch_names),
        )

        # Build RawArray (MNE expects data in Volts)
        raw = mne.io.RawArray(signal_data.data, info, verbose=False)

        # Set standard 10-20 montage (channels must match standard names)
        try:
            montage = mne.channels.make_standard_montage("standard_1020")
            raw.set_montage(montage, on_missing="warn", verbose=False)
        except Exception:
            pass

        # Set annotations
        if signal_data.annotations:
            onsets = np.array([a[0] for a in signal_data.annotations])
            durations = np.array([a[1] for a in signal_data.annotations])
            descriptions = [a[2] for a in signal_data.annotations]
            annotations = mne.Annotations(
                onset=onsets,
                duration=durations,
                description=descriptions,
            )
            raw.set_annotations(annotations)

        # Export to EDF+
        mne.export.export_raw(output_path, raw, fmt="edf", overwrite=True, verbose=False)

        return os.path.abspath(output_path)
