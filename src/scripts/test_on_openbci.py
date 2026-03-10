"""Test a pre-trained EEGNet model on OpenBCI recordings.

Maps 8 OpenBCI channels to the 64-channel PhysioNet layout
(zero-padding for absent channels).

Usage:
    python -m src.scripts.test_on_openbci --model models/eegnet_physionet.pt --recording path/to/recording.npy
"""

import argparse
import json
import numpy as np
import torch
from scipy.signal import resample

from src.models.eegnet import EEGNet
from src.services.eeg_alignment_service import EEGAlignmentService

# Mapping: OpenBCI channel name → index in PhysioNet 64-channel layout
# These indices come from the EEGMMIDB electrode order (see protocole-physionet-eegmmidb.md)
OPENBCI_TO_PHYSIONET = {
    "C3": 8,
    "Cz": 10,
    "C4": 12,
    "FC3": 1,
    "FC4": 5,
    "CP3": 15,
    "CP4": 19,
    "FCz": 3,
}

# PhysioNet 64-channel names in EDF order
PHYSIONET_CH_NAMES = [
    "Fc5", "Fc3", "Fc1", "Fcz", "Fc2", "Fc4", "Fc6",
    "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
    "Cp5", "Cp3", "Cp1", "Cpz", "Cp2", "Cp4", "Cp6",
    "Fp1", "Fpz", "Fp2",
    "Af7", "Af3", "Afz", "Af4", "Af8",
    "F7", "F5", "F3", "F1", "Fz", "F2", "F4", "F6", "F8",
    "Ft7", "Ft8",
    "T7", "T8", "T9", "T10",
    "Tp7", "Tp8",
    "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8",
    "Po7", "Po3", "Poz", "Po4", "Po8",
    "O1", "Oz", "O2", "Iz",
]


def load_openbci_recording(npy_path: str, json_path: str):
    """Load an OpenBCI recording (.npy + .json metadata)."""
    data = np.load(npy_path)  # (n_channels, n_samples)
    with open(json_path, "r") as f:
        meta = json.load(f)
    return data, meta


def map_to_64_channels(
    X_8ch: np.ndarray,
    openbci_ch_names: list[str],
    channel_map: dict[str, int] | None = None,
) -> np.ndarray:
    """Map 8-channel OpenBCI data to 64-channel PhysioNet layout.

    Parameters
    ----------
    X_8ch : (n_trials, 8, n_times) or (8, n_times)
    openbci_ch_names : list of OpenBCI channel names
    channel_map : mapping from channel name to PhysioNet index

    Returns
    -------
    X_64 : same shape but with 64 channels (zeros for absent)
    """
    if channel_map is None:
        channel_map = OPENBCI_TO_PHYSIONET

    single = X_8ch.ndim == 2
    if single:
        X_8ch = X_8ch[np.newaxis]

    n_trials, _, n_times = X_8ch.shape
    X_64 = np.zeros((n_trials, 64, n_times), dtype=X_8ch.dtype)

    for ch_name, physionet_idx in channel_map.items():
        ch_name_upper = ch_name.upper()
        for i, name in enumerate(openbci_ch_names):
            if name.upper() == ch_name_upper:
                X_64[:, physionet_idx, :] = X_8ch[:, i, :]
                break

    return X_64[0] if single else X_64


def resample_to_160hz(data: np.ndarray, sfreq_original: float) -> np.ndarray:
    """Resample data from original sfreq to 160 Hz.

    Parameters
    ----------
    data : (..., n_times) — last axis is time
    sfreq_original : original sampling frequency

    Returns
    -------
    resampled data at 160 Hz
    """
    if abs(sfreq_original - 160.0) < 0.1:
        return data
    ratio = 160.0 / sfreq_original
    n_new = int(round(data.shape[-1] * ratio))
    return resample(data, n_new, axis=-1).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Test EEGNet on OpenBCI data")
    parser.add_argument("--model", required=True, help="Path to .pt model file")
    parser.add_argument("--recording", required=True, help="Path to .npy recording")
    parser.add_argument("--metadata", default=None, help="Path to .json metadata (auto-detected)")
    args = parser.parse_args()

    if args.metadata is None:
        args.metadata = args.recording.replace(".npy", ".json")

    # Load model
    checkpoint = torch.load(args.model, map_location="cpu", weights_only=False)
    model = EEGNet(
        n_channels=checkpoint["n_channels"],
        n_times=checkpoint["n_times"],
        n_classes=checkpoint["n_classes"],
        channel_dropout=0.0,  # no dropout at test time
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Load recording
    data, meta = load_openbci_recording(args.recording, args.metadata)
    print(f"Recording: {meta.get('subject', '?')}, {meta.get('test_type', '?')}")
    print(f"  Channels: {meta.get('channels', [])}")
    print(f"  Sfreq: {meta.get('sfreq', 250)} Hz")
    print(f"  Duration: {meta.get('duration_sec', 0):.1f}s")

    # Resample to 160 Hz
    sfreq = meta.get("sfreq", 250)
    data = resample_to_160hz(data, sfreq)

    # Map to 64 channels
    ch_names = meta.get("channels", [f"CH{i+1}" for i in range(data.shape[0])])
    data_64 = map_to_64_channels(data, ch_names)

    # Segment into fixed windows (3.5s = 560 samples at 160 Hz)
    n_times = checkpoint["n_times"]
    n_windows = data_64.shape[1] // n_times
    if n_windows == 0:
        print("Recording too short for even one window")
        return

    windows = np.stack(
        [data_64[:, i * n_times : (i + 1) * n_times] for i in range(n_windows)]
    )  # (n_windows, 64, 560)

    # Normalize
    for i in range(windows.shape[0]):
        for ch in range(windows.shape[1]):
            std = windows[i, ch].std()
            if std > 0:
                windows[i, ch] = (windows[i, ch] - windows[i, ch].mean()) / std

    # Euclidean alignment
    if windows.shape[0] > 1:
        windows, _ = EEGAlignmentService.euclidean_alignment(windows)

    # Inference
    X_tensor = torch.FloatTensor(windows[:, np.newaxis, :, :])
    with torch.no_grad():
        logits = model(X_tensor)
        probs = torch.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)

    print(f"\nPredictions ({n_windows} windows):")
    for i, (pred, prob) in enumerate(zip(preds, probs)):
        conf = prob[pred].item()
        print(f"  Window {i+1}: class {pred.item()} (confidence {conf:.2f})")

    # Summary
    unique, counts = np.unique(preds.numpy(), return_counts=True)
    print(f"\nSummary: {dict(zip(unique.tolist(), counts.tolist()))}")


if __name__ == "__main__":
    main()
