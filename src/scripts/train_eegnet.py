"""Train EEGNet on PhysioNet dataset with cross-subject validation.

Usage:
    python -m src.scripts.train_eegnet
    python -m src.scripts.train_eegnet --dataset models/physionet_dataset.npz --epochs 500
    python -m src.scripts.train_eegnet --n-folds 5 --batch-size 64
"""

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, confusion_matrix, cohen_kappa_score

from src.models.eegnet import EEGNet


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    n_batches = 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


def evaluate(model, loader, criterion, device):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0
    n_batches = 0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            total_loss += loss.item()
            n_batches += 1
            all_preds.append(logits.argmax(dim=1).cpu().numpy())
            all_labels.append(y_batch.cpu().numpy())
    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    acc = accuracy_score(labels, preds)
    loss_avg = total_loss / max(n_batches, 1)
    return loss_avg, acc, preds, labels


def main():
    parser = argparse.ArgumentParser(description="Train EEGNet on PhysioNet MI data")
    parser.add_argument("--dataset", default="models/physionet_dataset.npz")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--channel-dropout", type=float, default=0.0)
    parser.add_argument("--dropout-rate", type=float, default=0.25)
    parser.add_argument("--output", default="models/eegnet_physionet.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    print(f"Device: {device}")

    data = np.load(args.dataset)
    X, y, subject_ids = data["X"], data["y"], data["subject_ids"]
    n_classes = len(np.unique(y))
    n_channels, n_times = X.shape[1], X.shape[2]
    print(f"Dataset: {X.shape[0]} trials, {n_channels}ch, {n_times}t, {n_classes} classes")

    # Add channel dimension for Conv2d: (N, C, H, W) = (N, 1, n_channels, n_times)
    X = X[:, np.newaxis, :, :]

    gkf = GroupKFold(n_splits=args.n_folds)
    fold_results = []

    best_global_acc = 0
    best_model_state = None

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=subject_ids)):
        print(f"\n{'='*60}")
        print(f"Fold {fold + 1}/{args.n_folds}")
        train_subjects = np.unique(subject_ids[train_idx])
        val_subjects = np.unique(subject_ids[val_idx])
        print(f"  Train: {len(train_idx)} trials ({len(train_subjects)} subjects)")
        print(f"  Val:   {len(val_idx)} trials ({len(val_subjects)} subjects)")

        train_ds = TensorDataset(
            torch.FloatTensor(X[train_idx]),
            torch.LongTensor(y[train_idx]),
        )
        val_ds = TensorDataset(
            torch.FloatTensor(X[val_idx]),
            torch.LongTensor(y[val_idx]),
        )
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size)

        model = EEGNet(
            n_channels=n_channels,
            n_times=n_times,
            n_classes=n_classes,
            dropout_rate=args.dropout_rate,
            channel_dropout=args.channel_dropout,
        ).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=1e-5
        )
        criterion = nn.CrossEntropyLoss()

        best_val_loss = float("inf")
        patience_counter = 0
        best_fold_state = None

        for epoch in range(args.epochs):
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
            val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
            scheduler.step()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_fold_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1

            if (epoch + 1) % 25 == 0 or patience_counter == 0:
                lr = optimizer.param_groups[0]["lr"]
                print(
                    f"  Epoch {epoch+1:3d} | "
                    f"train_loss={train_loss:.4f} | "
                    f"val_loss={val_loss:.4f} | "
                    f"val_acc={val_acc:.3f} | "
                    f"lr={lr:.1e}"
                )

            if patience_counter >= args.patience:
                print(f"  Early stopping at epoch {epoch + 1}")
                break

        # Evaluate best model for this fold
        model.load_state_dict(best_fold_state)
        _, val_acc, preds, labels = evaluate(model, val_loader, criterion, device)
        kappa = cohen_kappa_score(labels, preds)
        cm = confusion_matrix(labels, preds)

        print(f"\n  Fold {fold + 1} results:")
        print(f"    Accuracy: {val_acc:.3f}")
        print(f"    Kappa:    {kappa:.3f}")
        print(f"    Confusion matrix:\n{cm}")

        fold_results.append({"acc": val_acc, "kappa": kappa})

        if val_acc > best_global_acc:
            best_global_acc = val_acc
            best_model_state = best_fold_state

    # Summary
    accs = [r["acc"] for r in fold_results]
    kappas = [r["kappa"] for r in fold_results]
    print(f"\n{'='*60}")
    print(f"Cross-validation results ({args.n_folds} folds):")
    print(f"  Accuracy: {np.mean(accs):.3f} ± {np.std(accs):.3f}")
    print(f"  Kappa:    {np.mean(kappas):.3f} ± {np.std(kappas):.3f}")

    # Save best model
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    torch.save(
        {
            "model_state_dict": best_model_state,
            "n_channels": n_channels,
            "n_times": n_times,
            "n_classes": n_classes,
            "sfreq": 160.0,
            "dropout_rate": args.dropout_rate,
            "channel_dropout": args.channel_dropout,
            "cv_accuracy": np.mean(accs),
            "cv_kappa": np.mean(kappas),
            "preprocessing": {
                "bandpass": (4.0, 40.0),
                "notch": 60.0,
                "reref": "average",
                "epoch_tmin": 0.5,
                "epoch_tmax": 4.0,
            },
        },
        args.output,
    )
    print(f"\nBest model saved to {args.output}")


if __name__ == "__main__":
    main()
