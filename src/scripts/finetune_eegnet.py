"""Fine-tune a pre-trained EEGNet on subject-specific OpenBCI data.

Freezes feature extraction layers (block1 + block2), retrains only
the separable convolution and classifier on your data.

Usage:
    python -m src.scripts.finetune_eegnet \
        --model models/eegnet_physionet.pt \
        --dataset models/openbci_dataset.npz \
        --output models/eegnet_finetuned.pt
"""

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, cohen_kappa_score

from src.models.eegnet import EEGNet


def main():
    parser = argparse.ArgumentParser(description="Fine-tune EEGNet on OpenBCI data")
    parser.add_argument("--model", required=True, help="Path to pre-trained .pt")
    parser.add_argument("--dataset", required=True, help="Path to .npz (X, y)")
    parser.add_argument("--output", default="models/eegnet_finetuned.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--n-folds", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    print(f"Device: {device}")

    checkpoint = torch.load(args.model, map_location="cpu", weights_only=False)

    data = np.load(args.dataset)
    X, y = data["X"], data["y"]
    n_classes = len(np.unique(y))
    print(f"Dataset: {X.shape[0]} trials, {n_classes} classes")
    print(f"  Class distribution: {np.unique(y, return_counts=True)}")

    X = X[:, np.newaxis, :, :]  # (N, 1, n_ch, n_times)

    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=42)
    fold_results = []
    best_global_acc = 0
    best_model_state = None

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\nFold {fold + 1}/{args.n_folds}")

        model = EEGNet(
            n_channels=checkpoint["n_channels"],
            n_times=checkpoint["n_times"],
            n_classes=n_classes,
            channel_dropout=0.0,
        ).to(device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        model.freeze_feature_extractor()

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"  Trainable: {trainable}/{total} parameters")

        train_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X[train_idx]), torch.LongTensor(y[train_idx])),
            batch_size=args.batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X[val_idx]), torch.LongTensor(y[val_idx])),
            batch_size=args.batch_size,
        )

        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            weight_decay=1e-4,
        )
        criterion = nn.CrossEntropyLoss()

        best_val_loss = float("inf")
        patience_counter = 0
        best_fold_state = None

        for epoch in range(args.epochs):
            model.train()
            for X_b, y_b in train_loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                optimizer.zero_grad()
                loss = criterion(model(X_b), y_b)
                loss.backward()
                optimizer.step()

            model.eval()
            val_preds, val_labels = [], []
            val_loss = 0
            n_b = 0
            with torch.no_grad():
                for X_b, y_b in val_loader:
                    X_b, y_b = X_b.to(device), y_b.to(device)
                    logits = model(X_b)
                    val_loss += criterion(logits, y_b).item()
                    n_b += 1
                    val_preds.append(logits.argmax(1).cpu().numpy())
                    val_labels.append(y_b.cpu().numpy())

            val_loss /= max(n_b, 1)
            val_acc = accuracy_score(np.concatenate(val_labels), np.concatenate(val_preds))

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_fold_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1

            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1:3d} | val_loss={val_loss:.4f} | val_acc={val_acc:.3f}")

            if patience_counter >= args.patience:
                print(f"  Early stopping at epoch {epoch + 1}")
                break

        model.load_state_dict(best_fold_state)
        model.eval()
        all_p, all_l = [], []
        with torch.no_grad():
            for X_b, y_b in val_loader:
                all_p.append(model(X_b.to(device)).argmax(1).cpu().numpy())
                all_l.append(y_b.numpy())
        preds = np.concatenate(all_p)
        labels = np.concatenate(all_l)
        acc = accuracy_score(labels, preds)
        kappa = cohen_kappa_score(labels, preds)

        print(f"  Fold {fold+1}: acc={acc:.3f}, kappa={kappa:.3f}")
        print(f"  {confusion_matrix(labels, preds)}")
        fold_results.append({"acc": acc, "kappa": kappa})

        if acc > best_global_acc:
            best_global_acc = acc
            best_model_state = best_fold_state

    accs = [r["acc"] for r in fold_results]
    kappas = [r["kappa"] for r in fold_results]
    print(f"\nFine-tuning results ({args.n_folds} folds):")
    print(f"  Accuracy: {np.mean(accs):.3f} ± {np.std(accs):.3f}")
    print(f"  Kappa:    {np.mean(kappas):.3f} ± {np.std(kappas):.3f}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    save_data = dict(checkpoint)
    save_data["model_state_dict"] = best_model_state
    save_data["finetune_accuracy"] = np.mean(accs)
    save_data["finetune_kappa"] = np.mean(kappas)
    torch.save(save_data, args.output)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
