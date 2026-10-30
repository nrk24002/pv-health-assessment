"""
CNN transfer-learning model for the Photovoltaic Health Assessment project
(Milestone 2.1).

Fine-tunes a ResNet18 backbone pretrained on ImageNet on the same
manifest/split used by the classical baseline (src/baseline_model.py), so
the two models are directly comparable. This script needs unrestricted
internet access the first time it runs, to download the pretrained
ImageNet weights via torchvision; after that they are cached locally
(~/.cache/torch) and no further downloads are needed.

Run:
    python3 src/train_cnn.py

Outputs:
    reports/cnn_metrics.json
    reports/cnn_classification_report.txt
    reports/figures/cnn_confusion_matrix.png
    reports/figures/baseline_vs_cnn_comparison.png
"""

import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, f1_score)
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights, resnet18

from dataset import MANIFEST_PATH, PROJECT_ROOT, PVDataset

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
SEED = 42
BATCH_SIZE = 16
MAX_EPOCHS = 12
PATIENCE = 3  # early stopping on validation loss (validation accuracy saturates at 1.000)
LR = 1e-4
WEIGHT_DECAY = 1e-4


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_model(num_classes: int):
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    in_feats = model.fc.in_features
    model.fc = nn.Linear(in_feats, num_classes)
    return model


def class_weights_from_dataset(ds: PVDataset, num_classes: int) -> torch.Tensor:
    counts = np.zeros(num_classes, dtype=np.float64)
    for row in ds.rows:
        counts[ds.class_to_idx[row["class"]]] += 1
    counts = np.maximum(counts, 1)
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []
    torch.set_grad_enabled(train)
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        if train:
            optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        if train:
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * images.size(0)
        all_preds.extend(outputs.argmax(dim=1).detach().cpu().numpy().tolist())
        all_labels.extend(labels.detach().cpu().numpy().tolist())
    torch.set_grad_enabled(True)
    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    return avg_loss, acc


def main():
    torch.manual_seed(SEED)
    device = get_device()
    print(f"Using device: {device}")

    train_ds = PVDataset(MANIFEST_PATH, split="train")
    val_ds = PVDataset(MANIFEST_PATH, split="val")
    test_ds = PVDataset(MANIFEST_PATH, split="test")
    classes = train_ds.classes
    num_classes = len(classes)
    print(f"Train/val/test sizes: {len(train_ds)}/{len(val_ds)}/{len(test_ds)}, classes: {classes}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = build_model(num_classes).to(device)
    weights = class_weights_from_dataset(train_ds, num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_val_acc = -1.0
    best_val_loss = 1e9
    best_state = None
    epochs_without_improve = 0
    history = []

    t0 = time.time()
    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        history.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
                         "val_loss": val_loss, "val_acc": val_acc})
        print(f"Epoch {epoch:2d}: train_loss={train_loss:.4f} train_acc={train_acc:.3f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}")

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= PATIENCE:
                print(f"Early stopping at epoch {epoch} (no val-loss improvement for {PATIENCE} epochs)")
                break

    train_time = time.time() - t0
    model.load_state_dict(best_state)

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            all_preds.extend(outputs.argmax(dim=1).cpu().numpy().tolist())
            all_labels.extend(labels.numpy().tolist())

    test_acc = accuracy_score(all_labels, all_preds)
    test_f1_macro = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    report_txt = classification_report(all_labels, all_preds, target_names=classes, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))

    print("\n=== TEST SET RESULTS (ResNet18, ImageNet-pretrained, fine-tuned) ===")
    print(f"Test accuracy: {test_acc:.3f}")
    print(f"Test macro-F1: {test_f1_macro:.3f}")
    print(report_txt)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    metrics = {
        "model": "resnet18_imagenet_pretrained_finetuned",
        "seed": SEED,
        "device": str(device),
        "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS,
        "epochs_run": len(history),
        "patience": PATIENCE,
        "learning_rate": LR,
        "weight_decay": WEIGHT_DECAY,
        "train_time_seconds": round(train_time, 1),
        "train_size": len(train_ds),
        "val_size": len(val_ds),
        "test_size": len(test_ds),
        "classes": classes,
        "best_val_accuracy": round(best_val_acc, 4),
        "test_accuracy": round(test_acc, 4),
        "test_macro_f1": round(test_f1_macro, 4),
        "history": history,
    }
    with open(REPORTS_DIR / "cnn_metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=2)

    with open(REPORTS_DIR / "cnn_classification_report.txt", "w") as fh:
        fh.write("CNN model: ResNet18 (ImageNet-pretrained, fine-tuned)\n\n")
        fh.write(f"Best validation accuracy: {best_val_acc:.3f}\n")
        fh.write(f"Test accuracy: {test_acc:.3f}   Test macro-F1: {test_f1_macro:.3f}\n\n")
        fh.write(report_txt)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(num_classes)); ax.set_yticks(range(num_classes))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted label"); ax.set_ylabel("True label")
    ax.set_title(f"CNN (ResNet18) confusion matrix (test set, acc={test_acc:.2f})")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "cnn_confusion_matrix.png", dpi=150)
    plt.close(fig)

    # Comparison figure against the classical baseline, if its metrics exist.
    baseline_path = REPORTS_DIR / "baseline_metrics.json"
    if baseline_path.exists():
        with open(baseline_path) as fh:
            baseline = json.load(fh)
        labels_cmp = ["Baseline\n(RF, color/texture)", "CNN\n(ResNet18 fine-tuned)"]
        accs = [baseline["test_accuracy"], test_acc]
        f1s = [baseline["test_macro_f1"], test_f1_macro]
        x = np.arange(2)
        width = 0.35
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.bar(x - width / 2, accs, width, label="Test accuracy")
        ax.bar(x + width / 2, f1s, width, label="Test macro-F1")
        ax.set_xticks(x); ax.set_xticklabels(labels_cmp)
        ax.set_ylim(0, 1.05)
        ax.set_title("Baseline vs. CNN performance")
        ax.legend()
        for i, (a, f1v) in enumerate(zip(accs, f1s)):
            ax.text(i - width / 2, a + 0.01, f"{a:.3f}", ha="center", fontsize=8)
            ax.text(i + width / 2, f1v + 0.01, f"{f1v:.3f}", ha="center", fontsize=8)
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "baseline_vs_cnn_comparison.png", dpi=150)
        plt.close(fig)
        print(f"\nSaved: {FIGURES_DIR / 'baseline_vs_cnn_comparison.png'}")
    else:
        print("\nNote: reports/baseline_metrics.json not found, skipped comparison figure.")

    print(f"Saved: {REPORTS_DIR / 'cnn_metrics.json'}")
    print(f"Saved: {REPORTS_DIR / 'cnn_classification_report.txt'}")
    print(f"Saved: {FIGURES_DIR / 'cnn_confusion_matrix.png'}")


if __name__ == "__main__":
    main()
