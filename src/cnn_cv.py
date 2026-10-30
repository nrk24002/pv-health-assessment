"""
Panel-grouped 5-fold cross-validation of CNN models (Milestones 2.1, 2.2 and 3.2).

The folds are the same as for the Random Forest baseline (GroupKFold over the
150 panel groups, rows in manifest order), so no image of a panel is ever in
the training and the test part of the same fold. Inside each fold, about 15% of
the training panels (per class) are held out as a validation set, and the
checkpoint with the lowest validation LOSS is kept (validation accuracy
saturates at 1.000 on this data, so it cannot separate checkpoints).

Besides accuracy and macro-F1 per fold, the script reports panel-level accuracy
(majority vote over the images of a panel), the panels with errors, and, for the
edge-deployment comparison, the number of parameters, the model size and the
CPU inference time per image.

Run (the first run downloads the ImageNet weights, ~10 s per model):
    python3 src/cnn_cv.py --model resnet18
    python3 src/cnn_cv.py --model mobilenet_v3_small
    python3 src/cnn_cv.py --model efficientnet_b0

Outputs (per model):
    reports/cnn_cv_<model>.json
    reports/cnn_cv_<model>_oof_predictions.csv
    reports/figures/cnn_cv_<model>_confusion_matrix.png
    models/<model>_fold<k>.pt        (not tracked in git)
A finished fold is cached in reports/cnn_cv_cache/, so an interrupted run can be
restarted and continues with the next fold.
"""
import argparse
import csv
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader, Dataset
from torchvision import models

from dataset import MANIFEST_PATH, PROJECT_ROOT, get_transforms

REPORTS = PROJECT_ROOT / "reports"
FIGURES = REPORTS / "figures"
MODELS = PROJECT_ROOT / "models"
SEED = 42
LR = 1e-4
WEIGHT_DECAY = 1e-4
BATCH = 16


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_model(name: str, num_classes: int, pretrained: bool = True):
    if name == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif name == "mobilenet_v3_small":
        m = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
    elif name == "efficientnet_b0":
        m = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
    else:
        raise ValueError(name)
    return m


class Images(Dataset):
    def __init__(self, images, labels, indices, train):
        self.images, self.labels, self.idx = images, labels, list(indices)
        self.tf = get_transforms(train=train)

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = self.idx[i]
        return self.tf(self.images[j]), int(self.labels[j])


def run_epoch(model, loader, criterion, optimizer, device, train):
    model.train() if train else model.eval()
    torch.set_grad_enabled(train)
    total, preds, labs = 0.0, [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if train:
            optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        if train:
            loss.backward()
            optimizer.step()
        total += loss.item() * x.size(0)
        preds += out.argmax(1).detach().cpu().tolist()
        labs += y.detach().cpu().tolist()
    torch.set_grad_enabled(True)
    return total / len(loader.dataset), accuracy_score(labs, preds)


def predict(model, loader, device):
    model.eval()
    preds = []
    with torch.no_grad():
        for x, _ in loader:
            preds += model(x.to(device)).argmax(1).cpu().tolist()
    return preds


def split_validation(train_idx, groups, labels, seed):
    """Hold out ~15% of the training panels of every class as validation panels."""
    rng = random.Random(seed)
    panels = defaultdict(set)
    for i in train_idx:
        panels[labels[i]].add(groups[i])
    val_panels = set()
    for c, ps in panels.items():
        ps = sorted(ps)
        rng.shuffle(ps)
        val_panels.update(ps[:max(1, round(0.15 * len(ps)))])
    tr = [i for i in train_idx if groups[i] not in val_panels]
    va = [i for i in train_idx if groups[i] in val_panels]
    return tr, va


def benchmark_cpu(model, runs=100, warmup=20):
    model = model.to("cpu").eval()
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        t0 = time.perf_counter()
        for _ in range(runs):
            model(x)
    return (time.perf_counter() - t0) / runs * 1000.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="resnet18",
                    choices=["resnet18", "mobilenet_v3_small", "efficientnet_b0"])
    ap.add_argument("--max-epochs", type=int, default=12)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--no-pretrained", action="store_true", help="only for testing the code")
    ap.add_argument("--group-file", default=None,
                    help="CSV (filepath, group_merged) that replaces the panel groups of the manifest; "
                         "used for the grouping sensitivity analysis")
    ap.add_argument("--tag", default="", help="suffix for all output files (use with --group-file)")
    a = ap.parse_args()
    name = a.model + (f"_{a.tag}" if a.tag else "")

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = get_device()
    print(f"model={a.model} device={device}", flush=True)

    with open(MANIFEST_PATH, newline="") as fh:
        rows = list(csv.DictReader(fh))
    classes = sorted({r["class"] for r in rows})
    cidx = {c: i for i, c in enumerate(classes)}
    labels = np.array([cidx[r["class"]] for r in rows])
    groups = np.array([int(r["group"]) for r in rows])
    if a.group_file:
        with open(a.group_file, newline="") as fh:
            alt = {r["filepath"]: int(r["group_merged"]) for r in csv.DictReader(fh)}
        groups = np.array([alt[r["filepath"]] for r in rows])
        print(f"groups from {a.group_file}: {len(set(groups.tolist()))} panel groups", flush=True)
    print("loading images ...", flush=True)
    images = [Image.open(PROJECT_ROOT / r["cache"]).convert("RGB") for r in rows]

    cache_dir = REPORTS / "cnn_cv_cache" / name
    cache_dir.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    oof = np.full(len(rows), -1)
    fold_of = np.zeros(len(rows), dtype=int)
    folds = []
    splits = list(GroupKFold(n_splits=a.folds).split(np.zeros(len(rows)), labels, groups=groups))
    for k, (tr_all, te) in enumerate(splits, start=1):
        fold_of[te] = k
        cf = cache_dir / f"fold{k}.json"
        if cf.exists():
            d = json.load(open(cf))
            oof[te] = d["preds"]
            folds.append(d["metrics"])
            print(f"fold {k}: loaded from cache (acc {d['metrics']['accuracy']:.4f})", flush=True)
            continue

        tr, va = split_validation(tr_all, groups, labels, SEED + k)
        tl = DataLoader(Images(images, labels, tr, True), batch_size=BATCH, shuffle=True)
        vl = DataLoader(Images(images, labels, va, False), batch_size=BATCH)
        el = DataLoader(Images(images, labels, te, False), batch_size=BATCH)

        model = build_model(a.model, len(classes), not a.no_pretrained).to(device)
        counts = np.maximum(np.bincount(labels[tr], minlength=len(classes)), 1)
        w = torch.tensor(counts.sum() / (len(classes) * counts), dtype=torch.float32).to(device)
        crit = nn.CrossEntropyLoss(weight=w)
        opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

        best_loss, best_state, best_ep, bad, hist = 1e9, None, 0, 0, []
        t0 = time.time()
        for ep in range(1, a.max_epochs + 1):
            trl, tra = run_epoch(model, tl, crit, opt, device, True)
            val, vaa = run_epoch(model, vl, crit, opt, device, False)
            hist.append({"epoch": ep, "train_loss": trl, "train_acc": tra, "val_loss": val, "val_acc": vaa})
            print(f"fold {k} epoch {ep:2d}: train_loss {trl:.4f} acc {tra:.3f} | val_loss {val:.4f} acc {vaa:.3f}",
                  flush=True)
            if val < best_loss - 1e-4:
                best_loss, best_ep, bad = val, ep, 0
                best_state = {n: v.detach().cpu().clone() for n, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= a.patience:
                    break
        train_time = time.time() - t0
        model.load_state_dict(best_state)
        torch.save(best_state, MODELS / f"{name}_fold{k}.pt")
        preds = predict(model, el, device)
        oof[te] = preds
        yt = labels[te]
        m = {"fold": k, "n_train_images": len(tr), "n_val_images": len(va), "n_test_images": len(te),
             "n_test_panels": int(len(set(groups[te]))),
             "accuracy": float(accuracy_score(yt, preds)),
             "macro_f1": float(f1_score(yt, preds, average="macro", zero_division=0)),
             "epochs_run": len(hist), "best_epoch": best_ep, "best_val_loss": float(best_loss),
             "train_time_seconds": round(train_time, 1), "history": hist}
        folds.append(m)
        json.dump({"preds": [int(p) for p in preds], "metrics": m}, open(cf, "w"))
        print(f"fold {k}: test acc {m['accuracy']:.4f} macro-F1 {m['macro_f1']:.4f} "
              f"(best epoch {best_ep}, {train_time:.0f} s)", flush=True)
        del model, opt
        if device.type == "mps":
            torch.mps.empty_cache()

    # ---- aggregate ----
    acc = accuracy_score(labels, oof)
    f1 = f1_score(labels, oof, average="macro", zero_division=0)
    fa = [f["accuracy"] for f in folds]
    ff = [f["macro_f1"] for f in folds]
    cm = confusion_matrix(labels, oof, labels=list(range(len(classes))))
    recall = {c: float(cm[i, i] / cm[i].sum()) for i, c in enumerate(classes)}

    panel_stats, panel_ok, wrong_panels = {}, 0, {}
    for g in sorted(set(groups)):
        ix = np.where(groups == g)[0]
        true = int(labels[ix[0]])
        vote = Counter(oof[ix].tolist()).most_common(1)[0][0]
        n_wrong = int((oof[ix] != true).sum())
        panel_ok += int(vote == true)
        if n_wrong:
            wrong_panels[int(g)] = {"class": classes[true], "n_images": int(len(ix)), "n_wrong": n_wrong,
                                    "majority_vote_correct": bool(vote == true),
                                    "predicted_as": {classes[p]: int(c) for p, c in Counter(oof[ix].tolist()).items()
                                                     if p != true}}
    n_panels = len(set(groups))

    # model size and CPU inference time (edge comparison)
    ref = build_model(a.model, len(classes), False)
    n_params = sum(p.numel() for p in ref.parameters())
    tmp = MODELS / f"_size_{a.model}.pt"
    torch.save(ref.state_dict(), tmp)
    size_mb = tmp.stat().st_size / 1e6
    tmp.unlink()
    cpu_ms = benchmark_cpu(ref)

    result = {
        "model": a.model, "seed": SEED, "device": str(device), "pretrained": not a.no_pretrained,
        "lr": LR, "weight_decay": WEIGHT_DECAY, "batch_size": BATCH, "max_epochs": a.max_epochs,
        "patience": a.patience, "checkpoint_selection": "lowest validation loss",
        "classes": classes, "n_folds": a.folds,
        "oof_accuracy": float(acc), "oof_macro_f1": float(f1),
        "fold_accuracy_mean": float(np.mean(fa)), "fold_accuracy_std": float(np.std(fa, ddof=1)),
        "fold_macro_f1_mean": float(np.mean(ff)), "fold_macro_f1_std": float(np.std(ff, ddof=1)),
        "panel_accuracy_majority_vote": panel_ok / n_panels, "panels_correct": panel_ok, "n_panels": n_panels,
        "panels_with_errors": wrong_panels, "n_panels_with_errors": len(wrong_panels),
        "n_misclassified_images": int((oof != labels).sum()),
        "per_class_recall": recall, "confusion_matrix": cm.tolist(),
        "n_parameters": int(n_params), "model_size_mb": round(size_mb, 2),
        "cpu_inference_ms_per_image": round(cpu_ms, 2),
        "total_train_time_seconds": round(sum(f["train_time_seconds"] for f in folds), 1),
        "folds": folds,
    }
    json.dump(result, open(REPORTS / f"cnn_cv_{name}.json", "w"), indent=2)
    with open(REPORTS / f"cnn_cv_{name}_oof_predictions.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["filepath", "class", "group", "fold", "pred"])
        for r, g, f_, p in zip(rows, groups, fold_of, oof):
            w.writerow([r["filepath"], r["class"], int(g), int(f_), classes[int(p)]])

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(f"{a.model}: confusion matrix, panel-grouped 5-fold CV (acc={acc:.3f})")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(FIGURES / f"cnn_cv_{name}_confusion_matrix.png", dpi=150)
    plt.close(fig)

    print("\n=== PANEL-GROUPED 5-FOLD CV:", a.model, "===")
    print(f"OOF accuracy {acc:.4f}  macro-F1 {f1:.4f}")
    print(f"fold accuracy {np.mean(fa):.4f} +/- {np.std(fa, ddof=1):.4f}: {[round(x, 4) for x in fa]}")
    print(f"panel accuracy (majority vote) {panel_ok}/{n_panels} = {panel_ok / n_panels:.4f}")
    print(f"panels with errors: {len(wrong_panels)}; misclassified images: {int((oof != labels).sum())}")
    print(f"params {n_params / 1e6:.2f} M, size {size_mb:.1f} MB, CPU {cpu_ms:.1f} ms/image")
    print(f"saved reports/cnn_cv_{name}.json", flush=True)


if __name__ == "__main__":
    main()
