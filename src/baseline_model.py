"""
Baseline for the Photovoltaic Health Assessment project: hand-crafted colour/contrast
features + Random Forest, evaluated on the duplicate-safe split from dataset.py.

Features (55 per image, from the 224x224 cache): per RGB channel the mean, the
standard deviation and a 16-bin normalised histogram (3 x 18 = 54), plus the
standard deviation of the grayscale image.

Evaluation:
  * train / validation / test split (group-aware, see dataset.py)
  * 5-fold GroupKFold cross-validation over all images (duplicate groups never
    cross a fold boundary)
  * for comparison only: 5-fold stratified cross-validation that ignores the
    duplicate groups, to show how much duplicates inflate the accuracy

Run:
    python3 src/baseline_model.py

Outputs (reports/): baseline_metrics.json, baseline_classification_report.txt,
baseline_test_predictions.csv, figures/baseline_confusion_matrix.png,
figures/baseline_feature_importance.png
"""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold, StratifiedKFold

from dataset import MANIFEST_PATH, PROJECT_ROOT

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
SEED = 42
N_BINS = 16
RF_PARAMS = dict(n_estimators=300, max_depth=None, min_samples_leaf=2,
                 class_weight="balanced", random_state=SEED, n_jobs=-1)

FEATURE_NAMES = []
for _c in ("R", "G", "B"):
    FEATURE_NAMES += [f"{_c}_mean", f"{_c}_std"] + [f"{_c}_hist_{i}" for i in range(N_BINS)]
FEATURE_NAMES.append("gray_std")


def extract_features(path: Path) -> np.ndarray:
    arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    feats = []
    for c in range(3):
        ch = arr[:, :, c]
        feats += [ch.mean(), ch.std()]
        hist, _ = np.histogram(ch, bins=N_BINS, range=(0.0, 1.0), density=True)
        feats += hist.tolist()
    feats.append(arr.mean(axis=2).std())
    return np.array(feats, dtype=np.float32)


def cv_scores(X, y, splitter, groups=None):
    accs, f1s = [], []
    for tr, te in splitter.split(X, y, groups=groups):
        clf = RandomForestClassifier(**RF_PARAMS).fit(X[tr], y[tr])
        p = clf.predict(X[te])
        accs.append(accuracy_score(y[te], p))
        f1s.append(f1_score(y[te], p, average="macro", zero_division=0))
    return np.array(accs), np.array(f1s)


def main():
    with open(MANIFEST_PATH, newline="") as fh:
        rows = list(csv.DictReader(fh))
    classes = sorted({r["class"] for r in rows})
    c2i = {c: i for i, c in enumerate(classes)}
    print(f"Extracting features for {len(rows)} images ...")
    X = np.vstack([extract_features(PROJECT_ROOT / r["cache"]) for r in rows])
    y = np.array([c2i[r["class"]] for r in rows])
    split = np.array([r["split"] for r in rows])
    groups = np.array([int(r["group"]) for r in rows])
    dup_groups = np.array([int(r["dup_group"]) for r in rows])

    tr, va, te = split == "train", split == "val", split == "test"
    print(f"train/val/test = {tr.sum()}/{va.sum()}/{te.sum()}, features = {X.shape[1]}")

    clf = RandomForestClassifier(**RF_PARAMS).fit(X[tr], y[tr])
    val_acc = accuracy_score(y[va], clf.predict(X[va]))
    pred = clf.predict(X[te])
    test_acc = accuracy_score(y[te], pred)
    test_f1 = f1_score(y[te], pred, average="macro", zero_division=0)
    report = classification_report(y[te], pred, target_names=classes, zero_division=0, digits=3)
    cm = confusion_matrix(y[te], pred, labels=list(range(len(classes))))
    print(f"val acc {val_acc:.3f} | test acc {test_acc:.3f} | test macro-F1 {test_f1:.3f}")
    print(report)

    print("Group-aware 5-fold CV ...")
    g_acc, g_f1 = cv_scores(X, y, GroupKFold(n_splits=5), groups=groups)
    print("  acc", np.round(g_acc, 3), "mean", round(g_acc.mean(), 3), "std", round(g_acc.std(), 3))
    print("5-fold CV grouped by identical pHash only (rotated variants of a panel are NOT grouped) ...")
    d_acc, d_f1 = cv_scores(X, y, GroupKFold(n_splits=5), groups=dup_groups)
    print("  acc", np.round(d_acc, 3), "mean", round(d_acc.mean(), 3), "std", round(d_acc.std(), 3))
    print("Stratified 5-fold CV that ignores duplicate groups (comparison only) ...")
    l_acc, l_f1 = cv_scores(X, y, StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED))
    print("  acc", np.round(l_acc, 3), "mean", round(l_acc.mean(), 3), "std", round(l_acc.std(), 3))

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    metrics = {
        "model": "handcrafted_colour_features_random_forest",
        "seed": SEED, "rf_params": {k: v for k, v in RF_PARAMS.items() if k != "n_jobs"},
        "feature_dim": int(X.shape[1]), "classes": classes,
        "train_size": int(tr.sum()), "val_size": int(va.sum()), "test_size": int(te.sum()),
        "chance_level_accuracy": round(1 / len(classes), 4),
        "val_accuracy": round(val_acc, 4),
        "test_accuracy": round(test_acc, 4), "test_macro_f1": round(test_f1, 4),
        "confusion_matrix": cm.tolist(),
        "group_cv_5fold": {"accuracy": [round(a, 4) for a in g_acc], "macro_f1": [round(a, 4) for a in g_f1],
                           "mean_accuracy": round(float(g_acc.mean()), 4), "std_accuracy": round(float(g_acc.std()), 4),
                           "mean_macro_f1": round(float(g_f1.mean()), 4), "std_macro_f1": round(float(g_f1.std()), 4)},
        "phash_group_cv_5fold": {"accuracy": [round(a, 4) for a in d_acc],
                                 "mean_accuracy": round(float(d_acc.mean()), 4), "std_accuracy": round(float(d_acc.std()), 4),
                                 "mean_macro_f1": round(float(d_f1.mean()), 4)},
        "n_panel_groups": int(len(set(groups.tolist()))),
        "stratified_cv_ignoring_duplicates_5fold": {
            "accuracy": [round(a, 4) for a in l_acc],
            "mean_accuracy": round(float(l_acc.mean()), 4), "std_accuracy": round(float(l_acc.std()), 4),
            "mean_macro_f1": round(float(l_f1.mean()), 4)},
    }
    (REPORTS_DIR / "baseline_metrics.json").write_text(json.dumps(metrics, indent=2))
    (REPORTS_DIR / "baseline_classification_report.txt").write_text(
        f"Baseline: colour/contrast features + Random Forest\n\nVal accuracy {val_acc:.3f}\n"
        f"Test accuracy {test_acc:.3f}  Test macro-F1 {test_f1:.3f}\n\n{report}")
    with open(REPORTS_DIR / "baseline_test_predictions.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["filepath", "true", "pred"])
        for r, t, p in zip([r for r, s in zip(rows, te) if s], y[te], pred):
            w.writerow([r["filepath"], classes[t], classes[p]])

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes))); ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right"); ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted label"); ax.set_ylabel("True label")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout(); fig.savefig(FIGURES_DIR / "baseline_confusion_matrix.png", dpi=200); plt.close(fig)

    order = np.argsort(clf.feature_importances_)[::-1][:15]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.barh([FEATURE_NAMES[i] for i in order][::-1], clf.feature_importances_[order][::-1])
    ax.set_xlabel("Feature importance")
    fig.tight_layout(); fig.savefig(FIGURES_DIR / "baseline_feature_importance.png", dpi=200); plt.close(fig)
    print("Saved results to", REPORTS_DIR)


if __name__ == "__main__":
    main()
