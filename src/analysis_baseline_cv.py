"""
Extra analysis for the baseline (reports): out-of-fold predictions of the
panel-grouped 5-fold cross-validation, the resulting confusion matrix and
per-class recall, the dataset structure table, and an example figure of the
variants of one source panel.

Run (after dataset.py and baseline_model.py):
    python3 src/analysis_baseline_cv.py
"""
import csv
import json
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from sklearn.model_selection import GroupKFold

from baseline_model import RF_PARAMS, extract_features
from dataset import INDEX_PATH, MANIFEST_PATH, PROJECT_ROOT

REPORTS = PROJECT_ROOT / "reports"
FIG = REPORTS / "figures"


def main():
    rows = list(csv.DictReader(open(MANIFEST_PATH, newline="")))
    idx = {r["filepath"]: r for r in csv.DictReader(open(INDEX_PATH, newline=""))}
    classes = sorted({r["class"] for r in rows})
    c2i = {c: i for i, c in enumerate(classes)}
    X = np.vstack([extract_features(PROJECT_ROOT / r["cache"]) for r in rows])
    y = np.array([c2i[r["class"]] for r in rows])
    g = np.array([int(r["group"]) for r in rows])

    oof = np.zeros_like(y)
    folds = []
    for k, (tr, te) in enumerate(GroupKFold(n_splits=5).split(X, y, groups=g)):
        p = RandomForestClassifier(**RF_PARAMS).fit(X[tr], y[tr]).predict(X[te])
        oof[te] = p
        folds.append({"fold": k + 1, "n_test_images": int(len(te)), "n_test_panels": int(len(set(g[te]))),
                      "accuracy": round(float(accuracy_score(y[te], p)), 4),
                      "classes_wrong": {classes[c]: int(((y[te] == c) & (p != c)).sum()) for c in range(len(classes))}})
    cm = confusion_matrix(y, oof)
    recall = (cm.diagonal() / cm.sum(1)).round(4).tolist()
    # misclassified panels
    wrong_panels = defaultdict(lambda: [0, 0])
    for gi, t, p in zip(g, y, oof):
        wrong_panels[gi][1] += 1
        wrong_panels[gi][0] += int(t != p)
    bad = {int(k): v for k, v in wrong_panels.items() if v[0] > 0}
    out = {"oof_accuracy": round(float(accuracy_score(y, oof)), 4),
           "oof_macro_f1": round(float(f1_score(y, oof, average="macro")), 4),
           "classes": classes, "confusion_matrix": cm.tolist(), "per_class_recall": dict(zip(classes, recall)),
           "folds": folds, "panels_with_errors": {"count": len(bad), "of": int(len(wrong_panels)),
                                                  "detail_[wrong,total]": bad}}
    (REPORTS / "baseline_groupcv_analysis.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "panels_with_errors"}, indent=1))
    print("panels with errors:", out["panels_with_errors"])

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
    fig.tight_layout(); fig.savefig(FIG / "baseline_groupcv_confusion_matrix.png", dpi=200); plt.close(fig)

    # dataset structure table
    tab = []
    for c in classes:
        rc = [r for r in idx.values() if r["class"] == c]
        panels = defaultdict(list)
        for r in rc:
            panels[(r["width"], r["height"])].append(r)
        rep = max(panels.values(), key=len)
        tab.append({"class": c, "files": len(rc), "distinct_md5": len({r["md5"] for r in rc}),
                    "distinct_phash": len({r["phash"] for r in rc}), "panels": len(panels),
                    "largest_panel_group": len(rep), "largest_group_distinct_md5": len({r["md5"] for r in rep})})
    (REPORTS / "dataset_structure.json").write_text(json.dumps(tab, indent=2))
    print(json.dumps(tab, indent=1))

    # example figure: 8 variants of one source panel
    key = ("Burnmarks", "844", "2113")
    v = [r for r in idx.values() if (r["class"], r["width"], r["height"]) == key][:8]
    fig, axs = plt.subplots(2, 4, figsize=(6.5, 3.5))
    for ax, r in zip(axs.ravel(), v):
        ax.imshow(Image.open(PROJECT_ROOT / r["cache"]).convert("RGB")); ax.axis("off")
    fig.tight_layout(pad=0.3); fig.savefig(FIG / "panel_variants_example.png", dpi=200); plt.close(fig)


if __name__ == "__main__":
    main()
