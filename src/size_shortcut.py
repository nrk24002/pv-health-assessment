"""
Shortcut check: how much of the class can be predicted from the original image
size alone (width, height, aspect ratio), without looking at any pixel?

The same Random Forest settings and the same fold definitions as in
baseline_model.py are used:
  * GroupKFold over the merged panel groups   (column "group")
  * GroupKFold over identical-pHash groups    (column "dup_group")
Chance level for six balanced classes is 0.167.

Run:
    python3 src/size_shortcut.py
Output: reports/size_shortcut.json
"""
import csv
import json

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupKFold

from dataset import MANIFEST_PATH, PROJECT_ROOT

SEED = 42
RF_PARAMS = dict(n_estimators=300, max_depth=None, min_samples_leaf=2,
                 class_weight="balanced", random_state=SEED, n_jobs=-1)


def main():
    with open(MANIFEST_PATH, newline="") as fh:
        rows = list(csv.DictReader(fh))
    classes = sorted({r["class"] for r in rows})
    y = np.array([classes.index(r["class"]) for r in rows])
    with open(PROJECT_ROOT / "data" / "processed" / "image_index.csv", newline="") as fh:
        size = {r["filepath"]: (int(r["width"]), int(r["height"])) for r in csv.DictReader(fh)}
    X = np.array([[*size[r["filepath"]], size[r["filepath"]][0] / size[r["filepath"]][1]]
                  for r in rows], dtype=float)
    out = {"features": ["width", "height", "aspect_ratio"], "chance_level": round(1 / len(classes), 4)}
    for name, col in (("panel_grouped_5fold", "group"), ("phash_grouped_5fold", "dup_group")):
        g = np.array([int(r[col]) for r in rows])
        accs = []
        for tr, te in GroupKFold(n_splits=5).split(X, y, groups=g):
            clf = RandomForestClassifier(**RF_PARAMS).fit(X[tr], y[tr])
            accs.append(accuracy_score(y[te], clf.predict(X[te])))
        out[name] = {"fold_accuracy": [round(a, 4) for a in accs],
                     "mean_accuracy": round(float(np.mean(accs)), 4),
                     "std_accuracy": round(float(np.std(accs)), 4)}
        print(name, out[name])
    (PROJECT_ROOT / "reports" / "size_shortcut.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
