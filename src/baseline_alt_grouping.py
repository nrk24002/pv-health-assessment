"""
Grouping sensitivity analysis for the Random Forest baseline.

Runs the panel-grouped 5-fold cross-validation of baseline_model.py with an
alternative grouping file (default: threshold 0.995, written by merge_sensitivity.py).

Run:
    python3 src/baseline_alt_grouping.py [--group-file data/processed/group_merge_t0995.csv]
Output: reports/sensitivity_t0995_baseline.json
"""
import argparse
import csv
import json

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold

from baseline_model import RF_PARAMS, extract_features
from dataset import MANIFEST_PATH, PROJECT_ROOT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group-file", default=str(PROJECT_ROOT / "data/processed/group_merge_t0995.csv"))
    a = ap.parse_args()
    with open(MANIFEST_PATH, newline="") as fh:
        rows = list(csv.DictReader(fh))
    with open(a.group_file, newline="") as fh:
        alt = {r["filepath"]: int(r["group_merged"]) for r in csv.DictReader(fh)}
    classes = sorted({r["class"] for r in rows})
    y = np.array([classes.index(r["class"]) for r in rows])
    g = np.array([alt[r["filepath"]] for r in rows])
    X = np.vstack([extract_features(PROJECT_ROOT / r["cache"]) for r in rows])
    accs, f1s, wrong = [], [], 0
    gwrong = {}
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups=g):
        p = RandomForestClassifier(**RF_PARAMS).fit(X[tr], y[tr]).predict(X[te])
        accs.append(accuracy_score(y[te], p))
        f1s.append(f1_score(y[te], p, average="macro", zero_division=0))
        for gi in set(g[te][p != y[te]].tolist()):
            gwrong[gi] = int(((p != y[te]) & (g[te] == gi)).sum())
    out = {"group_file": a.group_file, "n_groups": int(len(set(g.tolist()))),
           "fold_accuracy": [round(x, 4) for x in accs],
           "mean_accuracy": round(float(np.mean(accs)), 4), "std_accuracy": round(float(np.std(accs)), 4),
           "mean_macro_f1": round(float(np.mean(f1s)), 4),
           "n_groups_with_errors": len(gwrong), "n_errors": int(sum(gwrong.values()))}
    print(out)
    (PROJECT_ROOT / "reports" / "sensitivity_t0995_baseline.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
