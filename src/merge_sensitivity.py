"""
Sensitivity of the panel grouping to the merge threshold (see merge_groups.py).

For several correlation thresholds and minimum shares, count how many panel groups
remain after merging groups of the same class that contain the same panel image.
Does not change data/processed/group_merge.csv; writes data/processed/group_merge_t0995.csv
(threshold 0.995, share 0.5) for the sensitivity analysis.

Run:
    python3 src/merge_sensitivity.py
Output: reports/merge_sensitivity.json
"""
import csv
import json

import numpy as np
from PIL import Image

from merge_groups import PROCESSED, PROJECT_ROOT, THUMB, norm


def main():
    with open(PROCESSED / "image_index.csv", newline="") as fh:
        rows = list(csv.DictReader(fh))
    ids, base = {}, []
    for r in rows:
        base.append(ids.setdefault((r["class"], r["width"], r["height"]), len(ids)))
    base = np.array(base)
    cls = np.array([r["class"] for r in rows])
    T = np.stack([np.asarray(Image.open(PROJECT_ROOT / r["cache"]).convert("RGB").resize((THUMB, THUMB), Image.BILINEAR),
                             dtype=np.float32) for r in rows])
    V = []
    for k in range(4):
        rot = np.rot90(T, k, axes=(1, 2))
        V += [norm(rot), norm(rot[:, :, ::-1])]
    N = V[0]
    corr = {}
    for c in sorted(set(cls)):
        ix = np.where(cls == c)[0]
        corr[c] = (ix, np.max([N[ix] @ v[ix].T for v in V], axis=0))
    out = []
    for thr in (0.999, 0.995, 0.99, 0.98):
        for share in (0.5, 0.25):
            parent = list(range(base.max() + 1))

            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x
            for c, (ix, C) in corr.items():
                g = base[ix]
                gs = sorted(set(g))
                for a in gs:
                    for b in gs:
                        if a >= b:
                            continue
                        ma, mb = g == a, g == b
                        sa = float(np.mean(C[np.ix_(ma, mb)].max(1) > thr))
                        sb = float(np.mean(C[np.ix_(mb, ma)].max(1) > thr))
                        if max(sa, sb) >= share:
                            parent[find(a)] = find(b)
            merged = np.array([find(x) for x in base])
            n = len(set(merged.tolist()))
            if (thr, share) == (0.995, 0.5):
                remap = {m: i for i, m in enumerate(sorted(set(merged.tolist())))}
                with open(PROCESSED / "group_merge_t0995.csv", "w", newline="") as fh:
                    w = csv.writer(fh)
                    w.writerow(["filepath", "group_merged"])
                    for r, m in zip(rows, merged):
                        w.writerow([r["filepath"], remap[int(m)]])
            out.append({"threshold": thr, "min_share": share, "groups_after_merging": n})
            print(out[-1])
    (PROJECT_ROOT / "reports" / "merge_sensitivity.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
