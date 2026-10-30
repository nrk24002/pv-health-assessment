"""
Merge panel groups that contain the same panel image (Milestone 3.2, second leakage check).

Grouping by (class, original width, height) puts all rotated/flipped copies of a
panel in one group, but a panel can also appear again in a different size (for
example a copy squeezed to 700 x 700 pixels, or a slightly rescaled copy). Such
copies end up in different groups and therefore possibly in different folds.

This script compares 64 x 64 RGB thumbnails of all images of the same class
(correlation after mean subtraction, in all 8 rotations/flips of the image). Two
groups are merged when at least half of the images in the smaller group have a
near-identical match (correlation > 0.999) in the other group. The result is
written to data/processed/group_merge.csv (filepath, group_merged) and is read by
dataset.py.

Run:
    python3 src/merge_groups.py
"""
import csv
import collections
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED = PROJECT_ROOT / "data" / "processed"
THUMB = 64
THRESHOLD = 0.999
MIN_SHARE = 0.5


def norm(x):
    x = x.reshape(len(x), -1)
    x = x - x.mean(1, keepdims=True)
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-6)


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

    parent = list(range(base.max() + 1))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    merges = []
    for c in sorted(set(cls)):
        ix = np.where(cls == c)[0]
        gs = sorted(set(base[ix]))
        C = np.max([N[ix] @ v[ix].T for v in V], axis=0)
        g = base[ix]
        for a in gs:
            for b in gs:
                if a >= b:
                    continue
                ma, mb = g == a, g == b
                # share of images of each group with a near-identical match in the other group
                sa = float(np.mean(C[np.ix_(ma, mb)].max(1) > THRESHOLD))
                sb = float(np.mean(C[np.ix_(mb, ma)].max(1) > THRESHOLD))
                if max(sa, sb) >= MIN_SHARE:
                    merges.append((c, a, b, int(ma.sum()), int(mb.sum()), round(sa, 2), round(sb, 2)))
                    parent[find(a)] = find(b)
    merged = np.array([find(x) for x in base])
    remap = {m: i for i, m in enumerate(sorted(set(merged)))}
    merged = np.array([remap[m] for m in merged])
    with open(PROCESSED / "group_merge.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["filepath", "group_merged"])
        for r, m in zip(rows, merged):
            w.writerow([r["filepath"], int(m)])
    print(f"groups before: {base.max() + 1}, after merging: {merged.max() + 1}")
    for m in merges:
        print("merged:", m)


if __name__ == "__main__":
    main()
