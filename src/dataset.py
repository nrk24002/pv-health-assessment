"""
Data utilities for the Photovoltaic Health Assessment project.

Works on the complete PV Module Dataset (6 classes x 525 images). The images are
first indexed and cached at 224x224 by build_index.py; this module then builds a
duplicate-safe train/val/test split and provides the PyTorch Dataset.

Duplicate handling: the dataset contains many byte-identical or visually
identical images (only 1891 of 3150 files are distinct by MD5) and, as found in
this project, many rotated/flipped variants of the same source panel. Within
each class the images fall into 26 groups of identical pixel dimensions, and the
images in such a group are variants of one source panel (the colour statistics
inside a group differ by about 3 grey levels, between groups by 14-19). Images
are therefore grouped into "panel groups" by (class, width, height) of the
original file. A second check (merge_groups.py) merges groups that contain the
same panel in another size (a squeezed copy), which gives 150 panel groups, 25
per class. Every panel group is assigned to exactly one of train/val/test, so
that variants of the same panel never appear on both sides of a split. (A finer grouping by identical pHash, "dup_group", is stored as well
for comparison; it leaves rotated variants of a panel in different groups.)

Split: stratified per class, about 70/15/15 (train/val/test), seed 42. Groups
are placed largest first on the split that is currently furthest below its
target share (greedy), which keeps the proportions close to the targets.

Preprocessing (applied to the 224x224 cache): ImageNet mean/std normalization.
Training-set augmentation: horizontal flip (p=0.5), rotation up to 10 degrees,
brightness/contrast jitter 0.1. No vertical flip and no strong colour changes.
"""

import csv
import random
from collections import defaultdict
from pathlib import Path

from PIL import Image

try:
    import torch  # noqa: F401
    from torch.utils.data import Dataset
    from torchvision import transforms
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False
    Dataset = object

IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
INDEX_PATH = PROCESSED_DIR / "image_index.csv"
MANIFEST_PATH = PROCESSED_DIR / "manifest.csv"
MANIFEST_FIELDS = ["filepath", "class", "split", "group", "dup_group", "md5", "cache"]


def read_index(path: Path = INDEX_PATH):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def duplicate_groups(rows):
    """Return one integer group id per row (same class and identical pHash -> same group)."""
    parent = list(range(len(rows)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    first = {}
    for i, r in enumerate(rows):
        key = (r["class"], r["phash"])
        if key in first:
            parent[find(i)] = find(first[key])
        else:
            first[key] = i
    return [find(i) for i in range(len(rows))]


def panel_groups(rows):
    """One group id per row: same class and same original width x height -> same source panel.
    If data/processed/group_merge.csv exists (written by merge_groups.py), groups that contain
    near-identical copies of the same panel (for example a copy squeezed to 700 x 700 pixels)
    are merged, so that such copies can never end up in different splits."""
    ids, out = {}, []
    for r in rows:
        key = (r["class"], r["width"], r["height"])
        out.append(ids.setdefault(key, len(ids)))
    merge_csv = PROCESSED_DIR / "group_merge.csv"
    if merge_csv.exists():
        with open(merge_csv, newline="") as fh:
            merged = {r["filepath"]: int(r["group_merged"]) for r in csv.DictReader(fh)}
        out = [merged[r["filepath"]] for r in rows]
    return out


def build_split_manifest(index_path: Path = INDEX_PATH, out_csv: Path = MANIFEST_PATH,
                         train_frac: float = 0.70, val_frac: float = 0.15, seed: int = 42):
    rows = read_index(index_path)
    gids = panel_groups(rows)
    dgids = duplicate_groups(rows)
    for r, g, d in zip(rows, gids, dgids):
        r["group"] = g
        r["dup_group"] = d

    rng = random.Random(seed)
    test_frac = 1.0 - train_frac - val_frac
    by_class = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by_class[r["class"]][r["group"]].append(r)

    out = []
    for cls in sorted(by_class):
        groups = list(by_class[cls].values())
        n = sum(len(g) for g in groups)
        targets = {"train": n * train_frac, "val": n * val_frac, "test": n * test_frac}
        counts = {"train": 0, "val": 0, "test": 0}
        rng.shuffle(groups)
        groups.sort(key=len, reverse=True)
        for g in groups:
            split = min(counts, key=lambda s: counts[s] - targets[s])
            counts[split] += len(g)
            for r in g:
                out.append({"filepath": r["filepath"], "class": cls, "split": split,
                            "group": r["group"], "dup_group": r["dup_group"],
                            "md5": r["md5"], "cache": r["cache"]})
        if counts["val"] == 0 or counts["test"] == 0:
            raise RuntimeError(f"Class {cls} has an empty val/test split: {counts}")

    out.sort(key=lambda r: (r["class"], r["filepath"]))
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        w.writerows(out)
    return out


def get_transforms(train: bool):
    if not _HAS_TORCH:
        raise ImportError("torch/torchvision are required for get_transforms()")
    if train:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class PVDataset(Dataset):
    """One split ('train'/'val'/'test') of the manifest, read from the 224x224 cache.
    Class indices are built from the full manifest so they are identical in all splits."""

    def __init__(self, manifest_csv: Path, split: str, root_dir: Path = PROJECT_ROOT, transform=None):
        if not _HAS_TORCH:
            raise ImportError("torch/torchvision are required for PVDataset")
        self.root_dir = Path(root_dir)
        self.transform = transform or get_transforms(train=(split == "train"))
        with open(manifest_csv, newline="") as fh:
            all_rows = list(csv.DictReader(fh))
        self.classes = sorted({r["class"] for r in all_rows})
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.rows = [r for r in all_rows if r["split"] == split]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        image = Image.open(self.root_dir / row["cache"]).convert("RGB")
        return self.transform(image), self.class_to_idx[row["class"]]


def summarize_manifest(rows):
    from collections import Counter
    return Counter((r["class"], r["split"]) for r in rows)


if __name__ == "__main__":
    rows = build_split_manifest()
    summary = summarize_manifest(rows)
    print(f"Wrote manifest with {len(rows)} rows to {MANIFEST_PATH}")
    print(f"{'class':16s} {'train':>6s} {'val':>6s} {'test':>6s}   unique groups (train/val/test)")
    for cls in sorted({c for c, _ in summary}):
        cnt = {s: summary.get((cls, s), 0) for s in ("train", "val", "test")}
        grp = {s: len({r['group'] for r in rows if r['class'] == cls and r['split'] == s}) for s in cnt}
        print(f"{cls:16s} {cnt['train']:6d} {cnt['val']:6d} {cnt['test']:6d}   {grp['train']}/{grp['val']}/{grp['test']}")
    tot = {s: sum(1 for r in rows if r['split'] == s) for s in ("train", "val", "test")}
    totg = {s: len({r['group'] for r in rows if r['split'] == s}) for s in tot}
    print("TOTAL", tot, "unique groups", totg)
    sets = {s: {r['group'] for r in rows if r['split'] == s} for s in tot}
    print("group overlap train/val, train/test, val/test:",
          len(sets['train'] & sets['val']), len(sets['train'] & sets['test']), len(sets['val'] & sets['test']))
