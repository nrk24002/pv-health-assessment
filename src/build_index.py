"""
Step 1: index the full PV Module Dataset and cache 224x224 copies.

For every image in <dataset>/<class>/*.jpg this records the MD5 checksum of the
file, a perceptual hash (pHash, 8x8) of the original image and the original size,
and stores a 224x224 RGB copy (PNG, lossless) in data/processed/cache224/.
The 224x224 copies are what the baseline and the CNN read, so both models see
exactly the same pixels.

The script is resumable: records that already exist in image_index.csv are
skipped, and it stops after --budget seconds, so it can be run in short slices.

Run:
    python3 src/build_index.py [--data DIR] [--budget 150]
"""
import argparse
import csv
import hashlib
import time
from multiprocessing import Pool
from pathlib import Path

import imagehash
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = PROJECT_ROOT.parent.parent.parent / "PV Module Dataset"
PROCESSED = PROJECT_ROOT / "data" / "processed"
CACHE = PROCESSED / "cache224"
INDEX = PROCESSED / "image_index.csv"
IMG_SIZE = 224
FIELDS = ["filepath", "class", "md5", "phash", "width", "height", "cache"]


def process(args):
    path, cls, data_dir = args
    rel = path.relative_to(data_dir)
    raw = path.read_bytes()
    md5 = hashlib.md5(raw).hexdigest()
    img = Image.open(path).convert("RGB")
    w, h = img.size
    ph = str(imagehash.phash(img))
    out = CACHE / cls / (path.stem + ".png")
    out.parent.mkdir(parents=True, exist_ok=True)
    img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR).save(out)
    return {"filepath": str(rel), "class": cls, "md5": md5, "phash": ph,
            "width": w, "height": h, "cache": str(out.relative_to(PROJECT_ROOT))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--budget", type=float, default=150.0)
    a = ap.parse_args()

    PROCESSED.mkdir(parents=True, exist_ok=True)
    done = {}
    if INDEX.exists():
        with open(INDEX, newline="") as fh:
            for r in csv.DictReader(fh):
                done[r["filepath"]] = r

    todo = []
    for cdir in sorted(p for p in a.data.iterdir() if p.is_dir()):
        for f in sorted(list(cdir.glob("*.jpg")) + list(cdir.glob("*.JPG"))):
            if str(f.relative_to(a.data)) not in done:
                todo.append((f, cdir.name, a.data))
    total = len(done) + len(todo)
    print(f"already indexed: {len(done)}, remaining: {len(todo)}")

    t0 = time.time()
    with Pool(4) as pool:
        for i in range(0, len(todo), 40):
            if time.time() - t0 > a.budget:
                break
            for rec in pool.map(process, todo[i:i + 40]):
                done[rec["filepath"]] = rec

    with open(INDEX, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(done[k] for k in sorted(done))
    print(f"indexed {len(done)} of {total} images")


if __name__ == "__main__":
    main()
