"""
Grad-CAM check for the ResNet18 models trained in the panel-grouped CV.

For every class, two correctly classified images and (if any) one
misclassified image are taken from the HELD-OUT part of a fold, i.e. from panels
the model has never seen, and shown with the Grad-CAM map of the last
convolutional block. The question is whether the model looks at the faulty
region of the panel or at something unrelated to the fault (background,
borders, panel edges).

Needs the fold models and predictions written by:  python3 src/cnn_cv.py --model resnet18

Run:
    python3 src/gradcam.py

Outputs:
    reports/figures/gradcam_examples.png
    reports/gradcam_examples.csv    (which images are shown)
"""
import csv
import random
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from cnn_cv import build_model, get_device
from dataset import PROJECT_ROOT, get_transforms

REPORTS = PROJECT_ROOT / "reports"
FIGURES = REPORTS / "figures"
MODELS = PROJECT_ROOT / "models"
SEED = 42


def grad_cam(model, x, target_class):
    store = {}
    layer = model.layer4[-1]
    h1 = layer.register_forward_hook(lambda m, i, o: store.__setitem__("a", o))
    h2 = layer.register_full_backward_hook(lambda m, gi, go: store.__setitem__("g", go[0]))
    model.zero_grad()
    out = model(x)
    pred = int(out.argmax(1))
    out[0, target_class].backward()
    h1.remove()
    h2.remove()
    w = store["g"].mean(dim=(2, 3), keepdim=True)
    cam = F.relu((w * store["a"]).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)[0, 0]
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    return cam.detach().cpu().numpy(), pred


def main():
    device = get_device()
    rng = random.Random(SEED)
    with open(REPORTS / "cnn_cv_resnet18_oof_predictions.csv", newline="") as fh:
        rows = list(csv.DictReader(fh))
    with open(PROJECT_ROOT / "data" / "processed" / "manifest.csv", newline="") as fh:
        cache = {r["filepath"]: r["cache"] for r in csv.DictReader(fh)}
    classes = sorted({r["class"] for r in rows})
    tf = get_transforms(train=False)

    picks = []
    for c in classes:
        ok = [r for r in rows if r["class"] == c and r["pred"] == c]
        bad = [r for r in rows if r["class"] == c and r["pred"] != c]
        # one image per panel, different panels
        def one_per_panel(rs):
            seen, out = set(), []
            rng.shuffle(rs)
            for r in rs:
                if r["group"] not in seen:
                    seen.add(r["group"])
                    out.append(r)
            return out
        picks += [(r, "correct") for r in one_per_panel(ok)[:2]]
        picks += [(r, "error") for r in one_per_panel(bad)[:1]]

    models_cache = {}
    ncls = len(classes)
    fig, axes = plt.subplots(3, ncls, figsize=(2.3 * ncls, 7.4))
    out_rows = []
    by_class = defaultdict(list)
    for r, kind in picks:
        by_class[r["class"]].append((r, kind))
    for ci, c in enumerate(classes):
        slots = [p for p in by_class[c] if p[1] == "correct"][:2]
        errs = [p for p in by_class[c] if p[1] == "error"][:1]
        cells = slots + [None] * (2 - len(slots)) + (errs if errs else [None])
        for ri, item in enumerate(cells):
            ax = axes[ri, ci]
            ax.axis("off")
            if item is None:
                ax.text(0.5, 0.5, "no errors" if ri == 2 else "", ha="center", va="center", fontsize=9,
                        color="gray", transform=ax.transAxes)
                continue
            r, kind = item
            k = int(r["fold"])
            if k not in models_cache:
                m = build_model("resnet18", ncls, pretrained=False)
                m.load_state_dict(torch.load(MODELS / f"resnet18_fold{k}.pt", map_location="cpu"))
                models_cache[k] = m.to(device).eval()
            img = Image.open(PROJECT_ROOT / cache[r["filepath"]]).convert("RGB")
            x = tf(img).unsqueeze(0).to(device)
            cam, pred = grad_cam(models_cache[k], x, classes.index(r["pred"]))
            ax.imshow(img)
            ax.imshow(cam, cmap="jet", alpha=0.40)
            ax.set_title(f"{r['class']}\npredicted: {classes[pred]}", fontsize=8,
                         color="red" if kind == "error" else "black")
            out_rows.append({"filepath": r["filepath"], "true": r["class"], "pred": classes[pred],
                             "fold": k, "panel": r["group"], "kind": kind})
    fig.tight_layout()
    fig.savefig(FIGURES / "gradcam_examples.png", dpi=130)
    plt.close(fig)
    with open(REPORTS / "gradcam_examples.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out_rows[0]))
        w.writeheader()
        w.writerows(out_rows)
    print(f"saved {FIGURES / 'gradcam_examples.png'} ({len(out_rows)} examples)")


if __name__ == "__main__":
    main()
