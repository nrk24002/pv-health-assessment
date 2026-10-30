# Photovoltaic Health Assessment

Image-based fault classification of photovoltaic (PV) panels for maintenance decision support.
Course project in D7016B Industrial AI and eMaintenance, Part II: Practical Implementation
(Luleå University of Technology, autumn 2026).

Author: Noel Rydback. Supervisor: Naveen Venkatesh Sridharan.

## Summary

The dataset has 3,150 labeled images in 6 classes (Burnmarks, Delamination, Discoloration,
Glass breakage, Good Panel, Snailtrail), but only 1,891 distinct files, and the images are
rotated, flipped or rescaled variants of 150 panel groups (25 per class). All models are
therefore evaluated with a panel-grouped hold-out split and a panel-grouped 5-fold
cross-validation, so that no panel appears in both training and test data.

| Model | 5-fold CV accuracy (panel-grouped) | Image errors | Panel groups correct |
|---|---|---|---|
| Random Forest, 55 colour features | 0.980 ± 0.038 | 64 | 149 / 150 |
| ResNet18 (fine-tuned) | 1.000 ± 0.000 | 0 | 150 / 150 |
| MobileNetV3-Small | 0.998 ± 0.004 | 7 | 150 / 150 |
| EfficientNet-B0 | 0.999 ± 0.001 | 2 | 150 / 150 |

Standard deviations are population standard deviations over the five folds.
All results are from one seed (42) and have no external test set. Image size alone predicts
the class with an accuracy of 0.498 (chance 0.167), so a shortcut cannot be excluded.
See the final report for the full discussion and limitations.

## Repository structure

```
src/        Python scripts (see "Reproducing the results")
data/
  raw/                  not tracked
  processed/            image_index.csv, manifest.csv, group_merge*.csv are tracked;
                        the 224x224 image cache is not
reports/    metrics (JSON), predictions (CSV) and figures used in the report
models/     fold checkpoints (*.pt, not tracked)
run_experiments.command   runs all CNN experiments on macOS (double-click)
run_sensitivity.command   ResNet18 with the alternative grouping (125 groups)
```

## Setup

```
conda create -n pv-health python=3.10
conda activate pv-health
pip install -r requirements.txt
```

## Data

The PV Module Dataset (6 class folders, 525 images each) was provided by the supervisor and is
not included in this repository. Place it in a folder `PV Module Dataset/` three levels above
the project folder, or pass `--data DIR` to `src/build_index.py`.

## Reproducing the results

Run from the project folder, in this order:

```
python3 src/build_index.py            # MD5, pHash, original size, 224x224 cache
python3 src/merge_groups.py           # merge squeezed copies -> 150 panel groups
python3 src/dataset.py                # panel-grouped train/val/test manifest (70/15/15, seed 42)
python3 src/baseline_model.py         # colour features + Random Forest, hold-out and CV
python3 src/analysis_baseline_cv.py   # baseline error analysis, dataset structure, example figure
python3 src/size_shortcut.py          # image-size shortcut check
python3 src/train_cnn.py              # ResNet18 on the hold-out split
python3 src/cnn_cv.py --model resnet18
python3 src/cnn_cv.py --model mobilenet_v3_small
python3 src/cnn_cv.py --model efficientnet_b0
python3 src/gradcam.py                # Grad-CAM for the ResNet18 fold models
python3 src/merge_sensitivity.py      # grouping sensitivity (merge threshold)
python3 src/baseline_alt_grouping.py  # baseline with 125 groups
python3 src/cnn_cv.py --model resnet18 --group-file data/processed/group_merge_t0995.csv --tag t0995
```

The CNN scripts use the Apple MPS GPU when available and otherwise the CPU. The first run
downloads the ImageNet weights through torchvision.

## Notes

* The panel grouping is a heuristic (image similarity, threshold 0.999) and was not checked
  against metadata from the data provider. `reports/merge_sensitivity.json` shows how the
  number of groups depends on the threshold (150 groups at 0.999, 125 at 0.995). With the 125 groups
  the Random Forest reaches 0.991 ± 0.012 and ResNet18 1.000 (`reports/sensitivity_t0995_baseline.json`,
  `reports/cnn_cv_resnet18_t0995.json`).
* `reports/` contains the final results. Earlier results from a first grouping (156 groups)
  were superseded and are not part of the repository.
