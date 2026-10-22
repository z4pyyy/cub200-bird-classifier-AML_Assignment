"""
overfit_test.py - Measure overfitting of every trained model.

Overfitting = the model fits the images it trained on much better than new images.
To measure it fairly we evaluate each best checkpoint on three splits with the SAME
clean evaluation transform (resize + centre crop, no augmentation):

    train (4,104 images the model learned from)
    val   (725 held-out images, used for early stopping / model choice)
    test  (1,204 images, never used for any decision)

The training accuracy logged during training is NOT usable for this: it was measured
on augmented (harder) images, and it is missing for MixUp/CutMix runs.

Reported per model:
    gap = train Top-1 - test Top-1    (the generalisation gap; large = overfitting)
    NLL = cross-entropy without label smoothing (how confident and right the model is)
    conf = mean top-1 probability     (over-confidence shows as conf >> accuracy)

Usage:
    python overfit_test.py                 # all runs in runs/
    python overfit_test.py runs/m3_resnet50
Output: figures/overfit_test.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from data import ROOT, BirdDataset, build_transforms, read_annotations, split_train_val
from evaluate import compute_metrics, load_run, predict


def clean_loaders(img_size):
    """Train / val / test loaders that all use the evaluation transform."""
    _, eval_tf = build_transforms(img_size, augment=False)
    names, labels = read_annotations(ROOT / "train.txt")
    (tr_n, tr_y), (va_n, va_y) = split_train_val(names, labels)  # same split, seed 42
    te_n, te_y = read_annotations(ROOT / "test.txt")
    sets = {
        "train": BirdDataset(ROOT / "Train", tr_n, tr_y, eval_tf),
        "val": BirdDataset(ROOT / "Train", va_n, va_y, eval_tf),
        "test": BirdDataset(ROOT / "Test", te_n, te_y, eval_tf),
    }
    return {k: DataLoader(ds, batch_size=64, num_workers=8, pin_memory=True) for k, ds in sets.items()}


def main():
    run_dirs = [Path(p) for p in sys.argv[1:]] or sorted(p.parent for p in Path("runs").glob("*/best.pt"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for run in run_dirs:
        model, args = load_run(run, device)
        row = {"run": run.name}
        for split, loader in clean_loaders(args["img_size"]).items():
            logp, y = predict(model, loader, device)  # log-probabilities
            m = compute_metrics(logp, y)
            row[f"{split}_top1"] = m["top1"]
            row[f"{split}_nll"] = float(-logp[np.arange(len(y)), y].mean())
            row[f"{split}_conf"] = float(np.exp(logp.max(1)).mean())
        row["gap_train_test"] = row["train_top1"] - row["test_top1"]
        row["gap_val_test"] = row["val_top1"] - row["test_top1"]  # selection bias check
        rows.append(row)
        print(f"{run.name:26s} train {row['train_top1']:.3f}  val {row['val_top1']:.3f}  "
              f"test {row['test_top1']:.3f}  gap {row['gap_train_test']:+.3f}  "
              f"NLL tr/te {row['train_nll']:.2f}/{row['test_nll']:.2f}  "
              f"conf te {row['test_conf']:.3f}", flush=True)
        del model
        torch.cuda.empty_cache()
    Path("figures").mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv("figures/overfit_test.csv", index=False)


if __name__ == "__main__":
    main()
