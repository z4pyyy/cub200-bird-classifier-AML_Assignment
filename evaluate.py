"""
evaluate.py - Run a trained model on a data split and report the metrics.

Usage:
    python evaluate.py runs/resnet50                # test set, plain
    python evaluate.py runs/resnet50 --tta          # + test-time augmentation
    python evaluate.py runs/resnet50 --split val

Outputs (written into the run folder):
    test_metrics.json      Top-1, average per-class accuracy, Top-5
    test_per_class.csv     accuracy of every class (for best/worst class analysis)
    test_logits.npz        raw model scores (for confusion analysis / ensembles)
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from data import ROOT, get_loaders, read_annotations
from metrics import average_class_accuracy, per_class_accuracy, top1_accuracy, topk_accuracy
from models import build_model


@torch.no_grad()  # no gradients needed for evaluation: saves memory and time
def predict(model, loader, device, tta=False):
    """Return (logits, targets) for every image in the loader.

    tta=True: test-time augmentation. We also run the horizontally flipped
    image and average the two softmax outputs. A flipped bird is still the
    same bird, so averaging two "views" gives a slightly more stable prediction.
    """
    model.eval()  # eval mode: dropout off, BatchNorm uses its stored statistics
    all_logits, all_targets = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            out = model(x).float().softmax(1)
            if tta:
                out = (out + model(torch.flip(x, dims=[3])).float().softmax(1)) / 2
        # Store log-probabilities: argmax is unchanged and they can be averaged later.
        all_logits.append(out.clamp_min(1e-8).log().cpu())
        all_targets.append(y)
    return torch.cat(all_logits).numpy(), torch.cat(all_targets).numpy()


def compute_metrics(logits, targets):
    preds = logits.argmax(1)
    return {
        "top1": top1_accuracy(preds, targets),
        "avg_class_acc": average_class_accuracy(preds, targets),
        "top5": topk_accuracy(logits, targets, k=5),
    }


def class_names():
    """Recover readable species names from the training file names.

    e.g. 'Black_footed_Albatross_0019_416160254.jpg' -> 'Black footed Albatross'
    """
    names, labels = read_annotations(ROOT / "train.txt")
    out = {}
    for n, y in zip(names, labels):
        if y not in out:
            out[y] = " ".join(n.split("_")[:-2])
    return [out[i] for i in range(len(out))]


def load_run(run_dir, device):
    """Rebuild the model from a run folder and load its best weights."""
    ckpt = torch.load(Path(run_dir) / "best.pt", map_location=device, weights_only=False)
    model, _ = build_model(ckpt["args"]["model"], drop=ckpt["args"]["drop"], pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).to(memory_format=torch.channels_last), ckpt["args"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_dir")
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--tta", action="store_true")
    p.add_argument("--workers", type=int, default=12)
    a = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, args = load_run(a.run_dir, device)
    _, val_loader, test_loader = get_loaders(img_size=args["img_size"], batch_size=64,
                                             num_workers=a.workers)
    loader = test_loader if a.split == "test" else val_loader

    logits, targets = predict(model, loader, device, tta=a.tta)
    m = compute_metrics(logits, targets)
    tag = f"{a.split}{'_tta' if a.tta else ''}"
    print(f"{a.run_dir} [{tag}]  top1 {m['top1']:.4f}  avg-class {m['avg_class_acc']:.4f}  top5 {m['top5']:.4f}")

    run = Path(a.run_dir)
    (run / f"{tag}_metrics.json").write_text(json.dumps(m, indent=2))
    np.savez(run / f"{tag}_logits.npz", logits=logits, targets=targets)
    pd.DataFrame({
        "class_id": range(200),
        "species": class_names(),
        "accuracy": per_class_accuracy(logits.argmax(1), targets),
        "n_images": np.bincount(targets, minlength=200),
    }).to_csv(run / f"{tag}_per_class.csv", index=False)


if __name__ == "__main__":
    main()
