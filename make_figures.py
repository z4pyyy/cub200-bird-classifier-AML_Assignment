"""
make_figures.py - All report figures and tables, from the saved runs.

Usage:
    python make_figures.py --best runs/m4_convnext_tiny

Writes into figures/:
    fig_models.png      model ladder: test Top-1 and average per-class accuracy
    fig_curves.png      (a) validation Top-1 per epoch, (b) overfitting gap with / without augmentation
    fig_ablation.png    change in test Top-1 when one technique is removed or added (vs. M3)
    fig_classes.png     the 10 hardest and 10 easiest species for the best model
    fig_cam.png         class activation maps: correct and wrong test predictions
    results.csv         one row per run (for the report tables)
    confusions.csv      most frequent (true -> predicted) mistakes of the best model
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # render to files, no window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image

from data import ROOT, build_transforms, read_annotations
from evaluate import class_names, load_run
from export_onnx import DeployWrapper
from metrics import per_class_accuracy

# Validated categorical palette (fixed order) + chart ink, light mode (print).
S = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
NEG = "#e34948"  # diverging: worse than reference
POS = "#2a78d6"  # diverging: better than reference

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "DejaVu Sans"],
    "font.size": 9, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "figure.dpi": 200,
    "savefig.bbox": "tight", "axes.titlesize": 10, "axes.titleweight": "bold",
    "axes.titlecolor": INK, "axes.titlelocation": "left",
})

LADDER = [  # (run folder, label in figures)
    ("m1_scratch_cnn", "M1 Scratch CNN"),
    ("m2_resnet50_frozen", "M2 ResNet-50 frozen"),
    ("m3_resnet50", "M3 ResNet-50 fine-tuned"),
    ("m4_efficientnet_b3", "M4 EfficientNet-B3"),
    ("m4_convnext_tiny", "M4 ConvNeXt-Tiny"),
    ("m4_convnext_tiny_mixcut", "M4 ConvNeXt-Tiny + MixUp/CutMix"),
]
ABLATIONS = [  # (run folder, what changed vs. M3)
    ("abl_r50_no_aug", "No augmentation"),
    ("abl_r50_light_aug", "Light augmentation only\n(crop + flip)"),
    ("abl_r50_no_ls", "No label smoothing"),
    ("abl_r50_same_lr", "Same LR for backbone\nand head"),
    ("abl_r50_mixcut", "+ MixUp / CutMix"),
]


def load_metrics(run):
    p = ROOT / "runs" / run / "test_metrics.json"
    return json.loads(p.read_text()) if p.exists() else None


def results_table(out):
    rows = []
    for run, label in LADDER + ABLATIONS:
        m = load_metrics(run)
        if m:
            rows.append({"run": run, "label": label.replace("\n", " "), **m})
    df = pd.DataFrame(rows)
    df.to_csv(out / "results.csv", index=False)
    return df


def fig_models(out):
    runs = [(r, l) for r, l in LADDER if load_metrics(r)]
    labels = [l for _, l in runs][::-1]
    top1 = [load_metrics(r)["top1"] * 100 for r, _ in runs][::-1]
    avg = [load_metrics(r)["avg_class_acc"] * 100 for r, _ in runs][::-1]
    y = np.arange(len(runs))
    h = 0.36
    fig, ax = plt.subplots(figsize=(6.2, 0.55 * len(runs) + 0.9))
    b1 = ax.barh(y + h / 2 + 0.01, top1, h, color=S[0], label="Top-1 accuracy")
    b2 = ax.barh(y - h / 2 - 0.01, avg, h, color=S[1], label="Average per-class accuracy")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_width() + 0.6, b.get_y() + b.get_height() / 2, f"{b.get_width():.1f}",
                    va="center", fontsize=8, color=INK2)
    ax.set_yticks(y, labels, color=INK)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Test accuracy (%)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper right", frameon=False, fontsize=8)
    ax.set_title("Model ladder: test accuracy (1,204 images, 200 classes)")
    fig.savefig(out / "fig_models.png")
    plt.close(fig)


def fig_curves(out):
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.9))
    # (a) validation Top-1 per epoch, one line per model, labelled at the line end
    short = {"m1_scratch_cnn": "M1 Scratch CNN", "m2_resnet50_frozen": "M2 ResNet-50 frozen",
             "m3_resnet50": "M3 ResNet-50", "m4_efficientnet_b3": "M4 EfficientNet-B3",
             "m4_convnext_tiny": "M4 ConvNeXt-Tiny"}
    for (run, label), color in zip(LADDER[:5], S):
        p = ROOT / "runs" / run / "history.csv"
        if not p.exists():
            continue
        h = pd.read_csv(p)
        a.plot(h.epoch, h.val_top1 * 100, color=color, lw=1.6, label=short[run])
    a.legend(frameon=False, fontsize=7, loc="lower right")
    a.set_xlabel("Epoch"); a.set_ylabel("Validation Top-1 (%)")
    a.set_ylim(0, 100)
    a.set_title("(a) Validation accuracy per epoch")

    # (b) overfitting gap: train vs. validation accuracy, ResNet-50 with and without augmentation
    for run, label, color in [("m3_resnet50", "With augmentation (M3)", S[0]),
                              ("abl_r50_no_aug", "No augmentation", S[1])]:
        p = ROOT / "runs" / run / "history.csv"
        if not p.exists():
            continue
        h = pd.read_csv(p)
        b.plot(h.epoch, h.train_acc * 100, color=color, lw=1.6, label=f"{label}: train")
        b.plot(h.epoch, h.val_top1 * 100, color=color, lw=1.6, ls="--", label=f"{label}: validation")
    b.set_xlabel("Epoch"); b.set_ylabel("Accuracy (%)")
    b.set_ylim(0, 102)
    b.legend(frameon=False, fontsize=7, loc="lower right")
    b.set_title("(b) Train vs. validation (ResNet-50)")
    fig.tight_layout()
    fig.savefig(out / "fig_curves.png")
    plt.close(fig)


def fig_ablation(out):
    ref = load_metrics("m3_resnet50")["top1"] * 100
    items = [(l, load_metrics(r)["top1"] * 100 - ref) for r, l in ABLATIONS if load_metrics(r)]
    labels = [l for l, _ in items][::-1]
    deltas = [d for _, d in items][::-1]
    fig, ax = plt.subplots(figsize=(6.2, 0.5 * len(items) + 0.9))
    y = np.arange(len(items))
    bars = ax.barh(y, deltas, 0.55, color=[POS if d >= 0 else NEG for d in deltas])
    for bar, d in zip(bars, deltas):
        ax.text(d + (0.12 if d >= 0 else -0.12), bar.get_y() + bar.get_height() / 2, f"{d:+.1f}",
                va="center", ha="left" if d >= 0 else "right", fontsize=8, color=INK2)
    ax.axvline(0, color=AXIS, lw=1)
    ax.set_yticks(y, labels, color=INK)
    lim = max(abs(min(deltas)), abs(max(deltas))) + 1.2
    ax.set_xlim(-lim, lim)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(f"Change in test Top-1 vs. M3 reference ({ref:.1f}%), percentage points")
    ax.set_title("Ablation: one change at a time (ResNet-50)")
    fig.savefig(out / "fig_ablation.png")
    plt.close(fig)


@torch.no_grad()
def best_model_outputs(best):
    """Run the best model (via the CAM wrapper) on every test image."""
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, args = load_run(best, dev)
    wrap = DeployWrapper(model.to(memory_format=torch.contiguous_format)).eval()
    _, tf = build_transforms(args["img_size"])
    names, labels = read_annotations(ROOT / "test.txt")
    probs, cams = [], []
    for i in range(0, len(names), 32):
        x = torch.stack([tf(Image.open(ROOT / "Test" / n).convert("RGB")) for n in names[i:i + 32]]).to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            p, c = wrap(x)
        probs.append(p.float().cpu())
        cams.append(c.float().cpu())
    return names, np.array(labels), torch.cat(probs).numpy(), torch.cat(cams).numpy(), args["img_size"]


def fig_classes(out, labels, probs, cls):
    acc = per_class_accuracy(probs.argmax(1), labels) * 100
    order = np.argsort(acc)
    worst, best = order[:10], order[::-1][:10]
    n100 = int((acc == 100).sum())
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.4, 3.0))
    easy_title = f"10 of the {n100} species at 100%" if n100 > 10 else "10 easiest species"
    for ax, idx, color, title in [(a, worst, NEG, "10 hardest species"), (b, best, POS, easy_title)]:
        y = np.arange(len(idx))[::-1]
        ax.barh(y, acc[idx], 0.6, color=color)
        for yy, v in zip(y, acc[idx]):
            ax.text(v + 1.5, yy, f"{v:.0f}", va="center", fontsize=7.5, color=INK2)
        ax.set_yticks(y, [cls[i] for i in idx], fontsize=7.5, color=INK)
        ax.set_xlim(0, 112)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("Test accuracy (%)")
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out / "fig_classes.png")
    plt.close(fig)


def confusions(out, labels, probs, cls):
    pred = probs.argmax(1)
    wrong = pred != labels
    pairs = pd.DataFrame({"true": [cls[i] for i in labels[wrong]], "predicted": [cls[i] for i in pred[wrong]]})
    tab = pairs.value_counts().reset_index(name="count")
    tab.to_csv(out / "confusions.csv", index=False)
    return tab


def overlay(img_path, cam, size):
    """Same view the web app shows: centre crop + jet CAM overlay."""
    _, tf_eval = build_transforms(size)
    im = Image.open(img_path).convert("RGB")
    r = int(size * 256 / 224)
    im = im.resize((round(im.width * r / min(im.size)), round(im.height * r / min(im.size))), Image.BILINEAR)
    left, top = (im.width - size) // 2, (im.height - size) // 2
    im = np.asarray(im.crop((left, top, left + size, top + size))) / 255.0
    c = np.maximum(cam, 0)
    c = c / (c.max() + 1e-8)
    c = np.asarray(Image.fromarray((c * 255).astype(np.uint8)).resize((size, size), Image.BICUBIC)) / 255.0
    heat = plt.get_cmap("jet")(c)[..., :3]
    return 0.55 * im + 0.45 * heat


def fig_cam(out, names, labels, probs, cams, size, cls):
    pred = probs.argmax(1)
    conf = probs.max(1)
    rng = np.random.default_rng(3)
    right = rng.choice(np.where(pred == labels)[0], 4, replace=False)
    wrong = np.where(pred != labels)[0]
    wrong = wrong[np.argsort(-conf[wrong])][:4]  # the most confident mistakes
    fig, axes = plt.subplots(2, 4, figsize=(7.4, 4.9))
    for row, idxs, tag in [(0, right, "Correct"), (1, wrong, "Wrong")]:
        for ax, i in zip(axes[row], idxs):
            ax.imshow(overlay(ROOT / "Test" / names[i], cams[i, pred[i]], size))
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            for s in ax.spines.values():
                s.set_visible(False)
            if tag == "Correct":
                ax.set_title(f"{cls[pred[i]]}\n{conf[i] * 100:.0f}%", fontsize=7, fontweight="normal", color=INK, loc="center")
            else:
                ax.set_title(f"Pred: {cls[pred[i]]} ({conf[i] * 100:.0f}%)\nTrue: {cls[labels[i]]}",
                             fontsize=7, fontweight="normal", color=INK, loc="center")
        axes[row, 0].set_ylabel(tag, fontsize=9, color=INK, fontweight="bold")
    fig.suptitle("Class activation maps of the best model (red = most evidence)", fontsize=10,
                 fontweight="bold", color=INK, x=0.02, ha="left")
    fig.tight_layout(h_pad=2.2)
    fig.savefig(out / "fig_cam.png")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--best", default="runs/m4_convnext_tiny")
    a = p.parse_args()
    out = ROOT / "figures"
    out.mkdir(exist_ok=True)

    df = results_table(out)
    print(df[["label", "top1", "avg_class_acc", "top5", "best_epoch", "epochs_run"]].to_string(index=False))
    fig_models(out)
    fig_curves(out)
    fig_ablation(out)

    cls = class_names()
    names, labels, probs, cams, size = best_model_outputs(a.best)
    fig_classes(out, labels, probs, cls)
    tab = confusions(out, labels, probs, cls)
    print("\nMost frequent confusions:\n", tab.head(10).to_string(index=False))
    fig_cam(out, names, labels, probs, cams, size, cls)
    print(f"\nfigures written to {out}")


if __name__ == "__main__":
    main()
