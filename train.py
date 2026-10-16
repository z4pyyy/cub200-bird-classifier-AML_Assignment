"""
train.py - Training scheme for all models.

Usage examples:
    python train.py --model scratch_cnn --epochs 80 --lr 1e-3 --name m1_scratch
    python train.py --model resnet50_frozen --epochs 30 --lr 1e-3 --name m2_resnet50_frozen
    python train.py --model resnet50 --epochs 40 --lr 1e-3 --name m3_resnet50
    python train.py --model convnext_tiny --epochs 40 --lr 4e-4 --name m4_convnext_tiny

One training epoch:
    for each batch:
        1. (optional) MixUp / CutMix the batch
        2. forward pass under mixed precision -> logits
        3. loss = cross-entropy (with label smoothing / soft targets)
        4. backward pass -> gradients
        5. AdamW step updates the weights; scheduler updates the learning rate
    then evaluate on the validation split, keep the best weights, maybe stop early.

Outputs in runs/<name>/:
    best.pt        weights of the epoch with the best validation Top-1
    history.csv    per-epoch losses / accuracies / learning rate (for the curves)
    config.json    every hyperparameter of this run (reproducibility)
    test_metrics.json  final test result of best.pt (test used once, at the end)
"""

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy

from data import ROOT, get_loaders
from evaluate import compute_metrics, predict
from models import build_model, count_params


def parse_args():
    p = argparse.ArgumentParser(description="Train a CNN on CUB-200")
    p.add_argument("--model", required=True,
                   help="scratch_cnn | resnet50_frozen | resnet50 | convnext_tiny | efficientnet_b3")
    p.add_argument("--name", default=None, help="run folder name (default: model name)")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--bs", type=int, default=32, help="batch size")
    p.add_argument("--lr", type=float, default=1e-3, help="learning rate of the new head")
    p.add_argument("--backbone-lr-mult", type=float, default=0.1,
                   help="backbone LR = lr * this. Pretrained layers get a smaller LR.")
    p.add_argument("--wd", type=float, default=0.05, help="AdamW weight decay")
    p.add_argument("--warmup", type=int, default=3, help="warmup epochs")
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--drop", type=float, default=0.3, help="dropout before the classifier")
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--mixup", type=float, default=0.0, help="MixUp alpha (0 = off)")
    p.add_argument("--cutmix", type=float, default=0.0, help="CutMix alpha (0 = off)")
    p.add_argument("--no-aug", action="store_true", help="ablation: no data augmentation")
    p.add_argument("--light-aug", action="store_true", help="ablation: only crop + flip")
    p.add_argument("--patience", type=int, default=10, help="early stopping patience (epochs)")
    p.add_argument("--full-train", action="store_true", help="train on train+val (final model only)")
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def set_seed(seed):
    """Fix all random number generators so a run can be repeated."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_optimizer(model, head_names, lr, backbone_mult, wd):
    """AdamW with 4 parameter groups.

    1. Discriminative learning rates: the new head starts from random weights
       and needs a large LR. The pretrained backbone already holds good
       features; a large LR would destroy them ("catastrophic forgetting"),
       so it gets lr * backbone_mult (default 10x smaller).
    2. Weight decay (L2-style regularisation, AdamW's decoupled version) is
       applied to conv / linear weights only. Biases and normalisation
       parameters are excluded, which is standard practice: decaying them
       does not reduce overfitting and can hurt accuracy.
    """
    groups = {(h, d): [] for h in (True, False) for d in (True, False)}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue  # frozen backbone weights (M2) are not optimised at all
        is_head = n in head_names
        use_decay = p.ndim > 1  # weights of conv/linear are >=2-D; bias / norm are 1-D
        groups[(is_head, use_decay)].append(p)

    param_groups = []
    for (is_head, use_decay), params in groups.items():
        if params:
            param_groups.append({
                "params": params,
                "lr": lr if is_head else lr * backbone_mult,
                "weight_decay": wd if use_decay else 0.0,
            })
    return torch.optim.AdamW(param_groups)


def make_scheduler(optimizer, warmup_steps, total_steps):
    """Linear warmup, then cosine decay to ~0 (updated every batch).

    Warmup: the first steps use a small LR that rises linearly. At the start
    the head is random and produces large, noisy gradients; a full LR at this
    point could damage the pretrained backbone.
    Cosine decay: the LR falls smoothly. Large steps early to learn fast,
    tiny steps at the end to settle into a good minimum.
    """
    def factor(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def main():
    args = parse_args()
    set_seed(args.seed)
    name = args.name or args.model
    run_dir = ROOT / "runs" / name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(vars(args), indent=2))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Let cuDNN pick the fastest convolution algorithm for our fixed input size.
    torch.backends.cudnn.benchmark = True

    # ---------------- data ----------------
    train_loader, val_loader, test_loader = get_loaders(
        img_size=args.img_size, batch_size=args.bs, augment=not args.no_aug,
        strong=not args.light_aug, num_workers=args.workers, seed=args.seed,
        full_train=args.full_train)

    # ---------------- model ----------------
    model, head_names = build_model(args.model, drop=args.drop)
    # channels_last memory layout: faster convolutions on modern NVIDIA GPUs.
    model = model.to(device).to(memory_format=torch.channels_last)
    total, trainable = count_params(model)
    print(f"{args.model}: {total / 1e6:.1f}M params, {trainable / 1e6:.2f}M trainable")
    frozen_backbone = trainable < total

    # ---------------- loss ----------------
    # MixUp blends two images and their labels (e.g. 70% bird A + 30% bird B).
    # CutMix pastes a patch of bird B onto bird A, with labels mixed by area.
    # Both act as strong regularisers: the model cannot be 100% confident on
    # any training image, so it overfits much less on small datasets.
    mixup_fn = None
    if args.mixup > 0 or args.cutmix > 0:
        mixup_fn = Mixup(mixup_alpha=args.mixup, cutmix_alpha=args.cutmix,
                         label_smoothing=args.label_smoothing, num_classes=200)
        # Mixed labels are probability vectors ("soft targets"), so the loss
        # is cross-entropy against a soft target distribution.
        criterion = SoftTargetCrossEntropy()
    else:
        # Label smoothing: the target is 0.9 for the true class and the
        # remaining 0.1 is spread over the other classes. It stops the model
        # pushing its logits to extremes (over-confidence), which helps
        # generalisation, especially for similar-looking species.
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    # ---------------- optimiser + LR schedule ----------------
    optimizer = make_optimizer(model, head_names, args.lr, args.backbone_lr_mult, args.wd)
    steps_per_epoch = len(train_loader)
    scheduler = make_scheduler(optimizer, args.warmup * steps_per_epoch, args.epochs * steps_per_epoch)

    # ---------------- training loop ----------------
    history, best_top1, best_epoch, bad_epochs = [], -1.0, -1, 0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        # train() turns on dropout and BatchNorm batch statistics. For the
        # frozen model we keep eval(): the frozen BatchNorm layers must keep
        # their ImageNet statistics, otherwise the "frozen" features would drift.
        model.eval() if frozen_backbone else model.train()

        run_loss, run_correct, seen = 0.0, 0, 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            y = y.to(device, non_blocking=True)
            y_hard = y
            if mixup_fn is not None:
                x, y = mixup_fn(x, y)  # y becomes a soft label matrix [B, 200]

            # Mixed precision (bfloat16): about 2x faster and half the memory on
            # an RTX 3060, with no loss in accuracy. Weights stay in float32.
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model(x)
                loss = criterion(logits.float(), y)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            # Gradient clipping: caps rare, very large gradients that could
            # make one bad batch destroy the pretrained weights.
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            scheduler.step()

            run_loss += loss.item() * x.size(0)
            seen += x.size(0)
            if mixup_fn is None:  # training accuracy is meaningless on mixed images
                run_correct += (logits.argmax(1) == y_hard).sum().item()

        # ---- validation (held-out 15% of Train) ----
        val_logits, val_targets = predict(model, val_loader, device)
        vm = compute_metrics(val_logits, val_targets)
        val_loss = nn.functional.nll_loss(torch.from_numpy(val_logits), torch.from_numpy(val_targets)).item()

        row = {
            "epoch": epoch,
            "train_loss": run_loss / seen,
            "train_acc": run_correct / seen if mixup_fn is None else float("nan"),
            "val_loss": val_loss,
            "val_top1": vm["top1"],
            "val_avg_class_acc": vm["avg_class_acc"],
            "lr": optimizer.param_groups[0]["lr"],
            "time_s": time.time() - t0,
        }
        history.append(row)
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
        print(f"ep {epoch:3d} | train loss {row['train_loss']:.3f} acc {row['train_acc']:.3f} | "
              f"val loss {val_loss:.3f} top1 {vm['top1']:.3f} avg {vm['avg_class_acc']:.3f} | "
              f"{row['time_s']:.0f}s", flush=True)

        # ---- checkpoint + early stopping ----
        # Keep the weights of the epoch with the best validation Top-1.
        # Early stopping: if validation Top-1 has not improved for `patience`
        # epochs, the model is only memorising the training set -> stop.
        if vm["top1"] > best_top1:
            best_top1, best_epoch, bad_epochs = vm["top1"], epoch, 0
            torch.save({"state_dict": model.state_dict(), "args": vars(args), "epoch": epoch},
                       run_dir / "best.pt")
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"Early stopping at epoch {epoch} (best epoch {best_epoch})")
                break

    # ---------------- final test evaluation (once) ----------------
    ckpt = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    test_logits, test_targets = predict(model, test_loader, device)
    tm = compute_metrics(test_logits, test_targets)
    tm.update({"best_epoch": best_epoch, "best_val_top1": best_top1,
               "epochs_run": len(history), "params_M": total / 1e6,
               "trainable_M": trainable / 1e6,
               "train_time_min": sum(r["time_s"] for r in history) / 60})
    (run_dir / "test_metrics.json").write_text(json.dumps(tm, indent=2))
    print(f"TEST  top1 {tm['top1']:.4f}  avg-class {tm['avg_class_acc']:.4f}  top5 {tm['top5']:.4f}")


if __name__ == "__main__":  # required on Windows for multi-process DataLoader workers
    main()
