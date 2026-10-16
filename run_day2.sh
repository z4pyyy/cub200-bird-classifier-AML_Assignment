#!/usr/bin/env bash
# Day 2 runs. Same split + seed as Day 1, so all numbers are comparable.
cd "$(dirname "$0")"
PY=./.venv/Scripts/python
# --- M4: modern pretrained CNNs (same recipe as M3) ---
$PY train.py --model convnext_tiny   --epochs 40 --lr 1e-3 --name m4_convnext_tiny
$PY train.py --model efficientnet_b3 --epochs 40 --lr 1e-3 --img-size 288 --name m4_efficientnet_b3
# --- Ablations on ResNet-50 (M3 = reference: strong aug + label smoothing 0.1 + 0.1x backbone LR) ---
$PY train.py --model resnet50 --epochs 40 --lr 1e-3 --no-aug               --name abl_r50_no_aug
$PY train.py --model resnet50 --epochs 40 --lr 1e-3 --light-aug            --name abl_r50_light_aug
$PY train.py --model resnet50 --epochs 40 --lr 1e-3 --label-smoothing 0    --name abl_r50_no_ls
$PY train.py --model resnet50 --epochs 40 --lr 1e-3 --backbone-lr-mult 1.0 --name abl_r50_same_lr
$PY train.py --model resnet50 --epochs 50 --lr 1e-3 --mixup 0.2 --cutmix 1.0 --patience 15 --name abl_r50_mixcut
# --- Best architecture + MixUp/CutMix ---
$PY train.py --model convnext_tiny --epochs 50 --lr 1e-3 --mixup 0.2 --cutmix 1.0 --patience 15 --name m4_convnext_tiny_mixcut
