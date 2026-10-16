#!/usr/bin/env bash
# Day 1 runs: the 3 baseline rungs of the model ladder (same split, same seed).
set -e
cd "$(dirname "$0")"
PY=./.venv/Scripts/python
$PY train.py --model scratch_cnn     --epochs 100 --lr 1e-3 --patience 25 --name m1_scratch_cnn
$PY train.py --model resnet50_frozen --epochs 30  --lr 3e-3 --name m2_resnet50_frozen
$PY train.py --model resnet50        --epochs 40  --lr 1e-3 --name m3_resnet50
