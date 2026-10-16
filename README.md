# CUB-200 Bird Species Classifier

COS30082 Applied Machine Learning, Assignment 1.
Terence Chin Seng Wong (104404059).

**Live demo:** https://z4pyyy.github.io/cub200-bird-classifier-AML_Assignment/

The demo runs a CNN in your browser with ONNX Runtime Web. Your photo is never uploaded.

## Models compared

| Model | Idea |
|---|---|
| M1 `scratch_cnn` | Small VGG-style CNN, random initial weights |
| M2 `resnet50_frozen` | ImageNet ResNet-50, frozen backbone, new linear head |
| M3 `resnet50` | ImageNet ResNet-50, fully fine-tuned |
| M4 `convnext_tiny`, `efficientnet_b3` | Modern pretrained CNNs, fully fine-tuned |

Results for each run are in `runs/<name>/test_metrics.json` and `runs/<name>/history.csv`.

## Files

| File | Purpose |
|---|---|
| `data.py` | Annotations, stratified train/val split, augmentation, DataLoaders |
| `models.py` | Model definitions (scratch CNN + timm pretrained CNNs) |
| `train.py` | Training loop: AdamW, warmup + cosine LR, label smoothing, MixUp/CutMix, early stopping |
| `evaluate.py` | Test evaluation, optional test-time augmentation, per-class CSV |
| `metrics.py` | Top-1 accuracy, average per-class accuracy, top-5 accuracy |
| `export_onnx.py` | Export to ONNX (prediction + class activation map) with a PyTorch parity check |
| `web/` | The static web app published to GitHub Pages |
| `run_day1.sh`, `run_day2.sh` | The exact commands for every run |

## Reproduce

1. Put the course dataset in this folder: `Train/`, `Test/`, `train.txt`, `test.txt`. The dataset is not in this repo.
2. Install PyTorch with CUDA, then `pip install -r requirements.txt`.
3. Train: `python train.py --model resnet50 --epochs 40 --lr 1e-3 --name m3_resnet50`
4. Evaluate: `python evaluate.py runs/m3_resnet50 --tta`
5. Export for the web: `python export_onnx.py runs/m3_resnet50 --fp16-weights`

## Data note

The reference photos in `web/refs/` and the examples in `web/examples/` are small thumbnails from the
[Caltech-UCSD Birds 200](http://www.vision.caltech.edu/visipedia/CUB-200.html) dataset, used for the demo only.
