"""
data.py - Data input, pre-processing and data loaders for CUB-200.

Pipeline:
    train.txt / test.txt  ->  (filename, label) lists
    train list            ->  stratified split into train (85%) and validation (15%)
    each split            ->  BirdDataset (opens image, applies transforms)
    each dataset          ->  DataLoader (batches, shuffles, loads in parallel workers)

Why a validation split?
    The test set must only be used ONCE at the end, to report final results.
    If we pick hyperparameters or the "best epoch" by looking at test accuracy,
    the test score becomes optimistic (we would be overfitting to the test set).
    So we hold out 15% of the training images as a validation set and use it
    for all model selection decisions (early stopping, comparing settings).
"""

from pathlib import Path

import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# Project root = folder that contains this file.
ROOT = Path(__file__).resolve().parent

# ImageNet statistics. Pretrained CNNs were trained on images normalised with
# these values, so we must normalise our images the same way for the pretrained
# features to make sense. (The scratch CNN also uses them; it does not hurt.)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

NUM_CLASSES = 200


def read_annotations(txt_path):
    """Read an annotation file. Each line is: '<image name> <class label>'.

    Returns two lists: file names and integer labels.
    """
    names, labels = [], []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:  # skip blank lines (e.g. trailing newline)
                continue
            name, label = line.rsplit(" ", 1)  # rsplit: safe even if a name had spaces
            names.append(name)
            labels.append(int(label))
    return names, labels


def build_transforms(img_size=224, augment=True, strong=True):
    """Return (train_transform, eval_transform).

    Train transform = data augmentation. Every epoch the network sees a slightly
    different version of each image (different crop, flip, colours). With only
    ~24 images per class, this is our main defence against overfitting: the model
    cannot simply memorise the exact pixels of each training image.

    Eval transform = deterministic. Resize, then centre crop. No randomness, so
    validation / test scores are repeatable.

    augment=False gives a "no augmentation" baseline for the ablation study.
    strong=False keeps only crop + flip (a "light augmentation" ablation).
    """
    resize = int(img_size * 256 / 224)  # standard ratio: resize to 256, crop 224

    eval_tf = transforms.Compose([
        transforms.Resize(resize),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),  # PIL image [0,255] -> float tensor [0,1], shape C x H x W
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    if not augment:
        return eval_tf, eval_tf

    train_steps = [
        # Random crop covering 35%-100% of the image, then resize to img_size.
        # Forces the model to recognise the bird from partial views and at
        # different scales. The lower bound is not too small, so the bird is
        # usually still visible in the crop.
        transforms.RandomResizedCrop(img_size, scale=(0.35, 1.0)),
        # A bird facing left is the same species as a bird facing right.
        transforms.RandomHorizontalFlip(),
    ]
    if strong:
        train_steps += [
            # Small colour changes simulate different lighting / cameras.
            # Hue is kept very small because plumage colour is a key feature
            # for telling species apart (e.g. a red vs. an orange breast).
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.02),
            # Small rotations: birds are photographed at slight angles.
            transforms.RandomRotation(15),
        ]
    train_steps += [
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
    if strong:
        # Random Erasing: blank out a random rectangle (after normalisation).
        # Simulates occlusion by branches / leaves and stops the model relying
        # on one single body part.
        train_steps.append(transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)))

    return transforms.Compose(train_steps), eval_tf


class BirdDataset(Dataset):
    """A PyTorch Dataset: returns (image_tensor, label) for a given index.

    Images are opened lazily (inside __getitem__), so we never hold all 6,000
    images in memory. The DataLoader calls __getitem__ from several worker
    processes in parallel, which keeps the GPU busy.
    """

    def __init__(self, image_dir, names, labels, transform):
        self.image_dir = Path(image_dir)
        self.names = list(names)
        self.labels = list(labels)
        self.transform = transform

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        path = self.image_dir / self.names[idx]
        # convert("RGB"): a few CUB images are greyscale or have an alpha
        # channel. The CNN expects exactly 3 channels.
        img = Image.open(path).convert("RGB")
        return self.transform(img), self.labels[idx]


def split_train_val(names, labels, val_frac=0.15, seed=42):
    """Stratified split: every class keeps the same train/val proportion.

    Stratification matters here because each class only has 16-31 images.
    A plain random split could leave some class with 0 validation images.
    The fixed seed makes the split identical in every run, so all models are
    compared on exactly the same validation images.
    """
    tr_n, va_n, tr_y, va_y = train_test_split(
        names, labels, test_size=val_frac, stratify=labels, random_state=seed
    )
    return (tr_n, tr_y), (va_n, va_y)


def get_loaders(img_size=224, batch_size=32, augment=True, strong=True,
                num_workers=6, val_frac=0.15, seed=42, full_train=False):
    """Build train / validation / test DataLoaders.

    full_train=True trains on ALL training images (train + val). It is used
    only after we fixed the hyperparameters on the validation split; then the
    validation loader is still returned but it is no longer a held-out set.
    """
    train_tf, eval_tf = build_transforms(img_size, augment, strong)

    names, labels = read_annotations(ROOT / "train.txt")
    (tr_n, tr_y), (va_n, va_y) = split_train_val(names, labels, val_frac, seed)
    if full_train:
        tr_n, tr_y = names, labels

    te_n, te_y = read_annotations(ROOT / "test.txt")

    train_ds = BirdDataset(ROOT / "Train", tr_n, tr_y, train_tf)
    val_ds = BirdDataset(ROOT / "Train", va_n, va_y, eval_tf)  # val images live in Train/
    test_ds = BirdDataset(ROOT / "Test", te_n, te_y, eval_tf)

    common = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),  # faster CPU -> GPU copies
        persistent_workers=num_workers > 0,    # keep workers alive between epochs (big win on Windows)
    )
    # drop_last=True on train: avoids a tiny last batch, which makes BatchNorm
    # statistics noisy (and MixUp needs an even batch size).
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **common)
    val_loader = DataLoader(val_ds, shuffle=False, **common)
    test_loader = DataLoader(test_ds, shuffle=False, **common)
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Quick self-check: python data.py
    tr, va, te = get_loaders(num_workers=0)
    print(f"train {len(tr.dataset)}  val {len(va.dataset)}  test {len(te.dataset)}")
    x, y = next(iter(tr))
    print("batch", tuple(x.shape), "labels", y[:8].tolist())
