"""
metrics.py - The two evaluation metrics required by the assignment.

Top-1 accuracy
    (1/N) * sum_k 1{ argmax(y_k) == ground_truth_k }
    Fraction of ALL test images whose highest-probability class is correct.

Average accuracy per class
    (1/C) * sum_i T_i,  where T_i = accuracy on the test images of class i.
    Every class counts equally, no matter how many test images it has.
    This is the fairer metric when classes have different sizes: in our test
    set classes have 4 to 8 images, so in Top-1 a class with 8 images counts
    twice as much as a class with 4.
"""

import numpy as np

from data import NUM_CLASSES


def top1_accuracy(preds, targets):
    preds, targets = np.asarray(preds), np.asarray(targets)
    return float((preds == targets).mean())


def per_class_accuracy(preds, targets, num_classes=NUM_CLASSES):
    """Return an array with the accuracy T_i of each class (NaN if a class has no images)."""
    preds, targets = np.asarray(preds), np.asarray(targets)
    acc = np.full(num_classes, np.nan)
    for c in range(num_classes):
        mask = targets == c
        if mask.any():
            acc[c] = (preds[mask] == c).mean()
    return acc


def average_class_accuracy(preds, targets, num_classes=NUM_CLASSES):
    # nanmean: ignore classes that have no images in this split.
    return float(np.nanmean(per_class_accuracy(preds, targets, num_classes)))


def topk_accuracy(logits, targets, k=5):
    """Top-5 accuracy: correct if the true class is among the k highest scores.

    Extra metric (not required). Useful for fine-grained birds, where the
    2nd guess is often a very similar-looking species.
    """
    logits, targets = np.asarray(logits), np.asarray(targets)
    topk = np.argsort(-logits, axis=1)[:, :k]
    return float((topk == targets[:, None]).any(axis=1).mean())
