"""
export_onnx.py - Export a trained model for the in-browser web app.

Usage:
    python export_onnx.py runs/m4_convnext_tiny            # writes into web/
    python export_onnx.py runs/m3_resnet50 --fp16-weights  # halve the file size

Why ONNX?
    ONNX is an open model format. ONNX Runtime Web can run an ONNX model
    directly inside the user's browser (WebAssembly / WebGPU). So the web app
    needs no Python server at all: GitHub Pages only serves static files, and
    the CNN runs on the visitor's own device. The photo never leaves it.

What is exported?
    The model is wrapped so that ONE forward pass returns two outputs:
        probs  [1, 200]        softmax class probabilities (the prediction)
        cam    [1, 200, h, w]  a class activation map for every class
    The web page picks the cam of the predicted class and overlays it on the
    photo, so the user sees WHERE the CNN looked.

How the CAM is computed (Class Activation Mapping, Zhou et al. 2016):
    A CNN classifier = feature extractor -> global average pool -> linear head.
    Average pooling and a linear layer commute, so applying the head to each
    spatial position of the feature map BEFORE pooling gives a score map whose
    average is exactly the final logit. High values = regions that pushed the
    prediction towards that class. We apply timm's own forward_head() to every
    position (as a 1x1 "image"), which works for ResNet, EfficientNet and
    ConvNeXt alike.

Parity check:
    After export we run ONNX Runtime and PyTorch on the same images and
    compare. This proves the deployed model is the model in the report.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
from onnx import helper, numpy_helper
from PIL import Image

from data import IMAGENET_MEAN, IMAGENET_STD, ROOT, read_annotations
from evaluate import class_names, compute_metrics, load_run, predict
from data import get_loaders
from models import ScratchCNN


class DeployWrapper(nn.Module):
    """Wrap a trained model so it outputs (probs, cam) in one pass."""

    def __init__(self, model):
        super().__init__()
        self.model = model.eval()

    def features(self, x):
        if isinstance(self.model, ScratchCNN):
            return self.model.features(x)
        return self.model.forward_features(x)  # timm: everything before the head

    def head(self, f):
        if isinstance(self.model, ScratchCNN):
            return self.model.head(self.model.pool(f))
        return self.model.forward_head(f)      # timm: pool -> (norm) -> linear

    def forward(self, x):
        f = self.features(x)                         # [B, C, h, w]
        logits = self.head(f)                        # [B, 200]
        b, c, h, w = f.shape
        # Treat every spatial position as its own 1x1 feature map, run the head
        # on it: [B, C, h, w] -> [B*h*w, C, 1, 1] -> [B*h*w, 200] -> [B, 200, h, w]
        per_pos = f.permute(0, 2, 3, 1).reshape(b * h * w, c, 1, 1)
        cam = self.head(per_pos).reshape(b, h, w, -1).permute(0, 3, 1, 2)
        return logits.softmax(1), cam


def fp16_weights(model_path):
    """Store large weight tensors as float16, but compute in float32.

    Each big float32 weight W becomes a float16 initializer plus a Cast node
    back to float32. The file is ~2x smaller (faster page load), while every
    operator still runs in float32, so all ONNX Runtime Web backends support it.
    """
    m = onnx.load(model_path)
    g = m.graph
    new_inits, cast_nodes = [], []
    for init in g.initializer:
        arr = numpy_helper.to_array(init)
        if arr.dtype == np.float32 and arr.size >= 1024:
            half = numpy_helper.from_array(arr.astype(np.float16), init.name + "_fp16")
            new_inits.append(half)
            cast_nodes.append(helper.make_node("Cast", [half.name], [init.name], to=onnx.TensorProto.FLOAT))
        else:
            new_inits.append(init)
    del g.initializer[:]
    g.initializer.extend(new_inits)
    # Cast nodes must come first so their outputs exist before use.
    old_nodes = list(g.node)
    del g.node[:]
    g.node.extend(cast_nodes + old_nodes)
    onnx.checker.check_model(m)
    onnx.save(m, model_path)


def make_assets(web_dir, img_size):
    """Copy the extra files the web page needs.

    refs/<id>.jpg   one reference photo per species (from the TRAINING set),
                    shown next to each of the top-5 predictions.
    examples/*.jpg  a few TEST images the user can click to try the app.
    """
    names, labels = read_annotations(ROOT / "train.txt")
    refs = web_dir / "refs"
    refs.mkdir(parents=True, exist_ok=True)
    done = set()
    for n, y in zip(names, labels):
        if y in done:
            continue
        done.add(y)
        im = Image.open(ROOT / "Train" / n).convert("RGB")
        im.thumbnail((200, 200))
        im.save(refs / f"{y}.jpg", quality=80)

    te_names, te_labels = read_annotations(ROOT / "test.txt")
    ex = web_dir / "examples"
    ex.mkdir(parents=True, exist_ok=True)
    picks = []
    rng = np.random.default_rng(7)
    for i in rng.choice(len(te_names), size=8, replace=False):
        im = Image.open(ROOT / "Test" / te_names[i]).convert("RGB")
        im.thumbnail((480, 480))
        im.save(ex / te_names[i], quality=85)
        picks.append({"file": f"examples/{te_names[i]}", "label": int(te_labels[i])})
    return picks


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_dir")
    p.add_argument("--web-dir", default="web")
    p.add_argument("--fp16-weights", action="store_true")
    a = p.parse_args()

    device = torch.device("cpu")  # export on CPU: simple and deterministic
    model, args = load_run(a.run_dir, device)
    model = model.to(memory_format=torch.contiguous_format).float()
    wrapper = DeployWrapper(model).eval()
    size = args["img_size"]

    web = ROOT / a.web_dir
    web.mkdir(exist_ok=True)
    onnx_path = web / "model.onnx"

    dummy = torch.randn(1, 3, size, size)
    torch.onnx.export(wrapper, (dummy,), str(onnx_path), input_names=["input"],
                      output_names=["probs", "cam"], opset_version=17, dynamo=False)
    if a.fp16_weights:
        fp16_weights(str(onnx_path))
    mb = onnx_path.stat().st_size / 1e6
    print(f"exported {onnx_path}  ({mb:.1f} MB)")

    # ---------------- parity check: ONNX Runtime vs PyTorch on the test set ----------------
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    _, _, test_loader = get_loaders(img_size=size, batch_size=1, num_workers=4)
    ort_probs, pt_probs, targets = [], [], []
    with torch.no_grad():
        for x, y in test_loader:
            o_probs, o_cam = sess.run(None, {"input": x.numpy()})
            t_probs, t_cam = wrapper(x)
            ort_probs.append(o_probs[0])
            pt_probs.append(t_probs[0].numpy())
            targets.append(int(y))
    ort_probs, pt_probs, targets = np.array(ort_probs), np.array(pt_probs), np.array(targets)
    parity = {
        "model": args["model"],
        "run": Path(a.run_dir).name,
        "file_mb": round(mb, 2),
        "fp16_weights": a.fp16_weights,
        "top1_agreement": float((ort_probs.argmax(1) == pt_probs.argmax(1)).mean()),
        "max_abs_prob_diff": float(np.abs(ort_probs - pt_probs).max()),
        "onnx_test": compute_metrics(np.log(ort_probs + 1e-8), targets),
        "pytorch_test": compute_metrics(np.log(pt_probs + 1e-8), targets),
        "cam_shape": list(o_cam.shape),
    }
    print(json.dumps(parity, indent=2))
    (Path(a.run_dir) / "onnx_parity.json").write_text(json.dumps(parity, indent=2))

    # ---------------- metadata + assets for the web page ----------------
    examples = make_assets(web, size)
    test_metrics = json.loads((Path(a.run_dir) / "test_metrics.json").read_text()) \
        if (Path(a.run_dir) / "test_metrics.json").exists() else {}
    meta = {
        "model": args["model"],
        "img_size": size,
        "resize": int(size * 256 / 224),
        "mean": IMAGENET_MEAN,
        "std": IMAGENET_STD,
        "classes": class_names(),
        "examples": examples,
        "test_metrics": test_metrics,
        "file_mb": round(mb, 1),
    }
    (web / "meta.json").write_text(json.dumps(meta, indent=1))
    print("wrote meta.json, refs/, examples/")


if __name__ == "__main__":
    main()
