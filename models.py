"""
models.py - The CNN architectures we compare.

    M1  scratch_cnn       Small VGG-style CNN, random initial weights.
    M2  resnet50_frozen   ImageNet-pretrained ResNet-50. Backbone frozen; only
                          the new 200-class head is trained ("linear probe").
    M3  resnet50          Same ResNet-50, but the whole network is fine-tuned.
    M4  convnext_tiny /   Modern pretrained CNNs, fully fine-tuned.
        efficientnet_b3

Each step up the ladder answers one question for the report:
    M1 -> M2 : how much do pretrained ImageNet features help on small data?
    M2 -> M3 : does adapting the backbone to birds help beyond a fixed feature extractor?
    M3 -> M4 : does a more modern CNN design help?
"""

import timm
import torch.nn as nn

from data import NUM_CLASSES

# timm model names. The ".tag" picks a specific set of pretrained weights.
TIMM_NAMES = {
    "resnet50": "resnet50.tv2_in1k",            # torchvision's improved ImageNet-1k recipe
    "resnet50_frozen": "resnet50.tv2_in1k",
    "convnext_tiny": "convnext_tiny.fb_in22k_ft_in1k",  # pretrained on ImageNet-22k, then 1k
    "efficientnet_b3": "efficientnet_b3.ra2_in1k",
}


def conv_block(c_in, c_out):
    """Two 3x3 conv layers + BatchNorm + ReLU, then 2x2 max-pool.

    - 3x3 convolutions learn local patterns (edges -> textures -> parts).
    - BatchNorm normalises activations, which makes training from scratch
      much more stable and allows a higher learning rate.
    - Max-pool halves the spatial size, so deeper layers "see" a larger area
      of the image (bigger receptive field) at lower cost.
    """
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),  # bias is redundant before BN
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class ScratchCNN(nn.Module):
    """M1: a small VGG-style CNN trained from random weights.

    Input 3x224x224 -> 5 conv blocks (224 -> 112 -> 56 -> 28 -> 14 -> 7)
    -> global average pooling -> dropout -> linear layer with 200 outputs.

    Global average pooling (instead of flattening 512x7x7 into a huge fully
    connected layer) keeps the parameter count low (~4.7M), which matters with
    only ~4,100 training images. Dropout randomly zeroes features during
    training so the classifier cannot depend on a few specific features.
    """

    def __init__(self, num_classes=NUM_CLASSES, drop=0.3):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(3, 32),
            conv_block(32, 64),
            conv_block(64, 128),
            conv_block(128, 256),
            conv_block(256, 512),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)  # 512x7x7 -> 512x1x1
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(drop), nn.Linear(512, num_classes))

        # He (Kaiming) initialisation: the standard choice for ReLU networks;
        # keeps activation variance stable through the layers at the start.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        return self.head(self.pool(self.features(x)))

    # Used by the Grad-CAM code: the last conv block is the layer we visualise.
    def gradcam_layer(self):
        return self.features[-1]


def build_model(name, drop=0.3, pretrained=True):
    """Create a model by name. Returns (model, head_param_names).

    head_param_names lets train.py give the new classifier head a higher
    learning rate than the pretrained backbone (see train.py).
    """
    if name == "scratch_cnn":
        model = ScratchCNN(drop=drop)
        return model, {n for n, _ in model.named_parameters()}  # everything is "new"

    if name not in TIMM_NAMES:
        raise ValueError(f"Unknown model '{name}'. Options: scratch_cnn, {', '.join(TIMM_NAMES)}")

    # timm downloads the pretrained ImageNet weights and replaces the original
    # 1000-class classifier with a fresh, randomly initialised 200-class one.
    # drop_rate adds dropout right before that classifier.
    model = timm.create_model(TIMM_NAMES[name], pretrained=pretrained,
                              num_classes=NUM_CLASSES, drop_rate=drop)

    # Names of the parameters that belong to the new classifier head.
    head_module_name = model.default_cfg["classifier"]  # e.g. "fc" or "head.fc"
    head_names = {n for n, _ in model.named_parameters() if n.startswith(head_module_name)}

    if name == "resnet50_frozen":
        # Freeze every backbone weight: no gradients -> they never change.
        # The network becomes a fixed ImageNet feature extractor + trainable linear head.
        for n, p in model.named_parameters():
            p.requires_grad = n in head_names

    return model, head_names


def gradcam_target_layer(model):
    """Return the last convolutional stage, which Grad-CAM visualises."""
    if isinstance(model, ScratchCNN):
        return model.gradcam_layer()
    if hasattr(model, "layer4"):   # ResNet
        return model.layer4[-1]
    if hasattr(model, "stages"):   # ConvNeXt
        return model.stages[-1]
    if hasattr(model, "conv_head"):  # EfficientNet
        return model.conv_head
    raise ValueError("No Grad-CAM layer known for this model")


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
