"""
Model factory — PyTorch/torchvision detector implementations.
================================================================================
No Ultralytics / no YOLO dependency. Every architecture here ships inside
torchvision and is pretrained on COCO, so the "person" concept is already
learned and we only fine-tune the detection head for our single class.

Available architectures (`--arch`):
    fasterrcnn      -> Faster R-CNN  ResNet50-FPN v2   (default, best accuracy)
    retinanet       -> RetinaNet     ResNet50-FPN v2   (single-stage)
    fcos            -> FCOS          ResNet50-FPN      (anchor-free)
    ssdlite         -> SSDLite       MobileNetV3-Large (lightweight / edge)

NOTE on class count:
    torchvision detectors reserve label 0 for the *background*. A single
    "person" foreground class therefore needs `num_classes = 2`.
"""

from functools import partial

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.retinanet import RetinaNetClassificationHead
from torchvision.models.detection.fcos import FCOSClassificationHead
from torchvision.models.detection.ssdlite import SSDLiteClassificationHead


SUPPORTED_ARCHS = ("fasterrcnn", "retinanet", "fcos", "ssdlite")


def _first_conv_in(module):
    """Return in_channels of the first Conv2d inside a head's conv stack
    (handles both plain Conv2d and Conv2dNormActivation blocks)."""
    for m in module.modules():
        if isinstance(m, torch.nn.Conv2d):
            return m.in_channels
    raise ValueError("no Conv2d found in classification head")


def build_model(arch: str = "fasterrcnn", num_classes: int = 2,
                pretrained: bool = True, min_size: int = 800, max_size: int = 1333):
    """
    Build a COCO-pretrained torchvision detector and swap its head so it
    predicts `num_classes` labels (background + foreground classes).

    Args:
        arch:        one of SUPPORTED_ARCHS.
        num_classes: total classes INCLUDING background (person-only => 2).
        pretrained:  load COCO-pretrained backbone + head weights.
        min_size/max_size: internal GeneralizedRCNNTransform resize bounds;
                     controls VRAM/accuracy trade-off. Ignored by ssdlite
                     (fixed 320x320).
    """
    arch = arch.lower()
    if arch not in SUPPORTED_ARCHS:
        raise ValueError(f"arch must be one of {SUPPORTED_ARCHS}, got '{arch}'")

    weights = "DEFAULT" if pretrained else None

    if arch == "fasterrcnn":
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(
            weights=weights, min_size=min_size, max_size=max_size,
        )
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    elif arch == "retinanet":
        model = torchvision.models.detection.retinanet_resnet50_fpn_v2(
            weights=weights, min_size=min_size, max_size=max_size,
        )
        in_features = _first_conv_in(model.head.classification_head.conv)
        num_anchors = model.head.classification_head.num_anchors
        model.head.classification_head = RetinaNetClassificationHead(
            in_features, num_anchors, num_classes,
            norm_layer=partial(torch.nn.GroupNorm, 32),
        )

    elif arch == "fcos":
        model = torchvision.models.detection.fcos_resnet50_fpn(
            weights=weights, min_size=min_size, max_size=max_size,
        )
        in_features = _first_conv_in(model.head.classification_head.conv)
        num_anchors = model.head.classification_head.num_anchors
        model.head.classification_head = FCOSClassificationHead(
            in_features, num_anchors, num_classes,
            norm_layer=partial(torch.nn.GroupNorm, 32),
        )

    else:  # ssdlite
        model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(
            weights=weights,
        )
        in_channels = [_first_conv_in(m) for m in model.head.classification_head.module_list]
        num_anchors = model.anchor_generator.num_anchors_per_location()
        norm_layer = partial(torch.nn.BatchNorm2d, eps=0.001, momentum=0.03)
        model.head.classification_head = SSDLiteClassificationHead(
            in_channels, num_anchors, num_classes, norm_layer,
        )

    return model


if __name__ == "__main__":
    # Quick smoke test: build each arch and count params.
    for a in SUPPORTED_ARCHS:
        m = build_model(a, num_classes=2, pretrained=False)
        n = sum(p.numel() for p in m.parameters()) / 1e6
        print(f"{a:12s} -> {n:6.1f}M params")
