# Inference example

Standalone person detection with the main fine-tuned Faster R-CNN (ResNet-50-FPN v2, torchvision; single class
`person`). It uses PyTorch and torchvision only, with no Ultralytics dependency.

## Install

Requires Python 3.9+.

```bash
pip install -r requirements.txt
```

For GPU, install a CUDA build of torch/torchvision from <https://pytorch.org> first.

## Weights

Download `faster_rcnn_100ep_MAIN_best.pth` and verify its SHA-256 against [`docs/MODEL_WEIGHTS.md`](../docs/MODEL_WEIGHTS.md):

```text
d49079caff104989affd7345f631f9f39a79104fa5a583609d13d2bafd2492ec
```

Place the file in `weights/` at the repository root.

## Run

```bash
# single image
python infer.py --source photo.jpg --model ../weights/faster_rcnn_100ep_MAIN_best.pth
# folder of images
python infer.py --source ./my_images --model ../weights/faster_rcnn_100ep_MAIN_best.pth --save-dir ./output
```

Annotated images (boxes, scores and a person count) are written to `--save-dir`.

| Option | Default | Description |
|---|---|---|
| `--source` | required | image file or folder of images |
| `--model` | `best.pth` | path to the checkpoint |
| `--conf` | `0.5` | score threshold (lower detects more small or distant persons) |
| `--save-dir` | `./output` | output folder |
| `--device` | `cuda` if available, else `cpu` | inference device |

## Licence

PyTorch and torchvision are distributed under their own software licences. The fine-tuned weights derive from
torchvision's COCO-pretrained weights and from the "site dataset" (CC BY 4.0); see
[`docs/MODEL_WEIGHTS.md`](../docs/MODEL_WEIGHTS.md) for their redistribution status.
