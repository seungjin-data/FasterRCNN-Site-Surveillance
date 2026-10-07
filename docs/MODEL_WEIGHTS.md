# Model weights

The trained weights are **not stored in this Git repository**. The table below identifies each file exactly so that a
downloaded copy can be verified (`sha256sum <file>`). Each weight file will be included in the Zenodo archive of this
study only if its redistribution permission is clear (see "Redistribution status"). Otherwise, the file is documented
here with instructions to reproduce it.

## Files

| File | Model | Training | SHA-256 |
|---|---|---|---|
| `faster_rcnn_100ep_MAIN_best.pth` | Faster R-CNN ResNet-50-FPN v2 (torchvision), 43.3 M params | 100-epoch schedule; validation-selected checkpoint at epoch 65 (main model) | `d49079caff104989affd7345f631f9f39a79104fa5a583609d13d2bafd2492ec` |
| `faster_rcnn_15ep_best.pth` | Faster R-CNN ResNet-50-FPN v2 (torchvision) | 15-epoch comparison recipe, seed 42 | `572f34b8c2e6e91444cb89add70aa61df9b88c28484caff773bfffb12759fb02` |
| `retinanet_15ep_best.pth` | RetinaNet ResNet-50-FPN v2 (torchvision), 36.4 M params | 15-epoch comparison recipe, seed 42 | `3e0c855f971e430feaa78a7158a13e23303874937f1c3539d7eedeb9f9c46a54` |
| `fcos_15ep_best.pth` | FCOS ResNet-50-FPN (torchvision), 32.1 M params | 15-epoch comparison recipe, seed 42 | `f05179dda71cc71608e4ec78d3d4d3b11707d09d2863bb35292b8fa7c2962f69` |
| `ssdlite_15ep_best.pth` | SSDLite320 MobileNetV3-Large (torchvision), 2.2 M params | 15-epoch comparison recipe, seed 42 | `b29966bdd7f2d0fab4f24def4c19757d48782e918b345c2aab69bd2eb643477c` |
| `yolov8m_best.pt` | YOLOv8m (Ultralytics 8.4.87), 25.8 M params | Ultralytics default recipe, 15 epochs, 1280 px | `10f07acef34d47b14cc587bebf9bcf10ff369ae57e49e8ca4d8c22a764c83fa9` |
| `rtdetr_l_best.pt` | RT-DETR-L (Ultralytics 8.4.87), 32.0 M params | Ultralytics default recipe, 15 epochs, 1280 px | `a7d8612bfbf511f3c400f732ce2117ccf4cd263931d72cc7bd2424bc1050dd78` |

The historical scene-held-out checkpoint (`runs/scene/fasterrcnn/best.pth`) was not preserved and is not available.

## Upstream models and redistribution status

All seven models were fine-tuned on the "site dataset" (CC BY 4.0, see [DATASET.md](DATASET.md)), starting from
COCO-pretrained weights.

| File(s) | Upstream initialisation | Upstream terms | Redistribution status |
|---|---|---|---|
| 5 torchvision models (`*.pth`) | torchvision COCO-pretrained weights (`fasterrcnn_resnet50_fpn_v2`, `retinanet_resnet50_fpn_v2`, `fcos_resnet50_fpn`, `ssdlite320_mobilenet_v3_large`) | torchvision documentation: pre-trained models "may have their own licenses or terms and conditions derived from the dataset used for training. It is your responsibility to determine whether you have permission to use the models for your use case." | **Not explicitly granted.** Not redistributed until the author confirms permission. |
| `yolov8m_best.pt`, `rtdetr_l_best.pt` | Ultralytics COCO-pretrained `yolov8m.pt`, `rtdetr-l.pt` | Ultralytics licence page: trained models fall under AGPL-3.0 by default; distribution requires AGPL-3.0 compliance (or an Ultralytics Enterprise licence). | **Conditional (AGPL-3.0 obligations).** Author decision pending. |

Sources: <https://docs.pytorch.org/vision/stable/models.html>, <https://www.ultralytics.com/license> (checked
2026-10-07).

## Reproducing the weights

The training commands are in `paper_scripts/run_phase1.sh`, `run_phase2.sh`, `run_phase2b.sh` and `run_phase3.sh`, and
the Ultralytics reference runs are in `paper_scripts/ultra_run.py`. Training is not bit-reproducible (cuDNN benchmark
mode), so retrained weights will not match the SHA-256 values above. The evaluation results in `results/` were
produced with the files listed here.

## Using the weights

1. Place the downloaded files in `weights/` at the repository root. This folder is ignored by Git. You can also set
   `WEIGHTS_DIR`.
2. Check each file against the table above with `sha256sum`.
3. See [`inference_example/README.md`](../inference_example/README.md) for running the main model.
