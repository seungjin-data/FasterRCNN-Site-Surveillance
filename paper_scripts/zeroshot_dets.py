"""Dump detections of the off-the-shelf COCO-pretrained Faster R-CNN on the test split (for paired bootstrap).

Legacy filename and result key 'zeroshot' retained for compatibility; they denote the off-the-shelf
COCO-pretrained baseline (person is a COCO training class)."""
import json, sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _paths as P
import torchvision
from eval_ckpt import run, DEV

m = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights="DEFAULT").eval().to(DEV)
out = run(m, "test")
json.dump(out["dets"], open(P.RESULTS / "zeroshot_test_dets.json", "w"))
print("off-the-shelf COCO-pretrained baseline: test mAP", out["coco"]["mAP"])
