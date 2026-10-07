"""
Person Detection — Standalone Inference (images)
================================================================================
Self-contained. Runs the trained model on a single image or a whole folder and
saves copies with green boxes + confidence + a person count drawn on them.

This inference script uses PyTorch and torchvision and does not depend on the Ultralytics package.

USAGE
-----
    # one image
    python infer.py --source photo.jpg

    # a folder of images
    python infer.py --source ./my_images --save-dir ./output

    # catch more distant/small people (lower threshold)
    python infer.py --source ./my_images --conf 0.3

Defaults: --model best.pth  --conf 0.5  --save-dir ./output
"""

import argparse
import os
import time
from functools import partial

import cv2
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import v2
from torchvision.io import read_image, ImageReadMode

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
_to_float = v2.ToDtype(torch.float32, scale=True)


def build_model(num_classes, min_size, max_size):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(
        weights=None, min_size=min_size, max_size=max_size)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = build_model(ckpt.get("num_classes", 2),
                        ckpt.get("min_size", 800), ckpt.get("max_size", 1333))
    model.load_state_dict(ckpt["model"])
    model.eval().to(device)
    print(f"Loaded model: {ckpt_path} "
          f"(epoch {ckpt.get('epoch', '?')}, val mAP {ckpt.get('best_map', -1):.3f})")
    return model


def draw(frame_bgr, boxes, scores, conf):
    count = 0
    for box, score in zip(boxes, scores):
        if score < conf:
            continue
        count += 1
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 200, 0), 2)
        label = f"person {score:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame_bgr, (x1, y1 - th - 6), (x1 + tw + 2, y1), (0, 200, 0), -1)
        cv2.putText(frame_bgr, label, (x1 + 1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(frame_bgr, f"count: {count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)
    return frame_bgr, count


@torch.no_grad()
def main():
    p = argparse.ArgumentParser("Person detection — image inference")
    p.add_argument("--source", required=True, help="image file OR folder of images")
    p.add_argument("--model", default="best.pth", help="path to model checkpoint")
    p.add_argument("--conf", type=float, default=0.5, help="confidence threshold (0-1)")
    p.add_argument("--save-dir", default="./output", help="where to save results")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    device = torch.device(args.device)
    model = load_model(args.model, device)

    if os.path.isdir(args.source):
        files = [os.path.join(args.source, f) for f in sorted(os.listdir(args.source))
                 if f.lower().endswith(IMAGE_EXTS)]
    else:
        files = [args.source]
    if not files:
        print("No images found in source."); return

    os.makedirs(args.save_dir, exist_ok=True)
    total = 0
    for f in files:
        img = read_image(f, ImageReadMode.RGB)
        t0 = time.time()
        out = model([_to_float(img).to(device)])[0]
        dt = (time.time() - t0) * 1000
        boxes = out["boxes"].cpu().numpy()
        scores = out["scores"].cpu().numpy()
        frame = cv2.imread(f)
        frame, n = draw(frame, boxes, scores, args.conf)
        out_path = os.path.join(args.save_dir, os.path.basename(f))
        cv2.imwrite(out_path, frame)
        total += n
        print(f"{os.path.basename(f):45s} {n:3d} person(s)  {dt:6.1f} ms")

    print(f"\nDone. {len(files)} image(s), {total} detection(s) -> {args.save_dir}")


if __name__ == "__main__":
    main()
