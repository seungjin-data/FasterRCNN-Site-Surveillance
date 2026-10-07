"""
Testing / inference script — PERSON detection (PyTorch/torchvision implementation).
================================================================================
Two modes:

  1) INFERENCE  — draw boxes on an image / folder / video / webcam and save.
        python test.py --ckpt runs/person_detector/best.pth \
            --source path/to/img_or_folder_or_video.mp4 --conf 0.5

        python test.py --ckpt runs/person_detector/best.pth --source 0   # webcam

  2) EVALUATE   — compute COCO mAP on a held-out split (e.g. test/).
        python test.py --ckpt runs/person_detector/best.pth --evaluate \
            --data-dir "./site dataset.v1-person-detection.coco" --split test

The checkpoint stores its own arch / class info, so you don't have to repeat
--arch etc. that you used at training time.
"""

import argparse
import os
import time

import cv2
import torch
from torchvision.transforms import v2
from torchvision.io import read_image, ImageReadMode
from torch.utils.data import DataLoader

from model import build_model
from dataset import CocoPersonDataset, get_transforms, collate_fn
from engine import evaluate

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv")
CLASS_NAMES = ["__background__", "person"]
_to_float = v2.ToDtype(torch.float32, scale=True)


# --------------------------------------------------------------------------- #
# Model loading                                                               #
# --------------------------------------------------------------------------- #
def load_model(ckpt_path, device, arch=None):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    arch = arch or ckpt.get("arch", "fasterrcnn")
    model = build_model(
        arch, num_classes=ckpt.get("num_classes", 2), pretrained=False,
        min_size=ckpt.get("min_size", 800), max_size=ckpt.get("max_size", 1333))
    model.load_state_dict(ckpt["model"])
    model.eval().to(device)
    print(f"Loaded {arch} from {ckpt_path} "
          f"(trained {ckpt.get('epoch', '?')} epochs, best mAP {ckpt.get('best_map', -1):.4f})")
    return model


# --------------------------------------------------------------------------- #
# Drawing                                                                      #
# --------------------------------------------------------------------------- #
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
def predict_tensor(model, img_uint8_rgb, device):
    """img_uint8_rgb: CHW uint8 RGB tensor -> boxes, scores (numpy on CPU)."""
    img = _to_float(img_uint8_rgb).to(device)
    out = model([img])[0]
    return out["boxes"].cpu().numpy(), out["scores"].cpu().numpy()


# --------------------------------------------------------------------------- #
# Inference over images / video                                               #
# --------------------------------------------------------------------------- #
def run_images(model, source, save_dir, conf, device):
    if os.path.isdir(source):
        files = [os.path.join(source, f) for f in sorted(os.listdir(source))
                 if f.lower().endswith(IMAGE_EXTS)]
    else:
        files = [source]
    os.makedirs(save_dir, exist_ok=True)

    total = 0
    for f in files:
        img = read_image(f, ImageReadMode.RGB)
        t0 = time.time()
        boxes, scores = predict_tensor(model, img, device)
        dt = (time.time() - t0) * 1000
        frame = cv2.imread(f)
        frame, n = draw(frame, boxes, scores, conf)
        out_path = os.path.join(save_dir, os.path.basename(f))
        cv2.imwrite(out_path, frame)
        total += n
        print(f"{os.path.basename(f):40s} {n:3d} person(s)  {dt:6.1f} ms  -> {out_path}")
    print(f"\nDone. {len(files)} image(s), {total} detection(s) -> {save_dir}")


def run_video(model, source, save_dir, conf, device):
    is_cam = str(source).isdigit()
    cap = cv2.VideoCapture(int(source) if is_cam else source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {source}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, "output.mp4")
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    fid = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = torch.from_numpy(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).permute(2, 0, 1)
        t0 = time.time()
        boxes, scores = predict_tensor(model, rgb, device)
        frame, n = draw(frame, boxes, scores, conf)
        writer.write(frame)
        if is_cam:
            cv2.imshow("person-detector (press q)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        fid += 1
        if fid % 30 == 0:
            print(f"frame {fid}: {n} person(s)  {1/(time.time()-t0+1e-9):.1f} FPS")

    cap.release(); writer.release(); cv2.destroyAllWindows()
    print(f"Saved -> {out_path}")


# --------------------------------------------------------------------------- #
# COCO evaluation                                                             #
# --------------------------------------------------------------------------- #
def run_eval(model, data_dir, split, ann_name, workers, device):
    img_dir = os.path.join(data_dir, split)
    ds = CocoPersonDataset(img_dir, os.path.join(img_dir, ann_name),
                           transforms=get_transforms(train=False))
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=workers,
                        collate_fn=collate_fn, pin_memory=True)
    print(f"Evaluating on '{split}' ({len(ds)} images)...")
    stats = evaluate(model, loader, device)
    print("\n---- COCO metrics ----")
    for k in ("mAP", "mAP50", "mAP75", "mAP_small", "mAP_medium", "mAP_large", "AR100"):
        print(f"  {k:12s}: {stats[k]:.4f}")


def main():
    p = argparse.ArgumentParser("Person detector inference / evaluation")
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--arch", type=str, default=None, help="Override arch (else read from ckpt)")
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    # inference
    p.add_argument("--source", type=str, help="image / folder / video path or webcam index")
    p.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
    p.add_argument("--save-dir", type=str, default="./results")
    # evaluation
    p.add_argument("--evaluate", action="store_true")
    p.add_argument("--data-dir", type=str)
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--ann-name", type=str, default="_annotations.coco.json")
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()

    device = torch.device(args.device)
    model = load_model(args.ckpt, device, args.arch)

    if args.evaluate:
        if not args.data_dir:
            p.error("--evaluate requires --data-dir")
        run_eval(model, args.data_dir, args.split, args.ann_name, args.workers, device)
        return

    if not args.source:
        p.error("provide --source for inference (or use --evaluate)")

    src = args.source
    if str(src).isdigit() or src.lower().endswith(VIDEO_EXTS):
        run_video(model, src, args.save_dir, args.conf, device)
    else:
        run_images(model, src, args.save_dir, args.conf, device)


if __name__ == "__main__":
    main()
