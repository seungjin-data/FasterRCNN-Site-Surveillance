"""
Training script — single-class PERSON detection (PyTorch/torchvision implementation).
================================================================================
Backbone: torchvision detector implementations; no Ultralytics/YOLO dependency.

Dataset (Roboflow COCO export) expected at:
    <data-dir>/train/_annotations.coco.json  + images
    <data-dir>/valid/_annotations.coco.json  + images

Quick start (matches the client's spec: 100 epochs, batch 20):
    python train.py --data-dir "./site dataset.v1-person-detection.coco" \
        --arch fasterrcnn --epochs 100 --batch-size 20 --workers 4

Features:
    * COCO-pretrained fine-tuning (person already learned)
    * AMP mixed precision, gradient accumulation, gradient clipping
    * Linear warmup + cosine LR schedule
    * COCO mAP evaluation, best-checkpoint tracking, resume support
    * Full JSON training history + console/file logging
"""

import argparse
import datetime
import json
import os
import sys
import time

import torch
from torch.utils.data import DataLoader

from dataset import CocoPersonDataset, get_transforms, collate_fn
from model import build_model, SUPPORTED_ARCHS
from engine import train_one_epoch, evaluate


def parse_args():
    p = argparse.ArgumentParser("Person detector training (torchvision)")
    # Data
    p.add_argument("--data-dir", type=str, required=True,
                   help="Dataset root containing train/ and valid/")
    p.add_argument("--train-split", type=str, default="train")
    p.add_argument("--val-split", type=str, default="valid")
    p.add_argument("--ann-name", type=str, default="_annotations.coco.json")
    # Model
    p.add_argument("--arch", type=str, default="fasterrcnn", choices=SUPPORTED_ARCHS)
    p.add_argument("--no-pretrained", action="store_true",
                   help="Train from scratch instead of COCO-pretrained (not recommended)")
    p.add_argument("--min-size", type=int, default=800, help="Internal resize shorter side")
    p.add_argument("--max-size", type=int, default=1333, help="Internal resize longer side")
    # Optimization
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=20)
    p.add_argument("--grad-accum", type=int, default=1,
                   help="Accumulate N steps -> effective batch = batch-size * N (saves VRAM)")
    p.add_argument("--lr", type=float, default=0.005)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--optimizer", type=str, default="sgd", choices=["sgd", "adamw"])
    p.add_argument("--warmup-epochs", type=float, default=3.0)
    p.add_argument("--clip-grad", type=float, default=5.0, help="Max grad norm (0=off)")
    # Runtime
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--eval-interval", type=int, default=5, help="Evaluate every N epochs")
    p.add_argument("--no-amp", action="store_true", help="Disable mixed precision")
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output-dir", type=str, default="./runs/person_detector")
    p.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-online-aug", action="store_true",
                   help="Ablation: disable the online (box-aware) training augmentation")
    return p.parse_args()


class Tee:
    """Write to console AND a log file simultaneously."""
    def __init__(self, path):
        self.terminal = sys.stdout
        self.log = open(path, "a", encoding="utf-8")
    def write(self, msg):
        self.terminal.write(msg); self.log.write(msg)
    def flush(self):
        self.terminal.flush(); self.log.flush()


def build_scheduler(optimizer, warmup_iters, total_iters):
    """Linear warmup -> cosine decay, stepped per optimizer update."""
    def lr_lambda(step):
        if step < warmup_iters:
            return (step + 1) / max(warmup_iters, 1)
        progress = (step - warmup_iters) / max(total_iters - warmup_iters, 1)
        return 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159265)).item())
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    os.makedirs(args.output_dir, exist_ok=True)
    sys.stdout = Tee(os.path.join(args.output_dir, "train.log"))

    print("=" * 78)
    print(f"Person Detector Training  |  {datetime.datetime.now():%Y-%m-%d %H:%M:%S}")
    print("=" * 78)
    for k, v in vars(args).items():
        print(f"  {k:16s}: {v}")
    device = torch.device(args.device)
    print(f"  device (resolved): {device} "
          f"({torch.cuda.get_device_name(0) if device.type=='cuda' else 'CPU'})")

    # ---- Data ----
    train_dir = os.path.join(args.data_dir, args.train_split)
    val_dir = os.path.join(args.data_dir, args.val_split)
    train_ds = CocoPersonDataset(
        train_dir, os.path.join(train_dir, args.ann_name),
        transforms=get_transforms(train=not args.no_online_aug))
    val_ds = CocoPersonDataset(
        val_dir, os.path.join(val_dir, args.ann_name),
        transforms=get_transforms(train=False))
    print(f"\nDataset: {len(train_ds)} train / {len(val_ds)} val images")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, collate_fn=collate_fn,
        pin_memory=True, drop_last=True, persistent_workers=args.workers > 0)
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=args.workers, collate_fn=collate_fn, pin_memory=True)

    # ---- Model ----  (num_classes = background + person = 2)
    model = build_model(args.arch, num_classes=2, pretrained=not args.no_pretrained,
                        min_size=args.min_size, max_size=args.max_size).to(device)

    # ---- Optimizer ----
    params = [p for p in model.parameters() if p.requires_grad]
    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum,
                                    weight_decay=args.weight_decay, nesterov=True)
    else:
        optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    steps_per_epoch = max(len(train_loader) // args.grad_accum, 1)
    total_iters = steps_per_epoch * args.epochs
    warmup_iters = int(steps_per_epoch * args.warmup_epochs)
    scheduler = build_scheduler(optimizer, warmup_iters, total_iters)

    use_amp = (not args.no_amp) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp) if use_amp else None
    print(f"AMP: {use_amp} | steps/epoch: {steps_per_epoch} | "
          f"effective batch: {args.batch_size * args.grad_accum}")

    # ---- Resume ----
    start_epoch, best_map = 1, -1.0
    history = []
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        if scaler and ckpt.get("scaler"):
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        best_map = ckpt.get("best_map", -1.0)
        history = ckpt.get("history", [])
        print(f"\nResumed from {args.resume} @ epoch {ckpt['epoch']} (best mAP {best_map:.4f})")

    # ---- Train loop ----
    t_start = time.time()
    for epoch in range(start_epoch, args.epochs + 1):
        ep_t = time.time()
        train_loss = train_one_epoch(
            model, optimizer, train_loader, device, epoch,
            scaler=scaler, grad_accum=args.grad_accum,
            max_norm=args.clip_grad, lr_scheduler=scheduler)

        record = {"epoch": epoch, "train_loss": round(train_loss, 4),
                  "lr": optimizer.param_groups[0]["lr"],
                  "time_s": round(time.time() - ep_t, 1)}

        do_eval = (epoch % args.eval_interval == 0) or (epoch == args.epochs)
        if do_eval:
            stats = evaluate(model, val_loader, device)
            record.update({"mAP": round(stats["mAP"], 4),
                           "mAP50": round(stats["mAP50"], 4),
                           "mAP_small": round(stats["mAP_small"], 4)})
            print(f"[Epoch {epoch}] loss={train_loss:.4f} "
                  f"mAP={stats['mAP']:.4f} mAP50={stats['mAP50']:.4f} "
                  f"mAP_small={stats['mAP_small']:.4f} ({record['time_s']}s)")

            if stats["mAP"] > best_map:
                best_map = stats["mAP"]
                _save(model, optimizer, scheduler, scaler, epoch, best_map,
                      history, args, os.path.join(args.output_dir, "best.pth"))
                print(f"    * new best mAP {best_map:.4f} -> best.pth")
        else:
            print(f"[Epoch {epoch}] loss={train_loss:.4f} "
                  f"lr={record['lr']:.2e} ({record['time_s']}s)")

        history.append(record)
        _save(model, optimizer, scheduler, scaler, epoch, best_map,
              history, args, os.path.join(args.output_dir, "last.pth"))
        with open(os.path.join(args.output_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=2)

    elapsed = datetime.timedelta(seconds=int(time.time() - t_start))
    print("\n" + "=" * 78)
    print(f"Training done in {elapsed} | best mAP = {best_map:.4f}")
    print(f"Weights: {os.path.join(args.output_dir, 'best.pth')}")
    print("=" * 78)


def _save(model, optimizer, scheduler, scaler, epoch, best_map, history, args, path):
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict() if scaler else None,
        "epoch": epoch,
        "best_map": best_map,
        "history": history,
        "arch": args.arch,
        "num_classes": 2,
        "class_names": ["__background__", "person"],
        "min_size": args.min_size,
        "max_size": args.max_size,
    }, path)


if __name__ == "__main__":
    main()
