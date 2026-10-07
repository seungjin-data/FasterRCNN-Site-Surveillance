"""
Training / evaluation loops with COCO mAP evaluation.
================================================================================
Self-contained (no torchvision `references/` files needed). Evaluation uses
pycocotools' COCOeval, the standard metric for detection.
"""

import contextlib
import io
import math

import torch
from tqdm import tqdm
from pycocotools.cocoeval import COCOeval


def train_one_epoch(model, optimizer, loader, device, epoch, scaler=None,
                    grad_accum=1, max_norm=0.0, log_interval=20, lr_scheduler=None):
    """Run one training epoch. Returns the mean total loss."""
    model.train()
    running = 0.0
    n = 0
    optimizer.zero_grad(set_to_none=True)

    pbar = tqdm(loader, desc=f"Epoch {epoch} [train]", leave=False)
    for step, (images, targets) in enumerate(pbar):
        images = [img.to(device, non_blocking=True) for img in images]
        targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()}
                   for t in targets]

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())
            loss_value = loss.item()

        if not math.isfinite(loss_value):
            # Skip the (rare) exploding batch instead of poisoning weights.
            optimizer.zero_grad(set_to_none=True)
            print(f"  ! non-finite loss ({loss_value}); skipping batch {step}")
            continue

        loss = loss / grad_accum
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (step + 1) % grad_accum == 0:
            if max_norm > 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if lr_scheduler is not None:      # per-step warmup schedulers
                lr_scheduler.step()

        running += loss_value
        n += 1
        if step % log_interval == 0:
            pbar.set_postfix(loss=f"{loss_value:.3f}",
                             lr=f"{optimizer.param_groups[0]['lr']:.2e}")

    return running / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, device):
    """
    Run COCO evaluation over `loader`'s dataset.
    Returns dict with the 12 standard COCO stats (mAP, mAP50, ...).
    """
    model.eval()
    coco_gt = loader.dataset.coco
    results = []

    for images, targets in tqdm(loader, desc="Evaluating", leave=False):
        images = [img.to(device, non_blocking=True) for img in images]
        outputs = model(images)

        for target, output in zip(targets, outputs):
            image_id = int(target["image_id"].item())
            boxes = output["boxes"].cpu()
            scores = output["scores"].cpu().tolist()
            labels = output["labels"].cpu().tolist()
            # xyxy -> xywh for COCO
            for box, score, label in zip(boxes, scores, labels):
                x1, y1, x2, y2 = box.tolist()
                results.append({
                    "image_id": image_id,
                    "category_id": int(label),   # person == 1, matches GT json
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": float(score),
                })

    if not results:
        print("  ! no detections produced; mAP = 0")
        return {k: 0.0 for k in _STAT_KEYS}

    # Silence pycocotools' verbose prints, keep the summary table.
    coco_dt = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    with contextlib.redirect_stdout(io.StringIO()):
        coco_eval.evaluate()
        coco_eval.accumulate()
    coco_eval.summarize()

    return dict(zip(_STAT_KEYS, coco_eval.stats.tolist()))


_STAT_KEYS = [
    "mAP", "mAP50", "mAP75", "mAP_small", "mAP_medium", "mAP_large",
    "AR1", "AR10", "AR100", "AR_small", "AR_medium", "AR_large",
]
