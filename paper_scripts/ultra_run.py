"""Train and evaluate an Ultralytics detector (YOLOv8 or RT-DETR) on the person dataset.

Usage: python ultra_run.py <yolo|rtdetr> <name>
Evaluation uses the same pycocotools protocol as the torchvision models (score >= 0.05, <= 100 dets).
"""
import contextlib
import io
import json
import sys
import time

import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _paths as P
YOLO_DATA = str(P.WORK / "yolo_data")
YAML = f"{YOLO_DATA}/data.yaml"
GT = str(P.DATA / "test" / "_annotations.coco.json")
RUNS = str(P.RUNS)
KEYS = ["mAP", "mAP50", "mAP75", "mAP_small", "mAP_medium", "mAP_large",
        "AR1", "AR10", "AR100", "AR_small", "AR_medium", "AR_large"]
IMGSZ = 1280


def train(kind, name):
    from ultralytics import RTDETR, YOLO
    common = dict(data=YAML, epochs=15, imgsz=IMGSZ, workers=4, seed=42, project=f"{RUNS}/ultra",
                  name=name, exist_ok=True, plots=False, amp=True, device=0, warmup_epochs=1, close_mosaic=3)
    if kind == "yolo":
        YOLO("yolov8m.pt").train(batch=8, **common)
    else:
        for batch in (4, 2):  # fall back if the transformer runs out of memory
            try:
                RTDETR("rtdetr-l.pt").train(batch=batch, **common)
                break
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
    return f"{RUNS}/ultra/{name}/weights/best.pt"


def evaluate(kind, name, weights):
    from ultralytics import RTDETR, YOLO
    model = (YOLO if kind == "yolo" else RTDETR)(weights)
    gt = COCO(GT)
    dets = []
    for im in gt.dataset["images"]:
        r = model.predict(f"{YOLO_DATA}/images/test/{im['file_name']}", imgsz=IMGSZ, conf=0.05, iou=0.5,
                          max_det=100, verbose=False, device=0)[0]
        for (x1, y1, x2, y2), s in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist()):
            dets.append({"image_id": im["id"], "category_id": 1, "bbox": [x1, y1, x2 - x1, y2 - y1], "score": s})
    with contextlib.redirect_stdout(io.StringIO()):
        E = COCOeval(gt, gt.loadRes(dets), "bbox"); E.params.catIds = [1]
        E.evaluate(); E.accumulate(); E.summarize()
    stats = dict(zip(KEYS, [round(float(x), 4) for x in E.stats]))
    path = f"{YOLO_DATA}/images/test/{gt.dataset['images'][0]['file_name']}"
    for _ in range(10):
        model.predict(path, imgsz=IMGSZ, conf=0.05, verbose=False, device=0)
    torch.cuda.synchronize(); t = time.time()
    for _ in range(100):
        model.predict(path, imgsz=IMGSZ, conf=0.05, verbose=False, device=0)
    torch.cuda.synchronize()
    lat = round((time.time() - t) / 100 * 1000, 1)
    params = round(sum(p.numel() for p in model.model.parameters()) / 1e6, 1)
    out = {"name": name, "arch": kind, "params_M": params, "latency_ms_1080p_fp32": lat,
           "test": {"coco": stats}, "imgsz": IMGSZ}
    json.dump(out, open(P.RESULTS / f"{name}.json", "w"), indent=2)
    json.dump(dets, open(P.RESULTS / f"{name}_test_dets.json", "w"))
    print(name, "test mAP", stats["mAP"], "mAP50", stats["mAP50"], "latency", lat, "ms")


if __name__ == "__main__":
    kind, name = sys.argv[1], sys.argv[2]
    evaluate(kind, name, train(kind, name))
