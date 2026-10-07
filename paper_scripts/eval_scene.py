"""Compare a model trained on all scenes with one trained without the held-out scenes, on the unseen-scene test set."""
import json
import random
import sys

import numpy as np
import torch

import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _paths as P
sys.path.insert(0, str(P.CORE))
import eval_ckpt
from eval_ckpt import DEV, run
from model import build_model
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import contextlib
import io

SCENE = str(P.WORK / "scene_split")
eval_ckpt.ROOT = SCENE
P1 = str(P.RESULTS)
MODELS = {"all_scenes": str(P.ALL_SCENES_CKPT),
          "held_out_scenes": str(P.SCENE_CKPT)}


def load(path):
    ck = torch.load(path, map_location="cpu")
    m = build_model(ck["arch"], 2, pretrained=False, min_size=ck.get("min_size", 800), max_size=ck.get("max_size", 1333))
    m.load_state_dict(ck["model"])
    return m.eval().to(DEV)


out, dets = {}, {}
for k, p in MODELS.items():
    r = run(load(p), "test")
    dets[k] = r.pop("dets")
    out[k] = r
    print(k, "AP", r["coco"]["mAP"], "AP50", r["coco"]["mAP50"], "APs", r["coco"]["mAP_small"])

gt_json = json.load(open(f"{SCENE}/test/_annotations.coco.json"))
ids = [i["id"] for i in gt_json["images"]]
imgs = {i["id"]: i for i in gt_json["images"]}
anns_by = {}
for a in gt_json["annotations"]:
    anns_by.setdefault(a["image_id"], []).append(a)
dets_by = {k: {} for k in dets}
for k in dets:
    for d in dets[k]:
        dets_by[k].setdefault(d["image_id"], []).append(d)


def ap_on(sample, k):
    images, anns, dd = [], [], []
    aid = 1
    for n, orig in enumerate(sample, start=1):
        images.append({**imgs[orig], "id": n})
        for a in anns_by.get(orig, []):
            anns.append({**a, "id": aid, "image_id": n}); aid += 1
        for d in dets_by[k].get(orig, []):
            dd.append({**d, "image_id": n})
    gt = COCO(); gt.dataset = {"images": images, "annotations": anns, "categories": [{"id": 1, "name": "person"}]}
    with contextlib.redirect_stdout(io.StringIO()):
        gt.createIndex()
        E = COCOeval(gt, gt.loadRes(dd), "bbox"); E.params.catIds = [1]
        E.evaluate(); E.accumulate(); E.summarize()
    return np.array([E.stats[0], E.stats[1]])


rng = random.Random(0)
B = 1000
res = {k: [] for k in dets}
for _ in range(B):
    s = [rng.choice(ids) for _ in ids]
    for k in dets:
        res[k].append(ap_on(s, k))
res = {k: np.array(v) for k, v in res.items()}
for k in dets:
    lo, hi = np.percentile(res[k], [2.5, 97.5], axis=0)
    out[k]["bootstrap"] = {"AP": [round(float(lo[0]), 4), round(float(hi[0]), 4)],
                           "AP50": [round(float(lo[1]), 4), round(float(hi[1]), 4)]}
d = res["all_scenes"] - res["held_out_scenes"]
lo, hi = np.percentile(d, [2.5, 97.5], axis=0)
out["paired_all_minus_heldout"] = {"AP_mean": round(float(d[:, 0].mean()), 4), "AP_ci": [round(float(lo[0]), 4), round(float(hi[0]), 4)],
                                    "AP50_mean": round(float(d[:, 1].mean()), 4), "AP50_ci": [round(float(lo[1]), 4), round(float(hi[1]), 4)]}
out["split"] = {"test_images": len(ids), "test_boxes": len(gt_json["annotations"])}
json.dump(out, open(f"{P1}/scene_eval.json", "w"), indent=2)
print(json.dumps(out["paired_all_minus_heldout"]))
