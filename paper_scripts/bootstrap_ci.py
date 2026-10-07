"""Percentile bootstrap (over test images) for AP / AP50 / AP_S, plus paired differences."""
import contextlib, io, json, os, random, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _paths as P
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

P1 = str(P.RESULTS)
GT = json.load(open(P.DATA / "test" / "_annotations.coco.json"))
B = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
# "zeroshot" = legacy identifier of the off-the-shelf COCO-pretrained baseline (kept for file compatibility)
NAMES = ["zeroshot", "ssdlite", "retinanet", "fcos", "yolov8m", "rtdetr_l", "fasterrcnn", "fasterrcnn_100ep",
         "abl_noaug", "abl_res640"]
NAMES = [n for n in NAMES if os.path.exists(f"{P1}/{n}_test_dets.json")]
DETS = {n: json.load(open(f"{P1}/{n}_test_dets.json")) for n in NAMES}

img_ids = [i["id"] for i in GT["images"]]
imgs = {i["id"]: i for i in GT["images"]}
anns_by = {}
for a in GT["annotations"]:
    if a["category_id"] == 1:
        anns_by.setdefault(a["image_id"], []).append(a)
dets_by = {n: {} for n in NAMES}
for n in NAMES:
    for d in DETS[n]:
        dets_by[n].setdefault(d["image_id"], []).append(d)


def ap_stats(sample, name):
    images, anns, dets = [], [], []
    aid = 1
    for k, orig in enumerate(sample, start=1):
        images.append({**imgs[orig], "id": k})
        for a in anns_by.get(orig, []):
            anns.append({**a, "id": aid, "image_id": k}); aid += 1
        for d in dets_by[name].get(orig, []):
            dets.append({**d, "image_id": k})
    gt = COCO(); gt.dataset = {"images": images, "annotations": anns,
                               "categories": [{"id": 1, "name": "person"}]}
    with contextlib.redirect_stdout(io.StringIO()):
        gt.createIndex()
        if not dets:
            return np.zeros(3)
        E = COCOeval(gt, gt.loadRes(dets), "bbox"); E.params.catIds = [1]
        E.evaluate(); E.accumulate(); E.summarize()
    return np.array([E.stats[0], E.stats[1], E.stats[3]])


def ci(x):
    lo, hi = np.percentile(x, [2.5, 97.5], axis=0)
    return lo.round(4).tolist(), hi.round(4).tolist()


rng = random.Random(0)
samples = [[rng.choice(img_ids) for _ in img_ids] for _ in range(B)]
res = {n: np.array([ap_stats(s, n) for s in samples]) for n in NAMES}
point = {n: ap_stats(img_ids, n) for n in NAMES}

out = {"B": B, "metrics": ["AP", "AP50", "AP_S"], "models": {}, "paired": {}}
for n in NAMES:
    lo, hi = ci(res[n])
    out["models"][n] = {"point": point[n].round(4).tolist(), "lo": lo, "hi": hi}
pairs = [("fasterrcnn_100ep", "zeroshot"), ("fasterrcnn", "zeroshot"), ("fasterrcnn_100ep", "fasterrcnn"),
         ("fasterrcnn", "abl_noaug"), ("fasterrcnn", "abl_res640"),
         ("fasterrcnn", "fcos"), ("fasterrcnn", "retinanet"), ("fcos", "retinanet"),
         ("fasterrcnn", "yolov8m"), ("fasterrcnn", "rtdetr_l"), ("yolov8m", "fcos")]
for a, b in pairs:
    if a in res and b in res:
        d = res[a] - res[b]; lo, hi = ci(d)
        out["paired"][f"{a} - {b}"] = {"mean": d.mean(axis=0).round(4).tolist(), "lo": lo, "hi": hi,
                                        "frac_positive_AP": float((d[:, 0] > 0).mean())}
json.dump(out, open(f"{P1}/bootstrap.json", "w"), indent=2)
# unrounded companion file (same resamples), used for printed labels to avoid double rounding
full = {"B": B, "metrics": out["metrics"], "models": {n: {"point": point[n].tolist(), "lo": np.percentile(res[n], 2.5, axis=0).tolist(),
        "hi": np.percentile(res[n], 97.5, axis=0).tolist()} for n in NAMES}}
json.dump(full, open(f"{P1}/bootstrap_full.json", "w"), indent=1)
print(json.dumps(out, indent=1))
