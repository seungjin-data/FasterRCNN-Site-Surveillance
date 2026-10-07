"""Evaluate one checkpoint on valid+test (COCO metrics, P/R/F1) and measure 1080p latency."""
import argparse, contextlib, io, json, sys, time
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _paths as P
sys.path.insert(0, str(P.CORE))
import torch, torchvision
from torch.utils.data import DataLoader
from pycocotools.cocoeval import COCOeval
from dataset import CocoPersonDataset, get_transforms, collate_fn
from model import build_model

ROOT = str(P.DATA)
KEYS = ["mAP", "mAP50", "mAP75", "mAP_small", "mAP_medium", "mAP_large",
        "AR1", "AR10", "AR100", "AR_small", "AR_medium", "AR_large"]
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _sync():
    if DEV.type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def run(model, split):
    d = f"{ROOT}/{split}"
    ds = CocoPersonDataset(d, f"{d}/_annotations.coco.json", transforms=get_transforms(False))
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn)
    res, preds = [], {}
    for imgs, tg in dl:
        out = model([i.to(DEV) for i in imgs])[0]
        iid = int(tg[0]["image_id"].item())
        keep = out["labels"] == 1
        b, s = out["boxes"][keep].cpu(), out["scores"][keep].cpu()
        preds[iid] = (b, s)
        for bb, ss in zip(b.tolist(), s.tolist()):
            res.append({"image_id": iid, "category_id": 1,
                        "bbox": [bb[0], bb[1], bb[2] - bb[0], bb[3] - bb[1]], "score": ss})
    gt = ds.coco
    with contextlib.redirect_stdout(io.StringIO()):
        E = COCOeval(gt, gt.loadRes(res), "bbox"); E.params.catIds = [1]
        E.evaluate(); E.accumulate(); E.summarize()
    stats = dict(zip(KEYS, [round(float(x), 4) for x in E.stats]))
    prf = {}
    for conf in (0.3, 0.5, 0.7):
        tp = fp = fn = 0
        for iid, (b, s) in preds.items():
            b = b[s >= conf]
            g = gt.loadAnns(gt.getAnnIds(imgIds=iid, catIds=[1]))
            gb = torch.tensor([[a["bbox"][0], a["bbox"][1], a["bbox"][0] + a["bbox"][2],
                                a["bbox"][1] + a["bbox"][3]] for a in g]).reshape(-1, 4)
            m = 0
            if len(b) and len(gb):
                iou = torchvision.ops.box_iou(b, gb); used = set()
                for i in range(len(b)):
                    j = int(iou[i].argmax())
                    if iou[i, j] >= 0.5 and j not in used:
                        used.add(j); m += 1
            tp += m; fp += len(b) - m; fn += len(gb) - m
        p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1); f = 2 * p * r / max(p + r, 1e-9)
        prf[str(conf)] = {"P": round(p, 4), "R": round(r, 4), "F1": round(f, 4), "TP": tp, "FP": fp, "FN": fn}
    return {"coco": stats, "prf": prf, "dets": res}


@torch.no_grad()
def latency(model, n=100):
    x = [torch.rand(3, 1080, 1920, device=DEV)]
    for _ in range(10):
        model(x)
    _sync(); t = time.time()
    for _ in range(n):
        model(x)
    _sync()
    return round((time.time() - t) / n * 1000, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--name", required=True)
    a = ap.parse_args()
    ck = torch.load(a.ckpt, map_location="cpu")
    # torchvision builds a different SSDLite backbone (reduced tail) when weights=None, so the
    # structure must be built with pretrained=True for the checkpoint to load; weights are overwritten.
    m = build_model(ck["arch"], 2, pretrained=(ck["arch"] == "ssdlite"),
                    min_size=ck.get("min_size", 800), max_size=ck.get("max_size", 1333))
    m.load_state_dict(ck["model"]); m.eval().to(DEV)
    out = {"name": a.name, "arch": ck["arch"], "best_epoch": ck["epoch"],
           "best_valid_map": ck["best_map"], "params_M": round(sum(p.numel() for p in m.parameters()) / 1e6, 1),
           "gpu": torch.cuda.get_device_name(0) if DEV.type == "cuda" else "cpu"}
    for sp in ("valid", "test"):
        out[sp] = run(m, sp)
        if sp == "test":  # kept for bootstrap confidence intervals
            json.dump(out[sp]["dets"], open(P.RESULTS / f"{a.name}_test_dets.json", "w"))
        out[sp].pop("dets", None)
    out["latency_ms_1080p_fp32"] = latency(m)
    json.dump(out, open(P.RESULTS / f"{a.name}.json", "w"), indent=2)
    print(a.name, "test mAP", out["test"]["coco"]["mAP"], "mAP50", out["test"]["coco"]["mAP50"],
          "latency", out["latency_ms_1080p_fp32"], "ms")


if __name__ == "__main__":
    main()
