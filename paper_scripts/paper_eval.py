"""Paper evaluation: fine-tuned vs off-the-shelf COCO-pretrained Faster R-CNN, on valid and test.
(Legacy result key 'zeroshot_coco' retained for compatibility = off-the-shelf COCO-pretrained baseline.)"""
import sys, json, time, contextlib, io
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _paths as P
sys.path.insert(0, str(P.CORE))
import torch, torchvision
from torch.utils.data import DataLoader
from pycocotools.cocoeval import COCOeval
from dataset import CocoPersonDataset, get_transforms, collate_fn
from model import build_model

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_sync = (lambda: torch.cuda.synchronize()) if DEV.type == "cuda" else (lambda: None)
ROOT = str(P.DATA)
KEYS = ["mAP","mAP50","mAP75","mAP_small","mAP_medium","mAP_large","AR1","AR10","AR100","AR_small","AR_medium","AR_large"]

def load_ft():
    ck = torch.load(P.MAIN_CKPT, map_location="cpu")
    m = build_model("fasterrcnn", 2, pretrained=False)
    m.load_state_dict(ck["model"]); print("ckpt epoch", ck["epoch"], "best_map", ck["best_map"])
    return m.eval().to(DEV)

def load_zs():
    m = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights="DEFAULT")
    return m.eval().to(DEV)  # COCO label 1 == person, same id as our GT

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
            res.append({"image_id": iid, "category_id": 1, "bbox": [bb[0], bb[1], bb[2]-bb[0], bb[3]-bb[1]], "score": ss})
    gt = ds.coco
    with contextlib.redirect_stdout(io.StringIO()):
        E = COCOeval(gt, gt.loadRes(res), "bbox"); E.params.catIds = [1]
        E.evaluate(); E.accumulate(); E.summarize()
    stats = dict(zip(KEYS, [round(float(x), 4) for x in E.stats]))
    # P/R/F1 at conf thresholds, IoU 0.5 greedy matching
    prf = {}
    for conf in (0.3, 0.5, 0.7):
        tp = fp = fn = 0
        for iid, (b, s) in preds.items():
            b = b[s >= conf]
            g = gt.loadAnns(gt.getAnnIds(imgIds=iid, catIds=[1]))
            gb = torch.tensor([[a["bbox"][0], a["bbox"][1], a["bbox"][0]+a["bbox"][2], a["bbox"][1]+a["bbox"][3]] for a in g]).reshape(-1, 4)
            if len(b) and len(gb):
                iou = torchvision.ops.box_iou(b, gb); used = set(); m = 0
                for i in range(len(b)):
                    j = int(iou[i].argmax())
                    if iou[i, j] >= 0.5 and j not in used: used.add(j); m += 1
                tp += m; fp += len(b) - m; fn += len(gb) - m
            else:
                fp += len(b); fn += len(gb)
        p = tp / max(tp+fp, 1); r = tp / max(tp+fn, 1); f = 2*p*r / max(p+r, 1e-9)
        prf[str(conf)] = {"P": round(p,4), "R": round(r,4), "F1": round(f,4), "TP": tp, "FP": fp, "FN": fn}
    return stats, prf

@torch.no_grad()
def latency(model, n=100):
    x = [torch.rand(3, 1080, 1920, device=DEV)]
    for _ in range(10): model(x)
    _sync(); t = time.time()
    for _ in range(n): model(x)
    _sync(); return round((time.time()-t)/n*1000, 1)

out = {"gpu": torch.cuda.get_device_name(0) if DEV.type == "cuda" else "cpu", "torch": torch.__version__, "torchvision": torchvision.__version__}
for name, loader in (("finetuned", load_ft), ("zeroshot_coco", load_zs)):
    m = loader(); out[name] = {}
    for sp in ("valid", "test"):
        st, prf = run(m, sp); out[name][sp] = {"coco": st, "prf": prf}; print(name, sp, st["mAP"], st["mAP50"], prf["0.5"])
    out[name]["latency_ms_1080p_fp32"] = latency(m); print(name, "latency", out[name]["latency_ms_1080p_fp32"])
    out[name]["params_M"] = round(sum(p.numel() for p in m.parameters())/1e6, 1)
json.dump(out, open(P.RESULTS / "results_summary.json", "w"), indent=2)
