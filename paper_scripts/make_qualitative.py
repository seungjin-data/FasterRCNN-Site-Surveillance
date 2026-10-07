"""Qualitative figure: GT (green) vs detections >=0.5 (red) of the main model on test images."""
import sys, json
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _paths as P
sys.path.insert(0, str(P.CORE))
import torch, cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torchvision.io import read_image, ImageReadMode
from torchvision.transforms import v2
from model import build_model

ROOT = str(P.DATA / "test")
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ck = torch.load(P.MAIN_CKPT, map_location="cpu")
m = build_model("fasterrcnn", 2, pretrained=False); m.load_state_dict(ck["model"]); m.eval().to(dev)
coco = json.load(open(f"{ROOT}/_annotations.coco.json"))
by = {}
for a in coco["annotations"]:
    by.setdefault(a["image_id"], []).append(a["bbox"])
imgs = {i["id"]: i["file_name"] for i in coco["images"]}
order = sorted(by, key=lambda k: -len(by[k]))
picks = [order[2], order[12]]  # a crowded and a moderately busy scene
to_f = v2.ToDtype(torch.float32, scale=True)

plt.rcParams.update({"font.family": "STIXGeneral", "font.size": 8})
# 1) inference and crop windows first, so both panels can be drawn at the same height
panels = []
for iid in picks:
    img = read_image(f"{ROOT}/{imgs[iid]}", ImageReadMode.RGB)
    with torch.no_grad():
        o = m([to_f(img).to(dev)])[0]
    keep = o["scores"] >= 0.5
    pb = o["boxes"][keep].cpu().tolist()
    gb = [[x, y, x + w, y + h] for x, y, w, h in by[iid]]
    allb = gb + pb
    x0 = max(min(b[0] for b in allb) - 80, 0); y0 = max(min(b[1] for b in allb) - 60, 0)
    x1 = min(max(b[2] for b in allb) + 80, img.shape[2]); y1 = min(max(b[3] for b in allb) + 60, img.shape[1])
    panels.append((img, gb, pb, (x0, y0, x1, y1)))
# 2) width ratios = crop aspect ratios -> equal panel heights
ratios = [(c[2] - c[0]) / (c[3] - c[1]) for _, _, _, c in panels]
fig, axes = plt.subplots(1, 2, figsize=(7.1, 7.1 / (sum(ratios) * 1.04) + 0.3), gridspec_kw={"width_ratios": ratios})
for k, (ax, (img, gb, pb, (x0, y0, x1, y1))) in enumerate(zip(axes, panels)):
    ax.imshow(img.permute(1, 2, 0).numpy()); ax.set_xlim(x0, x1); ax.set_ylim(y1, y0); ax.axis("off")
    for b in gb:
        ax.add_patch(plt.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, ec="lime", lw=1.0))
    for b in pb:
        ax.add_patch(plt.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, ec="red", lw=0.9, ls="--"))
    ax.set_title(f"({'ab'[k]}) {len(gb)} annotated persons, {len(pb)} detections (score ≥ 0.5)", fontsize=8)
fig.tight_layout(pad=0.3, w_pad=0.8); fig.savefig(P.FIG / "qualitative.pdf", dpi=int(os.environ.get("QUAL_DPI", "400")), bbox_inches="tight"); fig.savefig(P.FIG / "qualitative.png", dpi=110, bbox_inches="tight")
print("saved", picks, [imgs[i] for i in picks])
