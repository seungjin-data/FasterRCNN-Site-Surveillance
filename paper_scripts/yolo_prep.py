"""Convert the COCO person dataset to YOLO format (hard links for images) for the Ultralytics baselines."""
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _paths as P

DATA = str(P.DATA)
OUT = str(P.WORK / "yolo_data")

for sp in ("train", "valid", "test"):
    d = json.load(open(f"{DATA}/{sp}/_annotations.coco.json"))
    os.makedirs(f"{OUT}/images/{sp}", exist_ok=True)
    os.makedirs(f"{OUT}/labels/{sp}", exist_ok=True)
    by = {}
    for a in d["annotations"]:
        if a["category_id"] == 1:
            by.setdefault(a["image_id"], []).append(a["bbox"])
    for im in d["images"]:
        src, dst = f"{DATA}/{sp}/{im['file_name']}", f"{OUT}/images/{sp}/{im['file_name']}"
        if not os.path.exists(dst):
            os.link(src, dst)
        W, H = im["width"], im["height"]
        lines = []
        for x, y, w, h in by.get(im["id"], []):
            if w <= 0 or h <= 0:
                continue
            lines.append(f"0 {(x + w / 2) / W:.6f} {(y + h / 2) / H:.6f} {w / W:.6f} {h / H:.6f}")
        stem = os.path.splitext(im["file_name"])[0]
        open(f"{OUT}/labels/{sp}/{stem}.txt", "w").write("\n".join(lines) + ("\n" if lines else ""))

open(f"{OUT}/data.yaml", "w").write(
    f"path: {OUT}\ntrain: images/train\nval: images/valid\ntest: images/test\nnames:\n  0: person\n")
print("yolo dataset ready:", OUT)
