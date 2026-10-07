"""Build the scene-held-out split from paper/phase1/scene_clusters.json.

train : training images whose scene cluster is NOT held out
valid : validation images of non-held-out clusters (model selection)
test  : valid+test images of held-out clusters (never seen scenes; unaugmented images only)
Images are hard-linked, so no disk space is used.
"""
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _paths as P

DATA = str(P.DATA)
OUT = str(P.WORK / "scene_split")
HELD_OUT = {3, 5, 8, 9}

cl = json.load(open(P.RESULTS / "scene_clusters.json"))
cluster = {(f["split"], f["file"]): f["cluster"] for f in cl["files"]}

target = {"train": [], "valid": [], "test": []}
for sp in ("train", "valid", "test"):
    d = json.load(open(f"{DATA}/{sp}/_annotations.coco.json"))
    for im in d["images"]:
        held = cluster[(sp, im["file_name"])] in HELD_OUT
        if sp == "train":
            if not held:
                target["train"].append((sp, im, d))
        elif held:
            target["test"].append((sp, im, d))
        else:
            target["valid"].append((sp, im, d))

stats = {}
for name, items in target.items():
    os.makedirs(f"{OUT}/{name}", exist_ok=True)
    images, anns = [], []
    ids_seen = set()
    for sp, im, d in items:
        new_id = len(images) + 1
        src = f"{DATA}/{sp}/{im['file_name']}"
        dst = f"{OUT}/{name}/{im['file_name']}"
        if not os.path.exists(dst):
            os.link(src, dst)
        images.append({**im, "id": new_id})
        ids_seen.add((sp, im["id"], new_id))
    lookup = {(sp, old): new for sp, old, new in ids_seen}
    for sp in ("train", "valid", "test"):
        d = json.load(open(f"{DATA}/{sp}/_annotations.coco.json"))
        for a in d["annotations"]:
            key = (sp, a["image_id"])
            if key in lookup and a["category_id"] == 1:
                anns.append({**a, "id": len(anns) + 1, "image_id": lookup[key]})
    cats = [{"id": 1, "name": "person", "supercategory": "site-dataset"}]
    json.dump({"images": images, "annotations": anns, "categories": cats},
              open(f"{OUT}/{name}/_annotations.coco.json", "w"))
    stats[name] = {"images": len(images), "boxes": len(anns)}
print(json.dumps(stats))
print("held-out clusters:", sorted(HELD_OUT))
