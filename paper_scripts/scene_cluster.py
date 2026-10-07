"""Group the 1,680 images into visual 'scenes' (colour-histogram clustering) to build a scene-held-out split.

Colour histograms are invariant to the provider's flips/rotations, so augmented copies of one source frame
fall into the same cluster. Output: paper/phase1/scene_clusters.json
"""
import json
import os
import re
import sys

import cv2
import numpy as np
from sklearn.cluster import KMeans

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _paths as P
DATA = str(P.DATA)
K = int(sys.argv[1]) if len(sys.argv) > 1 else 10


def hist_feature(path):
    img = cv2.imread(path, cv2.IMREAD_REDUCED_COLOR_8)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1, 2], None, [12, 4, 4], [0, 180, 0, 256, 0, 256]).ravel()
    return np.sqrt(h / h.sum())  # Hellinger map


rows = []
for sp in ("train", "valid", "test"):
    d = json.load(open(f"{DATA}/{sp}/_annotations.coco.json"))
    for im in d["images"]:
        src = re.match(r"(\d+)_jpg", im["file_name"]).group(1)
        rows.append({"split": sp, "file": im["file_name"], "src": f"{sp if sp != 'train' else 'tv'}:{src}"
                     if False else src, "id": im["id"]})

# source ids are unique across splits (verified earlier), so the numeric id identifies the source frame
feats = {}
for r in rows:
    f = hist_feature(f"{DATA}/{r['split']}/{r['file']}")
    r["feat"] = f
    feats.setdefault(r["src"], []).append(f)
srcs = sorted(feats)
X = np.stack([np.mean(feats[s], axis=0) for s in srcs])
km = KMeans(n_clusters=K, n_init=20, random_state=0).fit(X)
cluster_of = {s: int(c) for s, c in zip(srcs, km.labels_)}

summary = {}
for r in rows:
    c = cluster_of[r["src"]]
    summary.setdefault(c, {"train": 0, "valid": 0, "test": 0, "sources": set()})
    summary[c][r["split"]] += 1
    summary[c]["sources"].add(r["src"])
print(f"K={K}")
for c in sorted(summary):
    s = summary[c]
    print(f"cluster {c}: sources={len(s['sources']):3d} images train/valid/test = {s['train']}/{s['valid']}/{s['test']}")

out = {"K": K, "cluster_of_source": cluster_of,
       "files": [{"split": r["split"], "file": r["file"], "src": r["src"], "cluster": cluster_of[r["src"]]} for r in rows]}
json.dump(out, open(P.RESULTS / "scene_clusters.json", "w"))
