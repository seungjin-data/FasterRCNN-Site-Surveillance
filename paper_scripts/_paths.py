"""Single source of every path used by the paper scripts (no machine-specific absolute paths).

Default layout (relative to the repository root):

    core/               training / model / dataset code
    paper_scripts/      this folder
    results/            per-model JSON, *_test_dets.json, bootstrap.json, ...
    runs/               checkpoints written by core/train.py (run_*.sh); not tracked
    work/               generated scene_split/ and yolo_data/; not tracked
    annotations/{train,valid,test}/   Roboflow COCO export (_annotations.coco.json; place the images next to it)
    weights/            model weights downloaded from the archive (see docs/MODEL_WEIGHTS.md); not tracked
    figures/            figure output

Every location can be overridden with an environment variable (SITE_DATA, RESULTS_DIR, RUNS_DIR, WORK_DIR,
WEIGHTS_DIR, FIG_DIR, MAIN_CKPT, ALL_SCENES_CKPT, SCENE_CKPT).
"""
import os
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]   # repository root


def _p(var, default):
    return Path(os.environ.get(var, default)).resolve()


CORE = PKG / "core"
DATA = _p("SITE_DATA", PKG / "annotations")
RESULTS = _p("RESULTS_DIR", PKG / "results")
RUNS = _p("RUNS_DIR", PKG / "runs")
WORK = _p("WORK_DIR", PKG / "work")
WEIGHTS = _p("WEIGHTS_DIR", PKG / "weights")
FIG = _p("FIG_DIR", PKG / "figures")

# checkpoints used by the scripts that load a fixed model
MAIN_CKPT = _p("MAIN_CKPT", WEIGHTS / "faster_rcnn_100ep_MAIN_best.pth")         # 100-epoch schedule, epoch-65 checkpoint
ALL_SCENES_CKPT = _p("ALL_SCENES_CKPT", WEIGHTS / "faster_rcnn_15ep_best.pth")   # standard 15-epoch model
SCENE_CKPT = _p("SCENE_CKPT", RUNS / "scene" / "fasterrcnn" / "best.pth")         # scene-held-out model (not released)
