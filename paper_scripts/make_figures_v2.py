"""Publication figures (IEEE style) for paper.tex. Needs results/*.json, results/*_test_dets.json, results/bootstrap.json,
results/main_model_history_with_ep65.json and the dataset annotations (paths: _paths.py)."""
import contextlib
import io
import json
import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _paths as P
P1 = str(P.RESULTS)
FIG = str(P.FIG)
DATA = str(P.DATA)

plt.rcParams.update({
    "font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 8,
    "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.4,
    "legend.frameon": False, "pdf.fonttype": 42, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
})

# Okabe-Ito colour-blind-safe palette
# key "zeroshot" = legacy identifier of the off-the-shelf COCO-pretrained baseline (kept for file compatibility)
COL = {"zeroshot": "#7F7F7F", "ssdlite": "#CC79A7", "retinanet": "#E69F00", "fcos": "#009E73",
       "yolov8m": "#6A3D9A", "rtdetr_l": "#B2182B",
       "fasterrcnn": "#56B4E9", "fasterrcnn_100ep": "#0072B2", "abl_noaug": "#D55E00", "abl_res640": "#8C6D1F"}
LAB = {"zeroshot": "Off-the-shelf COCO-pretrained", "ssdlite": "SSDLite", "retinanet": "RetinaNet", "fcos": "FCOS",
       "yolov8m": "YOLOv8m", "rtdetr_l": "RT-DETR-L",
       "fasterrcnn": "Faster R-CNN (15 ep)", "fasterrcnn_100ep": "Faster R-CNN (100 ep, main)",
       "abl_noaug": "15 ep, no online aug.", "abl_res640": "15 ep, 640 px input"}
MAIN = ["zeroshot", "ssdlite", "retinanet", "fcos", "yolov8m", "rtdetr_l", "fasterrcnn", "fasterrcnn_100ep"]


def load_results():
    res = {n: json.load(open(f"{P1}/{n}.json")) for n in MAIN[1:]}
    summ = json.load(open(f"{P1}/results_summary.json"))
    res["zeroshot"] = {"test": summ["zeroshot_coco"]["test"],
                       "latency_ms_1080p_fp32": json.load(open(f"{P1}/zeroshot_latency.json"))["zeroshot_latency_ms"]}
    return res


def save(fig, name):
    fig.savefig(f"{FIG}/{name}.pdf", bbox_inches="tight")
    fig.savefig(f"{FIG}/{name}.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def fig_pipeline():
    fig, ax = plt.subplots(figsize=(7.1, 2.05))
    ax.set_xlim(0, 100); ax.set_ylim(0, 34); ax.axis("off")
    blocks = [("Site image\nRGB, mostly\n1920\u00d71080", "#EEEEEE"),
              ("Online aug.\n(training only)\nphotometric,\nzoom-out,\nIoU crop, h-flip", "#FCE5CD"),
              ("Resize\nshorter side 800\nlonger \u2264 1,333", "#CFE2F3"),
              ("ResNet-50\n+ FPN\n(P2\u2013P6)", "#CFE2F3"),
              ("RPN\nanchors \u2192\nproposals", "#CFE2F3"),
              ("RoIAlign +\nbox head\n(person / bg)", "#CFE2F3"),
              ("NMS\nscore \u2265 0.05\n\u2264 100 boxes", "#D9EAD3")]
    n = len(blocks); gap = 2.2; w = (100 - 0.8 - (n - 1) * gap) / n; y0, h = 9.5, 14.5
    xs = [0.4 + i * (w + gap) for i in range(n)]
    for x, (txt, fc) in zip(xs, blocks):
        ax.add_patch(FancyBboxPatch((x, y0), w, h, boxstyle="round,pad=0.2,rounding_size=1.0",
                                    fc=fc, ec="#444444", lw=0.7))
        ax.text(x + w / 2, y0 + h / 2, txt, ha="center", va="center", fontsize=6.0, linespacing=1.25)
    for i in range(n - 1):
        ax.add_patch(FancyArrowPatch((xs[i] + w + 0.3, y0 + h / 2), (xs[i + 1] - 0.3, y0 + h / 2),
                                     arrowstyle="-|>", mutation_scale=6, lw=0.8, color="#444444"))
    # COCO-pretrained initialisation of backbone and heads (arrow from above, label to its right)
    xa = xs[3] + w / 2
    ax.add_patch(FancyArrowPatch((xa, 32.0), (xa, y0 + h + 0.5), arrowstyle="-|>", mutation_scale=6,
                                 lw=0.8, color="#0072B2"))
    ax.text(xa + 1.0, 29.0, "COCO-pretrained initialisation\n(new 2-class box predictor)", fontsize=6.0,
            color="#0072B2", ha="left", va="center", linespacing=1.2)

    def bracket(xa, xb, text, color):
        ax.plot([xa, xa, xb, xb], [7.6, 6.6, 6.6, 7.6], color=color, lw=0.8)
        ax.text((xa + xb) / 2, 4.6, text, ha="center", va="top", fontsize=6.1, color=color)
    bracket(xs[0], xs[1] + w, "Data: 1,680 images, 6,824 persons", "#666666")
    bracket(xs[2], xs[5] + w, "Faster R-CNN R50-FPN v2 (torchvision), 43.3 M parameters", "#0072B2")
    bracket(xs[6], xs[6] + w, "Evaluation", "#38761D")
    save(fig, "fig_pipeline")


def fig_dataset():
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.3))
    cols = {"train": "#0072B2", "valid": "#E69F00", "test": "#009E73"}
    bins_s = np.logspace(np.log10(4), np.log10(500), 36)
    bins_n = np.arange(0, 27) - 0.5
    for sp in ("train", "valid", "test"):
        d = json.load(open(f"{DATA}/{sp}/_annotations.coco.json"))
        scale = [np.sqrt(a["bbox"][2] * a["bbox"][3]) for a in d["annotations"] if a["category_id"] == 1]
        per = np.zeros(len(d["images"]), dtype=int)
        idx = {im["id"]: k for k, im in enumerate(d["images"])}
        for a in d["annotations"]:
            if a["category_id"] == 1:
                per[idx[a["image_id"]]] += 1
        n = f"{sp} ({len(scale):,} boxes)"
        axes[0].hist(scale, bins=bins_s, density=True, histtype="step", lw=1.2, color=cols[sp], label=n)
        axes[1].hist(np.clip(per, 0, 25), bins=bins_n, density=True, histtype="step", lw=1.2, color=cols[sp],
                     label=f"{sp} ({len(per)} img.)")
    a0, a1 = axes
    a0.set_xscale("log"); a0.set_xlabel(r"Box scale $\sqrt{wh}$ (pixels, log axis)"); a0.set_ylabel("Density")
    for v, t in ((32, "32"), (96, "96")):
        a0.axvline(v, color="k", ls=":", lw=0.7)
    ymax = a0.get_ylim()[1]
    a0.text(16, ymax * 0.93, "small", ha="center", fontsize=7); a0.text(55, ymax * 0.93, "medium", ha="center", fontsize=7)
    a0.text(200, ymax * 0.93, "large", ha="center", fontsize=7)
    a0.legend(fontsize=6.5, loc="center right"); a0.set_title("(a) Box scale", fontsize=8)
    a1.set_xlabel("Persons per image (25 = 25 or more)"); a1.set_ylabel("Density")
    a1.legend(fontsize=6.5); a1.set_title("(b) Persons per image", fontsize=8)
    fig.tight_layout(); save(fig, "fig_dataset")


def fig_training():
    # derived file: raw main_model_history.json + epoch 65 recovered from main_model_train.log (see PROVENANCE_history.md)
    hist = json.load(open(f"{P1}/main_model_history_with_ep65.json"))["history"]
    ev = [h for h in hist if "mAP" in h]
    fig, ax1 = plt.subplots(figsize=(3.5, 2.35))
    ax1.plot([h["epoch"] for h in hist], [h["train_loss"] for h in hist], color="#0072B2", lw=1.1)
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Training loss", color="#0072B2")
    ax2 = ax1.twinx(); ax2.spines["right"].set_visible(True); ax2.grid(False)
    ax2.plot([h["epoch"] for h in ev], [h["mAP"] for h in ev], "o-", color="#D55E00", ms=2.6, lw=1.0)
    ax2.set_ylabel("Validation AP", color="#D55E00"); ax2.set_ylim(0.38, 0.47)
    best = max(ev, key=lambda h: h["mAP"])
    ax2.annotate(f"best: epoch {best['epoch']}\nAP {best['mAP']:.3f}", (best["epoch"], best["mAP"]),
                 (best["epoch"] - 5, 0.405), fontsize=6.5, ha="center", arrowprops=dict(arrowstyle="-", lw=0.5))
    fig.tight_layout(); save(fig, "training_curves")


def pr_data(name):
    gt = COCO(f"{DATA}/test/_annotations.coco.json")
    dets = json.load(open(f"{P1}/{name}_test_dets.json"))
    with contextlib.redirect_stdout(io.StringIO()):
        E = COCOeval(gt, gt.loadRes(dets), "bbox"); E.params.catIds = [1]
        E.evaluate(); E.accumulate()
    prec = E.eval["precision"][:, :, 0, 0, 2]  # [T, R]
    ap_iou = [float(np.mean(p[p > -1])) for p in prec]
    return E.params.recThrs, prec[0], ap_iou


def fig_pr_iou():
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.55))
    out = {}
    for n in MAIN:
        r, p50, apiou = pr_data(n); out[n] = apiou
        lw = 1.8 if n == "fasterrcnn_100ep" else 1.1
        axes[0].plot(r, p50, color=COL[n], lw=lw, label=LAB[n])
        axes[1].plot(np.arange(0.5, 0.96, 0.05), apiou, "o-", color=COL[n], lw=lw, ms=2.5)
    axes[0].set_xlabel("Recall"); axes[0].set_ylabel("Precision"); axes[0].set_xlim(0, 1); axes[0].set_ylim(0, 1.02)
    axes[0].set_title("(a) Precision–recall at IoU 0.50 (test)", fontsize=8); axes[0].legend(fontsize=6, loc="lower left")
    axes[1].set_xlabel("IoU threshold"); axes[1].set_ylabel("AP at threshold"); axes[1].set_ylim(0, 0.9)
    axes[1].set_title("(b) AP as a function of IoU threshold (test)", fontsize=8)
    fig.tight_layout(); save(fig, "fig_pr_iou")
    json.dump(out, open(f"{P1}/ap_vs_iou.json", "w"), indent=1)


def fig_size(res):
    keys = [("mAP_small", "Small"), ("mAP_medium", "Medium"), ("mAP_large", "Large"), ("mAP", "All")]
    fig, ax = plt.subplots(figsize=(7.1, 2.35))
    n = len(MAIN); w = 0.8 / n
    for i, m in enumerate(MAIN):
        vals = [res[m]["test"]["coco"][k] for k, _ in keys]
        xs = np.arange(len(keys)) + (i - (n - 1) / 2) * w
        bars = ax.bar(xs, vals, w * 0.92, color=COL[m], label=LAB[m])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.008, f"{v:.2f}", ha="center", fontsize=5.2, rotation=90)
    ax.set_xticks(range(len(keys))); ax.set_xticklabels([k[1] for k in keys])
    ax.set_ylabel("AP@[.5:.95] (test)"); ax.set_ylim(0, 0.68); ax.grid(axis="x", visible=False)
    ax.legend(fontsize=6.3, ncol=4, loc="upper left")
    fig.tight_layout(); save(fig, "fig_size")


def fig_tradeoff(res):
    # unrounded companion (bootstrap_full.json) avoids double rounding in the printed 3-decimal labels
    bf = f"{P1}/bootstrap_full.json"
    boot = json.load(open(bf if os.path.exists(bf) else f"{P1}/bootstrap.json"))["models"]
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.6), gridspec_kw={"width_ratios": [1, 1.05]})
    a = axes[0]
    offs = {"zeroshot": (5, -10), "ssdlite": (5, 4), "retinanet": (-38, -3), "fcos": (4, -11),
            "yolov8m": (-37, 2), "rtdetr_l": (-12, -12),
            "fasterrcnn": (-70, -1), "fasterrcnn_100ep": (-30, 7)}
    for m in MAIN:
        x = res[m]["latency_ms_1080p_fp32"]; y = res[m]["test"]["coco"]["mAP"]
        a.scatter(x, y, s=34, color=COL[m], zorder=3, edgecolor="white", lw=0.5)
        a.annotate(LAB[m].replace(" (100 ep, main)", " (100 ep)"), (x, y), textcoords="offset points", xytext=offs[m], fontsize=6.3)
    a.set_xlabel("Latency per 1920×1080 input (ms, RTX 3090, FP32)"); a.set_ylabel("AP@[.5:.95] (test)")
    a.set_xlim(0, 62); a.set_ylim(-0.02, 0.52); a.set_title("(a) Accuracy–speed trade-off", fontsize=8)
    b = axes[1]
    order = ["zeroshot", "ssdlite", "retinanet", "fcos", "yolov8m", "rtdetr_l", "abl_res640", "abl_noaug",
             "fasterrcnn", "fasterrcnn_100ep"]
    order = [m for m in order if m in boot]  # models without finished bootstrap are skipped
    for k, m in enumerate(order[::-1]):
        p, lo, hi = boot[m]["point"][0], boot[m]["lo"][0], boot[m]["hi"][0]
        b.errorbar(p, k, xerr=[[p - lo], [hi - p]], fmt="o", color=COL[m], ms=4.5, capsize=2.2, lw=1.1)
        b.text(hi + 0.012, k, f"{p:.3f} [{lo:.3f}, {hi:.3f}]", va="center", fontsize=6)
    b.set_yticks(range(len(order))); b.set_yticklabels([LAB[m] for m in order[::-1]], fontsize=6.6)
    b.set_xlabel("AP@[.5:.95] with 95% bootstrap CI (test)"); b.set_xlim(-0.02, 0.72); b.grid(axis="y", visible=False)
    b.set_title("(b) Uncertainty from test-set sampling", fontsize=8)
    fig.tight_layout(); save(fig, "fig_tradeoff_ci")


if __name__ == "__main__":
    which = sys.argv[1:] or ["pipeline", "dataset", "training", "pr", "size", "tradeoff"]
    res = load_results()
    if "pipeline" in which: fig_pipeline()
    if "dataset" in which: fig_dataset()
    if "training" in which: fig_training()
    if "pr" in which: fig_pr_iou()
    if "size" in which: fig_size(res)
    if "tradeoff" in which: fig_tradeoff(res)
    print("done", which)
