# Reproducibility

All commands are run from the repository root. Every path is resolved in `paper_scripts/_paths.py` and can be
overridden with an environment variable. The overridable variables are `SITE_DATA`, `RESULTS_DIR`, `RUNS_DIR`,
`WORK_DIR`, `WEIGHTS_DIR`, `FIG_DIR`, `MAIN_CKPT`, `ALL_SCENES_CKPT` and `SCENE_CKPT`.

## 1. Environment

```bash
pip install -r requirements.txt
```

- **Study environment:** training ran on one NVIDIA GeForce RTX 3090 with PyTorch 2.5.1 and torchvision 0.20.1.
- **Matplotlib versions:** the figures were generated with matplotlib 3.10.9 (Fig. 1 and 3) and 3.11.2 (Fig. 2, 4–7).
  The same versions reproduce the PDFs pixel for pixel. Other versions may change cosmetic details only.

## 2. Dataset annotations

The COCO-format annotation files are in `annotations/{train,valid,test}/_annotations.coco.json`. Steps 3–5 need only
these files and `results/`. For training, re-inference and Fig. 7, also download the images of the same Roboflow export
(see [DATASET.md](DATASET.md)) and copy them next to the annotation files.

## 3. Evaluation from stored detections

The stored test-set detections of every evaluated model are in `results/<model>_test_dets.json`. The per-model metric
files are `results/<model>.json`; `results/results_summary.json` holds the off-the-shelf baseline. These outputs are
used as follows:

- `bootstrap_ci.py` and `make_figures_v2.py` recompute AP, AP50, APS, the precision–recall curves and AP versus IoU
  from the stored detections with pycocotools.
- `paper_scripts/eval_ckpt.py --ckpt <checkpoint> --name <model>` re-evaluates a checkpoint on the validation and test
  splits. This needs the images and the weights.

Model keys: `zeroshot` (legacy key for the off-the-shelf COCO-pretrained baseline), `fasterrcnn_100ep` (main model),
`fasterrcnn`, `fasterrcnn_s1`, `fasterrcnn_s2` (15-epoch seeds), `retinanet`, `fcos`, `ssdlite`, `yolov8m`,
`rtdetr_l`, `abl_noaug`, `abl_res640`.

## 4. Bootstrap

```bash
python paper_scripts/bootstrap_ci.py 1000
```

- **Method:** percentile bootstrap over the 94 test images (B = 1,000, `random.Random(0)`) for AP, AP50 and APS, plus
  paired differences.
- **Outputs:** `results/bootstrap.json` (rounded) and `results/bootstrap_full.json` (unrounded).
- **Warning:** the script overwrites `results/bootstrap.json`. Set `RESULTS_DIR` to a copy of `results/` to keep the
  shipped file.

## 5. Figures

```bash
python paper_scripts/make_figures_v2.py pipeline dataset training pr size tradeoff   # Fig. 1-6 -> figures/
python paper_scripts/make_qualitative.py                                            # Fig. 7 (images + main weights)
```

| Figure | File | Script and inputs |
|---|---|---|
| Fig. 1 | `figures/fig_dataset.pdf` | `make_figures_v2.py dataset`: annotations |
| Fig. 2 | `figures/fig_pipeline.pdf` | `make_figures_v2.py pipeline`: diagram only |
| Fig. 3 | `figures/training_curves.pdf` | `make_figures_v2.py training`: `results/main_model_history_with_ep65.json` |
| Fig. 4 | `figures/fig_size.pdf` | `make_figures_v2.py size`: `results/*.json` |
| Fig. 5 | `figures/fig_pr_iou.pdf` | `make_figures_v2.py pr`: stored detections + test annotations |
| Fig. 6 | `figures/fig_tradeoff_ci.pdf` | `make_figures_v2.py tradeoff`: `results/*.json`, `results/bootstrap.json` |
| Fig. 7 | `figures/qualitative.pdf` | `make_qualitative.py`: test images + `weights/faster_rcnn_100ep_MAIN_best.pth` (CPU or GPU inference) |

Fig. 7 is written at 400 dpi by default (`QUAL_DPI`), which keeps both image crops at or above their native
resolution. Setting `QUAL_DPI=220` reproduces the earlier 220-dpi version. Both versions draw identical boxes, crop
windows and titles. Panel (a) shows 17 annotated persons and 16 detections with score ≥ 0.5; panel (b) shows 9 and 7.
The counts are detections, not matched true positives.

## 6. Optional: inference and training

- **Inference:** see [`inference_example/README.md`](../inference_example/README.md) and
  [MODEL_WEIGHTS.md](MODEL_WEIGHTS.md).
- **Training queues (GPU and images required):** `paper_scripts/run_phase1.sh` (four torchvision detectors, 15
  epochs), `run_phase2.sh` / `run_phase2b.sh` (seeds, ablations, bootstrap) and `run_phase3.sh` (scene-held-out model).
- **Scene split:** `paper_scripts/scene_cluster.py` (HSV histograms, k-means with K = 10) and `scene_split.py`
  (held-out clusters 3, 5, 8, 9).
- **Ultralytics reference runs:** `paper_scripts/yolo_prep.py` and `ultra_run.py` (Ultralytics 8.4.87, AGPL-3.0).
- **Determinism:** training uses `torch.manual_seed` with cuDNN benchmark mode enabled. Runs are therefore not
  bit-identical.

## Provenance notes

- **Epoch-65 history record.** `results/main_model_history.json` is the raw training history (99 records). The 100-epoch
  run resumed from the epoch-65 checkpoint. The checkpoint is saved before the epoch's history record is appended, so
  the epoch-65 record is missing from the raw file. `results/main_model_history_with_ep65.json` adds that record from
  `results/main_model_train.log`, which contains:

  ```text
  [Epoch 65] loss=0.1817 mAP=0.4446 mAP50=0.8082 mAP_small=0.3668 (91.9s)
  ```

  The learning rate is not logged on evaluation epochs and is left as `null`. The derived file records the SHA-256 of
  both source files. Fig. 3 reads the derived file.
- **Scene-held-out model.** The historical scene-held-out checkpoint was not preserved. Its split definition
  (`results/scene_clusters.json`) and summary results (`results/scene_eval.json`) are kept, but the model cannot be
  independently re-inferred.

## Verification record (2026-10-07, this repository layout)

| Check | Result |
|---|---|
| `make_figures_v2.py` (Fig. 1–6) vs `figures/*.pdf`, rasterised at 150 dpi | 6/6 pixel-identical (max difference 0) |
| `make_qualitative.py` (default 400 dpi) vs `figures/qualitative.pdf` | byte-identical |
| `bootstrap_ci.py 20` (smoke test) vs shipped `results/bootstrap.json` | per-model point estimates identical for all 10 models; same 11 pairs |
| `inference_example/infer.py` on test image `479_jpg…` with the main weights (CPU, conf 0.5) | 16 persons detected, matching Fig. 7(a) |
