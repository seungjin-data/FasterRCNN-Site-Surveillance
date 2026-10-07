#!/bin/bash
# Phase 3: scene-held-out Faster R-CNN, YOLOv8m and RT-DETR-L baselines, then the 1080p ablation.
cd "$(dirname "$0")/.."   # = repository root (paths below are relative to it)
DATA_DIR="${SITE_DATA:-annotations}"
DATA="./work/scene_split"
echo "===== $(date +%T) TRAIN scene ====="
python -u core/train.py --data-dir "$DATA" --arch fasterrcnn --epochs 15 --warmup-epochs 1 --batch-size 4 --grad-accum 5 \
  --workers 4 --eval-interval 5 --output-dir runs/scene/fasterrcnn --seed 42
echo "===== $(date +%T) EVAL scene ====="
python -u paper_scripts/eval_scene.py
echo "===== $(date +%T) PREP yolo data ====="
python -u paper_scripts/yolo_prep.py
echo "===== $(date +%T) TRAIN yolov8m ====="
python -u paper_scripts/ultra_run.py yolo yolov8m
echo "===== $(date +%T) TRAIN rtdetr ====="
python -u paper_scripts/ultra_run.py rtdetr rtdetr_l
echo "===== $(date +%T) TRAIN abl_res1080 ====="
rm -rf runs/phase2/abl_res1080
python -u core/train.py --data-dir "$DATA_DIR" --arch fasterrcnn --epochs 15 --warmup-epochs 1 \
  --workers 4 --eval-interval 5 --output-dir runs/phase2/abl_res1080 --batch-size 2 --grad-accum 10 --seed 42 \
  --min-size 1080 --max-size 1920
python -u paper_scripts/eval_ckpt.py --ckpt runs/phase2/abl_res1080/best.pth --name abl_res1080
echo "===== $(date +%T) PHASE3 DONE ====="
