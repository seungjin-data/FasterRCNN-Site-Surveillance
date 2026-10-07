#!/bin/bash
cd "$(dirname "$0")/.."   # = repository root (paths below are relative to it)
DATA_DIR="${SITE_DATA:-annotations}"
echo "===== $(date +%T) TRAIN abl_res1080 ====="
python -u core/train.py --data-dir "$DATA_DIR" --arch fasterrcnn --epochs 15 --warmup-epochs 1 \
  --workers 4 --eval-interval 5 --output-dir runs/phase2/abl_res1080 --batch-size 2 --grad-accum 10 --seed 42 \
  --min-size 1080 --max-size 1920
echo "===== $(date +%T) EVAL abl_res1080 ====="
python -u paper_scripts/eval_ckpt.py --ckpt runs/phase2/abl_res1080/best.pth --name abl_res1080
echo "===== $(date +%T) RES1080 DONE ====="
