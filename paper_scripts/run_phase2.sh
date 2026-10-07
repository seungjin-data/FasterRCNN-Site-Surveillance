#!/bin/bash
# Phase 2: seeds + ablations for Faster R-CNN (15 epochs, same recipe as phase 1), then detection dumps.
cd "$(dirname "$0")/.."   # = repository root (paths below are relative to it)
DATA_DIR="${SITE_DATA:-annotations}"
DATA="$DATA_DIR"
COMMON="--data-dir \"$DATA\" --arch fasterrcnn --epochs 15 --warmup-epochs 1 --workers 4 --eval-interval 5"

job() {  # name, extra args...
  NAME=$1; shift
  echo "===== $(date +%T) TRAIN $NAME ====="
  eval python -u core/train.py $COMMON --output-dir runs/phase2/$NAME "$@"
  echo "===== $(date +%T) EVAL $NAME ====="
  python -u paper_scripts/eval_ckpt.py --ckpt runs/phase2/$NAME/best.pth --name $NAME
}

job fasterrcnn_s1   --batch-size 4 --grad-accum 5 --seed 1
job fasterrcnn_s2   --batch-size 4 --grad-accum 5 --seed 2
job abl_noaug       --batch-size 4 --grad-accum 5 --seed 42 --no-online-aug
job abl_res640      --batch-size 4 --grad-accum 5 --seed 42 --min-size 640 --max-size 1066
job abl_res1080     --batch-size 2 --grad-accum 10 --seed 42 --min-size 1080 --max-size 1920

echo "===== $(date +%T) DETS for bootstrap ====="
python -u paper_scripts/eval_ckpt.py --ckpt runs/phase1/fasterrcnn/best.pth --name fasterrcnn
python -u paper_scripts/eval_ckpt.py --ckpt runs/person_detector/best.pth --name fasterrcnn_100ep
python -u paper_scripts/zeroshot_dets.py
echo "===== $(date +%T) PHASE2 DONE ====="
