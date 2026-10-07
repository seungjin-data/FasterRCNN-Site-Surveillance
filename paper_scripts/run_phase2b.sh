#!/bin/bash
# Phase 2b: detection dumps for bootstrap, then bootstrap CIs (1080p ablation skipped).
cd "$(dirname "$0")/.."   # = repository root (paths below are relative to it)
DATA_DIR="${SITE_DATA:-annotations}"
echo "===== $(date +%T) DETS ====="
python -u paper_scripts/eval_ckpt.py --ckpt runs/phase1/fasterrcnn/best.pth --name fasterrcnn
python -u paper_scripts/eval_ckpt.py --ckpt runs/person_detector/best.pth --name fasterrcnn_100ep
python -u paper_scripts/zeroshot_dets.py
python -u paper_scripts/eval_ckpt.py --ckpt runs/phase2/abl_noaug/best.pth --name abl_noaug
python -u paper_scripts/eval_ckpt.py --ckpt runs/phase2/abl_res640/best.pth --name abl_res640
echo "===== $(date +%T) BOOTSTRAP ====="
python -u paper_scripts/bootstrap_ci.py 1000 > results/bootstrap_stdout.txt
echo "===== $(date +%T) PHASE2B DONE ====="
