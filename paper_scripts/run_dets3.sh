#!/bin/bash
cd "$(dirname "$0")/.."   # = repository root (paths below are relative to it)
DATA_DIR="${SITE_DATA:-annotations}"
for A in retinanet fcos ssdlite; do
  python -u paper_scripts/eval_ckpt.py --ckpt runs/phase1/$A/best.pth --name $A
done
echo "===== DETS3 DONE ====="
