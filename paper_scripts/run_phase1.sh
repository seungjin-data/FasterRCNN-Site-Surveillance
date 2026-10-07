#!/bin/bash
# Phase 1: same recipe (15 epochs, 1 warmup epoch, eff. batch 20, seed 42) for 4 architectures.
cd "$(dirname "$0")/.."   # = repository root (paths below are relative to it)
DATA_DIR="${SITE_DATA:-annotations}"
DATA="$DATA_DIR"
for ARCH in ssdlite fasterrcnn retinanet fcos; do
  if [ "$ARCH" = "ssdlite" ]; then BS=20; GA=1; else BS=4; GA=5; fi
  echo "===== $(date +%T) TRAIN $ARCH ====="
  python -u core/train.py --data-dir "$DATA" --arch $ARCH --epochs 15 --warmup-epochs 1 \
      --batch-size $BS --grad-accum $GA --workers 4 --eval-interval 5 \
      --output-dir runs/phase1/$ARCH --seed 42
  echo "===== $(date +%T) EVAL $ARCH ====="
  python -u paper_scripts/eval_ckpt.py --ckpt runs/phase1/$ARCH/best.pth --name $ARCH
done
echo "===== $(date +%T) PHASE1 DONE ====="
