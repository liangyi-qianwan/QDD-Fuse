#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-python}
DATA_ROOT=${DATA_ROOT:-data}
DEVICE=${DEVICE:-cuda}
BATCH_SIZE=${BATCH_SIZE:-64}
NUM_WORKERS=${NUM_WORKERS:-0}
OUT_DIR=${OUT_DIR:-results/retest}

mkdir -p "$OUT_DIR"

for ds in mosi mosei simsv2; do
  "$PYTHON" scripts/retest_pnr_checkpoint.py \
    --checkpoint "checkpoints/$ds/best.pt" \
    --output-json "$OUT_DIR/${ds}_retest.json" \
    --data-root "$DATA_ROOT" \
    --device "$DEVICE" \
    --batch-size "$BATCH_SIZE" \
    --num-workers "$NUM_WORKERS"
done
