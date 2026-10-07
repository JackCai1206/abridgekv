#!/bin/bash
# LightThinker baseline in our stack: thought boundaries at paragraph breaks, 9 cache tokens per thought at appended
# positions, same data, recipe and budget as every other arm.  bash launch/train_lightthinker.sh
set -euo pipefail
source "$(dirname "$0")/env.sh"; cd "$REPO"
RUN_DIR=${RUN_DIR:-$OUTPUT_DIR/runs/lightthinker}; mkdir -p "$RUN_DIR"
NPROC=${NPROC:-$(nvidia-smi -L | wc -l)}
torchrun --nproc_per_node="$NPROC" --master_port=$((29500 + RANDOM % 1000)) abridgekv/sft_lightthinker.py \
  --base "$BASE_MODEL" --pairs "$POOL" --C 512 --nmax 32 --evaln 192 --micro "${MICRO:-2}" --eff_batch 256 \
  --data_stream canonical:8x1x32 --optimizer zero_adamw --lr 1e-4 --lr_schedule wsd --decay_frac 0.2 \
  --max_resp_tok 2000000000 --ce_chunk 4096 --grad_ckpt 1 --log_every 10 \
  --save_dir "$RUN_DIR" --save_every 25 --eval_every 50 --n_keep 1 2>&1 | tee -a "$RUN_DIR/train.log"
