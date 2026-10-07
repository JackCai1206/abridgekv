#!/bin/bash
# Train one arm of the paper on one node (8x H100-80GB or 4x GB300; the data order is independent of the GPU count).
#   ARM=perlayer bash launch/train.sh     # AbridgeKV: C=512, K_max=256, per-layer quotas, mean 128 rows per block
#   ARM=uniform  bash launch/train.sh     # 128 rows in every layer
#   ARM=appended bash launch/train.sh     # ablation: uniform with code rows at appended positions
#   ARM=vanilla  bash launch/train.sh     # full-KV reference
# Optional: C (block size, default 512), K_MEAN (mean rows per block, default 128), SEED, MICRO, NPROC, RUN_DIR,
# MAX_RESP_TOK (default 2e9 response tokens; the prototyping runs of the coding appendix use 6e8), POOL.
# Re-running the same command resumes from the newest complete checkpoint in RUN_DIR.
set -euo pipefail
source "$(dirname "$0")/env.sh"; cd "$REPO"
ARM=${ARM:?ARM=perlayer|uniform|appended|vanilla}; C=${C:-512}; K_MEAN=${K_MEAN:-128}; SEED=${SEED:-0}
NPROC=${NPROC:-$(nvidia-smi -L | wc -l)}
export RAW_PROMPT=1 SEED GRAD_CKPT=1 DATA_STREAM=canonical:8x1x32 DECAY_FRAC=0.2 DATA_LIMIT=0 SAVE_OPT_EVERY=25 \
       ACTIVE_CODE_ROWS=auto CODE_POSITIONS=bitrev
case $ARM in
  vanilla)  ARGS=(--arm baseline --K 128) ;;
  uniform)  ARGS=(--arm compressed --K 256 --head_code_quotas "uniform:$K_MEAN" --head_code_quota_budget "$K_MEAN") ;;
  perlayer) ARGS=(--arm compressed --K 256 --head_code_quotas "${SCHEDULE:-$PER_LAYER_SCHEDULE}" --head_code_quota_budget "$K_MEAN") ;;
  appended) ARGS=(--arm compressed --K "$K_MEAN" --head_code_quotas "uniform:$K_MEAN" --head_code_quota_budget "$K_MEAN"); CODE_POSITIONS=appended ;;
  *) echo "unknown ARM=$ARM"; exit 2 ;;
esac
SUF=""; [ "$SEED" = 0 ] || SUF=_seed$SEED
RUN_DIR=${RUN_DIR:-$OUTPUT_DIR/runs/${ARM}_c${C}_k${K_MEAN}$SUF}; mkdir -p "$RUN_DIR"
CKPT=$(ls -d "$RUN_DIR"/step*/ 2>/dev/null | sort -V | while read d; do [ -e "$d/model.pt.complete" ] && echo "$d/model.pt"; done | tail -1)
echo "[train] arm=$ARM C=$C mean_rows=$K_MEAN seed=$SEED nproc=$NPROC run_dir=$RUN_DIR resume=${CKPT:-none}"
torchrun --nproc_per_node="$NPROC" --master_port=$((29500 + RANDOM % 1000)) abridgekv/sft_ddp.py \
  --base "$BASE_MODEL" "${ARGS[@]}" ${CKPT:+--ckpt "$CKPT"} --pairs "$POOL" --C "$C" --L 1 --pause_xattn full \
  --nmax $((16384 / C)) --evaln 192 --cross_block_loss 1 --micro "${MICRO:-8}" --eff_batch 256 --lr 1e-4 --lr_schedule wsd \
  --max_steps 2000000 --max_resp_tok "${MAX_RESP_TOK:-2000000000}" --compile 0 --ce_chunk "${CE_CHUNK:-4096}" \
  --optimizer zero_adamw --save "$RUN_DIR/model.pt" --save_dir "$RUN_DIR" --save_every 25 --eval_every 50 --n_keep 2 \
  --log_every 10 2>&1 | tee -a "$RUN_DIR/train.log"
