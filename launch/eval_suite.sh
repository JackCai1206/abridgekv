#!/bin/bash
# Evaluate one exported model on the 446-problem competition suite: 8 samples per problem at T=0.6, top-p 0.95,
# top-k 20, seed 1234, 31,744 new tokens (the 32K cap). One GPU; each set is written to $OUT/<set>.json.
#   MODEL=<hf dir> ARM=perlayer|uniform|vanilla|rkv TAG=<name> bash launch/eval_suite.sh
#   R-KV: ARM=rkv MODEL=<vanilla hf dir> RKV_BUDGET=4096, run inside R-KV's own vLLM port (github.com/Zefan-Cai/R-KV).
# Then: python3 scripts/summarize_suite.py --root $OUTPUT_DIR/eval --out <json> --md <md>
set -euo pipefail
source "$(dirname "$0")/env.sh"; cd "$REPO"
MODEL=${MODEL:?}; ARM=${ARM:?}; TAG=${TAG:?}
SETS=${SETS:-"aime2024_I aime2024_II aime2025 aime_2022 aime_2023 aime_2026 amc_2022_2023 hmmt_feb_2023 hmmt_feb_2024 hmmt_feb_2025 brumo_2025 cmimc_2025 smt_2025"}
export BOXED_INSTRUCTION=1 VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
EXTRA=(); MML=32768; EVAL_ARM=vanilla
case $ARM in
  perlayer) EXTRA=(--use-compresslm --compresslm-layer-quotas "${SCHEDULE:-$PER_LAYER_SCHEDULE}"); export COMPRESSLM_PHYSICAL_QUOTAS=1; MML=50176; EVAL_ARM=compressed ;;
  uniform)  EXTRA=(--use-compresslm); MML=43008; EVAL_ARM=compressed ;;
  vanilla)  ;;
  rkv)      export VLLM_V1_R_KV_BUDGET=${RKV_BUDGET:-4096} VLLM_V1_R_KV_BUFFER=64 VLLM_V1_R_KV_FREE_BLOCKS=1 VLLM_V1_R_KV_TRACE=1
            # the redundancy term's per-request scratch must fit under this cap, or the port silently keeps full KV
            export VLLM_V1_R_KV_SCORE_CHUNK_MB=$(python3 -c "print(int(2*(2*2+5)*8*(1.5*($VLLM_V1_R_KV_BUDGET+64))**2/2**20)+1)") ;;
  *) echo "unknown ARM=$ARM"; exit 2 ;;
esac
for SET in $SETS; do
  OUT=$OUTPUT_DIR/eval/${SET}_qwen3_4b_vllm_${TAG}_merged; mkdir -p "$OUT"
  [ -s "$OUT/$EVAL_ARM.json" ] && { echo "[suite] $SET done"; continue; }
  python3 scripts/eval_math500_compresslm.py --model-dir "$MODEL" --data data/math_eval/$SET.jsonl --output "$OUT/$EVAL_ARM.json" \
    --num-generations 8 --max-tokens 31744 --max-model-len $MML --max-num-seqs ${MAX_NUM_SEQS:-32} \
    --temperature 0.6 --top-p 0.95 --top-k 20 --seed 1234 --attention-backend FLASH_ATTN "${EXTRA[@]}"
done
