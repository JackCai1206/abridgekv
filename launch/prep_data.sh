#!/bin/bash
# Build the 400K-pair math SFT pool (nvidia/OpenMathReasoning, Qwen3-4B-Base tokenizer, 512-16K tokens) and the
# competition evaluation sets in data/math_eval/. Needs Hub access; the pool takes a few hours on a many-core CPU node.
#   bash launch/prep_data.sh            # CODE=1 additionally builds the 120K-pair coding pool (OpenCodeReasoning)
set -euo pipefail
source "$(dirname "$0")/env.sh"; cd "$REPO"
mkdir -p "$(dirname "$POOL")"
TOK_DIR=$BASE_MODEL N_KEEP=400000 LO=512 HI=16384 BANDS='4096:0.60,8192:0.25,16384:0.15' HOLDOUT_PER_BAND=64 \
  NPROC=${NPROC:-$(nproc)} OUT=$POOL python3 scripts/prep_math_sft.py
python3 scripts/build_math_eval_sets.py
python3 scripts/dl_math_eval.py
if [ "${CODE:-0}" = 1 ]; then
  OUT=$OUTPUT_DIR/data/code_sft_long_qwen3_120000.pkl NPROC=${NPROC:-8} PREP_CHUNK=5000 python3 scripts/prep_code_sft.py
fi
