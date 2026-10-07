#!/bin/bash
# Time to finish on one saturated GPU: replay 512 requests at each arm's own suite generation lengths, run to completion
# with asynchronous scheduling (needs the async shadow: bash third_party/install_shadow.sh async $VLLM_SHADOW).
#   MODEL=<hf dir> ARM=perlayer|uniform|vanilla TAG=<name> SEQS=<concurrency cap> bash launch/bench.sh
set -euo pipefail
source "$(dirname "$0")/env.sh"; cd "$REPO"
MODEL=${MODEL:?}; ARM=${ARM:?}; TAG=${TAG:?}; VLLM_SHADOW=${VLLM_SHADOW:?async-patched vLLM directory}
export PYTHONPATH=$VLLM_SHADOW:$PYTHONPATH VLLM_ASYNC_SCHED=1 VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
A=(); MML=50176
# LENGTHS_KEY: which arm's suite generation lengths to replay (keys of paper/data/workload_complete_async.json)
case $ARM in
  perlayer) A=(--compresslm --layer-quotas "${SCHEDULE:-$PER_LAYER_SCHEDULE}"); export COMPRESSLM_PHYSICAL_QUOTAS=1; KEY=tpl_physical ;;
  uniform)  A=(--compresslm); KEY=uni_plain ;;
  vanilla)  A=(--attention-backend FLASHINFER); MML=32768; KEY=vanilla_graphs ;;
  *) echo "unknown ARM=$ARM"; exit 2 ;;
esac
python3 scripts/bench_decode.py --model-dir "$MODEL" --tag "$TAG" --out "$OUTPUT_DIR/bench" --mode complete \
  --lengths-file "${LENGTHS:-paper/data/workload_complete_async.json}" --lengths-key "${LENGTHS_KEY:-$KEY}" \
  --requests 512 --len-scale 1 --max-model-len $MML --max-num-seqs "${SEQS:-128}" --rep 1 "${A[@]}"
