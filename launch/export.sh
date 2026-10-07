#!/bin/bash
# Export a trained run to a Hugging Face directory that the vLLM overlay serves.
#   RUN_DIR=<run> ARM=perlayer|uniform|appended|vanilla bash launch/export.sh     -> $RUN_DIR/hf
# Uniform arms are exported with only their K_MEAN code rows, so vLLM serves them on its stock attention path.
set -euo pipefail
source "$(dirname "$0")/env.sh"; cd "$REPO"
RUN_DIR=${RUN_DIR:?}; ARM=${ARM:?}; OUT=${OUT:-$RUN_DIR/hf}
case $ARM in
  vanilla) A=(--arm baseline) ;;
  perlayer) A=(--arm compressed) ;;
  uniform|appended) A=(--arm compressed --code-rows "${K_MEAN:-128}") ;;
  *) echo "unknown ARM=$ARM"; exit 2 ;;
esac
python3 scripts/convert_compresslm_checkpoint_to_hf.py --checkpoint "$RUN_DIR/model.pt" --output "$OUT" --base-model "$BASE_MODEL" "${A[@]}"
