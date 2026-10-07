#!/bin/bash
# Measure the per-layer schedule on a trained uniform model: halve one layer's code rows at a time (128 -> 64), record
# the held-out cross-entropy cost, give the 6 costliest layers 256 rows and the 12 cheapest 64 (mean 128).
#   RUN_DIR=<uniform run> bash launch/measure_schedule.sh          -> $RUN_DIR/perlayer_schedule.json
set -euo pipefail
source "$(dirname "$0")/env.sh"; cd "$REPO"
RUN_DIR=${RUN_DIR:?trained uniform run directory (with model.pt)}
export PROBE_RUN=$RUN_DIR PROBE_OUT=$RUN_DIR/quota_probe.jsonl PROBE_STAGES=${PROBE_STAGES:-layer} PAIRS=$POOL NMAX=32 EVALN=192
python3 scripts/probe_quota_ablation.py
PROBE_JSONL=$PROBE_OUT OUT=$RUN_DIR/perlayer_schedule.json python3 scripts/build_trunc_perlayer_schedule.py
