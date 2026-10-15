#!/bin/bash
# Build a patched copy ("shadow") of the installed vLLM package without modifying the environment itself.
#   bash third_party/install_shadow.sh async <dest>   # asynchronous scheduling + code-step KV trim (speed benchmarks)
#   bash third_party/install_shadow.sh lt <dest>      # LightThinker boundary-triggered compaction (baseline)
#   bash third_party/install_shadow.sh plain <dest>   # the overlay alone, to try it without reinstalling
# Use the result with PYTHONPATH=<dest>:$REPO. Requires the base overlay installed first
# (bash third_party/vllm_overlay/install_overlay.sh).
set -euo pipefail
VARIANT=${1:?variant: async | lt | plain}; DEST=${2:?destination directory}
HERE=$(cd "$(dirname "$0")" && pwd)
SITE=$(cd / && python3 -c "import os, vllm; print(os.path.dirname(vllm.__file__))")
mkdir -p "$DEST.new"
rsync -a --delete "$SITE/" "$DEST.new/vllm/"
rsync -a "$HERE/vllm_overlay/vllm/" "$DEST.new/vllm/"     # refresh from the overlay in this repository
case $VARIANT in
  async) for p in async_scheduling trim_code_kv; do (cd "$DEST.new" && patch -p1 --forward < "$HERE/vllm_overlay_async/$p.patch"); done ;;
  lt)    (cd "$DEST.new" && patch -p1 --forward < "$HERE/vllm_overlay_lt/lightthinker.patch")
         grep -q uses_boundary_trigger "$DEST.new/vllm/config/compresslm.py" ;;
  plain) ;;
  *) echo "unknown variant $VARIANT"; exit 2 ;;
esac
[ -d "$DEST" ] && mv "$DEST" "$DEST.old.$(date +%s)"
mv "$DEST.new" "$DEST"
echo "[shadow] installed $VARIANT vLLM at $DEST"
