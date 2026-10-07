# Shared settings for the launch scripts. Override any of these in the environment.
#   OUTPUT_DIR   runs, exports, evaluations and benchmarks (default: ./outputs)
#   POOL         tokenized SFT pool written by launch/prep_data.sh
#   VLLM_SHADOW  async-patched vLLM copy from `third_party/install_shadow.sh async` (speed benchmarks only)
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export PYTHONPATH=$REPO${PYTHONPATH:+:$PYTHONPATH} PYTHONUNBUFFERED=1
export OUTPUT_DIR=${OUTPUT_DIR:-$REPO/outputs}
export POOL=${POOL:-$OUTPUT_DIR/data/math_sft_long_qwen3_400000.pkl}
export BASE_MODEL=${BASE_MODEL:-Qwen/Qwen3-4B-Base}
export PER_LAYER_SCHEDULE=${PER_LAYER_SCHEDULE:-$REPO/results/schedules/sched_trunc_perlayer_mean128.json}
# FlexAttention settings used for every training run in the paper
export COMPRESSLM_ARITH_MASK=1 COMPRESSLM_FAST_BLOCKMASK=1
export COMPRESSLM_FLEX_KERNEL_OPTIONS=${COMPRESSLM_FLEX_KERNEL_OPTIONS:-"BLOCK_M=128;BLOCK_N=64;BLOCK_M1=64;BLOCK_N1=128;BLOCK_M2=128;BLOCK_N2=64;num_warps=8"}
mkdir -p "$OUTPUT_DIR"
