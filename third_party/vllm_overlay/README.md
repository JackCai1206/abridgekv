# vLLM overlay for AbridgeKV

Python-only changes to vLLM 0.13.0 that serve AbridgeKV models: every completed block of generated text is
compressed into the KV entries of K learned pause tokens, after which the block's own KV is freed and later tokens
attend to the pause-token entries. The overlay is copied on top of an installed vLLM wheel; nothing is compiled.

It started from the [microsoft/memento](https://github.com/microsoft/memento) vLLM overlay (block-masking scheduler,
request tracking, KV compaction) and adds the pause-token protocol below. See `../../NOTICE`.

## Install

```bash
pip install vllm==0.13.0
bash third_party/vllm_overlay/install_overlay.sh   # rsyncs vllm/ over the installed package
```

`third_party/install_shadow.sh async <dir>` and `third_party/install_shadow.sh lt <dir>` build patched copies for the
speed benchmark (asynchronous scheduling plus trimming of code-step KV) and for the LightThinker baseline.

## Decoding protocol

The exported model directory carries `compresslm_config.json` (block size C, number of pause tokens K, schedule) and
the pause tokens as special tokens `<|compress_code_000|>` ... with their trained embeddings. Pause tokens are never
sampled. With `--use-compresslm` (see `scripts/eval_math500_compresslm.py`):

1. After every C generated tokens the scheduler appends the K pause tokens and runs them in one step with
   bidirectional attention among themselves. Their positions lie inside the block they summarize (bit-reversed
   order), matching training.
2. The block's C content entries are then removed from the KV cache (compaction); the pause-token entries stay.
3. Generation continues. The token sampled at the boundary is returned to the client and fed after the pause tokens.

Prompt tokens are never compressed.

## Per-layer quotas

A per-layer schedule (`results/schedules/sched_trunc_perlayer_mean128.json`: 6 layers keep 256 rows per block, 18
keep 128, 12 keep 64) is served in one of two ways:

- **masked** (`--compresslm-layer-quotas <schedule>`): one KV cache; layer l attends to the first q_l rows of every
  block. Every layer still stores K_max rows per block, so memory matches a uniform K_max model.
- **physical** (also `COMPRESSLM_PHYSICAL_QUOTAS=1`): one KV cache group per distinct quota, each compacted to its
  own row count, so the cache shrinks with the quota. This is the configuration used for the paper's results and
  speed numbers.

Uniform models are exported with only their K code rows (`launch/export.sh`) and run on the same path with one group.

## Files

| Path | Role |
|---|---|
| `vllm/config/compresslm.py` | configuration, schedule loading and validation |
| `vllm/v1/core/compresslm/` | per-request block state, pause-token insertion, compaction spans |
| `vllm/v1/compacted_spans.py` | logical-to-physical position bookkeeping after compaction |
| `vllm/v1/worker/compresslm_masks.py` | attention masks for code steps and masked per-layer quotas |
| `vllm/v1/worker/compresslm_packed_kv.py` | packed per-head code storage (per-head analysis only) |
| `vllm/v1/core/sched/scheduler.py`, `vllm/v1/core/*kv_cache*` | scheduling and per-group compaction |
| `vllm/v1/worker/gpu_model_runner.py` | step construction, CUDA-graph safety checks |
| `vllm/v1/core/block_masking/` | delimiter-driven block masking inherited from microsoft/memento |

## Tests

CPU tests of the scheduler state and masks: `scripts/test_compresslm_vllm_state.py`,
`scripts/test_compresslm_layer_quotas.py`, `scripts/test_compresslm_physical_quotas.py`. GPU parity against the
training-time forward pass: `scripts/check_compresslm_vllm_parity.py`, `scripts/check_teacher_forced_parity.py`, and
the masking negative control `scripts/check_full_attention_control.py`.
