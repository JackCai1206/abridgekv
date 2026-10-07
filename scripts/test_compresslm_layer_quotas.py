"""CPU checks for per-layer code-quota serving on the stock paged cache (vLLM overlay).

1. CompressLMConfig validation of layer_code_quotas (bitrev-only, no packed KV / selective history) and the helpers.
2. CompressLMProcessor.code_physical_spans: the physical [start, end) of every resident code span after compaction,
   driven through several blocks with a fake request that tracks compacted spans like vllm.v1.request.Request.
3. build_layer_quota_masks against a dense per-request reference (causal content steps, noncausal code steps with an
   excluded static-tail range, mixed prefill/decode query lengths).
"""
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import test_compresslm_vllm_state as base  # noqa: E402  (installs the fake vllm package skeleton, loads config/processor)

OVERLAY = base.OVERLAY
masks_module = base.load_module(
    "vllm.v1.worker.compresslm_masks",
    OVERLAY / "vllm" / "v1" / "worker" / "compresslm_masks.py",
)
build_layer_quota_masks = masks_module.build_layer_quota_masks
CompressLMConfig = base.CompressLMConfig
CompressLMProcessor = base.CompressLMProcessor
CODE = base.CODE


def expect_error(**kwargs):
    try:
        CompressLMConfig(**kwargs)
    except ValueError:
        return
    raise AssertionError(f"config accepted {kwargs}")


def test_config():
    common = dict(enable=True, block_size=8, num_code_tokens=4, replay_tail=0, raw_prompt=True, code_positions="bitrev")
    cfg = CompressLMConfig(layer_code_quotas=(4, 2, 1, 4), **common)
    assert cfg.uses_layer_quotas and cfg.distinct_layer_quotas == (1, 2, 4)
    assert [cfg.layer_quota(l) for l in range(5)] == [4, 2, 1, 4, 4]
    full = CompressLMConfig(layer_code_quotas=(4, 4), **common)
    assert not full.uses_layer_quotas
    expect_error(layer_code_quotas=(4, 5), **common)
    expect_error(layer_code_quotas=(0, 4), **common)
    expect_error(layer_code_quotas=(2, 4), **{**common, "code_positions": "appended"})
    expect_error(layer_code_quotas=(2, 4), packed_kv=True, packed_kv_k_max=4, packed_kv_quotas=((2,), (4,)), **common)
    expect_error(layer_code_quotas=(2, 4), retain_full_history_layers=(0,), **common)
    print("[layer-quotas] config validation ok")


class TrackingRequest(base.FakeRequest):
    """FakeRequest plus the compacted-span bookkeeping of vllm.v1.request.Request.logical_to_physical."""

    def __init__(self, prompt, max_tokens):
        super().__init__(prompt, max_tokens)
        self.removed: list[tuple[int, int]] = []

    def logical_to_physical(self, logical_pos: int, group: int = 0) -> int:
        for start, end in self.removed:
            assert not start <= logical_pos < end, "queried a compacted position"
        return logical_pos - sum(end - start for start, end in self.removed if end <= logical_pos)

    def physical_token_ids(self) -> list[int]:
        return [t for i, t in enumerate(self._all_token_ids) if not any(s <= i < e for s, e in self.removed)]


def test_code_physical_spans():
    config = CompressLMConfig(enable=True, block_size=4, num_code_tokens=2, replay_tail=0, raw_prompt=True,
                              code_positions="bitrev", layer_code_quotas=(2, 1, 2))
    config._code_token_ids = [900, 901]
    config._initialized = True
    processor = CompressLMProcessor(config)
    prompt = [11, 12, 13, 14, 15, 16]          # two prompt blocks' worth (4 + 2)
    generated = [21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
    request = TrackingRequest(prompt, max_tokens=len(generated))
    state = processor.prepare_request(request)
    request.compresslm_state = state
    gi = 0
    checks = 0

    def mask_span(_rid, start, end, groups=None):
        request.removed.append((start, end))
        return SimpleNamespace(success=True, error_message=None)

    for _ in range(200):
        if request.status != base.RequestStatus.RUNNING:
            break
        proposed = request.num_tokens - request.num_computed_tokens
        count = processor.limit_num_new_tokens(request, proposed)
        role = state.roles[request.num_computed_tokens]
        # spans as the scheduler would emit them for this step (before the tokens are computed)
        spans = processor.code_physical_spans(request)
        staged = [b for b in state.blocks if b.code_span is not None]
        assert len(spans) == len(staged)
        phys = request.physical_token_ids()
        for (ps, pe), block in zip(spans, staged):
            assert pe - ps == 2 and phys[ps:pe] == [900, 901], (spans, phys)
            assert block.code_span[0] >= ps
        if role == CODE:
            assert spans and spans[-1][1] == request.num_computed_tokens + count - sum(e - s for s, e in request.removed)
        checks += 1
        skip = processor.should_skip_sampling(request, count)
        request.num_computed_tokens += count
        sampled = []
        if not skip:
            sampled = [generated[gi]]
            gi += 1
        processor.update_after_step(request, count, sampled, mask_span, max_model_len=256, pack_code_span=None)
    else:
        raise AssertionError("did not terminate")
    assert gi == len(generated) and len(request.removed) >= 2 and checks > 10
    print(f"[layer-quotas] code_physical_spans ok over {checks} steps, {len(request.removed)} compactions, "
          f"{len([b for b in state.blocks if b.code_span])} code spans")


def reference_masks(requests, quotas, K, causal):
    out = {}
    for q in quotas:   # q >= K: exclusion only, no code span hidden
        parts = []
        for query_len, seq_len, spans, excluded in requests:
            m = torch.ones((query_len, seq_len), dtype=torch.bool)
            if causal:
                for j in range(query_len):
                    m[j, seq_len - query_len + j + 1:] = False
            if excluded is not None:
                m[:, excluded[0]:excluded[1]] = False
            for s, e in spans:
                m[:, s + q:e] = False
            parts.append(m.flatten())
        out[q] = torch.cat(parts)
    return out


def test_masks():
    rng = random.Random(0)
    K = 4
    for trial in range(60):
        causal = trial % 2 == 0
        reqs = []
        for _ in range(rng.randint(1, 6)):
            nblocks = rng.randint(0, 4)
            spans, pos = [], rng.randint(1, 9)   # prompt
            for _ in range(nblocks):
                spans.append((pos, pos + K)); pos += K
            content = rng.randint(1, 12) if causal else 0
            seq_len = pos + content + (0 if causal else K)   # code step: the K code rows are the queries at the end
            if not causal:
                spans.append((seq_len - K, seq_len))
            query_len = rng.randint(1, min(seq_len, 5)) if causal else K
            excluded = None
            if not causal and pos - K >= 2 and rng.random() < 0.5:
                excluded = (pos - K - 2, pos - K)   # some content range before the current codes
            if not causal and nblocks == 0 and rng.random() < 0.5:
                excluded = (0, 1)
            reqs.append((query_len, seq_len, spans, excluded))
        quotas = sorted(set([1, 2, 4] + [rng.choice([1, 2, 3])]))
        got = build_layer_quota_masks(reqs, quotas, K, causal=causal, device=torch.device("cpu"))
        ref = reference_masks(reqs, quotas, K, causal)
        assert got.keys() == ref.keys(), (got.keys(), ref.keys())
        for q in ref:
            assert got[q].shape == ref[q].shape and torch.equal(got[q], ref[q]), (trial, q, reqs)
    assert build_layer_quota_masks([], [1], K, causal=True, device=torch.device("cpu")) == {}
    print("[layer-quotas] build_layer_quota_masks == dense reference over 60 random batches (causal and code steps)")


if __name__ == "__main__":
    test_config()
    test_code_physical_spans()
    test_masks()
    print("LAYER QUOTA SERVING PASS")
