"""CPU checks for CompressLM PHYSICAL per-layer quotas (vLLM overlay): one KV cache group per code quota, each
group compacting its own layout (content + code slots >= quota), so memory follows the schedule.

1. CompactedSpanIndex (per-group logical<->physical maps) against a brute-force reference.
2. CompressLMProcessor with group_quotas driven through several blocks with a fake per-group request: after every
   block compaction each group's physical token stream holds exactly [prompt-block rows | codes[:q_g] | ...] and
   code_physical_spans(request, g) returns only the block being coded.
3. (needs the real vllm package) get_kv_cache_groups splits 36 CodeQuotaFullAttentionSpec layers (6 x 256, 18 x 128,
   12 x 64) into 6 equal groups of one quota each, and CompressLMConfig.kv_group_quotas reads them back.
"""
import random
import sys
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
OVERLAY = HERE.parent / "third_party" / "vllm_overlay"
if "--real-vllm" not in sys.argv:
    import test_compresslm_vllm_state as base  # noqa: E402  (fake vllm skeleton + config/processor loaders)
    CompressLMConfig = base.CompressLMConfig
    CompressLMProcessor = base.CompressLMProcessor
    CODE = base.CODE
    spans_module = base.load_module("vllm.v1.compacted_spans", OVERLAY / "vllm" / "v1" / "compacted_spans.py")
    CompactedSpanIndex = spans_module.CompactedSpanIndex


def test_span_index():
    rng = random.Random(0)
    for _ in range(300):
        n = rng.randint(1, 60)
        removed = sorted(rng.sample(range(n), rng.randint(0, n // 2)))
        spans, i = [], 0
        while i < len(removed):
            j = i
            while j + 1 < len(removed) and removed[j + 1] == removed[j] + 1:
                j += 1
            spans.append((removed[i], removed[j] + 1)); i = j + 1
        index = CompactedSpanIndex(spans)
        active = [p for p in range(n) if p not in removed]
        assert index.total_removed == len(removed)
        assert index.get_active_positions(n) == active
        for phys, logical in enumerate(active):
            assert index.logical_to_physical(logical) == phys, (spans, logical)
            assert index.physical_to_logical(phys) == logical
            assert index.is_token_active(logical)
        for r in removed:
            assert not index.is_token_active(r)
    print("[physical-quotas] CompactedSpanIndex == brute force over 300 random layouts")


class GroupRequest(base.FakeRequest if "--real-vllm" not in sys.argv else object):
    """FakeRequest with one compacted layout per KV cache group (like vllm.v1.request.Request)."""

    def __init__(self, prompt, max_tokens, num_groups):
        super().__init__(prompt, max_tokens)
        self.removed = [[] for _ in range(num_groups)]          # per group: list of (start, end)

    def logical_to_physical(self, logical_pos, group=0):
        for start, end in self.removed[group]:
            assert not start <= logical_pos < end, f"queried a compacted position in group {group}"
        return logical_pos - sum(e - s for s, e in self.removed[group] if e <= logical_pos)

    def physical_token_ids(self, group):
        return [t for i, t in enumerate(self._all_token_ids) if not any(s <= i < e for s, e in self.removed[group])]


def test_processor_group_removals():
    # C=4 content, K=4 codes; groups: quota 4 (full), 2, 1, 2
    group_quotas = (4, 2, 1, 2)
    config = CompressLMConfig(enable=True, block_size=4, num_code_tokens=4, replay_tail=0, raw_prompt=True,
                              code_positions="bitrev", layer_code_quotas=(4, 2, 1, 2), physical_layer_quotas=True)
    config._code_token_ids = [900, 901, 902, 903]
    config._initialized = True
    processor = CompressLMProcessor(config)
    processor.set_group_quotas(group_quotas)
    prompt = [11, 12, 13, 14, 15, 16]
    generated = [21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33]
    request = GroupRequest(prompt, max_tokens=len(generated), num_groups=len(group_quotas))
    request.sampling_params.bad_words_token_ids = [[900], [901], [902], [903]]
    state = processor.prepare_request(request)
    request.compresslm_state = state
    gi = 0
    code_steps = 0

    def mask_span(_rid, start, end, groups=None):
        for g in (range(len(group_quotas)) if groups is None else groups):
            assert not any(s < end and start < e for s, e in request.removed[g]), "double removal"
            request.removed[g].append((start, end))
        return SimpleNamespace(success=True, error_message=None)

    for _ in range(400):
        if request.status != base.RequestStatus.RUNNING:
            break
        proposed = request.num_tokens - request.num_computed_tokens
        count = processor.limit_num_new_tokens(request, proposed)
        role = state.roles[request.num_computed_tokens]
        for g, q in enumerate(group_quotas):
            spans = processor.code_physical_spans(request, g)
            phys = request.physical_token_ids(g)
            # every span still carries all K rows in this group's layout
            for ps, pe in spans:
                assert pe - ps == 4 and phys[ps:pe] == [900, 901, 902, 903], (g, spans, phys)
            if role == CODE:
                assert len(spans) == 1, f"code step should expose only the block being coded (group {g}): {spans}"
        if role == CODE:
            code_steps += 1
        skip = processor.should_skip_sampling(request, count)
        request.num_computed_tokens += count
        sampled = []
        if not skip:
            sampled = [generated[gi]]
            gi += 1
        processor.update_after_step(request, count, sampled, mask_span, max_model_len=256, pack_code_span=None)
        # invariant after each step: in group g every COMPACTED block keeps exactly its first q_g codes and no
        # response content; the block being generated is untouched.
        for g, q in enumerate(group_quotas):
            phys = request.physical_token_ids(g)
            for block in state.blocks:
                if block.block_id not in state.compacted_block_ids or block.code_span is None:
                    continue
                cs, ce = block.code_span
                kept = [request._all_token_ids[p] for p in range(cs, ce) if not any(s <= p < e for s, e in request.removed[g])]
                assert kept == [900, 901, 902, 903][:q], (g, q, kept)
                for p in block.response_content_positions:
                    assert any(s <= p < e for s, e in request.removed[g]), (g, p)
            assert len(phys) == len(request._all_token_ids) - sum(e - s for s, e in request.removed[g])
    else:
        raise AssertionError("did not terminate")
    assert gi == len(generated) and code_steps >= 2
    n_compacted = len(state.compacted_block_ids)
    # group sizes: full group removed content only; quota-q groups removed content + (K - q) code rows per block
    for g, q in enumerate(group_quotas):
        code_rows_removed = sum(e - s for s, e in request.removed[g]) - sum(
            e - s for s, e in request.removed[0])
        assert code_rows_removed == n_compacted * (4 - q), (g, q, code_rows_removed, n_compacted)
    print(f"[physical-quotas] processor: {n_compacted} blocks compacted over {code_steps} code steps; per-group layouts "
          f"keep codes[:q] only (removed rows per group: {[sum(e - s for s, e in r) for r in request.removed]})")


def test_grouping_with_real_vllm():
    try:
        import torch
        from vllm.v1.kv_cache_interface import CodeQuotaFullAttentionSpec
        from vllm.v1.core.kv_cache_utils import get_kv_cache_groups
    except Exception as exc:
        raise AssertionError(f"real vllm import failed (install the overlay into the vllm env first): {exc}")
    quotas = {l: (256 if l in (0, 15, 18, 20, 21, 24) else 64 if l in (1, 2, 3, 4, 5, 6, 8, 10, 30, 31, 32, 35) else 128)
              for l in range(36)}
    specs = {f"model.layers.{l}.self_attn.attn": CodeQuotaFullAttentionSpec(
        block_size=16, num_kv_heads=8, head_size=128, dtype=torch.bfloat16, code_quota=quotas[l]) for l in range(36)}
    cfg = SimpleNamespace(scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=False))
    groups = get_kv_cache_groups(cfg, specs)
    sizes = sorted(len(g.layer_names) for g in groups)
    assert len(groups) == 6 and sizes == [6] * 6, [(len(g.layer_names), g.kv_cache_spec.code_quota) for g in groups]
    per_group = []
    for g in groups:
        qs = {quotas[int(n.split(".layers.")[1].split(".")[0])] for n in g.layer_names}
        assert len(qs) == 1 and qs == {g.kv_cache_spec.code_quota}
        per_group.append(qs.pop())
    assert sorted(per_group) == [64, 64, 128, 128, 128, 256]
    from vllm.config.compresslm import CompressLMConfig as RealConfig
    sched = tuple(quotas[l] for l in range(36))
    cfgc = RealConfig(enable=True, block_size=512, num_code_tokens=256, replay_tail=0, raw_prompt=True,
                      code_positions="bitrev", layer_code_quotas=sched, physical_layer_quotas=True)
    assert cfgc.kv_group_quotas([g.layer_names for g in groups]) == tuple(per_group)
    print(f"[physical-quotas] real vLLM grouping: 36 layers -> {len(groups)} groups of 6, quotas {per_group}")


def test_preemption_bounds_code_spans():
    """After a preemption the logical structure survives but the KV does not.

    The scheduler frees the blocks, sets num_computed_tokens = 0 and keeps state.blocks (so re-prefill replays the
    same token stream without restaging codes), but it also clears compacted_block_ids. code_physical_spans would
    then emit every block's codes at their original high positions, past the end of the partially rebuilt sequence,
    and the worker rejects them ("code span (1280, 1536) outside seq_len=768"). Bounding by the re-prefill progress
    is the fix; this checks the bound admits exactly the blocks whose codes are back in the KV.
    """
    group_quotas = (4, 2, 1, 2)
    config = CompressLMConfig(enable=True, block_size=4, num_code_tokens=4, replay_tail=0, raw_prompt=True,
                              code_positions="bitrev", layer_code_quotas=(4, 2, 1, 2), physical_layer_quotas=True)
    config._code_token_ids = [900, 901, 902, 903]
    config._initialized = True
    processor = CompressLMProcessor(config)
    processor.set_group_quotas(group_quotas)
    request = GroupRequest([11, 12, 13, 14, 15, 16], max_tokens=13, num_groups=len(group_quotas))
    request.sampling_params.bad_words_token_ids = [[900], [901], [902], [903]]
    state = processor.prepare_request(request)
    request.compresslm_state = state

    # Hand-build the post-preemption shape: several coded blocks, no compaction progress, KV empty.
    from types import SimpleNamespace as NS
    state.blocks = [NS(block_id=i, code_span=(100 * i + 20, 100 * i + 24), response_content_positions=[])
                    for i in range(4)]
    state.compacted_block_ids.clear()
    request.compacted_spans = []
    request.num_computed_tokens = 0

    unbounded = processor.code_physical_spans(request, 0)
    assert len(unbounded) == 4, "without a bound every block's codes are emitted (the bug)"
    # block i covers [100i+20, 100i+24), so a limit of 100i+24 admits exactly blocks 0..i
    for limit, expected in [(0, 0), (23, 0), (24, 1), (124, 2), (224, 3), (324, 4), (10_000, 4)]:
        got = processor.code_physical_spans(request, 0, limit)
        assert len(got) == expected, f"limit {limit}: expected {expected} spans, got {got}"
        for _, end in got:
            assert end <= limit, f"emitted a span ending past the rebuilt sequence: {got} limit {limit}"
    print("[physical-quotas] preemption: code spans are bounded by re-prefill progress "
          "(4 unbounded -> 0/1/2/3/4 as the request is rebuilt)")


if __name__ == "__main__":
    if "--real-vllm" in sys.argv:
        test_grouping_with_real_vllm()
    else:
        test_span_index()
        test_processor_group_removals()
        test_preemption_bounds_code_spans()
        print("PHYSICAL QUOTA SERVING PASS (CPU); run `--real-vllm` in the vllm env for the grouping check")
