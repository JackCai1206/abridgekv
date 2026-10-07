"""CPU-only state-machine checks for the CompressLM vLLM overlay."""

import enum
import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
OVERLAY = ROOT / "third_party" / "vllm_overlay"

# The overlay is installed over a stock vLLM wheel in production. These tests
# need only RequestStatus, so provide that one dependency on CPU-only hosts.
for package in ("vllm", "vllm.v1", "vllm.v1.core", "vllm.v1.core.compresslm"):
    module = types.ModuleType(package)
    module.__path__ = []
    sys.modules[package] = module
request_module = types.ModuleType("vllm.v1.request")


class RequestStatus(enum.IntEnum):
    RUNNING = 1
    FINISHED_STOPPED = 2
    FINISHED_LENGTH_CAPPED = 3


request_module.RequestStatus = RequestStatus
sys.modules.setdefault("vllm.v1.request", request_module)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


config_module = load_module(
    "compresslm_config_under_test",
    OVERLAY / "vllm" / "config" / "compresslm.py",
)
load_module(
    "vllm.v1.core.compresslm.state",
    OVERLAY / "vllm" / "v1" / "core" / "compresslm" / "state.py",
)
processor_module = load_module(
    "vllm.v1.core.compresslm.processor",
    OVERLAY / "vllm" / "v1" / "core" / "compresslm" / "processor.py",
)
state_module = sys.modules["vllm.v1.core.compresslm.state"]
CompressLMConfig = config_module.CompressLMConfig
CompressLMProcessor = processor_module.CompressLMProcessor
CODE, CONTENT, REPLAY = state_module.CODE, state_module.CONTENT, state_module.REPLAY


class FakeRequest:
    def __init__(self, prompt: list[int], max_tokens: int):
        self.request_id = "test"
        self.prompt_token_ids = prompt
        self.prompt_embeds = None
        self.pooling_params = None
        self.sampling_params = SimpleNamespace(
            frequency_penalty=0.0,
            presence_penalty=0.0,
            repetition_penalty=1.0,
            stop=None,
            ignore_eos=False,
            stop_token_ids=[],
            min_tokens=0,
            logprobs=None,
            prompt_logprobs=None,
            bad_words_token_ids=[[900], [901]],
        )
        self.eos_token_id = 2
        self.max_tokens = max_tokens
        self.status = RequestStatus.RUNNING
        self.stop_reason = None
        self._all_token_ids = prompt.copy()
        self._output_token_ids: list[int] = []
        self.num_prompt_tokens = len(prompt)
        self.num_computed_tokens = 0
        self.compresslm_state = None
        self.use_structured_output = False

    @property
    def num_tokens(self) -> int:
        return len(self._all_token_ids)

    def append_output_token_ids(self, token_ids):
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        self._output_token_ids.extend(token_ids)
        self._all_token_ids.extend(token_ids)


def run_case(
    prompt: list[int],
    generated: list[int],
    replay_tail: int = 2,
    retain_tail_kv: int = 0,
    retain_tail_policy: str = "rolling",
    code_exclude_replay_tail: bool = False,
    retain_full_history_layers: tuple[int, ...] = (),
    history_replaces_codes_layers: tuple[int, ...] = (),
    mask_events: list[tuple[int, int, int]] | None = None,
    packed_kv: bool = False,
    packed_events: list[tuple[int, int]] | None = None,
):
    config = CompressLMConfig(
        enable=True,
        block_size=4,
        num_code_tokens=2,
        replay_tail=replay_tail,
        retain_tail_kv=retain_tail_kv,
        retain_tail_policy=retain_tail_policy,
        code_exclude_replay_tail=code_exclude_replay_tail,
        retain_full_history_layers=retain_full_history_layers,
        history_replaces_codes_layers=history_replaces_codes_layers,
        packed_kv=packed_kv,
        packed_kv_k_max=2,
        raw_prompt=True,
    )
    config._code_token_ids = [900, 901]
    config._initialized = True
    processor = CompressLMProcessor(config)
    request = FakeRequest(prompt, max_tokens=len(generated))
    state = processor.prepare_request(request)
    request.compresslm_state = state

    masks: list[tuple[int, int]] = []
    scheduled_roles: list[tuple[int, int]] = []
    generated_index = 0

    def mask_span(_request_id, start, end):
        assert 0 <= start < end <= request.num_computed_tokens
        masks.append((start, end))
        if mask_events is not None:
            mask_events.append((start, end, request.num_computed_tokens))
        return SimpleNamespace(success=True, error_message=None)

    for _ in range(200):
        if request.status != RequestStatus.RUNNING:
            break
        proposed = request.num_tokens - request.num_computed_tokens
        assert proposed > 0
        count = processor.limit_num_new_tokens(request, proposed)
        assert count > 0
        start = request.num_computed_tokens
        role = state.roles[start]
        assert all(state.roles[pos] == role for pos in range(start, start + count))
        if role == CODE:
            assert count == config.num_code_tokens
        scheduled_roles.append((role, count))
        skip = processor.should_skip_sampling(request, count)
        request.num_computed_tokens += count
        sampled = []
        if not skip:
            sampled = [generated[generated_index]]
            generated_index += 1
        processor.update_after_step(
            request,
            count,
            sampled,
            mask_span,
            max_model_len=256,
            pack_code_span=(
                (lambda _request_id, start, end: packed_events.append(
                    (start, end)
                ))
                if packed_events is not None
                else None
            ),
        )
    else:
        raise AssertionError("state machine did not terminate")

    assert state.visible_output_token_ids == generated
    assert generated_index == len(generated)
    assert all(count == 2 for role, count in scheduled_roles if role == CODE)
    assert all(role in (CONTENT, CODE, REPLAY) for role, _ in scheduled_roles)

    for block in state.blocks:
        if block.code_span is None:
            continue
        code_start, code_end = block.code_span
        assert request._all_token_ids[code_start:code_end] == [900, 901]
        assert state.roles[code_start:code_end] == [CODE, CODE]
        if block.replay_positions:
            replay_ids = [
                request._all_token_ids[position]
                for position in block.replay_positions
            ]
            assert replay_ids == block.content_token_ids[-2:]
            assert all(state.roles[position] == REPLAY
                       for position in block.replay_positions)
        if retain_tail_kv:
            positions = block.content_positions[-retain_tail_kv:]
            assert block.retained_tail_positions == positions
            assert not block.replay_positions
        excluded_tail = replay_tail if code_exclude_replay_tail else retain_tail_kv
        if excluded_tail:
            positions = block.content_positions[-excluded_tail:]
            assert block.code_excluded_content_span == (
                positions[0],
                positions[-1] + 1,
            )
        else:
            assert block.code_excluded_content_span is None

    prompt_positions = {
        position
        for block in state.blocks
        for position in block.prompt_content_positions
    }
    assert not any(
        any(position in prompt_positions for position in range(start, end))
        for start, end in masks
    )
    return request, state, masks


def main() -> None:
    default_config = CompressLMConfig()
    assert default_config.retain_tail_policy == "rolling"
    try:
        CompressLMConfig(retain_tail_policy="invalid")
    except ValueError as error:
        assert "retain_tail_policy" in str(error)
    else:
        raise AssertionError("invalid retain_tail_policy was accepted")
    try:
        CompressLMConfig(
            enable=True,
            replay_tail=0,
            retain_tail_kv=0,
            retain_tail_policy="sticky",
        )
    except ValueError as error:
        assert "retain_tail_kv > 0" in str(error)
    else:
        raise AssertionError("sticky policy without a retained tail was accepted")
    try:
        CompressLMConfig(
            enable=True,
            block_size=4,
            num_code_tokens=2,
            replay_tail=0,
            packed_kv=True,
        )
    except ValueError as error:
        assert "packed_kv_k_max" in str(error)
    else:
        raise AssertionError("packed KV accepted mismatched K_max")

    # Exact-boundary prompt: the first sampled response token must be held
    # until code + replay are processed, then become the next content token.
    request, state, _ = run_case([10, 11, 12, 13], [20, 21, 22, 23, 24, 25])
    first_code_end = state.blocks[0].replay_positions[-1] + 1
    assert request._all_token_ids[first_code_end] == 20

    # Multi-block prompt plus a prompt/response-straddling content block.
    _, state, masks = run_case(
        [30, 31, 32, 33, 34, 35],
        [40, 41, 42, 43, 44, 45, 46],
    )
    assert len(state.blocks) >= 3
    assert masks

    # Evaluation-only selective history preserves every physical KV. Logical
    # expiration still advances, but no compaction callback may fire.
    _, selective_state, selective_masks = run_case(
        [10],
        [20, 21, 22, 23, 24, 25, 26, 27],
        retain_full_history_layers=(1, 2),
        history_replaces_codes_layers=(2,),
    )
    assert not selective_masks
    assert selective_state.compacted_block_ids
    assert selective_state.expired_replay_positions

    # Packed mode archives each complete code chunk before removing both its
    # ordinary code rows and the response content summarized by those codes.
    packed_events: list[tuple[int, int]] = []
    _, packed_state, packed_masks = run_case(
        [10, 11, 12, 13],
        [20, 21, 22, 23, 24, 25],
        replay_tail=0,
        packed_kv=True,
        packed_events=packed_events,
    )
    first_packed_block = packed_state.blocks[0]
    assert packed_events[0] == first_packed_block.code_span
    assert first_packed_block.code_span in packed_masks
    second_packed_block = packed_state.blocks[1]
    assert packed_events[1] == second_packed_block.code_span
    assert (
        second_packed_block.response_content_positions[0],
        second_packed_block.code_span[1],
    ) in packed_masks

    # STATIC retained tail + packed KV: the block's final T content rows stay resident forever (never expire, never
    # replayed); the summarized response content and the code span are removed as two separate spans around them.
    static_events: list[tuple[int, int]] = []
    _, static_state, static_masks = run_case(
        [10],
        [20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
        replay_tail=0,
        retain_tail_kv=2,
        retain_tail_policy="static",
        packed_kv=True,
        packed_events=static_events,
    )
    b0, b1 = static_state.blocks[0], static_state.blocks[1]
    assert b0.retained_tail_positions == [2, 3] and b0.code_span == (4, 6)
    assert static_events[0] == b0.code_span and static_events[1] == b1.code_span
    assert (1, 2) in static_masks and (4, 6) in static_masks          # block 0: response prefix [1,2), codes [4,6); tail [2,4) stays
    assert (2, 3) not in static_masks and (3, 4) not in static_masks and (2, 4) not in static_masks
    assert (b1.response_content_positions[0], b1.retained_tail_positions[0]) in static_masks
    assert b1.code_span in static_masks
    assert not static_state.expired_retained_tail_positions and not static_state.expired_sticky_retained_block_ids
    assert REPLAY not in static_state.roles
    # code exclusion range = the block's own tail (logical), independent of the policy
    assert b0.code_excluded_content_span == (2, 4) and b1.code_excluded_content_span == (
        b1.retained_tail_positions[0], b1.retained_tail_positions[-1] + 1)
    # STATIC without packed KV: only the response prefix is removed; codes and tails stay resident.
    _, static_dense_state, static_dense_masks = run_case(
        [10],
        [20, 21, 22, 23, 24, 25, 26, 27],
        replay_tail=0,
        retain_tail_kv=2,
        retain_tail_policy="static",
    )
    assert (1, 2) in static_dense_masks and all(m[0] >= 1 for m in static_dense_masks)
    assert (4, 6) not in static_dense_masks and (2, 4) not in static_dense_masks
    assert not static_dense_state.expired_retained_tail_positions
    # packed KV rejects the window policies but accepts static
    for policy in ("rolling", "sticky"):
        try:
            CompressLMConfig(enable=True, block_size=4, num_code_tokens=2, replay_tail=0, retain_tail_kv=2,
                             retain_tail_policy=policy, packed_kv=True, packed_kv_k_max=2)
        except ValueError as error:
            assert "static" in str(error)
        else:
            raise AssertionError(f"packed KV accepted a {policy} retained tail")
    static_cfg = CompressLMConfig(enable=True, block_size=8, num_code_tokens=4, replay_tail=0, retain_tail_kv=2,
                                  retain_tail_policy="static", packed_kv=True, packed_kv_k_max=4, code_positions="bitrev")
    assert static_cfg.code_excluded_tail == 2
    offs = static_cfg.bitrev_code_offsets().tolist()
    assert max(offs) < 6 and len(set(offs)) == 4 and offs[0] == 0, offs      # grid over [0, C - T) = [0, 6)

    # Tail-exclusive checkpoints persist the exact logical source span that
    # the worker must hide from code queries.
    run_case(
        [70, 71, 72, 73],
        [80, 81, 82, 83, 84, 85],
        code_exclude_replay_tail=True,
    )

    # True retention keeps the original pre-code positions, inserts no replay
    # token IDs, and expires one response KV after each next-block query.
    request, state, masks = run_case(
        [10],
        [20, 21, 22, 23, 24, 25, 26, 27],
        replay_tail=0,
        retain_tail_kv=2,
    )
    first_block = state.blocks[0]
    assert first_block.retained_tail_positions == [2, 3]
    assert request._all_token_ids[:7] == [10, 20, 21, 22, 900, 901, 23]
    assert REPLAY not in state.roles
    assert (1, 2) in masks
    assert (2, 3) in masks
    assert (3, 4) in masks
    assert state.expired_retained_tail_positions.issuperset({2, 3})

    # Re-prefill after preemption may expose several existing content tokens
    # at once. It must still schedule one query per retained-tail expiration.
    config = CompressLMConfig(
        enable=True,
        block_size=4,
        num_code_tokens=2,
        replay_tail=0,
        retain_tail_kv=2,
    )
    config._code_token_ids = [900, 901]
    config._initialized = True
    processor = CompressLMProcessor(config)
    bulk_state = SimpleNamespace(
        roles=[CONTENT] * 8,
        content_to_expired_replay={},
        content_to_expired_retained_tail={2: 0},
    )
    bulk_request = SimpleNamespace(
        compresslm_state=bulk_state,
        num_computed_tokens=2,
        num_tokens=8,
    )
    assert processor.limit_num_new_tokens(bulk_request, proposed=6) == 1
    bulk_state.content_to_expired_retained_tail = {3: 0}
    assert processor.limit_num_new_tokens(bulk_request, proposed=6) == 2

    # Sticky retention keeps every prior response-tail KV visible through all
    # next-block content and removes the response subset once, immediately
    # before the next block's codes. It never creates replay IDs or rolling
    # expirations.
    sticky_events: list[tuple[int, int, int]] = []
    request, state, masks = run_case(
        [10],
        list(range(20, 29)),
        replay_tail=0,
        retain_tail_kv=2,
        retain_tail_policy="sticky",
        mask_events=sticky_events,
    )
    first_block, second_block = state.blocks[:2]
    assert first_block.retained_tail_positions == [2, 3]
    assert not state.content_to_expired_retained_tail
    assert not state.expired_retained_tail_positions
    assert state.expired_sticky_retained_block_ids.issuperset({0})
    second_boundary = second_block.content_positions[-1] + 1
    assert sticky_events.count((2, 4, second_boundary)) == 1
    assert sticky_events.index(
        (2, 4, second_boundary)
    ) < sticky_events.index((6, 8, second_block.code_span[1]))
    assert not any(
        start < 4 and end > 2 and computed < second_boundary
        for start, end, computed in sticky_events
    )
    assert REPLAY not in state.roles
    assert not any(block.replay_positions for block in state.blocks)
    for token_id in [10, *range(20, 28)]:
        assert request._all_token_ids.count(token_id) == 1

    sticky_config = CompressLMConfig(
        enable=True,
        block_size=4,
        num_code_tokens=2,
        replay_tail=0,
        retain_tail_kv=2,
        retain_tail_policy="sticky",
    )
    sticky_config._code_token_ids = [900, 901]
    sticky_config._initialized = True
    sticky_processor = CompressLMProcessor(sticky_config)
    sticky_bulk_state = SimpleNamespace(
        roles=[CONTENT] * 8,
        content_to_expired_replay={},
        content_to_expired_retained_tail={},
    )
    sticky_bulk_request = SimpleNamespace(
        compresslm_state=sticky_bulk_state,
        num_computed_tokens=2,
        num_tokens=8,
    )
    assert sticky_processor.limit_num_new_tokens(
        sticky_bulk_request, proposed=6
    ) == 6

    # A retained tail can straddle prompt and response. Only its contiguous
    # response suffix is removed at the sticky boundary.
    mixed_events: list[tuple[int, int, int]] = []
    _, mixed_state, mixed_masks = run_case(
        [30, 31, 32],
        list(range(40, 49)),
        replay_tail=0,
        retain_tail_kv=2,
        retain_tail_policy="sticky",
        mask_events=mixed_events,
    )
    mixed_first, mixed_second = mixed_state.blocks[:2]
    assert mixed_first.retained_tail_positions == [2, 3]
    mixed_boundary = mixed_second.content_positions[-1] + 1
    assert mixed_events.count((3, 4, mixed_boundary)) == 1
    assert not any(start <= 2 < end for start, end in mixed_masks)

    # Preemption clears boundary progress because the emptied cache must replay
    # all block-level compactions during re-prefill. Drive the real state
    # machine to prove the prior sticky tail is removed before existing codes
    # are recomputed, rather than only testing the removal helper.
    sticky_processor.reset_for_preemption(state)
    assert not state.expired_sticky_retained_block_ids
    request.num_computed_tokens = 0
    request.status = RequestStatus.RUNNING
    reapplied: list[tuple[int, int, int]] = []

    def remask(_request_id, start, end):
        reapplied.append((start, end, request.num_computed_tokens))
        return SimpleNamespace(success=True, error_message=None)

    while request.num_computed_tokens < second_block.code_span[0]:
        proposed = request.num_tokens - request.num_computed_tokens
        count = sticky_processor.limit_num_new_tokens(request, proposed)
        request.num_computed_tokens += count
        sticky_processor.update_after_step(
            request,
            count,
            [],
            remask,
            max_model_len=256,
        )

    assert (2, 4, second_boundary) in reapplied
    assert state.expired_sticky_retained_block_ids == {0}

    # Without replay, the code step must compact and insert the held token in
    # the same scheduler update.
    request, state, masks = run_case(
        [50, 51, 52, 53],
        [60, 61, 62, 63, 64, 65],
        replay_tail=0,
    )
    first_block = state.blocks[0]
    assert first_block.code_span is not None
    assert not first_block.replay_positions
    assert request._all_token_ids[first_block.code_span[1]] == 60
    assert masks

    # A code chunk that cannot fit must length-cap instead of deadlocking in
    # an all-or-nothing code scheduling phase.
    config = CompressLMConfig(
        enable=True, block_size=4, num_code_tokens=2, replay_tail=2
    )
    config._code_token_ids = [900, 901]
    config._initialized = True
    processor = CompressLMProcessor(config)
    request = FakeRequest([10, 11, 12, 13], max_tokens=8)
    request.compresslm_state = processor.prepare_request(request)
    request.num_computed_tokens = 4
    visible, stopped = processor.update_after_step(
        request,
        4,
        [20],
        lambda *_: SimpleNamespace(success=True, error_message=None),
        max_model_len=8,
    )
    assert visible == [20] and stopped
    assert request.status == RequestStatus.FINISHED_LENGTH_CAPPED

    # Every scheduler-side internal layout transition must force a full token
    # synchronization to the worker exactly when its cached list becomes stale.
    config = CompressLMConfig(
        enable=True, block_size=4, num_code_tokens=2, replay_tail=2
    )
    config._code_token_ids = [900, 901]
    config._initialized = True
    processor = CompressLMProcessor(config)
    request = FakeRequest([10, 11, 12, 13], max_tokens=4)
    state = processor.prepare_request(request)
    request.compresslm_state = state
    assert not state.worker_token_ids_dirty
    request.num_computed_tokens = 4
    processor.update_after_step(
        request,
        4,
        [20],
        lambda *_: SimpleNamespace(success=True, error_message=None),
        max_model_len=256,
    )
    assert state.worker_token_ids_dirty
    state.worker_token_ids_dirty = False
    request.num_computed_tokens += 2
    processor.update_after_step(
        request,
        2,
        [],
        lambda *_: SimpleNamespace(success=True, error_message=None),
        max_model_len=256,
    )
    assert state.worker_token_ids_dirty
    state.worker_token_ids_dirty = False
    request.num_computed_tokens += 2
    processor.update_after_step(
        request,
        2,
        [],
        lambda *_: SimpleNamespace(success=True, error_message=None),
        max_model_len=256,
    )
    assert state.worker_token_ids_dirty
    print("CompressLM vLLM state-machine checks passed")


if __name__ == "__main__":
    main()
