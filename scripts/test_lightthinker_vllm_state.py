"""CPU check of the LightThinker boundary trigger in the vLLM CompressLM processor (LT shadow overlay).

Drives the scheduler-side state machine with a fake request and a scripted token stream, then simulates what every
scheduled query can attend to (keys already computed minus the spans compacted so far; code steps causal) and checks
it equals the TRAINING mask row of that position (abridgekv/lightthinker.py, itself tested against LightThinker's
create_attention_for_aug_data). Also checks the processed stream equals augment_generated(), splitter steps do not
sample, the continue step does, and prompt / cache / continue rows are never evicted.

  LT_SHADOW=<shadow dir from third_party/install_shadow.sh lt> python scripts/test_lightthinker_vllm_state.py
"""
import enum
import importlib.util
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from abridgekv.lightthinker import augment_generated, dense_mask, LTIds  # noqa: E402

SHADOW = Path(os.environ["LT_SHADOW"]) / "vllm"
for package in ("vllm", "vllm.v1", "vllm.v1.core", "vllm.v1.core.compresslm"):
    module = types.ModuleType(package); module.__path__ = []; sys.modules[package] = module
request_module = types.ModuleType("vllm.v1.request")


class RequestStatus(enum.IntEnum):
    RUNNING = 1
    FINISHED_STOPPED = 2
    FINISHED_LENGTH_CAPPED = 3


request_module.RequestStatus = RequestStatus
sys.modules["vllm.v1.request"] = request_module


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod; spec.loader.exec_module(mod)
    return mod


cfg_mod = load("lt_cfg", SHADOW / "config" / "compresslm.py")
state_mod = load("vllm.v1.core.compresslm.state", SHADOW / "v1" / "core" / "compresslm" / "state.py")
proc_mod = load("vllm.v1.core.compresslm.processor", SHADOW / "v1" / "core" / "compresslm" / "processor.py")
CODE, CONTENT = state_mod.CODE, state_mod.CONTENT

K = 3
IDS = LTIds(cache=(900, 901, 902), splitter=950, cont=951)
EOS = 2


class FakeRequest:
    def __init__(self, prompt, max_tokens):
        self.request_id = "lt"
        self.prompt_token_ids = list(prompt)
        self.prompt_embeds = None
        self.pooling_params = None
        self.sampling_params = SimpleNamespace(
            frequency_penalty=0.0, presence_penalty=0.0, repetition_penalty=1.0, stop=None, ignore_eos=False,
            stop_token_ids=[], min_tokens=0, logprobs=None, prompt_logprobs=None,
            bad_words_token_ids=[[t] for t in IDS.cache] + [[IDS.cont]])
        self.eos_token_id = EOS
        self.max_tokens = max_tokens
        self.status = RequestStatus.RUNNING
        self.stop_reason = None
        self._all_token_ids = list(prompt)
        self._output_token_ids = []
        self.num_prompt_tokens = len(prompt)
        self.num_computed_tokens = 0
        self.compresslm_state = None
        self.use_structured_output = False

    @property
    def num_tokens(self):
        return len(self._all_token_ids)

    def append_output_token_ids(self, token_ids):
        token_ids = [token_ids] if isinstance(token_ids, int) else list(token_ids)
        self._output_token_ids.extend(token_ids); self._all_token_ids.extend(token_ids)


def run(prompt, generated, max_model_len=10_000):
    config = cfg_mod.CompressLMConfig(enable=True, block_size=1 << 20, num_code_tokens=K, replay_tail=0,
                                      pause_xattn="causal", raw_prompt=True, boundary_token_id=IDS.splitter,
                                      continue_token_id=IDS.cont)
    assert config.code_step_causal and config.uses_boundary_trigger
    config._code_token_ids = list(IDS.cache); config._initialized = True
    proc = proc_mod.CompressLMProcessor(config)
    req = FakeRequest(prompt, max_tokens=len(generated))
    state = proc.prepare_request(req); req.compresslm_state = state
    evicted = set()
    visibility = {}                                   # query position -> set of visible key positions
    sampled_at = []                                   # position whose logits produced each sampled token
    gi = 0

    def mask_span(_rid, start, end, groups=None):
        assert 0 <= start < end <= req.num_computed_tokens
        evicted.update(range(start, end))
        return SimpleNamespace(success=True, error_message=None)

    for _ in range(10_000):
        if req.status != RequestStatus.RUNNING:
            break
        n = proc.limit_num_new_tokens(req, req.num_tokens - req.num_computed_tokens)
        assert n > 0
        s0 = req.num_computed_tokens
        role = state.roles[s0]
        if role == CODE:
            assert n == K
        for q in range(s0, s0 + n):                   # causal within the step (code steps too), minus evicted keys
            visibility[q] = {k for k in range(q + 1) if k not in evicted}
        skip = proc.should_skip_sampling(req, n)
        req.num_computed_tokens += n
        sampled = []
        if not skip:
            sampled = [generated[gi]]; gi += 1; sampled_at.append(s0 + n - 1)
        proc.update_after_step(req, n, sampled, mask_span, max_model_len=max_model_len)
    else:
        raise AssertionError("did not terminate")
    return req, state, evicted, visibility, sampled_at, gi


def check(prompt, generated):
    req, state, evicted, vis, sampled_at, gi = run(prompt, generated)
    assert gi == len(generated), (gi, len(generated))
    seq, hide, pred = augment_generated(prompt, generated, IDS)
    stream = req._all_token_ids
    # a trailing splitter's cache/continue are processed only if the request keeps running; compare the common part
    L = len(stream)
    assert stream == seq[:L].tolist(), (stream, seq.tolist())
    assert sampled_at == pred.tolist(), (sampled_at, pred.tolist())
    M = dense_mask(hide)
    for q, keys in vis.items():
        want = set(np.flatnonzero(M[q, :q + 1]).tolist())
        assert keys == want, f"query {q} ({stream[q]}): vLLM sees {sorted(keys)} training mask {sorted(want)}"
    never = {p for p in range(L) if p < len(prompt) or stream[p] in IDS.cache or stream[p] == IDS.cont}
    assert not (evicted & never), "prompt/cache/continue rows were evicted"
    assert state.visible_output_token_ids == list(generated)
    return len(vis), len(evicted)


def main():
    rng = np.random.default_rng(0)
    # hand case: two thoughts, then eos
    q, e = check([5, 6, 7], [10, 11, IDS.splitter, 12, IDS.splitter, 13, EOS])
    print(f"[ok] hand case: {q} queries match the training mask, {e} rows evicted")
    total = 0
    for i in range(300):
        n = int(rng.integers(1, 80))
        gen = rng.integers(10, 60, n).tolist()
        for j in range(n):
            if rng.random() < 0.15:
                gen[j] = IDS.splitter
        if rng.random() < 0.7:
            gen[-1] = EOS
        if gen[-1] == IDS.splitter:
            gen.append(int(rng.integers(10, 60)))       # the continue step samples one more token
        q, _ = check([int(x) for x in rng.integers(10, 60, int(rng.integers(1, 6)))], gen)
        total += q
    print(f"[ok] 300 random streams: every scheduled query's visible keys == training mask row ({total} queries)")
    # length cap: no room for splitter + cache + continue -> capped, not an overflow
    req, *_ = run([5, 6], [10, IDS.splitter, 11], max_model_len=6)
    assert req.status == RequestStatus.FINISHED_LENGTH_CAPPED
    print("[ok] length cap at a boundary finishes the request")
    # config guards
    try:
        cfg_mod.CompressLMConfig(enable=True, num_code_tokens=K, replay_tail=0, pause_xattn="full",
                                 boundary_token_id=IDS.splitter, continue_token_id=IDS.cont)
        raise AssertionError("boundary trigger with pause_xattn=full must be rejected")
    except ValueError:
        pass
    print("ALL LIGHTTHINKER VLLM STATE TESTS PASSED")


if __name__ == "__main__":
    main()
