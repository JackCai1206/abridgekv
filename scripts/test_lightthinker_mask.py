"""CPU tests: our LightThinker sequence builder + attention mask vs their released implementation.

The reference mask is THEIR function, imported from the cloned repo (zjunlp/LightThinker, lightthinker_v1/LightThinker/
utils.py: create_attention_for_aug_data) with the flags of their train.sh (see_current, bi_directional, diagonal,
exclude_continue, prefill_compress all false), fed the locate_index list their tokenizer would build for our sequence.
Also checks the FlexAttention BlockMask (batched, padded) densifies to the same mask, and the label layout.

  python scripts/test_lightthinker_mask.py
"""
import os
import sys
import types

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from abridgekv.lightthinker import (  # noqa: E402
    NEVER, augment_generated, build_example, dense_mask, lt_ids, mask_mod_factory, segments_for_reference,
)

THEIRS = os.environ["LIGHTTHINKER_REPO"]   # a clone of github.com/zjunlp/LightThinker
sys.modules.setdefault("jsonlines", types.ModuleType("jsonlines"))   # imported by their utils.py, unused here
sys.path.insert(0, THEIRS)
from utils import create_attention_for_aug_data  # noqa: E402  (their code)

VOCAB = 50
IDS = lt_ids(VOCAB)
EOS = 3
ENDS = np.zeros(VOCAB + 16, dtype=bool)
ENDS[[7, 8]] = True                               # toy "\n\n" and ".\n\n" tokens


def theirs(ex):
    segs = segments_for_reference(ex, IDS)
    return create_attention_for_aug_data(
        input_ids=ex["ids"].tolist(), locate_index_list=segs, locate_indicator_list=["compressed-output"] * len(segs),
        bi_directional=False, see_current=False, diagonal=False, exclude_continue=False, prefill_compress=False,
    ).numpy()


def their_labels(ex):
    """Their aug-wo-pc labels: prompt ignored; thought + splitter and eos labeled; cache + continue masked."""
    seq = ex["ids"]
    lab = np.array([0 if (t in IDS.cache or t == IDS.cont) else 1 for t in seq.tolist()])
    lab[: ex["prompt_len"]] = 0
    return lab


def random_example(rng, truncate=False):
    plen = int(rng.integers(1, 6))
    prompt = rng.integers(10, VOCAB, plen)
    n = int(rng.integers(1, 60))
    resp = rng.integers(10, VOCAB, n)
    resp[rng.random(n) < 0.2] = rng.choice([7, 8])  # boundaries, incl. back-to-back ones
    resp = np.concatenate([resp, [EOS]])
    max_orig = plen + len(resp) if not truncate else plen + int(rng.integers(1, len(resp) + 1))
    return prompt, resp, max_orig


def check_structure(prompt, resp, max_orig, ex):
    kept = resp[: max_orig - len(prompt)]
    body = ex["ids"][ex["orig"] == 1]
    assert np.array_equal(body, kept), "original tokens must survive in order"
    assert int(ex["orig"].sum()) == len(kept)
    # a thought closes after every kept boundary token that is not the last kept token, and before a kept eos
    closes = sum(1 for t in kept[:-1] if ENDS[t])
    if len(kept) and kept[-1] == EOS and len(kept) >= 2 and not ENDS[kept[-2]]:
        closes += 1
    assert int((ex["ids"] == IDS.splitter).sum()) == closes, (closes, ex["ids"])
    # every splitter is followed by the 9 cache tokens and one continue
    for p in np.flatnonzero(ex["ids"] == IDS.splitter):
        assert ex["ids"][p + 1:p + 1 + len(IDS.cache)].tolist() == list(IDS.cache)
        assert ex["ids"][p + 1 + len(IDS.cache)] == IDS.cont
    assert np.array_equal(ex["target"], their_labels(ex)), (ex["target"], their_labels(ex))


def test_against_theirs(n=400):
    rng = np.random.default_rng(0)
    for i in range(n):
        prompt, resp, max_orig = random_example(rng, truncate=(i % 3 == 0))
        ex = build_example(prompt, resp, max_orig, ENDS, IDS, EOS)
        check_structure(prompt, resp, max_orig, ex)
        ours, ref = dense_mask(ex["hide_at"]), theirs(ex)
        assert ours.shape == ref.shape and np.array_equal(ours, ref), f"mask mismatch on example {i}"
    print(f"[ok] dense mask == their create_attention_for_aug_data on {n} random examples (+ labels, layout)")


def test_hand_example():
    # prompt [11 12] | thought A = [20 7] | thought B = [21 22 8] | tail [23] + eos
    ex = build_example([11, 12], [20, 7, 21, 22, 8, 23, EOS], 100, ENDS, IDS, EOS)
    S, c = IDS.splitter, list(IDS.cache)
    expect = [11, 12, 20, 7, S] + c + [IDS.cont, 21, 22, 8, S] + c + [IDS.cont, 23, S] + c + [IDS.cont, EOS]
    assert ex["ids"].tolist() == expect, ex["ids"].tolist()
    m = dense_mask(ex["hide_at"])
    cont_a = 5 + len(c)
    assert m[cont_a - 1, 2:5].all(), "last cache token of A sees thought A incl. its splitter"
    assert not m[cont_a, 2:5].any() and m[cont_a, :2].all(), "continue of A: thought A hidden, prompt visible"
    assert m[-1, 5:cont_a + 1].all(), "A's cache + continue stay visible to the end"
    print("[ok] hand-built example: layout, hidden spans, retained cache/continue rows")


def test_flex_blockmask():
    from torch.nn.attention.flex_attention import create_block_mask
    rng = np.random.default_rng(1)
    exs = [build_example(*random_example(rng), ENDS, IDS, EOS) for _ in range(3)]
    S = 256                                        # > 128 so several sparse blocks exist
    hide = torch.full((len(exs), S), int(NEVER), dtype=torch.long)
    for b, ex in enumerate(exs):
        hide[b, :len(ex["ids"])] = torch.from_numpy(ex["hide_at"])
    bm = create_block_mask(mask_mod_factory(hide), B=len(exs), H=None, Q_LEN=S, KV_LEN=S, device="cpu", _compile=False)
    dense = bm.to_dense()                          # block-level; check element-wise via the mask_mod on the grid
    q = torch.arange(S)[:, None]
    k = torch.arange(S)[None, :]
    for b, ex in enumerate(exs):
        el = mask_mod_factory(hide)(torch.tensor(b), 0, q, k).numpy()
        L = len(ex["ids"])
        assert np.array_equal(el[:L, :L], dense_mask(ex["hide_at"])), "mask_mod != dense reference"
        assert not el[:L, L:].any(), "real queries must not see padding"
        # the block mask must not drop any allowed (q, k) block
        blk = dense[b, 0].numpy().astype(bool)
        need = np.zeros_like(blk)
        qi, ki = np.nonzero(el)
        need[qi // 128, ki // 128] = True
        assert not (need & ~blk).any(), "BlockMask skips a block the mask needs"
    # flex attention output == SDPA with the dense mask
    from torch.nn.attention.flex_attention import flex_attention
    torch.manual_seed(0)
    qh, kh, vh = (torch.randn(len(exs), 2, S, 16) for _ in range(3))
    out = flex_attention(qh, kh, vh, block_mask=bm)
    full = torch.stack([torch.from_numpy(mask_mod_factory(hide)(torch.tensor(b), 0, q, k).numpy()) for b in range(len(exs))])
    ref = torch.nn.functional.scaled_dot_product_attention(qh, kh, vh, attn_mask=full[:, None])
    for b, ex in enumerate(exs):
        L = len(ex["ids"])
        torch.testing.assert_close(out[b, :, :L], ref[b, :, :L], atol=1e-5, rtol=1e-5)
    print("[ok] FlexAttention BlockMask (batched, padded) == dense reference; flex output == SDPA with the dense mask")


def test_model_forward():
    """Whole-model check: HF Qwen3 with our FlexAttention BlockMask == HF eager attention with the dense mask."""
    from torch.nn.attention.flex_attention import create_block_mask
    from transformers import Qwen3Config, Qwen3ForCausalLM
    torch._dynamo.config.disable = True                        # CPU: eager flex (HF would compile it), as the trainer does
    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=VOCAB + 16, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=1024)
    rng = np.random.default_rng(2)
    exs = [build_example(*random_example(rng), ENDS, IDS, EOS) for _ in range(2)]
    S = max(len(e["ids"]) for e in exs)
    x = torch.zeros(len(exs), S, dtype=torch.long)
    hide = torch.full((len(exs), S), int(NEVER), dtype=torch.long)
    for b, e in enumerate(exs):
        x[b, :len(e["ids"])] = torch.from_numpy(e["ids"]); hide[b, :len(e["ids"])] = torch.from_numpy(e["hide_at"])
    pos = torch.arange(S)[None].expand(len(exs), S)
    import copy
    cf, ce = copy.deepcopy(cfg), copy.deepcopy(cfg)          # separate configs: the attention backend lives on the config
    cf._attn_implementation, ce._attn_implementation = "flex_attention", "eager"
    flex = Qwen3ForCausalLM(cf)
    eager = Qwen3ForCausalLM(ce); eager.load_state_dict(flex.state_dict())
    bm = create_block_mask(mask_mod_factory(hide), B=len(exs), H=None, Q_LEN=S, KV_LEN=S, device="cpu", _compile=False)
    q = torch.arange(S)[:, None]; k = torch.arange(S)[None, :]
    allowed = torch.stack([mask_mod_factory(hide)(torch.tensor(b), 0, q, k) for b in range(len(exs))])[:, None]
    add = torch.zeros(allowed.shape).masked_fill(~allowed, float("-inf"))
    with torch.no_grad():
        hf = flex.model(input_ids=x, position_ids=pos, attention_mask=bm).last_hidden_state
        he = eager.model(input_ids=x, position_ids=pos, attention_mask=add).last_hidden_state
    for b, e in enumerate(exs):
        L = len(e["ids"])
        torch.testing.assert_close(hf[b, :L], he[b, :L], atol=2e-5, rtol=2e-5)
    # and the mask matters: plain causal differs after the first continue token
    with torch.no_grad():
        hc = eager.model(input_ids=x, position_ids=pos).last_hidden_state
    first_cont = int(np.flatnonzero(exs[0]["ids"] == IDS.cont)[0]) if (exs[0]["ids"] == IDS.cont).any() else None
    if first_cont is not None:
        assert not torch.allclose(hc[0, first_cont], hf[0, first_cont], atol=1e-4)
    print("[ok] HF Qwen3 forward with the flex BlockMask == eager attention with the dense LightThinker mask")


def test_generation_layout():
    """The decoder-side layout (splitter sampled -> cache + continue follow) equals the training layout, so the
    vLLM parity check's HF reference runs exactly the trained mask."""
    rng = np.random.default_rng(3)
    for _ in range(200):
        prompt, resp, max_orig = random_example(rng)
        ex = build_example(prompt, resp, max_orig, ENDS, IDS, EOS)
        gen = [t for t in ex["ids"][len(prompt):].tolist() if t not in IDS.cache and t != IDS.cont]
        seq, hide, pred = augment_generated(prompt, gen, IDS)
        assert seq.tolist() == ex["ids"].tolist() and hide.tolist() == ex["hide_at"].tolist()
        assert all(seq[p + 1] == g for p, g in zip(pred, gen))
    print("[ok] generation-side layout == training layout on 200 examples")


if __name__ == "__main__":
    test_hand_example()
    test_against_theirs()
    test_flex_blockmask()
    test_generation_layout()
    test_model_forward()
    print("ALL LIGHTTHINKER MASK TESTS PASSED")
