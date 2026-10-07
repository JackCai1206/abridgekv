"""LightThinker (Zhang et al., EMNLP 2025; zjunlp/LightThinker v1) thought-level compression, rebuilt on our stack.

Algorithm (their `aug-wo-pc` training mode, `output_comp_level=sentence`, `n_token=9`):
  * the response is cut into thoughts at "\\n\\n";
  * after every thought the sequence carries  <splitter> <cache_0>..<cache_8> <continue>  (the splitter is a
    training target, so the model learns to end a thought; cache and continue tokens are never targets);
  * positions are the plain sequential index over this augmented sequence (their rolling_rope is off);
  * attention (their `utils.create_attention_for_aug_data`, see_current/bi_directional/diagonal/exclude_continue all
    False): causal, except that the continue token of thought i and EVERY later token cannot see thought i's raw
    tokens (including its splitter). The cache tokens of thought i still see thought i, earlier cache/continue
    tokens stay visible for good, and the prompt is never hidden.
  * decoding: when the model samples the splitter, one forward runs [splitter, cache_0..8, continue], the thought's
    KV (incl. the splitter) is evicted, and the next token is sampled from the continue token's logits. Each thought
    therefore leaves 9 cache rows + 1 continue row in the KV cache.

Differences from their release (all reported with the results):
  * segmentation is done on token ids: a thought ends after any response token whose text ends with "\\n\\n" (Qwen
    merges punctuation into newline tokens, e.g. ".\\n\\n"); they segment text offline (`thoughts_list`, precomputed,
    rule not in the repo) and append "\\n" + splitter, with "\\n\\n" registered as a special token;
  * a response truncated mid-thought keeps its open last thought (no insertion), as the vanilla arm keeps its tail;
  * base model, data, optimizer, steps and eval are ours (Qwen3-4B-Base, 16K responses) instead of R1-Distill at 4K.

Token ids: the 11 new tokens take the first unused embedding rows after the tokenizer's vocabulary, in the order the
exporter registers them as additional special tokens: cache tokens "<|compress_code_000|>".."<|compress_code_008|>"
(so the vLLM CompressLM overlay finds them with its default code_token_format), then "<|lt_splitter|>",
"<|lt_continue|>".
"""
from dataclasses import dataclass

import numpy as np

N_CACHE = 9
CODE_TOKEN_FORMAT = "<|compress_code_{:03d}|>"
SPLITTER_TOKEN = "<|lt_splitter|>"
CONTINUE_TOKEN = "<|lt_continue|>"
NEVER = np.iinfo(np.int64).max // 4          # kv_hide_at for keys that are never hidden


@dataclass(frozen=True)
class LTIds:
    cache: tuple
    splitter: int
    cont: int

    @property
    def special_tokens(self):
        return [CODE_TOKEN_FORMAT.format(i) for i in range(len(self.cache))] + [SPLITTER_TOKEN, CONTINUE_TOKEN]


def lt_ids(tokenizer_len, n_cache=N_CACHE):
    base = int(tokenizer_len)
    return LTIds(cache=tuple(range(base, base + n_cache)), splitter=base + n_cache, cont=base + n_cache + 1)


def boundary_table(tokenizer, size):
    """bool[size]: token id ends a thought (its decoded text ends with a blank line)."""
    table = np.zeros(size, dtype=bool)
    for t in range(len(tokenizer)):
        if tokenizer.decode([t]).endswith("\n\n"):
            table[t] = True
    return table


def build_example(prompt, response, max_orig, ends, ids, eos_id):
    """Augment one (prompt, response) pair.

    prompt+response are truncated to `max_orig` ORIGINAL tokens first (exactly the vanilla arm's `batch()`), so both
    arms see the same text. Returns int64 arrays over the augmented sequence:
      ids         token ids
      target      1 where the token is a CE target (predicted from the previous position): original response tokens
                  (incl. eos) and splitters
      orig        1 where the token is an original response token (the token budget / comparable CE)
      hide_at     for raw thought tokens (incl. the splitter): index of the thought's continue token (no query at or
                  after it may attend them); NEVER elsewhere
    """
    prompt = np.asarray(prompt, dtype=np.int64)
    response = np.asarray(response, dtype=np.int64)[: max(0, max_orig - len(prompt))]
    n_resp = len(response)
    out_ids = list(prompt.tolist())
    target = [0] * len(prompt)
    orig = [0] * len(prompt)
    hide_at = [int(NEVER)] * len(prompt)
    thought_start = len(out_ids)                  # index of the current open thought's first token
    open_thought = False

    def close():
        nonlocal open_thought
        out_ids.append(ids.splitter); target.append(1); orig.append(0); hide_at.append(int(NEVER))
        cont_at = len(out_ids) + len(ids.cache)
        for j in range(thought_start, len(out_ids)):
            hide_at[j] = cont_at
        for c in ids.cache:
            out_ids.append(c); target.append(0); orig.append(0); hide_at.append(int(NEVER))
        out_ids.append(ids.cont); target.append(0); orig.append(0); hide_at.append(int(NEVER))
        open_thought = False

    for i, t in enumerate(response.tolist()):
        if t == eos_id:
            if open_thought:
                close()                           # their data: last thought + splitter + cache + continue, then eos
            out_ids.append(t); target.append(1); orig.append(1); hide_at.append(int(NEVER))
            continue
        if not open_thought:
            thought_start = len(out_ids); open_thought = True
        out_ids.append(t); target.append(1); orig.append(1); hide_at.append(int(NEVER))
        if ends[t] and i + 1 < n_resp:            # a boundary that is the last kept token stays open (truncation)
            close()
    return {
        "ids": np.asarray(out_ids, dtype=np.int64),
        "target": np.asarray(target, dtype=np.int64),
        "orig": np.asarray(orig, dtype=np.int64),
        "hide_at": np.asarray(hide_at, dtype=np.int64),
        "prompt_len": len(prompt),
    }


def dense_mask(hide_at):
    """bool[S, S] allowed(q, k) for one sequence (reference for the flex mask_mod)."""
    S = len(hide_at)
    q = np.arange(S)[:, None]
    k = np.arange(S)[None, :]
    return (k <= q) & (q < np.asarray(hide_at)[None, :])


def mask_mod_factory(hide_at):
    """FlexAttention mask_mod over a padded [B, S] hide_at tensor (pads carry NEVER; pads sit after the sequence so
    causality alone keeps real queries off them)."""
    def mask_mod(b, h, q_idx, kv_idx):
        return (kv_idx <= q_idx) & (q_idx < hide_at[b, kv_idx])
    return mask_mod


def segments_for_reference(example, ids):
    """Their locate_index_list for our example: one [start, end, l_inst=0, n_comp, n_continue=1] per closed thought,
    with [start, end) the thought INCLUDING its splitter (their 'abandoned' segment = thought + '\\n' + splitter)."""
    seq = example["ids"]
    segs = []
    start = None
    for p in range(example["prompt_len"], len(seq)):
        t = int(seq[p])
        if t == ids.cont:
            start = None
        elif t == ids.splitter:
            segs.append([start, p + 1, 0, len(ids.cache), 1])
            start = None
        elif t not in ids.cache and start is None:
            start = p
    return segs


def augment_generated(prompt, generated, ids):
    """The sequence a LightThinker decoder actually processed for `generated` (visible tokens, splitters included):
    after every splitter the cache tokens and the continue token follow. Returns (ids, hide_at, pred_pos) where
    pred_pos[j] is the position whose logits produced generated[j] (the continue token for a thought's first token).
    A trailing splitter's cache/continue tokens are appended too (the decoder ran them before stopping or not; they
    predict nothing that is compared)."""
    seq = list(int(t) for t in prompt)
    hide = [int(NEVER)] * len(seq)
    pred = []
    start = len(seq)
    for t in generated:
        pred.append(len(seq) - 1)
        seq.append(int(t)); hide.append(int(NEVER))
        if int(t) == ids.splitter:
            cont_at = len(seq) + len(ids.cache)
            for j in range(start, len(seq)):
                hide[j] = cont_at
            seq.extend(ids.cache); hide.extend([int(NEVER)] * len(ids.cache))
            seq.append(ids.cont); hide.append(int(NEVER))
            start = len(seq)
    return np.asarray(seq, dtype=np.int64), np.asarray(hide, dtype=np.int64), np.asarray(pred, dtype=np.int64)
