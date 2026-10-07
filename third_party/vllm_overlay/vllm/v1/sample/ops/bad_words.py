# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bad-word masking with a vectorized path for single-token bans.

CompressLM forbids every learned code token (K_max of them, 512 for the
C=512 quota study) by passing them as single-token ``bad_words``. The stock
implementation loops over every banned word for every request on every
decode step in Python, which dominated packed-KV decode (about 20% of all
samples in a py-spy profile). Single-token bans are order-independent, so
they are applied here as one indexed ``-inf`` assignment for the whole batch;
multi-token bans keep the original prefix-matching loop.
"""

import torch

_SMALLEST_LOGIT = float("-inf")

# id(list) -> (list, singleton token-id tensor or None, multi-token words)
_SPLIT_CACHE: dict[int, tuple[list[list[int]], torch.Tensor | None, list[list[int]]]] = {}


def _split_bad_words(
    bad_words_token_ids: list[list[int]], device: torch.device
) -> tuple[torch.Tensor | None, list[list[int]]]:
    key = id(bad_words_token_ids)
    cached = _SPLIT_CACHE.get(key)
    if cached is not None and cached[0] is bad_words_token_ids:
        singles, multi = cached[1], cached[2]
        if singles is None or singles.device == device:
            return singles, multi
    single_ids = [w[0] for w in bad_words_token_ids if len(w) == 1]
    multi = [w for w in bad_words_token_ids if len(w) != 1]
    singles = (
        torch.tensor(single_ids, dtype=torch.long, device=device)
        if single_ids
        else None
    )
    if len(_SPLIT_CACHE) > 4096:
        _SPLIT_CACHE.clear()
    _SPLIT_CACHE[key] = (bad_words_token_ids, singles, multi)
    return singles, multi


def _apply_bad_words_single_batch(
    logits: torch.Tensor,
    bad_words_token_ids: list[list[int]],
    past_tokens_ids: list[int],
) -> None:
    for bad_word_ids in bad_words_token_ids:
        if len(bad_word_ids) > len(past_tokens_ids) + 1:
            continue

        prefix_length = len(bad_word_ids) - 1
        last_token_id = bad_word_ids[-1]
        actual_prefix = past_tokens_ids[-prefix_length:] if prefix_length > 0 else []
        expected_prefix = bad_word_ids[:prefix_length]

        assert len(actual_prefix) == len(expected_prefix)

        if actual_prefix == expected_prefix:
            logits[last_token_id] = _SMALLEST_LOGIT


def apply_bad_words(
    logits: torch.Tensor,
    bad_words_token_ids: dict[int, list[list[int]]],
    past_tokens_ids: list[list[int]],
) -> None:
    rows: list[torch.Tensor] = []
    cols: list[torch.Tensor] = []
    for i, bad_words_ids in bad_words_token_ids.items():
        singles, multi = _split_bad_words(bad_words_ids, logits.device)
        if singles is not None:
            rows.append(torch.full_like(singles, i))
            cols.append(singles)
        if multi:
            _apply_bad_words_single_batch(logits[i], multi, past_tokens_ids[i])
    if rows:
        logits[torch.cat(rows), torch.cat(cols)] = _SMALLEST_LOGIT


def apply_bad_words_with_drafts(
    logits: torch.Tensor,
    bad_words_token_ids: dict[int, list[list[int]]],
    past_tokens_ids: list[list[int]],
    num_draft_tokens: list[int],
) -> None:
    start_idx = 0
    for i, bad_words_ids in bad_words_token_ids.items():
        for draft_idx in range(num_draft_tokens[i]):
            _apply_bad_words_single_batch(
                logits[start_idx + draft_idx],
                bad_words_ids,
                past_tokens_ids[start_idx + draft_idx],
            )
        start_idx += num_draft_tokens[i]
