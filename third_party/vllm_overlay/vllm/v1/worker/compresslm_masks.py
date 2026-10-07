# SPDX-License-Identifier: Apache-2.0
"""Attention masks for evaluation-only layer-selective CompressLM history and per-layer code quotas."""

from __future__ import annotations

from dataclasses import dataclass

import torch


CONTENT = 0
CODE = 1
REPLAY = 2


@dataclass(frozen=True)
class CompressLMLayout:
    block_size: int
    num_code_tokens: int
    replay_tail: int
    retain_tail_kv: int
    retain_tail_policy: str
    code_excluded_tail: int

    @property
    def period(self) -> int:
        return self.block_size + self.num_code_tokens + self.replay_tail


def _logical_columns(
    seq_len: int,
    prompt_len: int,
    layout: CompressLMLayout,
    device: torch.device,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    positions = torch.arange(seq_len, device=device)
    block = positions // layout.period
    within = positions % layout.period
    role = torch.full_like(positions, REPLAY)
    role[within < layout.block_size] = CONTENT
    role[
        (within >= layout.block_size)
        & (within < layout.block_size + layout.num_code_tokens)
    ] = CODE
    content_abs = block * layout.block_size + within
    is_prompt = (role == CONTENT) & (content_abs < prompt_len)
    return positions, block, role, content_abs, is_prompt


def build_request_masks(
    *,
    seq_len: int,
    query_len: int,
    prompt_len: int,
    layout: CompressLMLayout,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return faithful, history-plus-codes, and history-replaces-codes masks."""
    if not 0 < query_len <= seq_len:
        raise ValueError(f"invalid query_len={query_len} for seq_len={seq_len}")
    positions, block, role, content_abs, is_prompt = _logical_columns(
        seq_len, prompt_len, layout, device
    )
    query_positions = positions[-query_len:]
    query_roles = role[-query_len:]
    if not bool(torch.all(query_roles == query_roles[0])):
        raise RuntimeError("CompressLM model step crosses a role boundary")

    faithful_rows = []
    history_rows = []
    replacement_rows = []
    for query_position in query_positions:
        query_block = block[query_position]
        query_role = role[query_position]
        query_within = query_position % layout.period
        causal = positions <= query_position
        is_content = role == CONTENT
        is_code = role == CODE
        is_replay = role == REPLAY
        own_block = block == query_block

        if query_role == CONTENT:
            own_content = is_content & own_block & causal
            prior_codes = is_code & causal
            active_replay = torch.zeros_like(causal)
            if layout.replay_tail and query_block > 0:
                active_replay = (
                    is_replay
                    & (block + 1 == query_block)
                    & ((positions % layout.period) - layout.block_size
                       - layout.num_code_tokens >= query_within)
                )
            active_retained = torch.zeros_like(causal)
            if layout.retain_tail_kv and query_block > 0:
                previous_tail = (
                    is_content
                    & (block + 1 == query_block)
                    & (content_abs % layout.block_size
                       >= layout.block_size - layout.retain_tail_kv)
                )
                if layout.retain_tail_policy == "rolling":
                    previous_tail &= (
                        query_block * layout.block_size + query_within
                        - content_abs
                        <= layout.retain_tail_kv
                    )
                active_retained = previous_tail

            faithful = (
                (is_prompt & causal)
                | own_content
                | prior_codes
                | active_replay
                | active_retained
            )
            history = (
                (is_content & causal)
                | prior_codes
                | active_replay
            )
            replacement = (is_content & causal) | active_replay
        elif query_role == CODE:
            own_content = is_content & own_block
            if layout.code_excluded_tail:
                own_content &= (
                    content_abs % layout.block_size
                    < layout.block_size - layout.code_excluded_tail
                )
            faithful = is_prompt | own_content | is_code
            history = faithful
            replacement = faithful
        else:
            own_content = is_content & own_block
            own_replay = is_replay & own_block & causal
            faithful = is_prompt | own_content | (is_code & causal) | own_replay
            history = faithful
            replacement = faithful

        faithful_rows.append(faithful)
        history_rows.append(history)
        replacement_rows.append(replacement)

    return (
        torch.stack(faithful_rows),
        torch.stack(history_rows),
        torch.stack(replacement_rows),
    )


def layer_index(layer_name: str) -> int | None:
    marker = ".layers."
    if marker not in layer_name:
        return None
    suffix = layer_name.split(marker, 1)[1]
    value = suffix.split(".", 1)[0]
    return int(value) if value.isdigit() else None


def build_layer_quota_masks(
    requests: list[tuple[int, int, list[tuple[int, int]], tuple[int, int] | None]],
    quotas: list[int],
    num_code_tokens: int,
    causal: bool,
    device: torch.device,
) -> dict[int, torch.Tensor]:
    """Per-quota FlashInfer custom masks for per-layer code quotas on the stock paged cache.

    ``requests`` holds one ``(query_len, seq_len, code_spans, excluded_range)`` per scheduled request, in batch order:
    ``code_spans`` are the physical ``[start, end)`` ranges of the resident code spans (row k = code slot k) and
    ``excluded_range`` the physical content range hidden from code queries (static tail) on a code step. For each
    quota q the returned mask (q >= num_code_tokens: exclusion only, no span hidden) is the row-major concatenation of the per-request (query_len x seq_len)
    matrices: the ordinary visibility of the step (causal for content steps; everything but the excluded range for the
    noncausal code step) with columns ``[start + q, end)`` of every code span hidden. Vectorized over the batch: one
    (total_queries x max_seq_len) matrix per quota, gathered into the ragged layout FlashInfer consumes.
    """
    if not requests:
        return {}
    query_lens = torch.tensor([q for q, _, _, _ in requests], dtype=torch.long)
    seq_lens = torch.tensor([s for _, s, _, _ in requests], dtype=torch.long)
    if bool((query_lens <= 0).any()) or bool((query_lens > seq_lens).any()):
        raise ValueError("invalid query_len/seq_len in layer-quota mask build")
    num_reqs = len(requests)
    max_seq = int(seq_lens.max())
    row_req = torch.repeat_interleave(torch.arange(num_reqs), query_lens)  # request index of every query row
    row_pos = (
        torch.repeat_interleave(seq_lens - query_lens, query_lens)
        + torch.cat([torch.arange(int(q)) for q in query_lens])
    )  # physical position of every query row
    row_req = row_req.to(device, non_blocking=True)
    row_pos = row_pos.to(device, non_blocking=True)
    seq_lens_d = seq_lens.to(device, non_blocking=True)
    col = torch.arange(max_seq, device=device)
    valid = col[None, :] < seq_lens_d[row_req][:, None]  # (rows, max_seq): real columns of the row's request
    allowed = valid & (col[None, :] <= row_pos[:, None]) if causal else valid.clone()
    ex_start = torch.zeros(num_reqs, dtype=torch.long)
    ex_end = torch.zeros(num_reqs, dtype=torch.long)
    for r, (_, _, _, excluded) in enumerate(requests):
        if excluded is not None:
            ex_start[r], ex_end[r] = excluded
    if bool((ex_end > ex_start).any()):
        ex_start = ex_start.to(device)[row_req][:, None]
        ex_end = ex_end.to(device)[row_req][:, None]
        allowed &= ~((col[None, :] >= ex_start) & (col[None, :] < ex_end))
    out: dict[int, torch.Tensor] = {}
    for q in quotas:
        if q >= num_code_tokens:
            # Nothing hidden by the quota: the group still needs the step's ordinary visibility with the
            # excluded (static tail) range removed. Skipping it raised KeyError on the code step of the
            # tail-excluded combined arm (quota-256 group, 2026-09-21).
            out[q] = allowed[valid]
            continue
        rows_idx: list[int] = []
        cols_idx: list[int] = []
        vals: list[int] = []
        for r, (_, _, spans, _) in enumerate(requests):
            for start, end in spans:
                if start + q < end:
                    rows_idx += [r, r]
                    cols_idx += [start + q, end]
                    vals += [1, -1]
        if rows_idx:
            diff = torch.zeros((num_reqs, max_seq + 1), dtype=torch.int32, device=device)
            diff.index_put_(
                (torch.tensor(rows_idx, device=device), torch.tensor(cols_idx, device=device)),
                torch.tensor(vals, dtype=torch.int32, device=device),
                accumulate=True,
            )
            hidden = diff.cumsum(1)[:, :max_seq] > 0  # (num_reqs, max_seq)
            mask = allowed & ~hidden[row_req]
        else:
            mask = allowed
        out[q] = mask[valid]  # row-major ragged concatenation: per request, query_len * seq_len entries
    return out
