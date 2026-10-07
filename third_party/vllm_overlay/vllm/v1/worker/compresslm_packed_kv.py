# SPDX-License-Identifier: Apache-2.0
"""Ragged per-layer/head KV storage for CompressLM pause codes."""

import os
from dataclasses import dataclass
from math import ceil
from typing import Sequence

import torch


@dataclass(frozen=True)
class PackedKVLayout:
    """Static ``K[layer, kv_head]`` layout for one compressed block."""

    quotas: tuple[tuple[int, ...], ...]
    k_max: int

    def __post_init__(self) -> None:
        if self.k_max <= 0:
            raise ValueError("packed KV k_max must be positive")
        if not self.quotas or not self.quotas[0]:
            raise ValueError("packed KV quotas must be a non-empty 2D schedule")
        num_heads = len(self.quotas[0])
        for layer, row in enumerate(self.quotas):
            if len(row) != num_heads:
                raise ValueError(
                    "packed KV quota rows must have equal length; "
                    f"layer 0 has {num_heads}, layer {layer} has {len(row)}"
                )
            for head, quota in enumerate(row):
                if not 0 < quota <= self.k_max:
                    raise ValueError(
                        f"packed KV quota[{layer}][{head}]={quota} must be "
                        f"in [1, {self.k_max}]"
                    )

    @classmethod
    def from_schedule(
        cls,
        schedule: Sequence[Sequence[int]] | None,
        *,
        num_layers: int,
        num_kv_heads: int,
        k_max: int,
    ) -> "PackedKVLayout":
        if num_layers <= 0 or num_kv_heads <= 0:
            raise ValueError("packed KV model dimensions must be positive")
        if schedule:
            quotas = tuple(tuple(int(value) for value in row) for row in schedule)
        else:
            quotas = tuple(
                (int(k_max),) * num_kv_heads for _ in range(num_layers)
            )
        if len(quotas) != num_layers:
            raise ValueError(
                f"packed KV schedule has {len(quotas)} layers; "
                f"model has {num_layers}"
            )
        if any(len(row) != num_kv_heads for row in quotas):
            widths = sorted(set(len(row) for row in quotas))
            raise ValueError(
                f"packed KV schedule row widths {widths} do not match "
                f"model num_kv_heads={num_kv_heads}"
            )
        return cls(quotas=quotas, k_max=int(k_max))

    @property
    def num_layers(self) -> int:
        return len(self.quotas)

    @property
    def num_kv_heads(self) -> int:
        return len(self.quotas[0])

    @property
    def offsets(self) -> tuple[tuple[int, ...], ...]:
        rows = []
        for quotas in self.quotas:
            offsets = [0]
            for quota in quotas:
                offsets.append(offsets[-1] + quota)
            rows.append(tuple(offsets))
        return tuple(rows)

    @property
    def packed_slots_per_layer(self) -> tuple[int, ...]:
        return tuple(sum(row) for row in self.quotas)

    @property
    def packed_slots(self) -> int:
        return sum(self.packed_slots_per_layer)

    @property
    def dense_slots(self) -> int:
        return self.num_layers * self.num_kv_heads * self.k_max

    @property
    def storage_ratio(self) -> float:
        return self.packed_slots / self.dense_slots

    def bytes_per_block(self, *, head_size: int, element_size: int) -> int:
        if head_size <= 0 or element_size <= 0:
            raise ValueError("head_size and element_size must be positive")
        return 2 * self.packed_slots * head_size * element_size

    def dense_bytes_per_block(self, *, head_size: int, element_size: int) -> int:
        if head_size <= 0 or element_size <= 0:
            raise ValueError("head_size and element_size must be positive")
        return 2 * self.dense_slots * head_size * element_size

    def pack_layer(self, layer: int, dense: torch.Tensor) -> torch.Tensor:
        """Pack ``[2, K_max, H_kv, D]`` into ``[2, sum_h K_lh, D]``."""
        expected = (2, self.k_max, self.num_kv_heads)
        if dense.ndim != 4 or tuple(dense.shape[:3]) != expected:
            raise ValueError(
                f"dense layer KV must start with shape {expected}; "
                f"got {tuple(dense.shape)}"
            )
        return torch.cat(
            [
                dense[:, :quota, head, :]
                for head, quota in enumerate(self.quotas[layer])
            ],
            dim=1,
        ).contiguous()

    def unpack_layer(
        self,
        layer: int,
        packed: torch.Tensor,
        *,
        fill_value: float = 0.0,
    ) -> torch.Tensor:
        """Expand a packed layer page to ``[2, K_max, H_kv, D]``."""
        expected_slots = self.packed_slots_per_layer[layer]
        if packed.ndim != 3 or packed.shape[0] != 2:
            raise ValueError(
                "packed layer KV must have shape [2, packed_slots, head_size]"
            )
        if packed.shape[1] != expected_slots:
            raise ValueError(
                f"packed layer {layer} has {packed.shape[1]} slots; "
                f"expected {expected_slots}"
            )
        dense = packed.new_full(
            (2, self.k_max, self.num_kv_heads, packed.shape[-1]),
            fill_value,
        )
        offsets = self.offsets[layer]
        for head, quota in enumerate(self.quotas[layer]):
            dense[:, :quota, head, :] = packed[
                :, offsets[head] : offsets[head + 1], :
            ]
        return dense


def _gather_cache_span(
    kv_cache: torch.Tensor,
    block_ids: Sequence[int] | torch.Tensor,
    *,
    start: int,
    end: int,
    block_size: int,
) -> torch.Tensor:
    """Gather a physical cache span as ``[2, tokens, H_kv, D]``."""
    if not 0 <= start < end:
        raise ValueError(f"invalid packed KV capture span [{start}, {end})")
    positions = torch.arange(start, end, device=kv_cache.device)
    table = torch.as_tensor(block_ids, dtype=torch.long, device=kv_cache.device)
    blocks = table[positions // block_size]
    offsets = positions % block_size
    if kv_cache.shape[0] == 2:
        return kv_cache[:, blocks, offsets].clone()
    if kv_cache.ndim != 5 or kv_cache.shape[1] != 2:
        raise ValueError(f"unsupported KV cache shape {tuple(kv_cache.shape)}")
    if kv_cache.shape[2] == block_size:
        return kv_cache[blocks, :, offsets].permute(1, 0, 2, 3).clone()
    if kv_cache.shape[3] == block_size:
        return kv_cache[blocks, :, :, offsets].permute(1, 0, 2, 3).clone()
    raise ValueError(
        f"cannot identify block-size axis in KV cache shape {tuple(kv_cache.shape)}"
    )


def _gather_cache_span_indexed(
    kv_cache: torch.Tensor,
    blocks: torch.Tensor,
    offsets: torch.Tensor,
    *,
    block_size: int,
) -> torch.Tensor:
    """``_gather_cache_span`` with precomputed block/offset indices.

    ``blocks``/``offsets`` may have any common shape ``I``; the result is
    ``[2, *I, H_kv, D]``."""
    lead = blocks.dim()
    if kv_cache.shape[0] == 2:
        return kv_cache[:, blocks, offsets]
    if kv_cache.ndim != 5 or kv_cache.shape[1] != 2:
        raise ValueError(f"unsupported KV cache shape {tuple(kv_cache.shape)}")
    # Advanced indices separated by a slice move to the front: [*I, 2, H, D].
    if kv_cache.shape[2] == block_size:
        return kv_cache[blocks, :, offsets].movedim(lead, 0)
    if kv_cache.shape[3] == block_size:
        return kv_cache[blocks, :, :, offsets].movedim(lead, 0)
    raise ValueError(
        f"cannot identify block-size axis in KV cache shape {tuple(kv_cache.shape)}"
    )


class PackedKVSidecar:
    """Worker-local packed pages and a correctness-first attention consumer."""

    def __init__(
        self,
        layout: PackedKVLayout,
        attention_layers: Sequence[object],
        *,
        block_size: int,
    ) -> None:
        if len(attention_layers) != layout.num_layers:
            raise ValueError(
                f"received {len(attention_layers)} attention layers for "
                f"{layout.num_layers}-layer packed layout"
            )
        self.layout = layout
        self.attention_layers = tuple(attention_layers)
        self.block_size = int(block_size)
        self.pages: dict[str, list[list[torch.Tensor]]] = {}
        self.active_request_ids: tuple[str, ...] = ()
        self.common_attn_metadata: object | None = None
        # Vectorized consumer (all KV heads of a request in one shot). Set
        # COMPRESSLM_PACKED_FAST=0 to fall back to the per-head reference loop.
        self.fast_path = os.environ.get("COMPRESSLM_PACKED_FAST", "1") != "0"
        self._side_index_cache: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}
        # Code step only: req_id -> physical [start, end) of the own-block content the codes must NOT attend
        # (static retained tail; the summarized span is [0, C - T) of the block). None on content steps.
        self.code_excluded_ranges: dict[str, tuple[int, int]] | None = None

    def set_active_request_ids(self, request_ids: Sequence[str]) -> None:
        self.active_request_ids = tuple(request_ids)

    def set_code_excluded_ranges(
        self, ranges: dict[str, tuple[int, int]] | None
    ) -> None:
        self.code_excluded_ranges = dict(ranges) if ranges else None

    def _code_excluded_range(self, request_id: str, seq_len: int, query_len: int) -> tuple[int, int] | None:
        if not self.code_excluded_ranges:
            return None
        rng = self.code_excluded_ranges.get(request_id)
        if rng is None:
            raise RuntimeError(f"packed KV code step: request {request_id} has no tail exclusion range")
        start, end = rng
        if not 0 <= start < end <= seq_len - query_len:
            raise RuntimeError(
                f"packed KV code step: invalid tail exclusion range {rng} for seq_len={seq_len}, query_len={query_len}"
            )
        return start, end

    def set_common_attn_metadata(self, metadata: object) -> None:
        self.common_attn_metadata = metadata
        # Main-span gather indices are identical for every layer within a
        # step; they are rebuilt lazily per request after each metadata update.
        self._span_index_cache: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}
        self._all_decode_cache = None
        self._batched_main_cache = None

    def _main_span_index(
        self, request_index: int, seq_len: int, block_table: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key = (request_index, seq_len)
        cached = getattr(self, "_span_index_cache", {}).get(key)
        if cached is not None:
            return cached
        num_blocks = ceil(seq_len / self.block_size)
        ids = block_table[request_index, :num_blocks]
        positions = torch.arange(seq_len, device=ids.device)
        table = ids.to(dtype=torch.long)
        cached = (table[positions // self.block_size], positions % self.block_size)
        if not hasattr(self, "_span_index_cache"):
            self._span_index_cache = {}
        self._span_index_cache[key] = cached
        return cached

    def free(self, request_id: str) -> None:
        self.pages.pop(request_id, None)

    def capture(
        self,
        request_id: str,
        *,
        physical_start: int,
        physical_end: int,
        block_ids: Sequence[int],
    ) -> None:
        if physical_end - physical_start != self.layout.k_max:
            raise ValueError(
                "packed KV capture must contain exactly k_max code rows; "
                f"got {physical_end - physical_start}, expected {self.layout.k_max}"
            )
        request_pages = self.pages.setdefault(
            request_id,
            [[] for _ in range(self.layout.num_layers)],
        )
        for layer_index, attention in enumerate(self.attention_layers):
            kv_cache = attention.kv_cache[0]
            dense = _gather_cache_span(
                kv_cache,
                block_ids,
                start=physical_start,
                end=physical_end,
                block_size=self.block_size,
            )
            request_pages[layer_index].append(
                self.layout.pack_layer(layer_index, dense)
            )

    @property
    def allocated_bytes(self) -> int:
        return sum(
            page.numel() * page.element_size()
            for layers in self.pages.values()
            for pages in layers
            for page in pages
        )

    def override_attention(
        self,
        *,
        layer_index: int,
        layer: object,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: object,
        output: torch.Tensor,
    ) -> None:
        if not self.active_request_ids:
            return
        if query.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise RuntimeError(
                f"packed KV reference attention does not support {query.dtype}"
            )
        common = self.common_attn_metadata
        if common is None:
            raise RuntimeError("packed KV common attention metadata is absent")
        query_starts = common.query_start_loc_cpu
        seq_lens = common._seq_lens_cpu
        block_table = common.block_table_tensor
        causal = bool(common.causal)
        num_heads = int(layer.num_heads)
        num_kv_heads = int(layer.num_kv_heads)
        groups = num_heads // num_kv_heads
        scale = float(layer.impl.scale)
        offsets = self.layout.offsets[layer_index]

        if self.fast_path and causal and self._all_decode(query_starts):
            self._attend_decode_batched(
                layer_index=layer_index,
                query=query,
                kv_cache=kv_cache,
                output=output,
                seq_lens=seq_lens,
                block_table=block_table,
                num_kv_heads=num_kv_heads,
                groups=groups,
                scale=scale,
            )
            return

        for request_index, request_id in enumerate(self.active_request_ids):
            request_pages = self.pages.get(request_id)
            layer_pages = (
                request_pages[layer_index] if request_pages is not None else []
            )
            query_start = int(query_starts[request_index])
            query_end = int(query_starts[request_index + 1])
            query_len = query_end - query_start
            if query_len <= 0:
                continue
            seq_len = int(seq_lens[request_index])
            if self.fast_path:
                blocks, offsets_in_block = self._main_span_index(
                    request_index, seq_len, block_table
                )
                main = _gather_cache_span_indexed(
                    kv_cache, blocks, offsets_in_block, block_size=self.block_size
                )
            else:
                num_blocks = ceil(seq_len / self.block_size)
                ids = block_table[request_index, :num_blocks]
                main = _gather_cache_span(
                    kv_cache,
                    ids,
                    start=0,
                    end=seq_len,
                    block_size=self.block_size,
                )
            request_query = query[query_start:query_end]
            request_output = output[query_start:query_end]
            prefix_len = seq_len - query_len

            if self.fast_path:
                self._attend_request_vectorized(
                    layer_index=layer_index,
                    layer_pages=layer_pages,
                    main=main,
                    request_query=request_query,
                    request_output=request_output,
                    seq_len=seq_len,
                    query_len=query_len,
                    prefix_len=prefix_len,
                    causal=causal,
                    scale=scale,
                    num_kv_heads=num_kv_heads,
                    groups=groups,
                    request_id=request_id,
                )
                continue

            for kv_head in range(num_kv_heads):
                side_start, side_end = offsets[kv_head : kv_head + 2]
                if layer_pages:
                    side_key = torch.cat(
                        [
                            page[0, side_start:side_end]
                            for page in layer_pages
                        ],
                        dim=0,
                    )
                    side_value = torch.cat(
                        [
                            page[1, side_start:side_end]
                            for page in layer_pages
                        ],
                        dim=0,
                    )
                else:
                    side_key = main.new_empty((0, main.shape[-1]))
                    side_value = main.new_empty((0, main.shape[-1]))
                key = torch.cat([side_key, main[0, :, kv_head]], dim=0)
                value = torch.cat([side_value, main[1, :, kv_head]], dim=0)
                head_start = kv_head * groups
                head_end = head_start + groups
                q = request_query[:, head_start:head_end]
                scores = torch.einsum("qhd,td->qht", q, key) * scale
                if causal and query_len > 1:
                    side_len = side_key.shape[0]
                    main_positions = torch.arange(
                        seq_len, device=query.device
                    )
                    visible_end = prefix_len + torch.arange(
                        1, query_len + 1, device=query.device
                    )
                    hidden = main_positions.unsqueeze(0) >= visible_end.unsqueeze(1)
                    scores[:, :, side_len:].masked_fill_(
                        hidden.unsqueeze(1), -torch.inf
                    )
                elif not causal:
                    if query_len != self.layout.k_max:
                        raise RuntimeError(
                            "packed KV noncausal step must contain exactly "
                            f"k_max={self.layout.k_max} code queries; "
                            f"got {query_len}"
                        )
                    quota = self.layout.quotas[layer_index][kv_head]
                    scores[
                        :,
                        :,
                        side_key.shape[0] + prefix_len + quota :,
                    ] = -torch.inf
                    excluded = self._code_excluded_range(request_id, seq_len, query_len)
                    if excluded is not None:
                        scores[:, :, side_key.shape[0] + excluded[0] : side_key.shape[0] + excluded[1]] = -torch.inf
                weights = torch.softmax(
                    scores, dim=-1, dtype=torch.float32
                ).to(q.dtype)
                attended = torch.einsum(
                    "qht,td->qhd", weights, value
                ).to(request_output.dtype)
                request_output[:, head_start:head_end].copy_(attended)

    # ------------------------------------------------------------------
    # Batched decode path: every active request contributes exactly one
    # causal query, so all requests are attended with one gather, one
    # score einsum, one softmax and one value einsum per layer.
    # ------------------------------------------------------------------
    def _all_decode(self, query_starts: torch.Tensor) -> bool:
        n = len(self.active_request_ids)
        if n == 0:
            return False
        cached = getattr(self, "_all_decode_cache", None)
        if cached is not None:
            return cached
        starts = query_starts[: n + 1].tolist()
        result = all(starts[i + 1] - starts[i] == 1 for i in range(n))
        self._all_decode_cache = result
        return result

    def _batched_main_index(
        self, seq_lens: torch.Tensor, block_table: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int]]:
        """Padded ``[R, S_max]`` block/offset indices plus a hidden mask for
        the ordinary paged cache of every active request (built once per
        step, shared by all layers)."""
        cached = getattr(self, "_batched_main_cache", None)
        if cached is not None:
            return cached
        n = len(self.active_request_ids)
        lens = seq_lens[:n].tolist()
        device = block_table.device
        s_max = max(lens)
        positions = torch.arange(s_max, device=device)
        lens_t = torch.as_tensor(lens, device=device)
        hidden = positions.unsqueeze(0) >= lens_t.unsqueeze(1)          # [R, S]
        block_idx = torch.clamp(positions // self.block_size, max=block_table.shape[1] - 1)
        blocks = block_table[:n, :].to(torch.long)[:, block_idx]           # [R, S]
        blocks = blocks.masked_fill(hidden, 0)
        offsets = (positions % self.block_size).unsqueeze(0).expand(n, -1)
        offsets = offsets.masked_fill(hidden, 0)
        cached = (blocks, offsets, hidden, lens)
        self._batched_main_cache = cached
        return cached

    def _batched_side_index(
        self, layer_index: int, page_counts: tuple[int, ...], device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Index ``[R, H_kv, S_side]`` into the per-layer flat page stack of
        all active requests (pages concatenated request by request), with a
        hidden mask; cached until the page counts change."""
        key = (layer_index, page_counts, str(device))
        cache = getattr(self, "_batched_side_cache", None)
        if cache is None:
            cache = self._batched_side_cache = {}
        cached = cache.get(key)
        if cached is not None:
            return cached
        quotas = torch.as_tensor(self.layout.quotas[layer_index], device=device)
        offsets = torch.as_tensor(
            self.layout.offsets[layer_index][:-1], device=device
        )
        slots = self.layout.packed_slots_per_layer[layer_index]
        counts = torch.as_tensor(page_counts, device=device)
        base = torch.cumsum(counts, 0) - counts                            # first page row per request
        max_pages = int(max(page_counts))
        width = max_pages * int(quotas.max().item())
        j = torch.arange(width, device=device)
        page = j.unsqueeze(0) // quotas.unsqueeze(1)                       # [H, W]
        slot = j.unsqueeze(0) % quotas.unsqueeze(1)
        hidden = page.unsqueeze(0) >= counts.view(-1, 1, 1)                # [R, H, W]
        index = (
            (base.view(-1, 1, 1) + page.unsqueeze(0)) * slots
            + offsets.view(1, -1, 1)
            + slot.unsqueeze(0)
        )
        index = index.masked_fill(hidden, 0)
        if len(cache) > 512:
            cache.clear()
        cached = (index, hidden, width)
        cache[key] = cached
        return cached

    def _attend_decode_batched(
        self,
        *,
        layer_index: int,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        output: torch.Tensor,
        seq_lens: torch.Tensor,
        block_table: torch.Tensor,
        num_kv_heads: int,
        groups: int,
        scale: float,
    ) -> None:
        n = len(self.active_request_ids)
        device = query.device
        blocks, offsets, main_hidden, _ = self._batched_main_index(seq_lens, block_table)
        main = _gather_cache_span_indexed(
            kv_cache, blocks, offsets, block_size=self.block_size
        )                                                                  # [2, R, S, H_kv, D]
        head_dim = main.shape[-1]
        main_key = main[0].permute(0, 2, 1, 3)                             # [R, H_kv, S, D]
        main_value = main[1].permute(0, 2, 1, 3)

        page_lists = [
            (self.pages.get(rid) or [[]] * self.layout.num_layers)[layer_index]
            for rid in self.active_request_ids
        ]
        page_counts = tuple(len(pages) for pages in page_lists)
        if any(page_counts):
            index, side_hidden, width = self._batched_side_index(
                layer_index, page_counts, device
            )
            stack = torch.cat([p for pages in page_lists for p in pages], dim=1)  # [2, total_rows, D]
            side_key = stack[0][index]                                     # [R, H_kv, W, D]
            side_value = stack[1][index]
            key = torch.cat([side_key, main_key], dim=2)
            value = torch.cat([side_value, main_value], dim=2)
            hidden = torch.cat(
                [side_hidden, main_hidden.unsqueeze(1).expand(-1, num_kv_heads, -1)],
                dim=2,
            )
        else:
            key, value = main_key, main_value
            hidden = main_hidden.unsqueeze(1).expand(-1, num_kv_heads, -1)

        q = query[:n].reshape(n, num_kv_heads, groups, head_dim)
        scores = torch.einsum("rhgd,rhtd->rhgt", q, key) * scale
        scores.masked_fill_(hidden.unsqueeze(2), -torch.inf)
        weights = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
        attended = torch.einsum("rhgt,rhtd->rhgd", weights, value)
        output[:n].copy_(
            attended.reshape(n, num_kv_heads * groups, head_dim).to(output.dtype)
        )

    def _side_gather_index(
        self, layer_index: int, num_pages: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Index/mask that expand ``num_pages`` packed pages of one layer into
        a head-major padded tensor ``[H_kv, num_pages * max_quota]``.

        Entry ``[h, j]`` addresses row ``off_h + (j mod q_h)`` of page
        ``j // q_h`` in the flattened ``[num_pages * packed_slots]`` page stack,
        matching the reference concatenation order. Rows ``j >= num_pages*q_h``
        are padding (masked)."""
        key = (layer_index, num_pages, str(device))
        cached = self._side_index_cache.get(key)
        if cached is not None:
            return cached
        quotas = self.layout.quotas[layer_index]
        offsets = self.layout.offsets[layer_index]
        packed_slots = self.layout.packed_slots_per_layer[layer_index]
        width = num_pages * max(quotas)
        index = torch.zeros((len(quotas), width), dtype=torch.long)
        hidden = torch.ones((len(quotas), width), dtype=torch.bool)
        for head, quota in enumerate(quotas):
            j = torch.arange(num_pages * quota)
            index[head, : num_pages * quota] = (
                (j // quota) * packed_slots + offsets[head] + (j % quota)
            )
            hidden[head, : num_pages * quota] = False
        cached = (index.to(device), hidden.to(device))
        self._side_index_cache[key] = cached
        return cached

    def _attend_request_vectorized(
        self,
        *,
        layer_index: int,
        layer_pages: Sequence[torch.Tensor],
        main: torch.Tensor,
        request_query: torch.Tensor,
        request_output: torch.Tensor,
        seq_len: int,
        query_len: int,
        prefix_len: int,
        causal: bool,
        scale: float,
        num_kv_heads: int,
        groups: int,
        request_id: str | None = None,
    ) -> None:
        """Same arithmetic as the per-head reference loop (bf16 scores,
        fp32 softmax, bf16 weights), but one batched op per stage for all
        KV heads of the request. Padded sidecar slots are masked to -inf."""
        device = request_query.device
        head_dim = main.shape[-1]
        main_key = main[0].permute(1, 0, 2)    # [H_kv, seq_len, D]
        main_value = main[1].permute(1, 0, 2)
        num_pages = len(layer_pages)
        if num_pages:
            index, side_hidden = self._side_gather_index(
                layer_index, num_pages, device
            )
            stacked = torch.stack(tuple(layer_pages))  # [P, 2, slots, D]
            flat = stacked.reshape(num_pages, 2, -1, head_dim)
            flat_key = flat[:, 0].reshape(-1, head_dim)
            flat_value = flat[:, 1].reshape(-1, head_dim)
            side_key = flat_key[index]        # [H_kv, S, D]
            side_value = flat_value[index]
            key = torch.cat([side_key, main_key], dim=1)
            value = torch.cat([side_value, main_value], dim=1)
            side_len = index.shape[1]
        else:
            key, value, side_len, side_hidden = main_key, main_value, 0, None
        total = side_len + seq_len

        q = request_query.reshape(query_len, num_kv_heads, groups, head_dim)
        scores = torch.einsum("qhgd,htd->qhgt", q, key) * scale

        hidden = None
        if side_hidden is not None and bool(side_hidden.any()):
            hidden = torch.zeros(
                (query_len, num_kv_heads, total), dtype=torch.bool, device=device
            )
            hidden[:, :, :side_len] = side_hidden.unsqueeze(0)
        if causal and query_len > 1:
            main_positions = torch.arange(seq_len, device=device)
            visible_end = prefix_len + torch.arange(1, query_len + 1, device=device)
            causal_hidden = main_positions.unsqueeze(0) >= visible_end.unsqueeze(1)
            if hidden is None:
                hidden = torch.zeros(
                    (query_len, num_kv_heads, total), dtype=torch.bool, device=device
                )
            hidden[:, :, side_len:] |= causal_hidden.unsqueeze(1)
        elif not causal:
            if query_len != self.layout.k_max:
                raise RuntimeError(
                    "packed KV noncausal step must contain exactly "
                    f"k_max={self.layout.k_max} code queries; got {query_len}"
                )
            quotas = torch.as_tensor(
                self.layout.quotas[layer_index], device=device
            )
            main_positions = torch.arange(seq_len, device=device)
            quota_hidden = main_positions.unsqueeze(0) >= (
                prefix_len + quotas
            ).unsqueeze(1)                                  # [H_kv, seq_len]
            if hidden is None:
                hidden = torch.zeros(
                    (query_len, num_kv_heads, total), dtype=torch.bool, device=device
                )
            hidden[:, :, side_len:] |= quota_hidden.unsqueeze(0)
            excluded = (
                self._code_excluded_range(request_id, seq_len, query_len)
                if request_id is not None else None
            )
            if excluded is not None:                        # own-block static tail: hidden from the codes
                hidden[:, :, side_len + excluded[0] : side_len + excluded[1]] = True
        if hidden is not None:
            scores.masked_fill_(hidden.unsqueeze(2), -torch.inf)

        weights = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
        attended = torch.einsum("qhgt,htd->qhgd", weights, value)
        request_output.copy_(
            attended.reshape(query_len, num_kv_heads * groups, head_dim).to(
                request_output.dtype
            )
        )


def maybe_override_attention(
    layer: object,
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    attn_metadata: object,
    output: torch.Tensor,
) -> None:
    sidecar = getattr(layer, "_compresslm_packed_sidecar", None)
    if sidecar is None or attn_metadata is None:
        return
    sidecar.override_attention(
        layer_index=layer._compresslm_packed_layer_index,
        layer=layer,
        query=query,
        kv_cache=kv_cache,
        attn_metadata=attn_metadata,
        output=output,
    )
