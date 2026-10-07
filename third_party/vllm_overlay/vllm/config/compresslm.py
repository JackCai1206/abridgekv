# SPDX-License-Identifier: Apache-2.0
"""Configuration for faithful CompressLM pause-code decoding."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class CompressLMConfig:
    """Fixed-block pause-code compression used by ``CompressLM`` checkpoints.

    The current implementation intentionally supports the production SFT
    configuration only: one bidirectional pause pass and either an optional
    legacy post-summary replay tail or an original-KV retained tail.
    """

    enable: bool = False
    block_size: int = 512
    num_code_tokens: int = 128
    replay_tail: int = 16
    retain_tail_kv: int = 0
    # rolling / sticky: window tails of the previous block (content queries only, expire). static: the final
    # retain_tail_kv content KVs of EVERY completed block stay resident forever, visible to all later queries
    # (content and code rows), and the codes summarize only the first block_size - retain_tail_kv tokens.
    retain_tail_policy: Literal["rolling", "sticky", "static"] = "rolling"
    code_exclude_replay_tail: bool = False
    raw_prompt: bool = True
    num_passes: int = 1
    pause_xattn: str = "full"
    raw_tail: int = 0
    code_token_format: str = "<|compress_code_{:03d}|>"
    debug: bool = False
    retain_full_history_layers: tuple[int, ...] = ()
    history_replaces_codes_layers: tuple[int, ...] = ()
    packed_kv: bool = False
    packed_kv_k_max: int = 2048
    packed_kv_quotas: tuple[tuple[int, ...], ...] = ()
    # RoPE layout of the code rows: "appended" = logical index (block period C+K+T); "bitrev" = K-independent
    # in-block layout (content b*C+j, code row k at b*C + bitreverse(k)*C/K), matching CompressLM(code_positions=).
    code_positions: str = "appended"
    # Per-layer code quotas for serving a per-layer trained schedule on the stock paged cache: layer l attends only
    # to code slots [0, layer_code_quotas[l]) of every block (the nested-prefix quota of training). The cache keeps
    # num_code_tokens rows per block; the extra slots are hidden by per-layer custom masks (attention groups are
    # split per distinct quota). Requires code_positions="bitrev" (prefix layouts are K-independent), no packed KV.
    layer_code_quotas: tuple[int, ...] = ()
    # Physical per-layer quotas: layers are grouped into KV cache groups by quota (own block tables), and a group
    # with quota q < num_code_tokens physically removes code slots [q, K) of every completed block, so memory
    # follows the schedule (mean quota) instead of K_max, and decode steps need no custom masks (only the code
    # step masks the block's own slots >= q and the static tail). Requires layer_code_quotas.
    physical_layer_quotas: bool = False

    _code_token_ids: list[int] = field(default_factory=list, repr=False)
    _initialized: bool = field(default=False, repr=False)

    def __post_init__(self) -> None:
        self.retain_full_history_layers = tuple(
            sorted(set(self.retain_full_history_layers))
        )
        self.history_replaces_codes_layers = tuple(
            sorted(set(self.history_replaces_codes_layers))
        )
        self.packed_kv_quotas = tuple(
            tuple(int(value) for value in row)
            for row in self.packed_kv_quotas
        )
        self.layer_code_quotas = tuple(int(q) for q in self.layer_code_quotas)
        if self.retain_tail_policy not in ("rolling", "sticky", "static"):
            raise ValueError(
                "retain_tail_policy must be one of: rolling, sticky, static"
            )
        if self.code_positions not in ("appended", "bitrev"):
            raise ValueError("code_positions must be appended or bitrev")
        if not self.enable:
            return
        if self.code_positions == "bitrev":
            C, K = self.block_size, self.num_code_tokens
            if (C & (C - 1)) or (K & (K - 1)) or self.replay_tail or self.raw_tail or self.num_passes != 1:
                raise ValueError("code_positions=bitrev needs power-of-two block_size and num_code_tokens, "
                                 "no replay/raw tail and num_passes=1")
        if self.block_size <= 0:
            raise ValueError("block_size must be positive")
        if self.num_code_tokens <= 0:
            raise ValueError("num_code_tokens must be positive")
        if not 0 <= self.replay_tail <= self.block_size:
            raise ValueError("replay_tail must be in [0, block_size]")
        if not 0 <= self.retain_tail_kv < self.block_size:
            raise ValueError("retain_tail_kv must be in [0, block_size)")
        if self.retain_tail_policy == "sticky" and not self.retain_tail_kv:
            raise ValueError(
                "retain_tail_policy='sticky' requires retain_tail_kv > 0"
            )
        if sum(
            bool(value)
            for value in (self.raw_tail, self.replay_tail, self.retain_tail_kv)
        ) > 1:
            raise ValueError(
                "raw_tail, replay_tail, and retain_tail_kv are mutually exclusive"
            )
        if self.code_exclude_replay_tail and not (
            0 < self.replay_tail < self.block_size
        ):
            raise ValueError(
                "code_exclude_replay_tail requires replay_tail in "
                "[1, block_size)"
            )
        if self.num_passes != 1:
            raise ValueError("vLLM CompressLM decoding currently supports L=1 only")
        if self.pause_xattn != "full":
            raise ValueError(
                "vLLM CompressLM decoding currently requires pause_xattn='full'"
            )
        if self.raw_tail:
            raise ValueError(
                "vLLM CompressLM decoding currently supports raw_tail=0 only"
            )
        if not self.raw_prompt:
            raise ValueError(
                "vLLM CompressLM decoding currently requires raw_prompt=True"
            )
        if any(layer < 0 for layer in self.retain_full_history_layers):
            raise ValueError("retained-history layer indices must be nonnegative")
        replacement_without_history = (
            set(self.history_replaces_codes_layers)
            - set(self.retain_full_history_layers)
        )
        if replacement_without_history:
            raise ValueError(
                "history_replaces_codes_layers requires retained history at "
                f"the same layers: {sorted(replacement_without_history)}"
            )
        if self.layer_code_quotas:
            for layer, quota in enumerate(self.layer_code_quotas):
                if not 0 < quota <= self.num_code_tokens:
                    raise ValueError(
                        f"layer_code_quotas[{layer}]={quota} must be in "
                        f"[1, {self.num_code_tokens}]"
                    )
            if self.uses_layer_quotas:
                if self.code_positions != "bitrev":
                    raise ValueError(
                        "layer_code_quotas below num_code_tokens need "
                        "code_positions=bitrev (nested-prefix code layout)"
                    )
                if self.packed_kv:
                    raise ValueError(
                        "layer_code_quotas and packed KV are mutually exclusive"
                    )
                if self.uses_selective_history:
                    raise ValueError(
                        "layer_code_quotas and selective full-history "
                        "evaluation are mutually exclusive"
                    )
        if self.physical_layer_quotas and not self.layer_code_quotas:
            raise ValueError("physical_layer_quotas needs layer_code_quotas")
        if self.packed_kv:
            if self.uses_selective_history:
                raise ValueError(
                    "packed KV and selective full-history evaluation are "
                    "mutually exclusive"
                )
            if self.raw_tail or (
                self.retain_tail_kv and self.retain_tail_policy != "static"
            ):
                raise ValueError(
                    "packed KV supports code/replay checkpoints and the static "
                    "retained tail, not window (rolling/sticky) or raw tails"
                )
            if self.code_exclude_replay_tail:
                raise ValueError(
                    "packed KV does not yet support code_exclude_replay_tail"
                )
            if self.packed_kv_k_max != self.num_code_tokens:
                raise ValueError(
                    "packed_kv_k_max must equal num_code_tokens so the dense "
                    "code pass computes every schedulable slot"
                )
            if self.packed_kv_k_max <= 0:
                raise ValueError("packed_kv_k_max must be positive")
            for layer, row in enumerate(self.packed_kv_quotas):
                if not row:
                    raise ValueError(
                        f"packed_kv_quotas[{layer}] must not be empty"
                    )
                for head, quota in enumerate(row):
                    if not 0 < quota <= self.packed_kv_k_max:
                        raise ValueError(
                            f"packed_kv_quotas[{layer}][{head}]={quota} must "
                            f"be in [1, {self.packed_kv_k_max}]"
                        )

    def bitrev_code_offsets(self):
        """In-block RoPE offsets of the K code rows (numpy int64): row k -> floor(bitreverse_{log2 K}(k) * span / K) with
        span = block_size - code_excluded_tail (the codes summarize only the non-retained prefix of the block; matches
        CompressLM._bitrev_code_offsets(span=...))."""
        import numpy as np
        K, C = self.num_code_tokens, self.block_size
        span = C - self.code_excluded_tail
        if span < K:
            raise ValueError(f"bitrev code span {span} < num_code_tokens {K}")
        bits = K.bit_length() - 1
        k = np.arange(K, dtype=np.int64)
        rev = np.zeros_like(k)
        for b in range(bits):
            rev |= ((k >> b) & 1) << (bits - 1 - b)
        return (rev * span) // K

    def bitrev_rope_positions(self, logical):
        """Map logical sequence positions (period C+K per block) to bitrev RoPE positions (numpy int64)."""
        import numpy as np
        C, K = self.block_size, self.num_code_tokens
        logical = np.asarray(logical, dtype=np.int64)
        blk, within = logical // (C + K), logical % (C + K)
        code = within >= C
        offs = self.bitrev_code_offsets()[np.clip(within - C, 0, K - 1)]
        return blk * C + np.where(code, offs, within)

    def initialize_token_ids(self, tokenizer) -> "CompressLMConfig":
        if self._initialized:
            return self
        if tokenizer is None:
            raise ValueError("CompressLM decoding requires an initialized tokenizer")

        token_ids: list[int] = []
        for index in range(self.num_code_tokens):
            token = self.code_token_format.format(index)
            ids = tokenizer.encode(token, add_special_tokens=False)
            if len(ids) != 1:
                raise ValueError(
                    f"CompressLM code token {token!r} must encode to one ID; got {ids}"
                )
            token_ids.append(ids[0])
        if len(set(token_ids)) != len(token_ids):
            raise ValueError("CompressLM code tokens must have distinct token IDs")

        self._code_token_ids = token_ids
        self._initialized = True
        return self

    @property
    def code_token_ids(self) -> list[int]:
        if not self._initialized:
            raise RuntimeError("Call initialize_token_ids() before decoding")
        return self._code_token_ids

    @property
    def code_tokens(self) -> list[str]:
        return [
            self.code_token_format.format(index)
            for index in range(self.num_code_tokens)
        ]

    @property
    def code_excluded_tail(self) -> int:
        if self.code_exclude_replay_tail:
            return self.replay_tail
        return self.retain_tail_kv

    @property
    def uses_selective_history(self) -> bool:
        return bool(self.retain_full_history_layers)

    @property
    def uses_physical_layer_quotas(self) -> bool:
        return bool(self.physical_layer_quotas) and self.uses_layer_quotas

    def kv_group_quotas(self, group_layer_names: "Sequence[Sequence[str]]") -> tuple[int, ...]:
        """Code quota of every KV cache group (each group must hold layers of ONE quota)."""
        quotas = []
        for names in group_layer_names:
            group = set()
            for name in names:
                index = layer_index(name)
                if index is None:
                    raise ValueError(f"physical per-layer quotas need indexed attention layers; got {name!r}")
                group.add(self.layer_quota(index))
            if len(group) != 1:
                raise ValueError(f"KV cache group mixes code quotas {sorted(group)}: {list(names)}")
            quotas.append(group.pop())
        return tuple(quotas)

    @property
    def uses_packed_kv(self) -> bool:
        return self.packed_kv

    @property
    def uses_layer_quotas(self) -> bool:
        """True when some layer sees fewer code slots than the cache holds."""
        return any(q < self.num_code_tokens for q in self.layer_code_quotas)

    def layer_quota(self, layer: int) -> int:
        """Code slots visible at `layer` (num_code_tokens for layers past the schedule)."""
        if 0 <= layer < len(self.layer_code_quotas):
            return self.layer_code_quotas[layer]
        return self.num_code_tokens

    @property
    def distinct_layer_quotas(self) -> tuple[int, ...]:
        return tuple(sorted(set(self.layer_code_quotas)))


def layer_index(layer_name: str) -> int | None:
    """Index n of an attention layer named '...layers.n....' (None when not indexed)."""
    marker = ".layers."
    if marker not in layer_name:
        return None
    suffix = layer_name.split(marker, 1)[1]
    value = suffix.split(".", 1)[0]
    return int(value) if value.isdigit() else None
