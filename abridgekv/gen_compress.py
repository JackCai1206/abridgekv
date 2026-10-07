"""Autoregressive GENERATION for CompressLM (abridgekv/compress_forward.py) via a manually-managed
KV cache with per-block compression + content EVICTION — the incremental image of the teacher-forced
`CompressLM.forward` (L=1, pause_xattn="full").

Cache mechanics reuse the KVStore/manual-cached-decode skeleton preserved in
archive/legacy/root_stack_2026-08-22/scripts/eval_looped_hf.py (RoPE applied per
token at its ABSOLUTE layout position so eviction is a pure column delete), but
the compression op is CompressLM-specific: each block of C content tokens is
summarised into K learned "pause code" KVs; later blocks attend prior blocks' K
codes instead of raw content.

Attention structure reproduced (verified against CompressLM._mask.mm, L=1):
  layout: block b -> content, codes, optional post-summary replay tail; RoPE = absolute layout pos.
  * a CONTENT token attends its own block's content (causal) + ALL prior blocks' K codes.
  * a CODE attends its own block's content + its own block's codes (pause_xattn="full" => bidirectional)
    + prior blocks' codes. Codes are computed by feeding pause_base (self.pause [+ seg_pool/pool_init])
    at the code positions through the SAME layers.

Entry points:
  python -m abridgekv.gen_compress --gate            # Deliverable-2 CE-consistency GATE (tiny CLM)
  python -m abridgekv.gen_compress --math500 --ckpt … [--baseline]   # Deliverable-3 MATH-500 eval
"""
import argparse, json, math, os, sys
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repo root
from abridgekv.compress_forward import CompressLM  # noqa: E402
from abridgekv.compress_forward import _code_content_tail_excluded  # noqa: E402
from abridgekv.compress_forward import _bitrev_code_offsets  # noqa: E402


# --- arch-agnostic attention primitives (mirror the archived looped evaluator) ----
def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q, k, cos, sin, unsqueeze_dim=1):
    cos, sin = cos.unsqueeze(unsqueeze_dim), sin.unsqueeze(unsqueeze_dim)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


def repeat_kv(h, n_rep):
    b, kv, s, d = h.shape
    if n_rep == 1:
        return h
    return h[:, :, None, :, :].expand(b, kv, n_rep, s, d).reshape(b, kv * n_rep, s, d)


def _qkn(attn, which, x):                # Qwen3 q_norm/k_norm if present, else identity (LLaMA/SmolLM2)
    n = getattr(attn, which, None)
    return n(x) if n is not None else x


# ----------------------------------------------------------------------------
# Per-layer K/V store with explicit EVICTION (identical semantics to eval_looped_hf.KVStore).
# ----------------------------------------------------------------------------
class KVStore:
    def __init__(self, num_layers):
        self.K = [None] * num_layers
        self.V = [None] * num_layers
        self.length = 0

    def append(self, li, k_new, v_new):
        if self.K[li] is None:
            self.K[li], self.V[li] = k_new, v_new
        else:
            self.K[li] = torch.cat([self.K[li], k_new], dim=2)
            self.V[li] = torch.cat([self.V[li], v_new], dim=2)

    def evict(self, keep_cols):
        for li in range(len(self.K)):
            if self.K[li] is not None:
                self.K[li] = self.K[li][:, :, keep_cols, :].contiguous()
                self.V[li] = self.V[li][:, :, keep_cols, :].contiguous()
        self.length = int(keep_cols.numel())


# ----------------------------------------------------------------------------
# One instance per sequence. Holds the KVStore + block bookkeeping; the cache always equals
# [prior blocks' K codes] ++ [raw prompt tokens] ++ [rolling previous-block
# raw/replay tail] ++ [current block's raw content so far].
# ----------------------------------------------------------------------------
class CompressGen:
    def __init__(
        self,
        clm: CompressLM,
        device,
        raw_prompt_len=0,
        capture_hidden=False,
        capture_attention=False,
        code_attend_current_content=True,
        code_init_override=None,
        raw_tail_override=None,
        oracle_full_history=False,
        oracle_history_layers=None,
        preserve_full_history=False,
        retain_full_history_layers=None,
        history_replaces_codes_layers=None,
        capture_content_attention=False,
        content_attention_offsets=None,
        content_head_callback=None,
        head_code_quotas=None,
        evict_policy=None,
        keep_per_block=0,
        evict_seed=0,
        skip_unused_codes=False,
    ):
        assert clm.L == 1, "gen_compress supports L=1 (real ckpt + gate are L=1)"
        self.clm = clm
        self.inner = clm.model.model          # LlamaModel (embed_tokens, layers, rotary_emb, norm)
        self.layers = self.inner.layers
        self.num_layers = len(self.layers)
        self.lm_head = clm.model.lm_head
        self.device = device
        self.dtype = next(clm.model.parameters()).dtype
        self.C, self.K = clm.C, clm.K
        self.raw_tail = (
            clm.raw_tail if raw_tail_override is None else int(raw_tail_override)
        )
        self.replay_tail = clm.replay_tail
        self.retain_tail_kv = clm.retain_tail_kv
        self.retain_tail_policy = clm.retain_tail_policy
        if raw_tail_override is not None and self.retain_tail_kv:
            raise ValueError(
                "raw_tail_override cannot be combined with retain_tail_kv"
            )
        self.content_tail_size = self.raw_tail or self.retain_tail_kv
        self.rolling_content_tail = (
            self.raw_tail
            or (
                self.retain_tail_kv
                if self.retain_tail_policy == "rolling"
                else 0
            )
        )
        self.code_exclude_replay_tail = clm.code_exclude_replay_tail
        self.code_excluded_tail = clm.code_excluded_tail
        self.raw_prompt_len = int(raw_prompt_len)
        self.d = clm.model.config.hidden_size
        self.xa = clm.pause_xattn
        self.store = KVStore(self.num_layers)
        self.b = 0                            # current block index
        self.f = 0                            # content tokens committed in current block (0..C)
        self._cur_ids = []                    # current block's content token ids (for pause_base + evict)
        self._col_role = torch.empty(0, dtype=torch.int8, device=device)   # 0=content, 1=code, 2=replay
        self._col_blk = torch.empty(0, dtype=torch.long, device=device)
        self._col_content_abs = torch.empty(0, dtype=torch.long, device=device)
        self._col_code_slot = torch.empty(
            0, dtype=torch.long, device=device
        )
        # Codes-free EVICTION BASELINES (length-stress comparison): no code rows are written; at every block boundary
        # the completed block's generated content is reduced to `keep_per_block` raw columns chosen by `evict_policy`
        # (recency = last tokens, random = fixed per-block permutation, score = most accumulated attention received
        # from this block's queries, summed over layers/heads, H2O-style). Prompt stays raw; positions are plain.
        self.evict_policy = evict_policy
        self.keep_per_block = int(keep_per_block)
        self.evict_seed = int(evict_seed)
        self.no_codes = evict_policy is not None
        if self.no_codes:
            if evict_policy not in ("recency", "random", "score"):
                raise ValueError(f"unsupported evict_policy={evict_policy!r} (recency|random|score)")
            if not 0 <= self.keep_per_block <= self.C:
                raise ValueError(f"keep_per_block must be in [0, {self.C}]")
            if self.raw_tail or self.replay_tail or self.retain_tail_kv:
                raise ValueError("evict_policy baselines are codes-free; raw/replay/retained tails must be 0")
        self._col_score = torch.empty(0, dtype=torch.float32, device=device)
        self.head_code_quotas = None
        if head_code_quotas is not None:
            quotas = torch.as_tensor(
                head_code_quotas, dtype=torch.long, device=device
            )
            num_kv_heads = clm.model.config.num_key_value_heads
            if tuple(quotas.shape) != (self.num_layers, num_kv_heads):
                raise ValueError(
                    "head_code_quotas must have shape "
                    f"[{self.num_layers}, {num_kv_heads}], got "
                    f"{tuple(quotas.shape)}"
                )
            if bool(((quotas <= 0) | (quotas > self.K)).any()):
                raise ValueError(
                    f"head_code_quotas must be in [1, {self.K}]"
                )
            self.head_code_quotas = quotas
        # Skip-when-unused (CompressLM.set_skip_unused_codes): on the code pass, code slot j passes through layer l unchanged
        # when j >= max_h quota[l][h]. Those rows are masked as keys at l for every query, so computing the layer and
        # restoring the input rows is exactly the trainer's torch.where identity.
        self.skip_unused_codes = bool(skip_unused_codes)
        if self.skip_unused_codes and self.head_code_quotas is None:
            raise ValueError("skip_unused_codes needs head_code_quotas")
        self.capture_hidden = bool(capture_hidden)
        self.hidden_trace = []
        self.capture_attention = bool(capture_attention)
        self.code_attend_current_content = bool(code_attend_current_content)
        if code_init_override not in (None, "segment_mean"):
            raise ValueError(
                f"unsupported code_init_override={code_init_override!r}"
            )
        self.code_init_override = code_init_override
        self.attention_trace = []
        self._attention_phase = None
        self.oracle_full_history = bool(oracle_full_history)
        self.oracle_history_layers = set(
            oracle_history_layers
            if oracle_history_layers is not None
            else ()
        )
        invalid_oracle_layers = self.oracle_history_layers - set(
            range(self.num_layers)
        )
        if invalid_oracle_layers:
            raise ValueError(
                "oracle_history_layers contains invalid layers: "
                f"{sorted(invalid_oracle_layers)}"
            )
        self.retain_full_history_layers = set(
            retain_full_history_layers
            if retain_full_history_layers is not None
            else ()
        )
        invalid_retained_layers = self.retain_full_history_layers - set(
            range(self.num_layers)
        )
        if invalid_retained_layers:
            raise ValueError(
                "retain_full_history_layers contains invalid layers: "
                f"{sorted(invalid_retained_layers)}"
            )
        self.history_replaces_codes_layers = set(
            history_replaces_codes_layers
            if history_replaces_codes_layers is not None
            else ()
        )
        history_visible_layers = (
            set(range(self.num_layers))
            if self.oracle_full_history
            else (
                self.oracle_history_layers
                | self.retain_full_history_layers
            )
        )
        invalid_replacement_layers = (
            self.history_replaces_codes_layers - history_visible_layers
        )
        if invalid_replacement_layers:
            raise ValueError(
                "history_replaces_codes_layers requires full history at the "
                f"same layers: {sorted(invalid_replacement_layers)}"
            )
        self.preserve_full_history = (
            self.oracle_full_history
            or bool(self.oracle_history_layers)
            or bool(preserve_full_history)
        )
        self._history_K = [None] * self.num_layers
        self._history_V = [None] * self.num_layers
        self._history_blk = torch.empty(
            0, dtype=torch.long, device=device
        )
        self._history_content_abs = torch.empty(
            0, dtype=torch.long, device=device
        )
        self.capture_content_attention = bool(capture_content_attention)
        self.content_head_callback = content_head_callback
        self.content_attention_offsets = set(
            content_attention_offsets
            if content_attention_offsets is not None
            else (1, 2, 4, 8, 16, 32, 64, 127, 128, 255, 511)
        )
        self.content_attention_trace = []
        self._attention_query_content_abs = None
        self._content_query_active = False

    # -- one decoder layer over a NEW chunk (1,n,d); visibility is split between retained-store columns
    # and the new chunk's own columns.
    def _layer(self, li, hidden, abs_pos, cos, sin, store_add, chunk_add, commit):
        layer = self.layers[li]
        attn = layer.self_attn
        normed = layer.input_layernorm(hidden)
        hs = (1, hidden.shape[1], -1, attn.head_dim)
        q = _qkn(attn, "q_norm", attn.q_proj(normed).view(hs)).transpose(1, 2)
        k = _qkn(attn, "k_norm", attn.k_proj(normed).view(hs)).transpose(1, 2)
        v = attn.v_proj(normed).view(hs).transpose(1, 2)
        q, k = apply_rope(q, k, cos, sin)
        ks, vs = self.store.K[li], self.store.V[li]
        history_visible = (
            self._content_query_active
            and li in self.retain_full_history_layers
            and self._history_K[li] is not None
        )
        history_k = self._history_K[li] if history_visible else None
        history_v = self._history_V[li] if history_visible else None
        T_history = history_k.shape[2] if history_k is not None else 0
        T_main = ks.shape[2] if ks is not None else 0
        k_parts = [
            part for part in (history_k, ks, k) if part is not None
        ]
        v_parts = [
            part for part in (history_v, vs, v) if part is not None
        ]
        k_full = torch.cat(k_parts, 2) if len(k_parts) > 1 else k_parts[0]
        v_full = torch.cat(v_parts, 2) if len(v_parts) > 1 else v_parts[0]
        T_store = T_history + T_main
        n, T = hidden.shape[1], k_full.shape[2]
        m = torch.zeros((1, 1, n, T), dtype=self.dtype, device=self.device)
        if store_add is not None and T_main:
            effective_store_add = store_add
            expose_oracle_history = (
                self._content_query_active
                and li in self.oracle_history_layers
            )
            replace_codes = (
                self._content_query_active
                and li in self.history_replaces_codes_layers
            )
            if expose_oracle_history or replace_codes:
                effective_store_add = store_add.clone()
            if expose_oracle_history:
                effective_store_add[..., self._col_role == 0] = 0
            if replace_codes:
                effective_store_add[..., self._col_role == 1] = (
                    torch.finfo(self.dtype).min
                )
            m[..., T_history:T_store] = effective_store_add
        if chunk_add is not None:
            m[..., T_store:] = chunk_add
        if self.head_code_quotas is not None:
            quota = self.head_code_quotas[li].repeat_interleave(
                attn.num_key_value_groups
            )
            n_heads = getattr(attn, "num_heads", None) or attn.config.num_attention_heads  # transformers 4.5x lacks num_heads
            m = m.expand(1, n_heads, n, T).clone()
            if T_main:
                invalid_store_codes = (
                    (self._col_role == 1).unsqueeze(0)
                    & (
                        self._col_code_slot.unsqueeze(0)
                        >= quota.unsqueeze(1)
                    )
                )
                m[..., T_history:T_store].masked_fill_(
                    invalid_store_codes.unsqueeze(0).unsqueeze(2),
                    torch.finfo(self.dtype).min,
                )
            if self._attention_phase == "code":
                if n != self.K:
                    raise RuntimeError(
                        "head-dependent code quota requires one full code chunk"
                    )
                invalid_chunk_codes = (
                    torch.arange(n, device=self.device).unsqueeze(0)
                    >= quota.unsqueeze(1)
                )
                m[..., T_store:].masked_fill_(
                    invalid_chunk_codes.unsqueeze(0).unsqueeze(2),
                    torch.finfo(self.dtype).min,
                )
        w = torch.matmul(q, repeat_kv(k_full, attn.num_key_value_groups).transpose(2, 3)) * attn.scaling + m
        w = F.softmax(w, dim=-1, dtype=torch.float32).to(q.dtype)
        if self.no_codes and self.evict_policy == "score" and self._content_query_active and T_main:
            self._col_score += w[0, :, :, T_history:T_store].float().sum((0, 1))   # attention mass received per column
        if self.capture_attention and self._attention_phase == "code":
            store_w = w[..., :T_store].float()
            chunk_w = w[..., T_store:].float()
            roles = self._col_role
            blocks = self._col_blk
            content_abs = self._col_content_abs
            current_content = (roles == 0) & (blocks == self.b)
            raw_prompt = (
                (roles == 0)
                & (content_abs < self.raw_prompt_len)
                & ~current_content
            )
            prior_codes = roles == 1
            replay = roles == 2

            def mass(mask):
                if not bool(mask.any()):
                    return 0.0
                return float(store_w[..., mask].sum(-1).mean())

            position_mass = [0.0] * self.C
            if bool(current_content.any()):
                per_column = store_w[..., current_content].mean((0, 1, 2))
                offsets = (content_abs[current_content] % self.C).tolist()
                for offset, value in zip(offsets, per_column.tolist()):
                    position_mass[offset] += value
                slot_mass = (
                    store_w[..., current_content]
                    .sum(-1)
                    .mean((0, 1))
                    .tolist()
                )
            else:
                slot_mass = [0.0] * hidden.shape[1]
            entropy = -(w.float() * w.float().clamp_min(1e-30).log())
            self.attention_trace.append(
                {
                    "block": self.b,
                    "layer": li,
                    "current_content_mass": mass(current_content),
                    "raw_prompt_mass": mass(raw_prompt),
                    "prior_code_mass": mass(prior_codes),
                    "replay_mass": mass(replay),
                    "new_code_mass": float(chunk_w.sum(-1).mean()),
                    "entropy": float(entropy.sum(-1).mean()),
                    "current_content_position_mass": position_mass,
                    "current_content_mass_by_code_slot": slot_mass,
                }
            )
        elif (
            self._attention_phase == "content"
            and (
                self.capture_content_attention
                or self.content_head_callback is not None
            )
        ):
            store_w = w[..., :T_store].float()
            if T_history:
                roles = torch.cat([
                    torch.zeros(
                        T_history, dtype=torch.int8, device=self.device
                    ),
                    self._col_role,
                ])
                blocks = torch.cat([
                    self._history_blk,
                    self._col_blk,
                ])
                content_abs = torch.cat([
                    self._history_content_abs,
                    self._col_content_abs,
                ])
                v_store = torch.cat([history_v, vs], dim=2)
            else:
                roles = self._col_role
                blocks = self._col_blk
                content_abs = self._col_content_abs
                v_store = vs
            query_abs = int(self._attention_query_content_abs)
            is_content = roles == 0
            raw_prompt = is_content & (content_abs < self.raw_prompt_len)
            own_block = is_content & (blocks == self.b)
            previous_block = is_content & (blocks + 1 == self.b)
            if self.retain_tail_policy == "static":
                faithful_tail = (
                    is_content
                    & (blocks < self.b)
                    & (content_abs % self.C >= self.C - self.retain_tail_kv)
                )
            elif self.retain_tail_policy == "sticky":
                faithful_tail = (
                    previous_block
                    & (content_abs % self.C >= self.C - self.retain_tail_kv)
                )
            else:
                faithful_tail = (
                    previous_block
                    & (query_abs - content_abs <= self.rolling_content_tail)
                )
            faithful_content = raw_prompt | own_block | faithful_tail
            oracle_only_content = is_content & ~faithful_content
            previous_prefix = previous_block & ~faithful_tail & ~raw_prompt
            older_content = (
                is_content
                & (blocks + 1 < self.b)
                & ~raw_prompt
            )
            prior_codes = roles == 1
            replay = roles == 2
            if self.content_head_callback is not None:
                self.content_head_callback(
                    layer=li,
                    block=self.b,
                    offset=query_abs % self.C,
                    weights=w,
                    values=v_full,
                    store_columns=T_store,
                    roles=roles,
                    blocks=blocks,
                    num_key_value_groups=attn.num_key_value_groups,
                )
            if not self.capture_content_attention:
                repeated_v = None
            else:
                repeated_v = (
                    repeat_kv(v_store, attn.num_key_value_groups)
                    if v_store is not None
                    else None
                )
                total_output = torch.matmul(
                    w.float(),
                    repeat_kv(
                        v_full, attn.num_key_value_groups
                    ).float(),
                )

            def source_stats(mask):
                if not self.capture_content_attention:
                    raise RuntimeError(
                        "aggregate content-attention capture is disabled"
                    )
                if not bool(mask.any()):
                    return {
                        "attention_mass": 0.0,
                        "attention_output_relative_l2": 0.0,
                    }
                contribution = torch.matmul(
                    store_w[..., mask],
                    repeated_v[..., mask, :].float(),
                )
                relative = (
                    torch.linalg.vector_norm(contribution, dim=-1)
                    / torch.linalg.vector_norm(
                        total_output, dim=-1
                    ).clamp_min(1e-30)
                )
                return {
                    "attention_mass": float(
                        store_w[..., mask].sum(-1).mean()
                    ),
                    "attention_output_relative_l2": float(relative.mean()),
                }

            if self.capture_content_attention:
                self.content_attention_trace.append(
                    {
                        "block": self.b,
                        "offset": query_abs % self.C,
                        "layer": li,
                        "oracle_full_history": self.oracle_full_history,
                        "oracle_history_visible": (
                            self.oracle_full_history
                            or li in self.oracle_history_layers
                        ),
                        "retained_history_visible": history_visible,
                        "history_replaces_codes": (
                            li in self.history_replaces_codes_layers
                        ),
                        "oracle_only_content": source_stats(
                            oracle_only_content
                        ),
                        "previous_block_prefix": source_stats(
                            previous_prefix
                        ),
                        "older_content": source_stats(older_content),
                        "faithful_exact_tail": source_stats(faithful_tail),
                        "raw_prompt": source_stats(raw_prompt),
                        "current_block_content": source_stats(own_block),
                        "prior_codes": source_stats(prior_codes),
                        "replay": source_stats(replay),
                    }
                )
        o = torch.matmul(w, repeat_kv(v_full, attn.num_key_value_groups))
        o = o.transpose(1, 2).contiguous().view(1, n, -1)
        h = hidden + attn.o_proj(o)
        h = h + layer.mlp(layer.post_attention_layernorm(h))
        if commit:
            self.store.append(li, k, v)
        return h

    def _append_col_meta(self, role, blk, content_abs):
        self._col_role = torch.cat([
            self._col_role,
            torch.full((len(content_abs),), role, dtype=torch.int8, device=self.device),
        ])
        self._col_blk = torch.cat([
            self._col_blk,
            torch.full((len(content_abs),), blk, dtype=torch.long, device=self.device),
        ])
        self._col_content_abs = torch.cat([
            self._col_content_abs,
            torch.tensor(content_abs, dtype=torch.long, device=self.device),
        ])
        code_slots = (
            torch.arange(len(content_abs), device=self.device)
            if role == 1
            else torch.full(
                (len(content_abs),),
                -1,
                dtype=torch.long,
                device=self.device,
            )
        )
        self._col_code_slot = torch.cat([
            self._col_code_slot,
            code_slots,
        ])
        self._col_score = torch.cat([
            self._col_score,
            torch.zeros(len(content_abs), dtype=torch.float32, device=self.device),
        ])

    @property
    def static_tail(self):
        """T of the static (permanent, per-block) retained tail; 0 for the window policies / no tail."""
        return self.retain_tail_kv if self.retain_tail_policy == "static" else 0

    def _static_tail_cols(self, before_block):
        """Content columns that are the static retained tail of a block < before_block (final T rows of each block)."""
        T = self.static_tail
        if not T:
            return torch.zeros_like(self._col_role, dtype=torch.bool)
        return (
            (self._col_role == 0)
            & (self._col_blk < before_block)
            & (self._col_content_abs % self.C >= self.C - T)
        )

    def _evict(self, keep):
        evicted_history = (
            ~keep
            & (self._col_role == 0)
            & (self._col_content_abs >= self.raw_prompt_len)
        )
        history_cols = evicted_history.nonzero(as_tuple=True)[0]
        if history_cols.numel() and self.retain_full_history_layers:
            for li in self.retain_full_history_layers:
                k_new = self.store.K[li][:, :, history_cols, :].contiguous()
                v_new = self.store.V[li][:, :, history_cols, :].contiguous()
                if self._history_K[li] is None:
                    self._history_K[li] = k_new
                    self._history_V[li] = v_new
                else:
                    self._history_K[li] = torch.cat(
                        [self._history_K[li], k_new], dim=2
                    )
                    self._history_V[li] = torch.cat(
                        [self._history_V[li], v_new], dim=2
                    )
            self._history_blk = torch.cat([
                self._history_blk,
                self._col_blk[history_cols],
            ])
            self._history_content_abs = torch.cat([
                self._history_content_abs,
                self._col_content_abs[history_cols],
            ])
        keep_cols = keep.nonzero(as_tuple=True)[0]
        self.store.evict(keep_cols)
        self._col_role = self._col_role[keep_cols]
        self._col_blk = self._col_blk[keep_cols]
        self._col_content_abs = self._col_content_abs[keep_cols]
        self._col_code_slot = self._col_code_slot[keep_cols]
        self._col_score = self._col_score[keep_cols]

    def cache_kv_stats(self):
        main_columns = [
            int(k.shape[2]) if k is not None else 0
            for k in self.store.K
        ]
        history_columns = [
            int(k.shape[2]) if k is not None else 0
            for k in self._history_K
        ]
        total_columns = sum(main_columns) + sum(history_columns)
        kv_heads = self.clm.model.config.num_key_value_heads
        head_dim = self.layers[0].self_attn.head_dim
        return {
            "main_columns_by_layer": main_columns,
            "history_columns_by_layer": history_columns,
            "total_columns": total_columns,
            "total_kv_elements": total_columns * 2 * kv_heads * head_dim,
        }

    def _content_store_add(self, query_content_abs):
        if self.store.length == 0:
            return None
        if self.no_codes:                     # eviction baselines: everything still stored is visible
            return None
        q = query_content_abs.view(-1, 1)
        is_code = self._col_role.view(1, -1) == 1
        is_content = self._col_role.view(1, -1) == 0
        is_replay = self._col_role.view(1, -1) == 2
        raw_prompt = is_content & (self._col_content_abs.view(1, -1) < self.raw_prompt_len)
        own_block = is_content & (self._col_blk.view(1, -1) == self.b)
        raw_tail = (
            is_content
            & (self._col_blk.view(1, -1) + 1 == self.b)
            & (
                q - self._col_content_abs.view(1, -1)
                <= self.rolling_content_tail
            )
        )
        sticky_tail = (
            is_content
            & (self._col_blk.view(1, -1) + 1 == self.b)
            & (
                self._col_content_abs.view(1, -1) % self.C
                >= self.C - self.retain_tail_kv
            )
            & (self.retain_tail_policy == "sticky")
        )
        replay_tail = (is_replay
                       & (self._col_blk.view(1, -1) + 1 == self.b)
                       & (q - self._col_content_abs.view(1, -1) <= self.replay_tail))
        static_tail = self._static_tail_cols(self.b).view(1, -1)         # earlier blocks' permanent tails
        if self.oracle_full_history:
            visible = is_code | is_content | replay_tail
        else:
            visible = (
                is_code | raw_prompt | own_block | raw_tail | sticky_tail
                | static_tail | replay_tail
            )
        neg = torch.finfo(self.dtype).min
        return torch.zeros(visible.shape, dtype=self.dtype, device=self.device).masked_fill_(~visible, neg)

    def _code_store_add(self):
        if self.store.length == 0:
            return None
        is_code = self._col_role == 1
        is_content = self._col_role == 0
        raw_prompt = is_content & (self._col_content_abs < self.raw_prompt_len)
        own_block = is_content & (self._col_blk == self.b)
        if self.code_attend_current_content:
            visible = is_code | raw_prompt | own_block | self._static_tail_cols(self.b)
            if self.code_excluded_tail:
                replayed_source = (
                    own_block
                    & (
                        self._col_content_abs % self.C
                        >= self.C - self.code_excluded_tail
                    )
                )
                visible = visible & ~replayed_source
        else:
            visible = is_code | (raw_prompt & ~own_block)
        neg = torch.finfo(self.dtype).min
        return torch.zeros((self.K, self.store.length), dtype=self.dtype, device=self.device).masked_fill_(
            ~visible.view(1, -1), neg)

    def _replay_store_add(self):
        if self.store.length == 0:
            return None
        is_code = self._col_role == 1
        is_content = self._col_role == 0
        raw_prompt = is_content & (self._col_content_abs < self.raw_prompt_len)
        own_block = is_content & (self._col_blk == self.b)
        visible = is_code | raw_prompt | own_block
        neg = torch.finfo(self.dtype).min
        return torch.zeros(
            (self.replay_tail, self.store.length),
            dtype=self.dtype,
            device=self.device,
        ).masked_fill_(~visible.view(1, -1), neg)

    def _evict_expired_tail(self):
        if self.no_codes or self.store.length == 0 or self.f >= self.C:
            return
        is_code = self._col_role == 1
        is_content = self._col_role == 0
        is_replay = self._col_role == 2
        raw_prompt = is_content & (self._col_content_abs < self.raw_prompt_len)
        current = is_content & (self._col_blk == self.b)
        next_content_abs = self.b * self.C + self.f
        if self.retain_tail_policy == "static":
            live_tail = self._static_tail_cols(self.b)
        elif self.retain_tail_policy == "sticky":
            live_tail = (
                is_content
                & (self._col_blk + 1 == self.b)
                & (
                    self._col_content_abs % self.C
                    >= self.C - self.retain_tail_kv
                )
            )
        else:
            live_tail = (
                is_content
                & (self._col_blk + 1 == self.b)
                & (
                    next_content_abs - self._col_content_abs
                    <= self.rolling_content_tail
                )
            )
        live_replay = (is_replay
                       & (self._col_blk + 1 == self.b)
                       & (next_content_abs - self._col_content_abs <= self.replay_tail))
        if self.preserve_full_history:
            self._evict(is_code | is_content | live_replay)
            return
        self._evict(is_code | raw_prompt | current | live_tail | live_replay)

    @property
    def code_positions(self):
        return getattr(self.clm, "code_positions", "appended")

    def _content_pos(self, n):
        """RoPE positions of the next n content tokens of block b (offset f). Must match CompressLM._rope_position_ids."""
        if self.code_positions == "bitrev" or self.no_codes:
            base = self.b * self.C
        else:
            base = self.b * (self.C + self.K + self.replay_tail)
        return torch.arange(base + self.f, base + self.f + n, device=self.device)

    def _code_pos(self):
        """RoPE positions of the K code rows of block b."""
        if self.code_positions == "bitrev":
            return self.b * self.C + _bitrev_code_offsets(
                self.K, self.C, self.device, span=self.C - self.code_excluded_tail
            )
        base = self.b * (self.C + self.K + self.replay_tail)
        return torch.arange(base + self.C, base + self.C + self.K, device=self.device)

    def _rope(self, abs_pos):
        ref = self.inner.embed_tokens.weight[:1]                              # dummy for dtype/device
        return self.inner.rotary_emb(ref.view(1, 1, -1), abs_pos.unsqueeze(0))

    def _causal_add(self, n):
        neg = torch.finfo(self.dtype).min
        i = torch.arange(n, device=self.device).unsqueeze(1)
        j = torch.arange(n, device=self.device).unsqueeze(0)
        return torch.zeros((n, n), dtype=self.dtype, device=self.device).masked_fill_(j > i, neg)

    # -- feed <= C-f content tokens as ONE chunk (must not cross a block boundary); returns final-normed (n,d)
    def feed_content(self, ids):
        n = len(ids)
        assert self.f + n <= self.C
        capture_content = (
            (
                self.capture_content_attention
                or self.content_head_callback is not None
            )
            and n == 1
            and self.b > 0
            and self.f in self.content_attention_offsets
        )
        abs_pos = self._content_pos(n)                                    # RoPE positions (appended | bitrev)
        cos, sin = self._rope(abs_pos)
        hidden = self.inner.embed_tokens(torch.tensor([ids], device=self.device))
        content_abs = torch.arange(self.b * self.C + self.f, self.b * self.C + self.f + n,
                                   device=self.device)
        sadd = self._content_store_add(content_abs)
        cadd = self._causal_add(n)
        if capture_content:
            self._attention_phase = "content"
            self._attention_query_content_abs = int(content_abs[0])
        self._content_query_active = True
        try:
            for li in range(self.num_layers):
                hidden = self._layer(
                    li, hidden, abs_pos, cos, sin, sadd, cadd, commit=True
                )
        finally:
            self._content_query_active = False
        if capture_content:
            self._attention_phase = None
            self._attention_query_content_abs = None
        self.store.length += n
        self._append_col_meta(0, self.b, content_abs.tolist())
        self.f += n
        self._cur_ids += list(ids)
        self._evict_expired_tail()
        normalized = self.inner.norm(hidden)[0]
        if self.capture_hidden:
            self.hidden_trace.append(
                {
                    "phase": "content",
                    "positions": abs_pos.detach().cpu(),
                    "token_ids": torch.tensor(ids),
                    "hidden_states": normalized.detach().cpu(),
                }
            )
        return normalized                                                     # (n,d) -> caller applies lm_head

    def _pause_base(self):
        """Code seeds (K,d) exactly as CompressLM.forward builds pause_base for ONE block (L=1)."""
        clm = self.clm
        pause = clm.pause.to(self.dtype)
        segment_mean = (
            self.code_init_override == "segment_mean"
            or getattr(clm, "seg_pool", False)
            or getattr(clm, "gated_seg_pool", False)
        )
        if not (segment_mean or getattr(clm, "pool_init", False)):
            return pause                                                      # default: generic per-slot seeds
        source_ids = (
            self._cur_ids[: self.C - self.code_excluded_tail]
            if self.code_excluded_tail
            else self._cur_ids
        )
        tok_e = self.inner.embed_tokens(
            torch.tensor([source_ids], device=self.device)
        )[0]
        K, d = self.K, self.d
        if segment_mean:
            seg = len(source_ids) // K
            if seg >= 1:
                tm = tok_e[: K * seg].reshape(K, seg, d).mean(1)
            else:
                tm = tok_e.mean(0, keepdim=True).expand(K, d)
            if getattr(clm, "gated_seg_pool", False):
                tm = tm * clm.seg_pool_gate.to(self.dtype).view(K, 1)
            return pause + tm.to(self.dtype)
        sc = torch.einsum('kd,cd->kc', clm.pool_q.to(self.dtype), clm.pool_k(tok_e)) * (d ** -0.5)  # (K,C)
        pooled = torch.einsum('kc,cd->kd', sc.float().softmax(-1).to(self.dtype), clm.pool_v(tok_e))
        return pause + pooled

    def _append_replay(self):
        T = self.replay_tail
        if not T:
            return
        C, K = self.C, self.K
        base = self.b * (C + K + T)
        abs_pos = torch.arange(base + C + K, base + C + K + T, device=self.device)
        cos, sin = self._rope(abs_pos)
        replay_ids = self._cur_ids[-T:]
        hidden = self.inner.embed_tokens(torch.tensor([replay_ids], device=self.device))
        sadd = self._replay_store_add()
        cadd = self._causal_add(T)
        for li in range(self.num_layers):
            hidden = self._layer(li, hidden, abs_pos, cos, sin, sadd, cadd, commit=True)
        self.store.length += T
        content_abs = list(range((self.b + 1) * C - T, (self.b + 1) * C))
        self._append_col_meta(2, self.b, content_abs)

    # -- compress the (now full) current block into K code KVs, then EVICT its content columns
    def evict_block(self):
        """Codes-free block boundary: keep `keep_per_block` generated columns of the completed block by policy."""
        C = self.C
        assert self.f == C
        is_content = self._col_role == 0
        cand = is_content & (self._col_blk == self.b) & (self._col_content_abs >= self.raw_prompt_len)
        keep = ~cand
        idx = cand.nonzero(as_tuple=True)[0]
        n_c = int(idx.numel())
        kb = self.keep_per_block
        if n_c <= kb:
            keep = keep | cand
        elif kb > 0:
            if self.evict_policy == "recency":
                sel = idx[-kb:]
            elif self.evict_policy == "random":
                g = torch.Generator(device="cpu").manual_seed(self.evict_seed * 1000003 + self.b)
                sel = idx[torch.randperm(n_c, generator=g)[:kb].to(idx.device)]
            else:                                                             # score: top accumulated attention received
                sel = idx[torch.topk(self._col_score[idx], kb).indices]
            keep[sel] = True
        self._evict(keep)
        self.b += 1
        self.f = 0
        self._cur_ids = []

    def compress(self):
        if self.no_codes:
            return self.evict_block()
        C, K = self.C, self.K
        assert self.f == C
        abs_pos = self._code_pos()                                        # RoPE positions of the K code rows
        cos, sin = self._rope(abs_pos)
        hidden = self._pause_base().to(self.dtype).view(1, K, self.d)
        if self.xa == "full":
            cadd = torch.zeros((K, K), dtype=self.dtype, device=self.device)  # bidirectional over the code set
        elif self.xa == "causal":
            cadd = self._causal_add(K)
        else:                                                                 # "none": codes ignore each other (& self)
            cadd = torch.full((K, K), torch.finfo(self.dtype).min, dtype=self.dtype, device=self.device)
        sadd = self._code_store_add()
        self._attention_phase = "code"
        for li in range(self.num_layers):
            new_hidden = self._layer(li, hidden, abs_pos, cos, sin, sadd, cadd, commit=True)
            if self.skip_unused_codes:
                used = int(self.head_code_quotas[li].max())
                if used < K:
                    new_hidden = torch.cat([new_hidden[:, :used], hidden[:, used:]], 1)
            hidden = new_hidden
        self._attention_phase = None
        if self.capture_hidden:
            self.hidden_trace.append(
                {
                    "phase": "code",
                    "positions": abs_pos.detach().cpu(),
                    "token_ids": torch.empty(0, dtype=torch.long),
                    "hidden_states": self.inner.norm(hidden)[0].detach().cpu(),
                }
            )
        self.store.length += K
        self._append_col_meta(1, self.b, [-1] * K)
        self._append_replay()
        is_code = self._col_role == 1
        is_content = self._col_role == 0
        is_replay = self._col_role == 2
        raw_prompt = is_content & (self._col_content_abs < self.raw_prompt_len)
        raw_tail = (
            is_content
            & (self._col_blk == self.b)
            & (
                self._col_content_abs
                >= (self.b + 1) * C - self.content_tail_size
            )
        )
        replay_tail = is_replay & (self._col_blk == self.b)
        static_tail = self._static_tail_cols(self.b)
        if self.preserve_full_history:
            self._evict(is_code | is_content | replay_tail)
        else:
            self._evict(is_code | raw_prompt | raw_tail | static_tail | replay_tail)
        self.b += 1
        self.f = 0
        self._cur_ids = []

    # -- feed one content token, compressing if it completes the block; returns its next-token logits (V,)
    def step(self, token_id):
        h = self.feed_content([int(token_id)])                               # (1,d)
        logits = self.lm_head(h[-1:]).float()[0]
        if self.f == self.C:
            self.compress()
        return logits

    # -- prefill a prompt (block-aligned chunks; compress completed blocks); returns last-token logits (V,)
    def prefill(self, prompt_ids):
        ids = list(prompt_ids)
        last = None
        i = 0
        while i < len(ids):
            take = min(self.C - self.f, len(ids) - i)
            h = self.feed_content(ids[i:i + take])
            last = self.lm_head(h[-1:]).float()[0]
            i += take
            if self.f == self.C:
                self.compress()
        return last

    @torch.no_grad()
    def generate(self, prompt_ids, max_new_tokens, eos_id=None, temperature=0.0,
                 top_p=1.0, top_k=0, gen=None):
        logits = self.prefill(prompt_ids)
        out = []
        for _ in range(max_new_tokens):
            nxt = _sample(logits, temperature, top_p, top_k, gen)
            out.append(nxt)
            if eos_id is not None and nxt == eos_id:
                break
            logits = self.step(nxt)
        return out


def _sample(logits, temperature, top_p, top_k, gen):
    if temperature is None or temperature <= 0:
        return int(logits.argmax())
    logits = logits / temperature
    if top_k and top_k > 0:
        kth = torch.topk(logits, min(top_k, logits.numel())).values[-1]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    probs = F.softmax(logits, dim=-1)
    if top_p and top_p < 1.0:
        sp, si = torch.sort(probs, descending=True)
        cdf = torch.cumsum(sp, dim=-1)
        cut = cdf > top_p
        cut[..., 1:] = cut[..., :-1].clone()
        cut[..., 0] = False
        sp = sp.masked_fill(cut, 0.0)
        probs = torch.zeros_like(probs).scatter_(-1, si, sp)
        probs = probs / probs.sum()
    return int(torch.multinomial(probs, 1, generator=gen))


# ============================================================================
# Deliverable 2 (THE GATE): incremental decode reproduces the teacher-forced forward.
# ============================================================================
def _dense_mask(clm, device, prompt_len=None):
    """Materialise CompressLM._mask.mm as a dense boolean (with_copy=False), replicating mm exactly."""
    C, N, K, L, T = clm.C, clm.N, clm.K, clm.L, clm.replay_tail
    S0 = N * (C + L * K + T)
    blk, role = clm.blk.to(device), clm.role.to(device)
    content_abs = clm.content_abs_pos.to(device)
    q = torch.arange(S0, device=device).view(S0, 1)
    kv = torch.arange(S0, device=device).view(1, S0)
    qr, kr = role.view(S0, 1), role.view(1, S0)
    qb, kb = blk.view(S0, 1), blk.view(1, S0)
    causal = kv <= q
    content_vis = (kr == 0) & (kb == qb)
    earlier_final = (kr == L) & (kb < qb)
    pause_q = (qr > 0) & (qr <= L)
    refine = pause_q & (kr == qr - 1) & (kb == qb)
    replay_q = qr == clm.replay_role
    replay_own_final = replay_q & (kr == L) & (kb == qb)
    replay_causal = replay_q & (kr == clm.replay_role) & (kb == qb)
    raw_tail = (
        (qr == 0)
        & (kr == 0)
        & (kb + 1 == qb)
        & (
            content_abs.view(S0, 1) - content_abs.view(1, S0)
            <= clm.raw_tail
        )
    )
    if clm.retain_tail_policy == "static":                               # permanent per-block tail rows, visible to every later query
        retained_tail = (
            (kr == 0)
            & (kb < qb)
            & (content_abs.view(1, S0) % C >= C - clm.retain_tail_kv)
            & (clm.retain_tail_kv > 0)
        )
    else:
        retained_tail = (
            (qr == 0)
            & (kr == 0)
            & (kb + 1 == qb)
            & (
                (
                    content_abs.view(1, S0) % C
                    >= C - clm.retain_tail_kv
                )
                if clm.retain_tail_policy == "sticky"
                else (
                    content_abs.view(S0, 1)
                    - content_abs.view(1, S0)
                    <= clm.retain_tail_kv
                )
            )
            & (clm.retain_tail_kv > 0)
        )
    replay_tail = ((qr == 0) & (kr == clm.replay_role) & (kb + 1 == qb)
                   & (content_abs.view(S0, 1) - content_abs.view(1, S0) <= T))
    base = causal & (
        content_vis | earlier_final | refine | replay_own_final
        | replay_causal | raw_tail | retained_tail | replay_tail
    )
    if clm.pause_xattn != "none":
        samepass = pause_q & (kr == qr) & (kb == qb)
        base = base | ((samepass & causal) if clm.pause_xattn == "causal" else samepass)
    if prompt_len is not None:
        prompt_raw = ((kr == 0).unsqueeze(0)
                      & (content_abs.view(1, 1, S0) < prompt_len.view(-1, 1, 1)))
        base = base.unsqueeze(0) | (causal.unsqueeze(0) & prompt_raw)
    if clm.code_excluded_tail:
        excluded = _code_content_tail_excluded(
            role,
            blk,
            content_abs,
            q,
            kv,
            C,
            clm.code_excluded_tail,
            clm.replay_role,
        )
        base = base & ~excluded
    return base


@torch.no_grad()
def _dense_ref_logits(clm, content_ids, device, prompt_len=None):
    """Teacher-forced reference via a materialised dense mask + manual (SDPA-style) attention.
    Returns logits at the loss_pos content positions (block-major, within-block 0..C-2)."""
    B, N, C = content_ids.shape
    K, T, d = clm.K, clm.replay_tail, clm.model.config.hidden_size
    S0 = N * (C + K + T)
    emb = clm.model.model.embed_tokens
    tok_e = emb(content_ids)                                                  # (B,N,C,d)
    pause_e = clm.pause.to(tok_e.dtype).view(1, 1, K, d).expand(B, N, K, d)   # default seeds (gate uses default)
    replay_e = tok_e[:, :, C - T:, :] if T else tok_e[:, :, :0, :]
    x = torch.cat([tok_e, pause_e, replay_e], dim=2).reshape(B, S0, d)
    mb = _dense_mask(clm, device, prompt_len)
    if mb.ndim == 2:
        mb = mb.unsqueeze(0)
    add = torch.zeros(mb.shape, dtype=x.dtype, device=device).masked_fill(
        ~mb, torch.finfo(x.dtype).min).unsqueeze(1)
    if getattr(clm, "code_positions", "appended") == "bitrev":
        assert clm.K_act == clm.K, "dense reference assumes the full-K layout"
        pos = clm._rope_position_ids(N, False, device).view(-1)
    else:
        pos = torch.arange(S0, device=device)
    cos, sin = clm.model.model.rotary_emb(x, pos.unsqueeze(0))
    h = x
    for layer in clm.model.model.layers:
        attn = layer.self_attn
        normed = layer.input_layernorm(h)
        hs = (B, S0, -1, attn.head_dim)
        q = _qkn(attn, "q_norm", attn.q_proj(normed).view(hs)).transpose(1, 2)
        k = _qkn(attn, "k_norm", attn.k_proj(normed).view(hs)).transpose(1, 2)
        v = attn.v_proj(normed).view(hs).transpose(1, 2)
        q, k = apply_rope(q, k, cos, sin)
        w = torch.matmul(q, repeat_kv(k, attn.num_key_value_groups).transpose(2, 3)) * attn.scaling + add
        w = F.softmax(w, dim=-1, dtype=torch.float32).to(q.dtype)
        o = torch.matmul(w, repeat_kv(v, attn.num_key_value_groups)).transpose(1, 2).contiguous().view(B, S0, -1)
        h = h + attn.o_proj(o)
        h = h + layer.mlp(layer.post_attention_layernorm(h))
    h = clm.model.model.norm(h)
    hid = h[:, clm.loss_pos, :].reshape(-1, d)
    return clm.model.lm_head(hid).float()


@torch.no_grad()
def _flex_ref_logits(clm, content_ids, device, prompt_len=None):
    """Teacher-forced reference via the EXISTING CompressLM path (FlexAttention mask)."""
    B, N, C = content_ids.shape
    K, T, d = clm.K, clm.replay_tail, clm.model.config.hidden_size
    S0 = N * (C + K + T)
    emb = clm.model.model.embed_tokens
    tok_e = emb(content_ids)
    pause_e = clm.pause.to(tok_e.dtype).view(1, 1, K, d).expand(B, N, K, d)
    replay_e = tok_e[:, :, C - T:, :] if T else tok_e[:, :, :0, :]
    x = torch.cat([tok_e, pause_e, replay_e], dim=2).reshape(B, S0, d)
    blockmask = (clm._mask_rawprompt(device, prompt_len, B)
                 if prompt_len is not None else clm.blockmask(device, False))
    pos_ids = None
    if getattr(clm, "code_positions", "appended") == "bitrev":           # explicit RoPE positions, as CompressLM.forward
        assert clm.K_act == clm.K, "flex reference assumes the full-K layout"
        pos_ids = clm._rope_position_ids(N, False, device)
    out = clm.model.model(inputs_embeds=x, attention_mask=blockmask, position_ids=pos_ids,
                          use_cache=False).last_hidden_state
    hid = out[:, clm.loss_pos, :].reshape(-1, d)
    return clm.model.lm_head(hid).float()


@torch.no_grad()
def _gen_ref_logits(clm, content_ids, device, prompt_len=None):
    """Incremental generator, teacher-forcing the known tokens; loss_pos logits (block-major, 0..C-2)."""
    B, N, C = content_ids.shape
    outs = []
    for bi in range(B):
        raw_prompt_len = int(prompt_len[bi]) if prompt_len is not None else 0
        g = CompressGen(clm, device, raw_prompt_len=raw_prompt_len)
        seq = []
        for blk in range(N):
            ids = content_ids[bi, blk].tolist()
            h = g.feed_content(ids)                                           # (C,d) final-normed
            logits = clm.model.lm_head(h).float()                            # (C,V)
            seq.append(logits[: C - 1])                                       # loss_pos: within-block 0..C-2
            g.compress()
        outs.append(torch.cat(seq, 0))
    return torch.cat(outs, 0)


def _gate_one(
    base,
    device,
    C,
    N,
    K,
    xa,
    raw_tail=0,
    replay_tail=0,
    retain_tail_kv=0,
    retain_tail_policy="rolling",
    raw_prompt_len=0,
    code_exclude_replay_tail=False,
):
    torch.manual_seed(0)
    clm = CompressLM(
        base,
        C,
        N,
        K,
        L=1,
        pause_xattn=xa,
        raw_tail=raw_tail,
        replay_tail=replay_tail,
        retain_tail_kv=retain_tail_kv,
        retain_tail_policy=retain_tail_policy,
        code_exclude_replay_tail=code_exclude_replay_tail,
    ).to(device).eval()
    clm.pause.data.normal_(0, 0.05)                                          # non-trivial random pause init
    ids = torch.randint(0, base.config.vocab_size, (2, N, C), device=device)
    prompt_len = (torch.full((ids.size(0),), raw_prompt_len, dtype=torch.int32, device=device)
                  if raw_prompt_len else None)
    dense = _dense_ref_logits(clm, ids, device, prompt_len)
    gen = _gen_ref_logits(clm, ids, device, prompt_len)
    gen_dense_diff = (gen - dense).abs().max().item()
    flex_ok, flex_diff = False, None
    try:
        flex = _flex_ref_logits(clm, ids, device, prompt_len)
        flex_diff = (flex - dense).abs().max().item()
        flex_ok = True
    except Exception as e:                                                    # e.g. FlexAttention unsupported on some HW
        flex = None
        print(f"[gate]   (flex forward unavailable: {type(e).__name__}: {str(e)[:80]}; using dense ref == flex on capable HW)")
    ref = flex if flex_ok else dense
    maxdiff = (gen - ref).abs().max().item()
    return maxdiff, flex_diff, gen_dense_diff, flex_ok


def run_gate(device):
    from transformers import AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM
    if str(device).startswith("cuda"):
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
    gate_base = os.environ.get("GATE_BASE", "").strip()
    if gate_base:
        base = AutoModelForCausalLM.from_pretrained(
            gate_base, dtype=torch.float32
        ).to(device)
        gate_label = gate_base
    else:
        config = LlamaConfig(
            vocab_size=257,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=4096,
        )
        base = LlamaForCausalLM(config).to(device)
        gate_label = "random-tiny-llama"
    # FlexAttention in some torch builds only compiles the first distinct
    # BlockMask shape in a process. Put the production retained-KV semantics
    # first and require Flex for that case; the remaining shape sweep still
    # requires exact dense/incremental agreement.
    headline_policy = os.environ.get(
        "RETAIN_TAIL_POLICY", "rolling"
    ).strip().lower()
    other_policy = "sticky" if headline_policy == "rolling" else "rolling"
    headline = (
        16, 4, 4, "full", 0, 0, 4, 19, False, headline_policy
    )
    configs = [
        headline,
        (16, 4, 4, "full", 0, 0, 4, 19, False, other_policy),
        (16, 4, 4, "full", 0, 0, 0, 0, False, "rolling"),
        (16, 4, 16, "full", 0, 0, 0, 0, False, "rolling"),
        (24, 5, 8, "full", 0, 0, 0, 0, False, "rolling"),
        (16, 4, 4, "causal", 0, 0, 0, 0, False, "rolling"),
        (16, 4, 4, "full", 4, 0, 0, 0, False, "rolling"),
        (16, 4, 4, "full", 4, 0, 0, 19, False, "rolling"),
        (16, 4, 4, "full", 16, 0, 0, 15, False, "rolling"),
        (16, 4, 4, "full", 16, 0, 0, 16, False, "rolling"),
        (16, 4, 4, "full", 16, 0, 0, 17, False, "rolling"),
        (16, 4, 4, "full", 0, 4, 0, 0, False, "rolling"),
        (16, 4, 4, "full", 0, 4, 0, 19, False, "rolling"),
        (16, 4, 4, "full", 0, 4, 0, 0, True, "rolling"),
        (16, 4, 4, "full", 0, 4, 0, 19, True, "rolling"),
        (16, 4, 4, "full", 0, 0, 4, 0, False, "rolling"),
    ]
    print(
        f"[gate] tiny CompressLM on {device}; dtype={next(base.parameters()).dtype} "
        f"matmul={torch.get_float32_matmul_precision()} tf32="
        f"{torch.backends.cuda.matmul.allow_tf32 if str(device).startswith('cuda') else 'n/a'}; "
        f"base={gate_label}, random pause init"
    )
    ok = True
    head = None
    for (
        C,
        N,
        K,
        xa,
        raw_tail,
        replay_tail,
        retain_tail_kv,
        raw_prompt_len,
        code_exclude_replay_tail,
        retain_tail_policy,
    ) in configs:
        maxdiff, flex_diff, gen_dense_diff, flex_ok = _gate_one(
            base,
            device,
            C,
            N,
            K,
            xa,
            raw_tail,
            replay_tail,
            retain_tail_kv,
            retain_tail_policy,
            raw_prompt_len,
            code_exclude_replay_tail,
        )
        config_key = (
            C,
            N,
            K,
            xa,
            raw_tail,
            replay_tail,
            retain_tail_kv,
            raw_prompt_len,
            code_exclude_replay_tail,
            retain_tail_policy,
        )
        is_headline = config_key == headline
        passed = (
            maxdiff < 1e-3
            and gen_dense_diff < 1e-3
            and (flex_ok or device == "cpu" or not is_headline)
        )
        ok = ok and passed
        fd = f"dense-vs-flex={flex_diff:.2e}" if flex_diff is not None else "dense-only"
        tag = "  <-- HEADLINE (retained-KV + raw prompt)" if is_headline else ""
        if is_headline:
            head = maxdiff
        print(f"[gate] C={C:2d} N={N} K={K:2d} xattn={xa:6s} "
              f"tail={raw_tail:2d} replay={replay_tail:2d} "
              f"retain_kv={retain_tail_kv:2d} prompt={raw_prompt_len:2d} "
              f"policy={retain_tail_policy:7s} "
              f"code_excl_replay={int(code_exclude_replay_tail)} "
              f"| GEN-vs-FORWARD max|Δlogit|={maxdiff:.3e} GEN-vs-DENSE={gen_dense_diff:.3e} "
              f"({fd}) -> {'PASS' if passed else 'FAIL'}{tag}")
    print(f"\n[gate] RESULT: {'PASS' if ok else 'FAIL'}  "
          f"(all configs < 1e-3; headline Flex required on GPU)  "
          f"headline max|Δ|={head:.3e}")
    return ok, head


# ============================================================================
# Deliverable 3: MATH-500 generation eval (compressed generator OR full-ctx baseline).
# ============================================================================
def _load_compresslm(ckpt_path, base_override, device, dtype):
    from transformers import AutoModelForCausalLM
    sd = torch.load(ckpt_path, map_location="cpu")
    meta = sd if isinstance(sd, dict) and "model" in sd else {"model": sd}
    state = meta["model"]
    base_name = base_override or meta.get("base", "HuggingFaceTB/SmolLM2-1.7B")
    C = int(meta.get("C", os.environ.get("C", 512)))
    K = int(meta.get("K", os.environ.get("K", 128)))
    L = int(meta.get("L", 1))
    xa = meta.get("pause_xattn", os.environ.get("PAUSE_XATTN", "full"))
    raw_prompt = bool(int(meta.get("raw_prompt", os.environ.get("RAW_PROMPT", 0))))
    raw_tail = int(meta.get("raw_tail", os.environ.get("RAW_TAIL", 0)))
    replay_tail = int(
        os.environ["REPLAY_TAIL"]
        if "REPLAY_TAIL" in os.environ
        else meta.get("replay_tail", 0)
    )
    retain_tail_kv = int(
        os.environ["RETAIN_TAIL_KV"]
        if "RETAIN_TAIL_KV" in os.environ
        else meta.get("retain_tail_kv", 0)
    )
    retain_tail_policy = (
        os.environ["RETAIN_TAIL_POLICY"]
        if "RETAIN_TAIL_POLICY" in os.environ
        else meta.get("retain_tail_policy", "rolling")
    )
    code_exclude_replay_tail = bool(
        int(meta.get("code_exclude_replay_tail", 0))
    )
    seg_pool = bool(int(meta.get("seg_pool", 0)))
    gated_seg_pool = bool(int(meta.get("gated_seg_pool", 0)))
    pool_init = bool(int(meta.get("pool_init", 0)))
    saved_replay_tail = int(meta.get("replay_tail", 0))
    if replay_tail != saved_replay_tail and not int(os.environ.get("ALLOW_REPLAY_TAIL_MISMATCH", 0)):
        raise ValueError(
            f"REPLAY_TAIL={replay_tail} does not match checkpoint replay_tail={saved_replay_tail}"
        )
    saved_retain_tail_kv = int(meta.get("retain_tail_kv", 0))
    if (
        retain_tail_kv != saved_retain_tail_kv
        and not int(os.environ.get("ALLOW_RETAIN_TAIL_KV_MISMATCH", 0))
    ):
        raise ValueError(
            f"RETAIN_TAIL_KV={retain_tail_kv} does not match checkpoint "
            f"retain_tail_kv={saved_retain_tail_kv}"
        )
    saved_retain_tail_policy = meta.get("retain_tail_policy", "rolling")
    if (
        retain_tail_policy != saved_retain_tail_policy
        and not int(os.environ.get("ALLOW_RETAIN_TAIL_POLICY_MISMATCH", 0))
    ):
        raise ValueError(
            f"RETAIN_TAIL_POLICY={retain_tail_policy!r} does not match checkpoint "
            f"retain_tail_policy={saved_retain_tail_policy!r}"
        )
    base = AutoModelForCausalLM.from_pretrained(base_name, dtype=dtype).to(device)
    clm = CompressLM(
        base,
        C,
        N=4,
        K=K,
        L=L,
        pause_xattn=xa,
        pool_init=pool_init,
        seg_pool=seg_pool,
        gated_seg_pool=gated_seg_pool,
        raw_tail=raw_tail,
        replay_tail=replay_tail,
        retain_tail_kv=retain_tail_kv,
        retain_tail_policy=retain_tail_policy,
        code_exclude_replay_tail=code_exclude_replay_tail,
        code_positions=str(meta.get("code_positions", "appended")),
    ).to(device)
    clm.raw_prompt = raw_prompt
    missing, unexpected = clm.load_state_dict(state, strict=False)
    print(f"[math500] loaded {ckpt_path}  base={base_name} C={C} K={K} L={L} xattn={xa} code_positions={clm.code_positions} "
          f"raw_prompt={int(raw_prompt)} raw_tail={raw_tail} replay_tail={replay_tail} "
          f"retain_tail_kv={retain_tail_kv} "
          f"retain_tail_policy={retain_tail_policy} "
          f"code_exclude_replay_tail={int(code_exclude_replay_tail)} "
          f"seg_pool={int(seg_pool)} pool_init={int(pool_init)} "
          f"gated_seg_pool={int(gated_seg_pool)} "
          f"| missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    return clm.eval(), base_name, C, K


def run_math500(args, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "scripts"))
    from eval_math500 import extract_boxed_answer, grade                     # reuse boxed grading
    from mv_grade import mv_grade_text, use_mv                               # SAME grader as the vLLM evaluators
    from prompt_format import format_user_content, boxed_instruction_enabled  # reuse prompt build

    dtype = torch.float32 if args.fp32 else torch.bfloat16
    rows = [json.loads(l) for l in open(args.data)]
    if args.limit:
        rows = rows[: args.limit]
    if args.nshard > 1:                                                      # strided shard (balances length across shards)
        rows = rows[args.shard::args.nshard]
        print(f"[math500] shard {args.shard}/{args.nshard}: {len(rows)} problems", flush=True)

    if args.baseline:                                                        # full-context causal LM (no compression)
        base_name = args.base or "HuggingFaceTB/SmolLM2-1.7B"
        model = AutoModelForCausalLM.from_pretrained(base_name, dtype=dtype).to(device).eval()
        if args.ckpt:                                                        # optional: load base weights from a CompressLM ckpt
            sd = torch.load(args.ckpt, map_location="cpu")
            state = sd["model"] if isinstance(sd, dict) and "model" in sd else sd
            base_sd = {k[len("model."):]: v for k, v in state.items() if k.startswith("model.")}
            m, u = model.load_state_dict(base_sd, strict=False)
            print(f"[math500] baseline loaded base weights from {args.ckpt} | missing={len(m)} unexpected={len(u)}", flush=True)
        C = int(os.environ.get("C", 512))
        tok = AutoTokenizer.from_pretrained(args.tokenizer or base_name)
    else:
        clm, base_name, C, _ = _load_compresslm(args.ckpt, args.base, device, dtype)
        model = clm.model
        tok = AutoTokenizer.from_pretrained(args.tokenizer or base_name)

    eos_id = tok.eos_token_id
    n_correct = n_multiblock = 0
    kv_full = kv_comp = 0
    Kc = (clm.K if not args.baseline else int(os.environ.get("K", "128")))
    records = []
    evict_policy = getattr(args, "evict_policy", "") or None
    gen_kwargs = {}
    if evict_policy:
        if args.baseline:
            raise ValueError("--evict-policy is a CompressGen mode; drop --baseline")
        gen_kwargs = dict(evict_policy=evict_policy, keep_per_block=args.keep_per_block, evict_seed=args.evict_seed)
        clm.raw_prompt = True                                                 # prompt always raw for the eviction baselines
        print(f"[math500] eviction baseline: policy={evict_policy} keep_per_block={args.keep_per_block} "
              f"evict_seed={args.evict_seed} (codes-free, prompt raw, plain positions)", flush=True)
    mode = "baseline" if args.baseline else (f"evict:{evict_policy}" if evict_policy else "compressed")
    raw_prompt = False if args.baseline else clm.raw_prompt
    raw_tail = 0 if args.baseline else clm.raw_tail
    replay_tail = 0 if args.baseline else clm.replay_tail
    retain_tail_kv = 0 if args.baseline else clm.retain_tail_kv
    retain_tail_policy = (
        "rolling" if args.baseline else clm.retain_tail_policy
    )
    output_identity = {
        "schema_version": 2,
        "mode": mode,
        "ckpt": args.ckpt,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "raw_prompt": raw_prompt,
        "raw_tail": raw_tail,
        "replay_tail": replay_tail,
        "retain_tail_kv": retain_tail_kv,
        "retain_tail_policy": retain_tail_policy,
        "shard": args.shard,
        "nshard": args.nshard,
        "n": len(rows),
        "evict_policy": evict_policy,
        "keep_per_block": (args.keep_per_block if evict_policy else None),
        "evict_seed": (args.evict_seed if evict_policy else None),
    }

    def make_summary(complete):
        processed = len(records)
        return {
            **output_identity,
            "complete": complete,
            "processed": processed,
            "base": base_name,
            "C": C,
            "accuracy": n_correct / max(processed, 1),
            "n_correct": n_correct,
            "multiblock_fraction": n_multiblock / max(processed, 1),
            "n_multiblock": n_multiblock,
            "boxed_instruction": boxed_instruction_enabled(),
            "nominal_rate": C / max(1, Kc),
            "effective_rate": kv_full / max(1, kv_comp),
            "kv_full": kv_full,
            "kv_comp": kv_comp,
        }

    def write_output(complete):
        if not args.output:
            return
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        tmp = f"{args.output}.tmp.{os.getpid()}"
        with open(tmp, "w") as fo:
            json.dump({**make_summary(complete), "per_problem": records}, fo, indent=2)
            fo.flush()
            os.fsync(fo.fileno())
        os.replace(tmp, args.output)

    def same_checkpoint_reference(prior, expected):
        if prior == expected:
            return True
        if not isinstance(prior, str) or not isinstance(expected, str):
            return False
        try:
            if os.path.samefile(prior, expected):
                return True
        except OSError:
            pass
        marker = f"{os.sep}outputs{os.sep}"
        prior_suffix = prior[prior.find(marker):] if marker in prior else None
        expected_suffix = expected[expected.find(marker):] if marker in expected else None
        return prior_suffix is not None and prior_suffix == expected_suffix

    if args.output and os.path.isfile(args.output):
        with open(args.output) as fi:
            prior = json.load(fi)
        for key, expected in output_identity.items():
            matches = (
                same_checkpoint_reference(prior.get(key), expected)
                if key == "ckpt"
                else prior.get(key) == expected
            )
            if not matches:
                raise RuntimeError(
                    f"cannot resume {args.output}: {key}={prior.get(key)!r}, expected {expected!r}"
                )
        records = prior.get("per_problem")
        if not isinstance(records, list) or [r.get("idx") for r in records] != list(range(len(records))):
            raise RuntimeError(f"cannot resume {args.output}: records are not a contiguous prefix")
        if len(records) > len(rows):
            raise RuntimeError(f"cannot resume {args.output}: {len(records)} records for {len(rows)} rows")
        if records and args.temperature > 0:
            raise RuntimeError("resuming sampled MATH-500 generation is unsupported; use temperature=0")
        n_correct = sum(int(r["correct"]) for r in records)
        n_multiblock = sum(int(r["multiblock"]) for r in records)
        kv_full = sum(int(r["full_cache_tokens"]) for r in records)
        kv_comp = sum(int(r["compressed_cache_tokens"] if not args.baseline else r["full_cache_tokens"])
                      for r in records)
        print(f"[math500] resuming {args.output} at {len(records)}/{len(rows)}", flush=True)

    g = torch.Generator(device=device).manual_seed(args.seed) if args.temperature > 0 else None
    for i, row in enumerate(rows[len(records):], start=len(records)):
        content = format_user_content(row["problem"], boxed_instruction_enabled())
        prompt_ids = tok(content, add_special_tokens=True)["input_ids"]
        if args.baseline:
            ids = torch.tensor([prompt_ids], device=device)
            with torch.no_grad():
                out = model.generate(ids, max_new_tokens=args.max_new_tokens,
                                     do_sample=args.temperature > 0, temperature=max(args.temperature, 1e-6),
                                     top_p=args.top_p, top_k=(args.top_k or None),
                                     pad_token_id=eos_id)
            gen_ids = out[0, len(prompt_ids):].tolist()
        else:
            gen = CompressGen(clm, device, raw_prompt_len=(len(prompt_ids) if clm.raw_prompt else 0), **gen_kwargs)
            gen_ids = gen.generate(prompt_ids, args.max_new_tokens, eos_id=eos_id,
                                   temperature=args.temperature, top_p=args.top_p, top_k=args.top_k, gen=g)
        text = tok.decode(gen_ids, skip_special_tokens=True)
        total_tokens = len(prompt_ids) + len(gen_ids)
        ended_with_eos = bool(gen_ids and eos_id is not None and gen_ids[-1] == eos_id)
        cache_total_tokens = total_tokens - int(ended_with_eos)
        multiblock = total_tokens > C
        # End-of-generation retained cache, including raw prompt and rolling tail when enabled.
        kv_full += cache_total_tokens
        kv_comp += cache_total_tokens if args.baseline else gen.store.length
        pred = extract_boxed_answer(text)
        if os.environ.get("DUMP_GENS"):                                       # diagnostic: is it truncation vs incoherence?
            print(f"[dump {i}] gen_tokens={len(gen_ids)} (max={args.max_new_tokens}) pred={pred!r} gold={row['answer']!r}\n"
                  f"  HEAD: {text[:400]!r}\n  TAIL: {text[-400:]!r}\n  has_boxed={'boxed' in text} "
                  f"has_solution={'begin_of_solution' in text}", flush=True)
        # Grade exactly as scripts/eval_math500_compresslm.py does (math-verify on the full text when USE_MATH_VERIFY=1
        # and the package is importable, legacy strict-boxed otherwise), so eviction baselines and the trained arms are
        # scored by one rule. Before this the HF path was legacy-only while the vLLM suite used math-verify.
        semantic = mv_grade_text(text, row["answer"], kind="math") if use_mv() else None
        ok = bool(semantic) if semantic is not None else grade(pred, row["answer"])
        n_correct += int(ok)
        n_multiblock += int(multiblock)
        records.append({"idx": i, "pred": pred, "gold": row["answer"], "correct": ok, "grader": ("mv" if semantic is not None else "legacy"),
                       "text": text,                                          # keep the completion: regradable offline
                       "prompt_tokens": len(prompt_ids), "gen_tokens": len(gen_ids),
                       "total_tokens": total_tokens, "multiblock": multiblock,
                       "full_cache_tokens": cache_total_tokens,
                       "compressed_cache_tokens": (None if args.baseline else gen.store.length)})
        write_output(complete=False)
        if (i + 1) % 10 == 0 or i == 0:
            print(f"[math500] {i+1}/{len(rows)}  acc={n_correct/(i+1):.3f}  "
                  f"multiblock={n_multiblock/(i+1):.3f}  last(pred={pred!r} gold={row['answer']!r} ok={ok})", flush=True)

    acc = n_correct / max(len(rows), 1)
    mbf = n_multiblock / max(len(rows), 1)
    summary = make_summary(complete=True)
    print(f"\n[math500] ==== SUMMARY ====")
    print(f"SHARD_RESULT shard={args.shard} correct={n_correct} n={len(rows)} multiblock={n_multiblock} kvfull={kv_full} kvcomp={kv_comp}", flush=True)  # mergeable
    print(f"[math500] mode={summary['mode']} n={len(rows)}  ACCURACY={acc:.4f} ({n_correct}/{len(rows)})")
    print(f"[math500] KV-COMPRESSION: nominal={C/max(1,Kc):.2f}x  effective={kv_full/max(1,kv_comp):.2f}x (full={kv_full} comp={kv_comp})")
    print(f"[math500] MULTI-BLOCK (>{C} tok) fraction = {mbf:.4f} ({n_multiblock}/{len(rows)})  "
          f"[single-block gens exercise NO compression]")
    if args.output:
        write_output(complete=True)
        print(f"[math500] wrote {args.output}", flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", action="store_true", help="run the CE-consistency validation gate (tiny CLM)")
    ap.add_argument("--math500", action="store_true", help="run MATH-500 generation eval")
    ap.add_argument("--ckpt", default=os.environ.get("CKPT", ""), help="CompressLM sft ckpt (model.pt)")
    ap.add_argument("--base", default=os.environ.get("BASE", ""), help="override base model name")
    ap.add_argument("--tokenizer", default=os.environ.get("TOKENIZER_DIR", ""), help="local tokenizer dir (offline-safe); default = base_name")
    ap.add_argument("--baseline", action="store_true", help="full-context causal baseline (no compression)")
    ap.add_argument("--evict-policy", default=os.environ.get("EVICT_POLICY", ""),
                    help="codes-free eviction baseline on a vanilla ckpt: recency|random|score (block-boundary, keep --keep-per-block)")
    ap.add_argument("--keep-per-block", type=int, default=int(os.environ.get("KEEP_PER_BLOCK", 128)))
    ap.add_argument("--evict-seed", type=int, default=int(os.environ.get("EVICT_SEED", os.environ.get("SEED", 0))))
    ap.add_argument("--data", default=os.environ.get("MATH_DATA",
                    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                 "data/math_eval/math500.jsonl")))
    ap.add_argument("--limit", type=int, default=int(os.environ.get("LIMIT", 0)))
    ap.add_argument("--shard", type=int, default=int(os.environ.get("SHARD", 0)))
    ap.add_argument("--nshard", type=int, default=int(os.environ.get("NSHARD", 1)))
    ap.add_argument("--max-new-tokens", type=int, default=int(os.environ.get("MAX_NEW_TOKENS", 3072)))
    ap.add_argument("--temperature", type=float, default=float(os.environ.get("TEMPERATURE", 0.0)))
    ap.add_argument("--top-p", type=float, default=float(os.environ.get("TOP_P", 0.95)))
    ap.add_argument("--top-k", type=int, default=int(os.environ.get("TOP_K", 0)))
    ap.add_argument("--seed", type=int, default=int(os.environ.get("SEED", 0)))
    ap.add_argument("--fp32", action="store_true", help="fp32 (V100/local); default bf16 (B200/A100)")
    ap.add_argument("--output", default=os.environ.get("OUTPUT", ""))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.gate:
        ok, _ = run_gate(device)
        sys.exit(0 if ok else 1)
    if args.math500:
        assert args.baseline or args.ckpt, "--math500 needs --ckpt (compressed) or --baseline"
        run_math500(args, device)
        return
    ap.error("pass --gate or --math500")


if __name__ == "__main__":
    main()
