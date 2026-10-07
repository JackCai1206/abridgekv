"""Pause-compression LM for FineWeb-Edu continue-pretraining (P setting), FAST + 3 objectives + L passes.

Block layout = [content(C), pause^1(K), ..., pause^L(K), optional replay_tail(T)]
  (block size C + L*K + T; S0 = N*(C+L*K+T)).
Stage-2 + inner-refine mask (FlexAttention, single forward):
  - content_i : own content (causal) + EARLIER blocks' final codes pause^L + an optional rolling
    original-KV tail.
  - pause_i^l : own content + own pause^{l-1} + EARLIER blocks' final codes.
    With retain_tail_kv, the final T own-content tokens are omitted because their original KVs remain available.
    Legacy code_exclude_replay_tail checkpoints retain their post-code token-replay semantics.
  - CONTENT is evicted for later blocks; only the K final codes (pause^L) carry forward.
Objectives (forward returns a dict): ce (compression LM), distill (KL full-ctx||compressed), copy (reconstruct
last block from its K final codes). L=1 reduces exactly to the plain single-pass compression.
"""
import os
import torch, torch.nn.functional as F
from torch import nn


def _roles(C, N, K, L, device, replay_tail=0):
    """Per-position roles: 0=content, 1..L=pause pass, L+1=post-summary tail replay."""
    per = C + L * K + replay_tail
    within = torch.arange(per, device=device).repeat(N)
    blk = torch.arange(N, device=device).repeat_interleave(per)
    role = torch.zeros(N * per, dtype=torch.long, device=device)
    p = within - C
    ispause = (p >= 0) & (p < L * K)
    role[ispause] = (p[ispause] // K) + 1                              # 1..L
    role[within >= C + L * K] = L + 1                                 # replay rows
    return blk, role


def _parse_kernel_options(text):
    """JSON ({"BLOCK_M":128,...}) or comma-free "BLOCK_M=128;BLOCK_N=64;num_warps=8" (sbatch --export splits on commas)."""
    text = text.strip()
    if text.startswith("{"):
        import json
        return json.loads(text)
    out = {}
    for item in text.split(";"):
        if item.strip():
            k, v = item.split("="); out[k.strip()] = int(v)
    return out


def _bitrev_code_offsets(K, C, device, span=None):
    """In-block RoPE offsets of the K code rows for code_positions='bitrev': row k -> floor(bitreverse_{log2 K}(k) * span / K),
    span = C by default. Any prefix of 2^m rows (a head quota) is spread evenly over [0, span), and the offsets of the first
    K' rows are the same for every K >= K' (power-of-two K; span >= K keeps offsets distinct). With a static retained tail
    (code_excluded_tail = T) the codes summarize only offsets [0, C - T), so span = C - T: no code shares a RoPE position
    with a retained raw token (C=512, T=16, K=256 -> offsets floor(1.9375 * rev), max 494)."""
    span = C if span is None else int(span)
    if span < K:
        raise ValueError(f"bitrev code span {span} < K={K}: offsets would collide")
    bits = K.bit_length() - 1
    k = torch.arange(K, device=device)
    rev = torch.zeros_like(k)
    for b in range(bits):
        rev |= ((k >> b) & 1) << (bits - 1 - b)
    return (rev * span) // K


def _ordered_to_dense_bool(num_blocks, indices):
    """Inverse of flex_attention._dense_to_ordered: (B,H,QB) counts + (B,H,QB,KVB) column ids -> bool grid."""
    kvb = indices.shape[-1]
    valid = torch.arange(kvb, device=indices.device).view(1, 1, 1, -1) < num_blocks.unsqueeze(-1)
    dense = torch.zeros(indices.shape, dtype=torch.bool, device=indices.device)
    dense.scatter_(-1, indices.long(), valid)
    return dense


def _content_abs_pos(C, N, K, L, device, replay_tail=0):
    """Per-S0-row (S0,): the ABSOLUTE content-token index (block*C + within-block offset) for content rows;
    replay rows map back to the original final T token positions; codes get N*C (>= any prompt_len).
    Used by raw-prompt and rolling-tail visibility tests."""
    per = C + L * K + replay_tail
    within = torch.arange(per, device=device).repeat(N)
    blk = torch.arange(N, device=device).repeat_interleave(per)
    iscontent = within < C
    cap = torch.full((N * per,), N * C, dtype=torch.long, device=device)
    cap[iscontent] = blk[iscontent] * C + within[iscontent]
    if replay_tail:
        isreplay = within >= C + L * K
        replay_offset = within[isreplay] - (C + L * K)
        cap[isreplay] = blk[isreplay] * C + (C - replay_tail) + replay_offset
    return cap


def _raw_tail_visible(role, blk, content_abs_pos, q, kv, raw_tail):
    """Whether content query q may see a raw token from the immediately preceding block.

    The tail is a rolling window: at position j in a new block, only the final
    raw_tail-j tokens from the previous block remain visible.
    """
    return ((role[q] == 0) & (role[kv] == 0) & (blk[kv] + 1 == blk[q])
            & (content_abs_pos[q] - content_abs_pos[kv] <= raw_tail))


def _retained_tail_visible(
    role,
    blk,
    content_abs_pos,
    q,
    kv,
    block_size,
    retain_tail_kv,
    retain_tail_policy,
):
    """Whether query q may see an original-KV tail (the final retain_tail_kv content rows) of an earlier block.
    rolling: content queries of the next block, sliding window; sticky: content queries of the next block, whole block;
    static: EVERY later-block query (content and code rows), forever -- the tail rows are permanent per-block state
    kept next to the block's codes (see docs/EXPERIMENT_QUEUE.md, 'static raw tail')."""
    if retain_tail_kv <= 0:
        return torch.zeros_like(q == kv)
    if retain_tail_policy == "rolling":
        return _raw_tail_visible(
            role, blk, content_abs_pos, q, kv, retain_tail_kv
        )
    source_tail = (
        content_abs_pos[kv] % block_size
        >= block_size - retain_tail_kv
    )
    if retain_tail_policy == "static":
        return (role[kv] == 0) & (blk[kv] < blk[q]) & source_tail
    return (
        (role[q] == 0)
        & (role[kv] == 0)
        & (blk[kv] + 1 == blk[q])
        & source_tail
    )


def _replay_tail_visible(role, blk, content_abs_pos, q, kv, replay_tail, replay_role):
    """Whether content query q may see a post-summary replay row from the preceding block."""
    return ((role[q] == 0) & (role[kv] == replay_role) & (blk[kv] + 1 == blk[q])
            & (content_abs_pos[q] - content_abs_pos[kv] <= replay_tail))


def _code_content_tail_excluded(
    role,
    blk,
    content_abs_pos,
    q,
    kv,
    block_size,
    replay_tail,
    replay_role,
):
    """Whether a code query must ignore an own-block exact tail carrier."""
    if replay_tail <= 0:
        return torch.zeros_like(q == kv)
    pause_q = (role[q] > 0) & (role[q] < replay_role)
    own_content = (role[kv] == 0) & (blk[kv] == blk[q])
    replayed_source = content_abs_pos[kv] % block_size >= block_size - replay_tail
    return pause_q & own_content & replayed_source


def _code_replay_tail_excluded(
    role,
    blk,
    content_abs_pos,
    q,
    kv,
    block_size,
    replay_tail,
    replay_role,
):
    """Backward-compatible name for legacy replay-tail callers."""
    return _code_content_tail_excluded(
        role,
        blk,
        content_abs_pos,
        q,
        kv,
        block_size,
        replay_tail,
        replay_role,
    )


class _QProjQRes(nn.Module):
    """Wraps a layer's q_proj to add a per-(pause-slot) learned QUERY residual at the pause rows (idea #4).
    A real submodule (NOT a forward hook) so torch.compile traces it and can't silently skip it. The residual is
    just a learned constant bias added to the pause-row queries — it does not touch the attention mask. ZERO-init
    => exact no-op at step 0 (added directly, so it still gets gradient and learns from step 0)."""
    def __init__(self, orig, LK, dq, pos, slot, S0):
        super().__init__()
        self.orig = orig                                          # original q_proj (nn.Linear); weights stay in place
        self.qres = nn.Parameter(torch.zeros(LK, dq))             # per pause-slot query residual
        self.register_buffer("pos", pos, persistent=False)        # pause-row indices (S0 layout)
        self.register_buffer("slot", slot, persistent=False)      # slot 0..LK-1 per pause row
        self.S0 = S0
    def forward(self, x):
        out = self.orig(x)                                        # (B, S, dq)
        if out.size(1) < self.S0:                                 # skip non-compress layouts (baseline/floor)
            return out
        res = out.new_zeros(out.size(1), out.size(2))
        res = res.index_copy(0, self.pos, self.qres[self.slot].to(out.dtype))
        return out + res                                          # content rows += 0; broadcast over batch


class _QProjCode(nn.Module):
    """Dedicated projection for the pause rows (any of q/k/v_proj; named for the query-side first use) (idea 1, 2026-09-14/26, query side): code tokens compute their
    attention query with their own W_q, initialised as a copy of the layer's q_proj (exact no-op at step 0), while
    content rows keep the original projection. The LM heads are then pulled less toward the "summarise this block"
    job. A real submodule (compile-safe), same pause-row bookkeeping as _QProjQRes. Serving is unchanged only for
    checkpoints that never enable it; the HF/vLLM code step would need the same swap (proto runs judge on held-out CE)."""
    def __init__(self, orig, pos, S0):
        super().__init__()
        self.orig = orig                                          # original q_proj, content rows
        self.code = nn.Linear(orig.in_features, orig.out_features, bias=orig.bias is not None)
        with torch.no_grad():
            self.code.weight.copy_(orig.weight)
            if orig.bias is not None: self.code.bias.copy_(orig.bias)
        self.register_buffer("pos", pos, persistent=False)        # pause-row indices (S0 layout)
        self.S0 = S0
    def forward(self, x):
        out = self.orig(x)                                        # (B, S, dq)
        if out.size(1) < self.S0:                                 # non-compress layouts (baseline/floor): untouched
            return out
        code = self.code(x.index_select(1, self.pos))             # (B, P, dq) queries of the pause rows
        return out.index_copy(1, self.pos, code.to(out.dtype))


def _preserve_norm_input_dtype(module, inputs, output):
    """Qwen3 RMSNorm's FP32 weight otherwise promotes BF16 Q/K back to FP32."""
    return output.to(inputs[0].dtype)


def _quota_key_allowed(code_slot, kv, head, query_head_quotas):
    """Keep non-code keys and only the first quota code slots for a query head."""
    clamped_kv = kv.clamp(max=code_slot.numel() - 1)
    slot = code_slot[clamped_kv]
    return (kv >= code_slot.numel()) | (slot < 0) | (
        slot < query_head_quotas[head]
    )


class CompressLM(nn.Module):
    def __init__(self, base_model, C, N, K, L=1, pause_xattn="full", pool_init=False, query_residual=False, code_q_heads=False,
                 code_kv_heads=False,
                 seg_pool=False, gated_seg_pool=False, raw_tail=0, replay_tail=0,
                 code_exclude_replay_tail=False, retain_tail_kv=0,
                 retain_tail_policy="rolling", head_code_quotas=None,
                 active_code_rows=None, code_positions="appended"):
        super().__init__()
        self.model = base_model
        self.model.config._attn_implementation = "flex_attention"
        self.C, self.N, self.K, self.L = C, N, K, L
        # CODE POSITIONS (RoPE): "appended" = raw layout index (block b spans C+K positions, codes after the content;
        # depends on K). "bitrev" = content keeps the vanilla numbering b*C+j and code row k sits INSIDE its block at
        # b*C + bitreverse_{log2 K}(k) * C / K, so any prefix of 2^m rows (a head quota) is evenly spread over the
        # block and the first 128 rows land on identical positions for every K >= 128 (K-independent layout).
        # Query and key of a code row share its position (standard RoPE), so code queries see the content after
        # their position at negative relative distance.
        if code_positions not in ("appended", "bitrev"):
            raise ValueError(f"code_positions must be 'appended' or 'bitrev', got {code_positions!r}")
        if code_positions == "bitrev" and (K & (K - 1) or C & (C - 1) or L != 1 or replay_tail or raw_tail):
            raise ValueError("code_positions='bitrev' needs power-of-two C and K, L=1 and no tails")
        self.code_positions = code_positions
        # ACTIVE CODE ROWS: with fixed head quotas, code slots >= max quota are invisible to every query in every
        # layer and carry no loss, so they need not be computed. K_act rows per block are placed in the sequence;
        # RoPE positions still follow the full-K layout (see _rope_position_ids), so the math is unchanged.
        self.K_act = int(active_code_rows) if active_code_rows else K
        if not 1 <= self.K_act <= K:
            raise ValueError(f"active_code_rows must be in [1, K={K}], got {active_code_rows}")
        if self.K_act != K and (pool_init or seg_pool or gated_seg_pool):
            raise ValueError("active_code_rows < K is not supported with pool_init/seg_pool")
        self._rope_pos_cache = {}
        if not 0 <= raw_tail <= C:
            raise ValueError(f"raw_tail must be in [0, C={C}], got {raw_tail}")
        if not 0 <= replay_tail <= C:
            raise ValueError(f"replay_tail must be in [0, C={C}], got {replay_tail}")
        if not 0 <= retain_tail_kv < C:
            raise ValueError(
                f"retain_tail_kv must be in [0, C={C}), got {retain_tail_kv}"
            )
        if sum(bool(value) for value in (raw_tail, replay_tail, retain_tail_kv)) > 1:
            raise ValueError(
                "raw_tail, replay_tail, and retain_tail_kv are mutually exclusive"
            )
        if retain_tail_policy not in ("rolling", "sticky", "static"):
            raise ValueError(
                "retain_tail_policy must be 'rolling', 'sticky' or 'static', got "
                f"{retain_tail_policy!r}"
            )
        if retain_tail_policy != "rolling" and not retain_tail_kv:
            raise ValueError(
                f"retain_tail_policy={retain_tail_policy!r} requires retain_tail_kv > 0"
            )
        if code_exclude_replay_tail and not 0 < replay_tail < C:
            raise ValueError(
                "code_exclude_replay_tail requires replay_tail in [1, C)"
            )
        self.raw_tail = raw_tail
        self.replay_tail = replay_tail
        self.retain_tail_kv = retain_tail_kv
        self.retain_tail_policy = retain_tail_policy
        self.content_tail_size = raw_tail or retain_tail_kv
        self.rolling_content_tail = (
            raw_tail
            or (
                retain_tail_kv
                if retain_tail_policy == "rolling"
                else 0
            )
        )
        self.code_exclude_replay_tail = bool(code_exclude_replay_tail)
        self.code_excluded_tail = (
            replay_tail if self.code_exclude_replay_tail else retain_tail_kv
        )
        self.replay_role = L + 1
        # within-pass code<->code attention: "none" (old: mutually-blind reads), "causal" (ordered codes),
        # "full" (bidirectional over the unordered code set — recommended; free in FlexAttention).
        assert pause_xattn in ("none", "causal", "full")
        self.pause_xattn = pause_xattn
        d = base_model.config.hidden_size
        std = getattr(base_model.config, "initializer_range", 0.02)
        self.pause = nn.Parameter(torch.empty(K, d).normal_(0, std))       # K learned code seeds (shared across passes)
        self.copy_token = nn.Parameter(torch.empty(d).normal_(0, std))
        # IDEA #7: Perceiver/Set-Transformer pooling init — a cheap single-head cross-attention pooler that makes each
        # code's INIT embedding CONTEXT-DEPENDENT: K DISTINCT learned queries (pool_q) attend the block's content
        # embeddings -> pooled content added to the per-slot learned bias (self.pause). pool_v ZERO-init => at step 0
        # pooled=0 => EXACT reduction to the generic-seed model, then it learns context-dependence as a residual.
        self.pool_init = pool_init
        # DETERMINISTIC segment-mean init: split each block's C content tokens into K contiguous segments and set
        # code k's seed = mean of segment k (+ learned per-slot bias). Guaranteed context-dependent, cannot collapse
        # (no params), and avoids the mean-washout that killed the learned pool (small K/C-sized segments preserve
        # content). Tests whether the pool null is failed-optimization (then this helps) or redundancy-with-forward-
        # attention (then this is null too, since codes already attend their block's content).
        self.seg_pool = seg_pool
        self.gated_seg_pool = gated_seg_pool
        if gated_seg_pool:
            self.seg_pool_gate = nn.Parameter(torch.zeros(K))
        if pool_init:
            self.pool_q = nn.Parameter(torch.empty(K, d).normal_(0, std))  # K distinct learned queries (per-slot specialization)
            self.pool_k = nn.Linear(d, d, bias=False)
            self.pool_v = nn.Linear(d, d, bias=False)
            nn.init.zeros_(self.pool_v.weight)                             # residual-from-zero: start == generic seeds, but
            # pool_v is OPTIMIZED DIRECTLY from step 0 (grad flows through the direct residual) so the pool self-activates.
            # (A single learned scalar gate g_pool init 0 was tried + REMOVED 2026-07-13: with ONE scalar it's a chicken-and-
            # egg -- pool params only learn if g_pool>0, but g_pool only grows if the still-random pool is useful -> traps
            # pool OFF -> false null (val_ce == ce exactly). Direct pool_v-zero-init has no such trap.)
        dev = next(base_model.parameters()).device
        blk, role = _roles(C, N, self.K_act, L, dev, replay_tail)
        self.register_buffer("blk", blk, persistent=False)
        self.register_buffer("role", role, persistent=False)
        self.register_buffer(
            "content_abs_pos",
            _content_abs_pos(C, N, self.K_act, L, dev, replay_tail),
            persistent=False,
        )
        content = role == 0
        self.register_buffer("loss_pos", content & torch.roll(content, -1), persistent=False)
        self.register_buffer("content_pos", content.nonzero(as_tuple=True)[0], persistent=False)  # all content rows (S0 layout, block order) for rep-distill
        pf = torch.tensor([k for k in range(N * C) if k % C != C - 1], device=dev, dtype=torch.long)
        self.register_buffer("predict_flat", pf, persistent=False)         # teacher-aligned content indices
        self.register_buffer("predict_block", pf // C, persistent=False)   # block index (0..N-1) of each predict position -> per-block loss
        self._bm, self._bm_copy = None, None
        self._quota_bm, self._quota_bm_copy = None, None
        self._quota_mask_dispatch = {}
        self.num_layers = len(self.model.model.layers)
        self.num_attention_heads = int(base_model.config.num_attention_heads)
        self.num_kv_heads = int(
            getattr(
                base_model.config,
                "num_key_value_heads",
                self.num_attention_heads,
            )
        )
        if self.num_attention_heads % self.num_kv_heads:
            raise ValueError(
                "num_attention_heads must be divisible by num_key_value_heads"
            )
        self.num_key_value_groups = (
            self.num_attention_heads // self.num_kv_heads
        )
        self.register_buffer(
            "head_code_quotas",
            torch.empty((0, 0), dtype=torch.long, device=dev),
            persistent=False,
        )
        self.register_buffer(
            "query_head_code_quotas",
            torch.empty((0, 0), dtype=torch.long, device=dev),
            persistent=False,
        )
        self._quota_hooks_installed = False
        self.set_head_code_quotas(head_code_quotas)
        for layer in self.model.model.layers:
            attn = layer.self_attn
            for name in ("q_norm", "k_norm"):
                norm = getattr(attn, name, None)
                if norm is not None and not getattr(norm, "_compresslm_dtype_guard", False):
                    norm.register_forward_hook(_preserve_norm_input_dtype)
                    norm._compresslm_dtype_guard = True
        # IDEA #4: per-(pause-slot, layer) learned QUERY residual. At every layer, the pause rows' query gets an
        # extra learned vector added AFTER q_proj: q_pause[slot,layer] = learned_q[slot,layer] + W_q.norm(x). Gives
        # each code a learned per-depth "attention program" (prefix-tuning-v2 on the query side). ZERO-init => starts
        # as an exact no-op (added directly, not gated -> learns from step 0, no deadzone). Applied via forward hooks
        # on each layer's q_proj; run with compile=0 (hooks under torch.compile are fragile).
        self.query_residual = query_residual
        self._S0 = int(N * (C + L * self.K_act + replay_tail))
        if query_residual:
            K = self.K_act
            per = C + L * K + replay_tail
            within = torch.arange(self._S0, device=dev) % per
            pmask = (within >= C) & (within < C + L * K)                   # pause rows only
            qpos = pmask.nonzero(as_tuple=True)[0]                         # S0-idx of pause rows
            qslot = (within[pmask] - C).long()                            # 0..L*K-1 per pause row
            for lyr in self.model.model.layers:                           # swap q_proj -> residual wrapper (each layer
                qp = lyr.self_attn.q_proj                                 # owns its own (L*K, dq) qres). compile-safe.
                lyr.self_attn.q_proj = _QProjQRes(qp, L * K, qp.out_features, qpos.clone(), qslot.clone(), self._S0)
        # IDEA 1 (query side): the pause rows read the block through their own q_proj copy per layer.
        self.code_q_heads = bool(code_q_heads)
        if code_q_heads:
            K = self.K_act
            per = C + L * K + replay_tail
            within = torch.arange(self._S0, device=dev) % per
            qpos = ((within >= C) & (within < C + L * K)).nonzero(as_tuple=True)[0]
            for lyr in self.model.model.layers:
                lyr.self_attn.q_proj = _QProjCode(lyr.self_attn.q_proj, qpos.clone(), self._S0)
        # IDEA 1 (key/value side): the pause rows WRITE their KV through their own k_proj/v_proj copies per layer, so the
        # summary a code row leaves for later blocks is produced by weights the content rows never use (same no-op init).
        self.code_kv_heads = bool(code_kv_heads)
        if code_kv_heads:
            K = self.K_act
            per = C + L * K + replay_tail
            within = torch.arange(self._S0, device=dev) % per
            kvpos = ((within >= C) & (within < C + L * K)).nonzero(as_tuple=True)[0]
            for lyr in self.model.model.layers:
                lyr.self_attn.k_proj = _QProjCode(lyr.self_attn.k_proj, kvpos.clone(), self._S0)
                lyr.self_attn.v_proj = _QProjCode(lyr.self_attn.v_proj, kvpos.clone(), self._S0)

    @property
    def has_head_code_quotas(self):
        return self.head_code_quotas.numel() > 0

    @property
    def head_code_quotas_are_dense(self):
        return self._head_code_quotas_are_dense

    def set_skip_unused_codes(self, flag=True):
        """SKIP-WHEN-UNUSED (mixture-of-depths for code rows). At layer l, code slot j with j >= max_h quota[l][h] is
        invisible as a key to every query (the quota mask), so the layer's only effect on it is its own residual update.
        With this on, such a slot passes through layer l unchanged (h_out = h_in), so a server need only run layer l on
        the rows that layer keeps: code-row compute becomes sum_l quota_l instead of K_max * num_layers. Off by default;
        a model trained without it sees a small change, which is exactly what probe_skip_unused.py measures."""
        self.skip_unused_codes = bool(flag)
        if flag and not getattr(self, "_skip_hooks_installed", False):
            for layer_index, layer in enumerate(self.model.model.layers):
                layer.register_forward_hook(self._skip_unused_hook(layer_index), with_kwargs=True)
            self._skip_hooks_installed = True

    def _skip_unused_mask(self, layer_index, S, device):
        """(S,) bool: True at code positions whose slot this layer does not keep (None if the layer keeps all)."""
        cache = self.__dict__.setdefault("_skip_mask_cache", {})
        key = (layer_index, S, str(device), self.N)
        if key in cache:
            return cache[key]
        mask = None
        if self.has_head_code_quotas:
            used = max(self.head_code_quotas_tuple[layer_index])
            C, L, K, T = self.C, self.L, self.K_act, self.replay_tail
            if used < K:
                per = C + L * K + T
                S0 = self.N * per
                pos = torch.arange(S, device=device)
                within = pos % per
                is_code = (pos < S0) & (within >= C) & (within < C + L * K)
                slot = (within - C) % K
                mask = is_code & (slot >= used)
        cache[key] = mask
        return mask

    def _skip_unused_hook(self, layer_index):
        def hook(module, args, kwargs, output):
            if not getattr(self, "skip_unused_codes", False):
                return output
            h_in = args[0] if args else kwargs["hidden_states"]
            h_out = output[0] if isinstance(output, tuple) else output
            mask = self._skip_unused_mask(layer_index, h_out.shape[1], h_out.device)
            if mask is None:
                return output
            new = torch.where(mask.view(1, -1, 1), h_in.to(h_out.dtype), h_out)
            return (new,) + tuple(output[1:]) if isinstance(output, tuple) else new
        return hook

    def _install_quota_hooks(self):
        if self._quota_hooks_installed:
            return
        for layer_index, layer in enumerate(self.model.model.layers):
            layer.self_attn.register_forward_pre_hook(
                self._quota_attention_mask_hook(layer_index),
                with_kwargs=True,
            )
        self._quota_hooks_installed = True

    def set_head_code_quotas(self, head_code_quotas):
        """Set fixed physical-KV-head quotas without adding persistent state."""
        device = self.pause.device
        if head_code_quotas is None:
            physical = torch.empty((0, 0), dtype=torch.long, device=device)
            expanded = torch.empty((0, 0), dtype=torch.long, device=device)
        else:
            try:
                raw_quotas = torch.as_tensor(
                    head_code_quotas,
                    device=device,
                )
            except (TypeError, ValueError, RuntimeError) as exc:
                raise ValueError(
                    "head_code_quotas must be a rectangular 2D integer array"
                ) from exc
            expected = (self.num_layers, self.num_kv_heads)
            if tuple(raw_quotas.shape) != expected:
                raise ValueError(
                    f"head_code_quotas must have shape {expected}, got "
                    f"{tuple(raw_quotas.shape)}"
                )
            if raw_quotas.dtype == torch.bool or (
                raw_quotas.is_floating_point()
                and bool((raw_quotas != raw_quotas.round()).any().item())
            ):
                raise ValueError("head_code_quotas must contain integers")
            physical = raw_quotas.to(dtype=torch.long)
            if bool(((physical <= 0) | (physical > self.K)).any().item()):
                raise ValueError(
                    f"head_code_quotas must be in [1, {self.K}]"
                )
            if bool((physical > self.K_act).any().item()):
                raise ValueError(
                    f"head_code_quotas exceed active_code_rows={self.K_act}"
                )
            expanded = physical.repeat_interleave(
                self.num_key_value_groups,
                dim=1,
            )
            self._install_quota_hooks()
        self.head_code_quotas = physical
        self.query_head_code_quotas = expanded
        self.head_code_quotas_tuple = tuple(
            tuple(int(value) for value in row)
            for row in physical.detach().cpu().tolist()
        )
        self._head_code_quotas_are_dense = bool(
            self.head_code_quotas_tuple
            and all(
                value == self.K
                for row in self.head_code_quotas_tuple
                for value in row
            )
        )
        self._quota_bm, self._quota_bm_copy = None, None
        self._quota_mask_dispatch = {}

    def _quota_attention_mask_hook(self, layer_index):
        def hook(module, args, kwargs):
            attention_mask = kwargs.get("attention_mask")
            dispatch = self._quota_mask_dispatch.get(id(attention_mask))
            if dispatch is not None and dispatch[0] is attention_mask:
                kwargs["attention_mask"] = dispatch[1][layer_index]
            return args, kwargs
        return hook

    def _mask_mod(self, device, with_copy, layer_index=None):
        C, N, K, L, T = self.C, self.N, self.K_act, self.L, self.replay_tail
        per = C + L * K + T
        S0 = N * per
        blk, role = _roles(C, N, K, L, device, T)
        cap = _content_abs_pos(C, N, K, L, device, T)
        within = torch.arange(S0, device=device) % per
        code_slot = torch.full(
            (S0,),
            -1,
            dtype=torch.long,
            device=device,
        )
        is_code = (within >= C) & (within < C + L * K)
        code_slot[is_code] = (within[is_code] - C) % K
        last_final_lo = (N - 1) * per + C + (L - 1) * K                    # first col of last block's pause^L
        Scopy = S0 + 1 + C if with_copy else S0
        query_quotas = (
            self.query_head_code_quotas[layer_index]
            if layer_index is not None
            else None
        )

        xa = self.pause_xattn
        def mm(b, h, q, kv):
            qc, kc = q.clamp(max=S0 - 1), kv.clamp(max=S0 - 1)
            in_cq, in_ck = q >= S0, kv >= S0
            qr, kr = role[qc], role[kc]; qb, kb = blk[qc], blk[kc]
            causal = kv <= q
            content_vis = (kr == 0) & (kb == qb)                          # content: own block only
            earlier_final = (kr == L) & (kb < qb)                         # earlier blocks' final codes
            pause_q = (qr > 0) & (qr <= L)
            refine = pause_q & (kr == qr - 1) & (kb == qb)                # pause^l <- own pause^{l-1}
            replay_q = qr == self.replay_role
            replay_own_final = replay_q & (kr == L) & (kb == qb)
            replay_causal = replay_q & (kr == self.replay_role) & (kb == qb)
            raw_tail_vis = _raw_tail_visible(
                role, blk, cap, qc, kc, self.raw_tail
            )
            retained_tail_vis = _retained_tail_visible(
                role,
                blk,
                cap,
                qc,
                kc,
                C,
                self.retain_tail_kv,
                self.retain_tail_policy,
            )
            replay_tail_vis = _replay_tail_visible(
                role, blk, cap, qc, kc, T, self.replay_role
            )
            base = (~in_cq) & (~in_ck) & causal & (
                content_vis | earlier_final | refine | replay_own_final
                | replay_causal | raw_tail_vis | retained_tail_vis
                | replay_tail_vis
            )
            if xa != "none":                                             # code<->code, same block & pass (incl self)
                samepass = (~in_cq) & (~in_ck) & pause_q & (kr == qr) & (kb == qb)
                base = base | (samepass & causal if xa == "causal" else samepass)  # "full" = bidirectional
            if self.code_excluded_tail:
                base = base & ~_code_content_tail_excluded(
                    role,
                    blk,
                    cap,
                    qc,
                    kc,
                    C,
                    self.code_excluded_tail,
                    self.replay_role,
                )
            # copy region q>=S0: attend ONLY last block's K final codes + causal within copy
            code_kv = (kv >= last_final_lo) & (kv < last_final_lo + K) & (~in_ck)
            copy = in_cq & (code_kv | (in_ck & (kv <= q)))
            allowed = base | copy
            if query_quotas is not None:
                allowed = allowed & _quota_key_allowed(
                    code_slot,
                    kv,
                    h,
                    query_quotas,
                )
            return allowed
        return mm, Scopy

    def _mask(self, device, with_copy, layer_index=None):
        from torch.nn.attention.flex_attention import create_block_mask
        mm, sequence_length = self._mask_mod(
            device,
            with_copy,
            layer_index,
        )
        heads = self.num_attention_heads if layer_index is not None else None
        return create_block_mask(
            mm,
            B=None,
            H=heads,
            Q_LEN=sequence_length,
            KV_LEN=sequence_length,
            device=device,
            _compile=True,
        )

    def blockmask(self, device, with_copy):
        if with_copy:
            if self._bm_copy is None: self._bm_copy = self._mask(device, True)
            return self._bm_copy
        if self._bm is None: self._bm = self._mask(device, False)
        return self._bm



    def _simple_mask_config(self):
        """L=1, no replay/raw/window tails, no copy, full code<->code attention: the mask is a closed-form function of
        (q, kv). A STATIC retained tail is also closed-form (earlier blocks' last T content rows visible to everyone,
        own-block tail hidden from the codes), so it stays on the fast path."""
        static_only = self.retain_tail_kv == 0 or self.retain_tail_policy == "static"
        return (
            os.environ.get("COMPRESSLM_ARITH_MASK", "1") != "0"
            and self.L == 1 and self.replay_tail == 0 and self.raw_tail == 0
            and static_only and self.code_excluded_tail == self._static_tail
            and self.pause_xattn == "full"
        )

    @property
    def _static_tail(self):
        """T of the static retained tail (0 for the window policies / no tail)."""
        return self.retain_tail_kv if self.retain_tail_policy == "static" else 0

    def _arith_mask_mod(self, device, prompt_len, layer_index):
        """Same visibility as _mask_mod/_mask_rawprompt_mod for the simple config, computed with integer
        arithmetic instead of role/blk/cap gathers -- ~4x cheaper inside the flex partial-block kernels."""
        C, K, N = self.C, self.K_act, self.N
        per = C + K
        S0 = N * per
        quotas = (
            self.query_head_code_quotas[layer_index].to(device)
            if layer_index is not None else None
        )
        pl = prompt_len
        T = self._static_tail

        def mm(b, h, q, kv):
            qb = q // per
            qw = q - qb * per
            kb = kv // per
            kw = kv - kb * per
            k_content = kw < C
            same_blk = kb == qb
            vis = (k_content & same_blk) | ((~k_content) & (kb < qb))        # own content | earlier blocks' codes
            if T:
                vis = vis | (k_content & (kb < qb) & (kw >= C - T))           # earlier blocks' static tails (all queries)
            if pl is not None:
                vis = vis | (k_content & (kb * C + kw < pl[b]))               # raw prompt visible to everyone
            vis = (kv <= q) & vis
            vis = vis | ((qw >= C) & (~k_content) & same_blk)                 # code<->code, same block (full)
            if T:
                vis = vis & ~((qw >= C) & same_blk & k_content & (kw >= C - T))   # codes never see their own block's tail
            if quotas is not None:
                vis = vis & (k_content | ((kw - C) < quotas[h]))              # per-head code-slot quota
            return vis
        return mm, S0

    def _mask_mods_for(self, device, prompt_len, layer_index):
        if self._simple_mask_config():
            return self._arith_mask_mod(device, prompt_len, layer_index)
        if prompt_len is not None:
            return self._mask_rawprompt_mod(device, prompt_len, layer_index)
        return self._mask_mod(device, False, layer_index)

    def _fast_blockmasks_supported(self, with_copy):
        from torch.nn.attention.flex_attention import _DEFAULT_SPARSE_BLOCK_SIZE as BS
        C, K, T = self.C, self.K_act, self.replay_tail
        return (
            os.environ.get("COMPRESSLM_FAST_BLOCKMASK", "1") != "0"
            and not with_copy
            and C % BS == 0 and K % BS == 0 and T % BS == 0
        )

    def _quota_block_ok(self, S0, BS, quotas, device):
        """Per (query head, kv block): is the quota term all-true / any-true over the block? (H, KVB) bools."""
        C, K, L, T = self.C, self.K_act, self.L, self.replay_tail
        per = C + L * K + T
        kvb = (S0 + BS - 1) // BS
        start = torch.arange(kvb, device=device) * BS
        within = start % per
        is_code = (within >= C) & (within < C + L * K)
        slot0 = ((within - C) % K).clamp(min=0)
        q = quotas.view(-1, 1)
        not_code = (~is_code).view(1, -1)
        any_ok = not_code | (slot0.view(1, -1) < q)
        full_ok = not_code | (slot0.view(1, -1) + BS <= q)
        return any_ok, full_ok

    def _fast_layer_blockmasks(self, device, B, prompt_len=None):
        """All layers' quota BlockMasks from one quota-free H=1 block mask + analytic per-head block classes.
        Exactly the block classification create_block_mask would produce for the full mask (every kv block is
        aligned to one code region and the base mask is constant over a code block's slots)."""
        from torch.nn.attention.flex_attention import (
            BlockMask, create_block_mask, _dense_to_ordered, _DEFAULT_SPARSE_BLOCK_SIZE as BS,
        )
        mm_base, S0 = self._mask_mods_for(device, prompt_len, None)
        Bm = B if prompt_len is not None else None
        base = create_block_mask(mm_base, B=Bm, H=None, Q_LEN=S0, KV_LEN=S0, device=device, _compile=True)
        partial = _ordered_to_dense_bool(base.kv_num_blocks, base.kv_indices)
        full = _ordered_to_dense_bool(base.full_kv_num_blocks, base.full_kv_indices)
        anyb = partial | full
        kvb = anyb.shape[-1]
        by_row, masks = {}, []
        for layer_index in range(self.num_layers):
            key = self.head_code_quotas_tuple[layer_index]
            if key not in by_row:
                quotas = self.query_head_code_quotas[layer_index].to(device)
                any_ok, full_ok = self._quota_block_ok(S0, BS, quotas, device)
                full_h = full & full_ok.view(1, -1, 1, kvb)
                partial_h = (anyb & any_ok.view(1, -1, 1, kvb)) & ~full_h
                pn, pi = _dense_to_ordered(partial_h)
                fn, fi = _dense_to_ordered(full_h)
                mm_h, _ = self._mask_mods_for(device, prompt_len, layer_index)
                by_row[key] = BlockMask.from_kv_blocks(
                    pn, pi, fn, fi, BLOCK_SIZE=BS, mask_mod=mm_h, seq_lengths=(S0, S0),
                )
            masks.append(by_row[key])
        return masks

    def layer_blockmasks(self, device, with_copy):
        if not self.has_head_code_quotas or self.head_code_quotas_are_dense:
            base = self.blockmask(device, with_copy)
            return [base] * self.num_layers
        cache_name = "_quota_bm_copy" if with_copy else "_quota_bm"
        cached = getattr(self, cache_name)
        if cached is None and self._fast_blockmasks_supported(with_copy):
            cached = self._fast_layer_blockmasks(device, None)
            setattr(self, cache_name, cached)
        if cached is None:
            by_quota_row = {}
            cached = []
            for layer_index in range(self.num_layers):
                quota_key = self.head_code_quotas_tuple[layer_index]
                if quota_key not in by_quota_row:
                    by_quota_row[quota_key] = self._mask(
                        device,
                        with_copy,
                        layer_index,
                    )
                cached.append(by_quota_row[quota_key])
            setattr(self, cache_name, cached)
        return cached

    def _mask_rawprompt_mod(self, device, prompt_len, layer_index=None):
        """POST-TRAINING mask: identical to _mask (no copy region) but ADDS raw-prompt visibility — any query may
        attend to the actual prompt tokens (abs content pos < prompt_len[b]), so the prompt is NOT compressed;
        only the response is block-compressed. prompt_len (B,) is per-example -> the mask is batch-dependent
        (B=batch, not None) and rebuilt per microbatch (not cached). Purely ADDITIVE to content_vis/earlier_final."""
        C, N, K, L, T = self.C, self.N, self.K_act, self.L, self.replay_tail
        per = C + L * K + T
        S0 = N * per
        blk, role, cap = self.blk, self.role, self.content_abs_pos
        within = torch.arange(S0, device=device) % per
        code_slot = torch.full(
            (S0,),
            -1,
            dtype=torch.long,
            device=device,
        )
        is_code = (within >= C) & (within < C + L * K)
        code_slot[is_code] = (within[is_code] - C) % K
        pl = prompt_len
        xa = self.pause_xattn
        query_quotas = (
            self.query_head_code_quotas[layer_index]
            if layer_index is not None
            else None
        )
        def mm(b, h, q, kv):
            qr, kr = role[q], role[kv]; qb, kb = blk[q], blk[kv]
            causal = kv <= q
            content_vis = (kr == 0) & (kb == qb)                          # own block content
            earlier_final = (kr == L) & (kb < qb)                         # earlier blocks' final codes (compressed carry)
            pause_q = (qr > 0) & (qr <= L)
            refine = pause_q & (kr == qr - 1) & (kb == qb)                # pause^l <- own pause^{l-1}
            replay_q = qr == self.replay_role
            replay_own_final = replay_q & (kr == L) & (kb == qb)
            replay_causal = replay_q & (kr == self.replay_role) & (kb == qb)
            prompt_raw = (kr == 0) & (cap[kv] < pl[b])                    # RAW prompt tokens visible to everyone
            raw_tail_vis = _raw_tail_visible(
                role, blk, cap, q, kv, self.raw_tail
            )
            retained_tail_vis = _retained_tail_visible(
                role,
                blk,
                cap,
                q,
                kv,
                C,
                self.retain_tail_kv,
                self.retain_tail_policy,
            )
            replay_tail_vis = _replay_tail_visible(
                role, blk, cap, q, kv, T, self.replay_role
            )
            base = causal & (
                content_vis | earlier_final | refine | replay_own_final
                | replay_causal | prompt_raw | raw_tail_vis
                | retained_tail_vis | replay_tail_vis
            )
            if xa != "none":                                             # code<->code, same block & pass (incl self)
                samepass = pause_q & (kr == qr) & (kb == qb)
                base = base | (samepass & causal if xa == "causal" else samepass)
            if self.code_excluded_tail:
                base = base & ~_code_content_tail_excluded(
                    role,
                    blk,
                    cap,
                    q,
                    kv,
                    C,
                    self.code_excluded_tail,
                    self.replay_role,
                )
            if query_quotas is not None:
                base = base & _quota_key_allowed(
                    code_slot,
                    kv,
                    h,
                    query_quotas,
                )
            return base
        return mm, S0

    def _mask_rawprompt(
        self,
        device,
        prompt_len,
        B,
        layer_index=None,
    ):
        from torch.nn.attention.flex_attention import create_block_mask
        mm, sequence_length = self._mask_rawprompt_mod(
            device,
            prompt_len,
            layer_index,
        )
        heads = self.num_attention_heads if layer_index is not None else None
        return create_block_mask(
            mm,
            B=B,
            H=heads,
            Q_LEN=sequence_length,
            KV_LEN=sequence_length,
            device=device,
            _compile=True,
        )

    def rawprompt_layer_blockmasks(self, device, prompt_len, B):
        if not self.has_head_code_quotas or self.head_code_quotas_are_dense:
            base = self._mask_rawprompt(device, prompt_len, B)
            return [base] * self.num_layers
        if self._fast_blockmasks_supported(False):
            return self._fast_layer_blockmasks(device, B, prompt_len)
        by_quota_row = {}
        layer_masks = []
        for layer_index in range(self.num_layers):
            quota_key = self.head_code_quotas_tuple[layer_index]
            if quota_key not in by_quota_row:
                by_quota_row[quota_key] = self._mask_rawprompt(
                    device,
                    prompt_len,
                    B,
                    layer_index,
                )
            layer_masks.append(by_quota_row[quota_key])
        return layer_masks

    def _embed_for_plain_forward(self, ids):
        """Token embeddings for the plain (non-compressed) HF forward, cast to the active autocast dtype.
        With FP32 master weights under BF16 autocast, embed_tokens returns FP32; transformers >= 5 then builds
        FP32 RoPE cos/sin from the hidden dtype and (no longer casting back) promotes Q/K to FP32 while V stays
        BF16 -> SDPA raises 'query, key, and value to have the same dtype'. The compressed path already casts
        its inputs_embeds (see forward); the baseline/teacher paths must do the same."""
        emb = self.model.model.embed_tokens(ids)
        if torch.is_autocast_enabled(emb.device.type):
            emb = emb.to(torch.get_autocast_dtype(emb.device.type))
        return emb

    @torch.no_grad()
    def full_ctx_logits(self, content_ids):
        B, N, C = content_ids.shape
        emb = self._embed_for_plain_forward(content_ids.reshape(B, N * C))
        return self.model(inputs_embeds=emb, use_cache=False).logits

    def _lm_ce(self, hid_flat, tgt_flat, chunk):
        if chunk <= 0:
            return F.cross_entropy(self.model.lm_head(hid_flat).float(), tgt_flat)
        tot = hid_flat.new_zeros((), dtype=torch.float32)
        for i in range(0, hid_flat.size(0), chunk):
            tot = tot + F.cross_entropy(self.model.lm_head(hid_flat[i:i + chunk]).float(),
                                        tgt_flat[i:i + chunk], reduction="sum")
        return tot / hid_flat.size(0)

    def _masked_lm_ce(self, hid_flat, tgt_flat, mask_flat, chunk):
        den = mask_flat.sum().clamp_min(1.0)
        if chunk <= 0:
            ce = F.cross_entropy(self.model.lm_head(hid_flat).float(), tgt_flat, reduction="none")
            return (ce * mask_flat).sum() / den
        tot = hid_flat.new_zeros((), dtype=torch.float32)
        for i in range(0, hid_flat.size(0), chunk):
            ce = F.cross_entropy(self.model.lm_head(hid_flat[i:i + chunk]).float(),
                                 tgt_flat[i:i + chunk], reduction="none")
            tot = tot + (ce * mask_flat[i:i + chunk]).sum()
        return tot / den

    def _masked_ce_kl_chunked(self, hid_flat, tgt_flat, mask_flat, teacher_rows, teacher_head, temp, chunk):
        """Response-masked CE + KL(teacher || student) at temperature `temp`, row-chunked and activation-checkpointed.

        teacher_rows: (R, d) frozen-teacher final hidden states aligned row-for-row with hid_flat (full-context
        forward gathered at the same predict positions); teacher_head: the teacher's lm_head. Each chunk recomputes
        both lm_heads in backward, so peak memory is O(chunk x vocab) instead of O(R x vocab) (R ~ 1e5 rows at
        micro 8 x 16K tokens; vocab 152K -> ~80 GB fp32 if materialized). Returns (ce_mean, kl_mean) over mask.
        """
        import torch.utils.checkpoint as _ckpt
        if teacher_rows.shape[0] != hid_flat.shape[0]:
            raise ValueError(f"teacher_rows {tuple(teacher_rows.shape)} misaligned with student rows {tuple(hid_flat.shape)}")
        den = mask_flat.sum().clamp_min(1.0)
        T = float(temp)
        lm_head = self.model.lm_head

        def _chunk(h, t, m, tr):
            logits = lm_head(h).float()
            ce = F.cross_entropy(logits, t, reduction="none")
            with torch.no_grad():
                tlog = teacher_head(tr).float()
                tp = F.softmax(tlog / T, -1)
                tlp = F.log_softmax(tlog / T, -1)
            slp = F.log_softmax(logits / T, -1)
            kl = (tp * (tlp - slp)).sum(-1) * (T * T)
            return (ce * m).sum(), (kl * m).sum()

        n = hid_flat.size(0)
        step = n if chunk <= 0 else chunk
        ce_tot = hid_flat.new_zeros((), dtype=torch.float32)
        kl_tot = hid_flat.new_zeros((), dtype=torch.float32)
        for i in range(0, n, step):
            sl = slice(i, i + step)
            if torch.is_grad_enabled():
                c, k = _ckpt.checkpoint(_chunk, hid_flat[sl], tgt_flat[sl], mask_flat[sl], teacher_rows[sl],
                                        use_reentrant=False)
            else:
                c, k = _chunk(hid_flat[sl], tgt_flat[sl], mask_flat[sl], teacher_rows[sl])
            ce_tot = ce_tot + c
            kl_tot = kl_tot + k
        return ce_tot / den, kl_tot / den

    def _per_token_lm_ce(self, hid_flat, tgt_flat, chunk):
        if chunk <= 0:
            return F.cross_entropy(
                self.model.lm_head(hid_flat).float(), tgt_flat, reduction="none"
            )
        parts = []
        for i in range(0, hid_flat.size(0), chunk):
            parts.append(F.cross_entropy(
                self.model.lm_head(hid_flat[i:i + chunk]).float(),
                tgt_flat[i:i + chunk],
                reduction="none",
            ))
        return torch.cat(parts)

    def _perblock_ce(self, hid_flat, tgt_flat, blk_idx, nblk, reps):
        """FREE per-block CE: one lm_head over the SAME predict positions, grouped by block index.
        Returns (mean_ce, per_block_ce[nblk]). blk_idx (P,) is the block of each predict position for ONE
        sample; reps tiles it over the flattened (reps*P,) rows. -> quantify loss growth across block position."""
        ce = F.cross_entropy(self.model.lm_head(hid_flat).float(), tgt_flat, reduction="none")   # (reps*P,)
        blk = blk_idx.repeat(reps)
        s = ce.new_zeros(nblk); c = ce.new_zeros(nblk)
        s.scatter_add_(0, blk, ce); c.scatter_add_(0, blk, torch.ones_like(ce))
        return ce.mean(), s / c.clamp_min(1.0)


    def _rope_position_ids(self, N, with_copy, device):
        """(1, S) RoPE positions of the reduced (K_act rows/block) layout, numbered as in the full-K layout."""
        key = (int(N), bool(with_copy), str(device))
        cached = self._rope_pos_cache.get(key)
        if cached is not None:
            return cached
        C, K, Ka, L, T = self.C, self.K, self.K_act, self.L, self.replay_tail
        per_full, per_act = C + L * K + T, C + L * Ka + T
        S0 = N * per_act
        idx = torch.arange(S0, device=device)
        within, blk = idx % per_act, idx // per_act
        is_code = (within >= C) & (within < C + L * Ka)
        p = within[is_code] - C
        if self.code_positions == "bitrev":                               # K-independent: content b*C+j, codes spread in-block
            pos = blk * C + within                                        # content rows (within < C)
            pos[is_code] = blk[is_code] * C + _bitrev_code_offsets(K, C, device, span=C - self.code_excluded_tail)[p]
            if with_copy:
                pos = torch.cat([pos, N * C + torch.arange(1 + C, device=device)])
        else:                                                             # "appended": raw index of the full-K layout
            pos = blk * per_full + within                                 # content rows (within < C)
            pos[is_code] = blk[is_code] * per_full + C + (p // Ka) * K + (p % Ka)
            is_replay = within >= C + L * Ka
            pos[is_replay] = blk[is_replay] * per_full + C + L * K + (within[is_replay] - C - L * Ka)
            if with_copy:
                pos = torch.cat([pos, N * per_full + torch.arange(1 + C, device=device)])
        pos = pos.view(1, -1)
        self._rope_pos_cache[key] = pos
        return pos

    def set_N(self, N):
        """Rebuild the N-dependent (non-persistent) buffers + mask cache for a new block count. Params
        (base model, pause, pool) are N-independent -> enables VARIABLE-LENGTH batches (bucket by N)."""
        if N == self.N:
            return
        self.N = N
        dev = self.pause.device
        C, K, L, T = self.C, self.K_act, self.L, self.replay_tail
        blk, role = _roles(C, N, K, L, dev, T)
        content = role == 0
        pf = torch.tensor([k for k in range(N * C) if k % C != C - 1], device=dev, dtype=torch.long)
        self.blk, self.role = blk, role
        self.content_abs_pos = _content_abs_pos(C, N, K, L, dev, T)
        self.loss_pos = content & torch.roll(content, -1)
        self.content_pos = content.nonzero(as_tuple=True)[0]
        self.predict_flat = pf
        self.predict_block = pf // C
        self._S0 = int(N * (C + L * K + T))
        self._bm, self._bm_copy = None, None                               # invalidate FlexAttention mask cache
        self._quota_bm, self._quota_bm_copy = None, None
        self._quota_mask_dispatch = {}

    def baseline_loss(self, content_ids, ce_chunk=0, ctx_blocks=0, block_ce=False, resp_mask=None,
                      per_token=False, include_cross_block=False):
        """Plain causal LM baseline (no compression), CE on the SAME within-block predict positions as the
        compressed forward. ctx_blocks = #blocks sharing one attention context: N (default) = full 2048-ctx
        CEILING; 1 = 512-ctx FLOOR (no cross-block info); any divisor of N = intermediate. Same tokens/batch,
        only the context length (how blocks are grouped into sequences) changes."""
        B, N, C = content_ids.shape
        G = ctx_blocks if ctx_blocks else N
        flat = content_ids.reshape(B * (N // G), G * C)                              # each seq = G contiguous blocks
        out = self.model.model(inputs_embeds=self._embed_for_plain_forward(flat),
                               use_cache=False).last_hidden_state                   # causal over G*C
        ar = torch.arange(G * C, device=flat.device)
        if include_cross_block:
            pos = ar[:-1]                                                  # include C-1 -> next block's position 0
        else:
            pos = ar[(ar % C) != (C - 1)]                                  # within-block next-token positions
        hid = out[:, pos, :].reshape(-1, out.size(-1))
        tgt = flat[:, pos + 1].reshape(-1)
        if block_ce:                                                       # per-block (0..G-1); ceiling uses G=N
            m, bc = self._perblock_ce(hid, tgt, pos // C, G, flat.size(0))
            return {"ce": m, "block_ce": bc}
        if resp_mask is not None:                                          # SFT: masked CE on response tokens (full-ctx baseline)
            rm = resp_mask.reshape(flat.shape)[:, pos + 1].reshape(-1).float()
            if per_token:                                                  # DIAG: per-token CE tagged by (block, pos-in-block)
                ce = self._per_token_lm_ce(hid, tgt, ce_chunk)
                reps = flat.size(0)
                if include_cross_block:
                    target = pos + 1
                    return {
                        "ce_pt": ce,
                        "tmask": rm,
                        "blk": (target // C).repeat(reps),
                        "pos": (target % C).repeat(reps),
                        "supervised": ((pos % C) != C - 1).repeat(reps),
                    }
                return {"ce_pt": ce, "tmask": rm, "blk": (pos // C).repeat(reps), "pos": (pos % C).repeat(reps)}
            return {"ce": self._masked_lm_ce(hid, tgt, rm, ce_chunk)}
        return {"ce": self._lm_ce(hid, tgt, ce_chunk)}

    def forward(self, content_ids, distill_lambda=0.0, copy_lambda=0.0, distill_temp=1.0,
                ce_chunk=0, baseline=False, ctx_blocks=0, teacher_logits=None,
                teacher_hidden=None, block_ce=False, resp_mask=None, raw_prompt=False,
                blockmask=None, per_token=False, include_cross_block=False,
                boundary_recovery_width=0, boundary_recovery_weight=1.0,
                teacher_rows=None, teacher_head=None):
        if baseline:                       # ctx_blocks: N (or 0)=ceiling(2048-ctx), 1=floor(512-ctx), divisor of N=intermediate
            return self.baseline_loss(
                content_ids, ce_chunk, ctx_blocks, block_ce=block_ce,
                resp_mask=resp_mask, per_token=per_token,
                include_cross_block=include_cross_block,
            )
        B, N, C = content_ids.shape
        K, L, T = self.K_act, self.L, self.replay_tail
        dev = content_ids.device
        emb = self.model.model.embed_tokens
        tok_e = emb(content_ids)                                          # (B,N,C,d)
        d = tok_e.size(-1); mdt = tok_e.dtype
        code_source_e = (
            tok_e[:, :, : C - self.code_excluded_tail, :]
            if self.code_excluded_tail
            else tok_e
        )
        code_source_len = code_source_e.size(2)
        if self.seg_pool or self.gated_seg_pool:                          # fine-grained segment-mean init
            seg = code_source_len // K
            if seg >= 1:
                tm = code_source_e[:, :, :K * seg, :].reshape(B, N, K, seg, d).mean(3)
            else:                                                         # K>C: fewer tokens than codes -> broadcast block mean
                tm = code_source_e.mean(2, keepdim=True).expand(B, N, K, d)
            if self.gated_seg_pool:
                tm = tm * self.seg_pool_gate.to(mdt).view(1, 1, K, 1)
            pause_base = self.pause.to(mdt).view(1, 1, K, d) + tm.to(mdt)
        elif self.pool_init:                                              # #7 context-dependent code seeds (LEARNED pool)
            sc = torch.einsum('kd,bncd->bnkc', self.pool_q.to(mdt), self.pool_k(code_source_e)) * (d ** -0.5)
            pooled = torch.einsum(
                'bnkc,bncd->bnkd',
                sc.float().softmax(-1).to(mdt),
                self.pool_v(code_source_e),
            )
            pause_base = self.pause.to(mdt).view(1, 1, K, d) + pooled     # generic per-slot bias + content pool (=self.pause @ init)
        else:
            pause_base = self.pause[:K].to(mdt).view(1, 1, K, d).expand(B, N, K, d)
        pause_e = pause_base.repeat(1, 1, L, 1)                           # (B,N,L*K,d)
        replay_e = tok_e[:, :, C - T:, :] if T else tok_e[:, :, :0, :]
        x = torch.cat([tok_e, pause_e, replay_e], dim=2).reshape(
            B, N * (C + L * K + T), d
        )
        with_copy = copy_lambda > 0
        S0 = N * (C + L * K + T)
        if with_copy:
            copy_in = torch.cat([self.copy_token.to(mdt).view(1, 1, d).expand(B, 1, d),
                                 emb(content_ids[:, -1, :])], dim=1)
            x = torch.cat([x, copy_in], dim=1)
        if torch.is_autocast_enabled(x.device.type):
            # Embedding lookups retain the FP32 master-weight dtype. Cast their
            # outputs so Qwen RoPE and Q/K/V all use the active BF16/FP16 dtype.
            x = x.to(torch.get_autocast_dtype(x.device.type))
        need_rep = teacher_hidden is not None                             # #REP-DISTILL: match per-layer hidden states
        layer_masks = None
        if self.has_head_code_quotas:
            if blockmask is not None:
                raise ValueError(
                    "an external blockmask cannot encode layer/head quotas; "
                    "omit blockmask when head_code_quotas are enabled"
                )
            if raw_prompt and resp_mask is not None and not with_copy:
                rmf = resp_mask.reshape(B, N * C)
                prompt_len = (rmf.cumsum(1) == 0).sum(1).to(torch.int32)
                layer_masks = self.rawprompt_layer_blockmasks(
                    dev,
                    prompt_len,
                    B,
                )
            else:
                layer_masks = self.layer_blockmasks(dev, with_copy)
            bmask = layer_masks[0]
            self._quota_mask_dispatch = {
                id(bmask): (bmask, layer_masks)
            }
        elif blockmask is not None:                                       # prebuilt outside the compiled graph (train loop) -> compile-safe
            bmask = blockmask
        elif raw_prompt and resp_mask is not None and not with_copy:      # POST-TRAIN eager path (eval): keep the prompt uncompressed
            rmf = resp_mask.reshape(B, N * C)
            prompt_len = (rmf.cumsum(1) == 0).sum(1).to(torch.int32)       # leading zeros = prompt length (pad trails the response)
            bmask = self._mask_rawprompt(dev, prompt_len, B)
        else:
            bmask = self.blockmask(dev, with_copy)
        pos_ids = None
        if self.K_act != self.K or self.code_positions != "appended":    # explicit RoPE positions (reduced layout / bitrev)
            pos_ids = self._rope_position_ids(N, with_copy, dev)           # (1, S): transformers mis-handles (B, S) ids for B > 1
        extra = {}
        ko = os.environ.get("COMPRESSLM_FLEX_KERNEL_OPTIONS")
        if ko:
            extra["kernel_options"] = _parse_kernel_options(ko)             # flex tile sizes (numerics-neutral)
        mo = self.model.model(inputs_embeds=x, attention_mask=bmask, position_ids=pos_ids,
                              use_cache=False, output_hidden_states=need_rep, **extra)
        out = mo.last_hidden_state

        if include_cross_block:
            pred_flat = torch.arange(N * C - 1, device=dev)
            content_hid = out[:, :S0, :][:, self.content_pos, :]
            hid = content_hid[:, :-1, :].reshape(-1, d)
            tgt = content_ids.reshape(B, N * C)[:, pred_flat + 1].reshape(-1)
        else:
            pred_flat = self.predict_flat
            hid = out[:, :S0, :][:, self.loss_pos, :].reshape(-1, d)
            tgt = content_ids.reshape(B, N * C)[:, pred_flat + 1].reshape(-1)
        if block_ce:                                                       # FREE per-block loss (eval-only); reuses this forward
            m, bc = self._perblock_ce(hid, tgt, pred_flat // C, N, B)
            return {"ce": m, "block_ce": bc}
        if resp_mask is not None:                                          # SFT: masked CE on RESPONSE tokens only
            tmask = resp_mask.reshape(B, N * C)[:, pred_flat + 1].reshape(-1).float()  # 1 where target is a response token
            if per_token and include_cross_block:
                ce = self._per_token_lm_ce(hid, tgt, ce_chunk)
                target = pred_flat + 1
                return {
                    "ce_pt": ce,
                    "tmask": tmask,
                    "blk": (target // C).repeat(B),
                    "pos": (target % C).repeat(B),
                    "supervised": ((pred_flat % C) != C - 1).repeat(B),
                }
            if boundary_recovery_width > 0 and boundary_recovery_weight != 1.0:
                target = pred_flat + 1
                recovery = (
                    (target // C > 0)
                    & (target % C >= 1)
                    & (target % C <= boundary_recovery_width)
                ).repeat(B)
                tmask = tmask * torch.where(
                    recovery,
                    tmask.new_tensor(boundary_recovery_weight),
                    tmask.new_tensor(1.0),
                )
            if teacher_rows is not None and not per_token:                 # FROZEN-teacher KD: chunked + checkpointed (never
                ce, kl = self._masked_ce_kl_chunked(                       # materializes full-vocab logits for all rows at once)
                    hid, tgt, tmask, teacher_rows, teacher_head, distill_temp, ce_chunk)
                return {"ce": ce, "distill": kl}
            if not per_token and teacher_logits is None:
                return {"ce": self._masked_lm_ce(hid, tgt, tmask, ce_chunk)}
            logits = self.model.lm_head(hid).float()
            ce = F.cross_entropy(logits, tgt, reduction="none")
            if per_token:                                                  # DIAG: per-token CE tagged by (block, pos-in-block)
                return {"ce_pt": ce, "tmask": tmask,
                        "blk": self.predict_block.repeat(B), "pos": (self.predict_flat % C).repeat(B)}
            den = tmask.sum().clamp_min(1.0)
            outd = {"ce": (ce * tmask).sum() / den}
            if teacher_logits is not None:                                 # RESPONSE-masked self-distill to the full-ctx teacher
                Tt = distill_temp                                         # (closes the compression gap: compressed logits -> full-ctx logits)
                kl = F.kl_div(F.log_softmax(logits / Tt, -1),
                              F.softmax(teacher_logits.float() / Tt, -1), reduction="none").sum(-1) * (Tt * Tt)
                outd["distill"] = (kl * tmask).sum() / den
            return outd
        if distill_lambda > 0:
            logits = self.model.lm_head(hid).float()
            losses = {"ce": F.cross_entropy(logits, tgt)}
            # teacher = FROZEN pretrained base's full-context logits at the predict positions, computed OUTSIDE
            # this (compiled/DDP) forward and passed in -> its weights never update, never in state_dict/optimizer.
            # Fallback (unit test, no teacher passed): self-teacher = current model full-ctx, detached (no_grad).
            if teacher_logits is None:
                with torch.no_grad():
                    teacher_logits = self.full_ctx_logits(content_ids)[:, pred_flat, :].reshape(-1, logits.size(-1))
            T = distill_temp
            losses["distill"] = F.kl_div(F.log_softmax(logits / T, -1),
                                         F.softmax(teacher_logits.float() / T, -1), reduction="batchmean") * (T * T)
        else:
            losses = {"ce": self._lm_ce(hid, tgt, ce_chunk)}
        if with_copy:
            losses["copy"] = self._lm_ce(out[:, S0:S0 + C, :].reshape(-1, d), content_ids[:, -1, :].reshape(-1), ce_chunk)
        if need_rep:
            # match student per-layer content hiddens to the frozen full-ctx teacher's, RMS-normed (scale-free).
            # teacher_hidden: (Lt, B, N*C, d), layers 1..Lt; student mo.hidden_states[li+1] at content_pos.
            rep = out.new_zeros(())
            for li in range(teacher_hidden.shape[0]):
                hs = mo.hidden_states[li + 1][:, self.content_pos, :].float()
                ht = teacher_hidden[li].float()
                hs = hs / hs.pow(2).mean(-1, keepdim=True).clamp_min(1e-12).sqrt()
                ht = ht / ht.pow(2).mean(-1, keepdim=True).clamp_min(1e-12).sqrt()
                rep = rep + (hs - ht).pow(2).mean()
            losses["rep"] = rep / teacher_hidden.shape[0]
        return losses


if __name__ == "__main__":
    import os, sys
    os.environ.setdefault("HF_HUB_OFFLINE", "0")
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from transformers import AutoModelForCausalLM
    dev = "cuda:0"; C, N, K = 32, 4, 16
    m = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M", torch_dtype=torch.bfloat16).to(dev)
    for Lp in (1, 2):
        net = CompressLM(m, C, N, K, L=Lp).to(dev)
        ids = torch.randint(0, 40000, (2, N, C), device=dev)
        Ls = net(ids, distill_lambda=1.0, copy_lambda=1.0, distill_temp=2.0)
        (Ls["ce"] + Ls["distill"] + Ls["copy"]).backward()
        print(f"L={Lp} S0={N*(C+Lp*K)} ok:", {k: round(v.item(), 3) for k, v in Ls.items()},
              "pause.grad", round(net.pause.grad.norm().item(), 2))
