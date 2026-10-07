"""Productionized variable-length reasoning SFT for pause-code compression — DDP-8 + torch.compile +
FlexAttention, mirroring pretrain_run.py's speed/precision setup. Keeps the SFT semantics of sft_run.py:
buckets prepped (prompt,response) pairs by block-count N, response-masked CE, warm-start from a compressed
CompressLM checkpoint (--ckpt), ARM=compressed|baseline. Adds, over sft_run.py:
  * DDP (torchrun --nproc_per_node=8): world_size x MICRO x ACCUM = effective batch (--eff_batch auto-picks ACCUM).
  * torch.compile + FlexAttention + fp32-master/bf16-autocast (same as pretrain_run.py).
  * TOKEN-WEIGHTED grad-accum (fixes a bias in sft_run.py): the response-masked CE is a masked MEAN, so
    averaging per-microbatch means (sft_run.py's `loss/ACCUM`) up-weights token-sparse microbatches. Here we
    pre-count the exact response-target tokens per microbatch (from data, no forward), all-reduce the global
    total T, and scale each microbatch loss by (n_tok * world / T) so the accumulated + DDP-averaged gradient
    EXACTLY equals the gradient of the single-big-batch masked mean (verified in scripts/test_sft_accum_parity.py).
  * --save/--save_every/--save_dir (periodic ckpt step<N>/model.pt) + --max_resp_tok early-stop (matched-token HP sweep).
  * Optional cosine LR schedule driven by RESPONSE TOKENS (so every BS arm follows the same LR(token) curve).

PADDING: bucket-by-N + last-block padding is KEPT (padding waste measured at ~8.4% on reasoning_pairs.pkl).
Padding-free document packing was deferred — see DEFERRED note at the bottom of this file for why + what it needs.

Launch (cluster, DDP-8):  torchrun --nproc_per_node=8 abridgekv/sft_ddp.py --arm compressed --ckpt <K128_distill/model.pt> ...
Local CPU smoke:          SFT_FORCE_CPU=1 python abridgekv/sft_ddp.py --base HuggingFaceTB/SmolLM2-135M --compile 0 --C 64 --K 8 ...
"""
import os
import gc, sys, math, time, argparse, contextlib, re, shutil, zipfile, json
from datetime import timedelta
os.environ.setdefault("HF_HUB_OFFLINE", os.environ.get("HF_OFFLINE", "0"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import torch, torch.nn.functional as F, torch.distributed as dist
from collections import defaultdict
from abridgekv.data_stream import CanonicalStream, pack_draws, parse_stream_spec
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoModelForCausalLM
from abridgekv.compress_forward import CompressLM
from abridgekv.lora import (
    DEFAULT_TARGET_MODULES,
    configure_lora_trainability,
    merged_state_dict,
    remap_full_state_for_lora,
    trainable_state_dict,
)
from abridgekv.muon import build_muon_adamw
from abridgekv.sft_quota_utils import (
    load_head_code_quotas,
    rebucket_pairs,
    validate_head_code_quotas,
)

torch._dynamo.config.cache_size_limit = 64          # variable-N -> one compiled graph per bucket (~6); raise the guard


def atomic_torch_save(payload, path):
    """Publish a checkpoint only after its bytes are fully written."""
    parent = os.path.dirname(path) or "."
    os.makedirs(parent, exist_ok=True)
    tmp = f"{path}.tmp"
    marker = f"{path}.complete"
    marker_tmp = f"{marker}.tmp"
    try:
        with open(tmp, "wb") as f:
            torch.save(payload, f)
            f.flush()
            os.fsync(f.fileno())
        if os.path.exists(marker):
            os.remove(marker)
        os.replace(tmp, path)
        with open(marker_tmp, "w", encoding="ascii") as f:
            f.write("complete\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(marker_tmp, marker)
    finally:
        for incomplete in (tmp, marker_tmp):
            if os.path.exists(incomplete):
                os.remove(incomplete)


def torch_checkpoint_complete(path):
    if os.path.isfile(f"{path}.complete"):
        return True
    try:
        with zipfile.ZipFile(path) as checkpoint:
            return bool(checkpoint.infolist())
    except (OSError, zipfile.BadZipFile):
        return False


def predict_positions(N, C, device, include_cross_block=False):
    """Next-token predictor positions, optionally including C-1 -> next-block offset 0."""
    ar = torch.arange(N * C, device=device)
    if include_cross_block:
        return ar[:-1]
    return ar[(ar % C) != (C - 1)]


def resp_target_tokens(rm, N, C, include_cross_block=False):
    """EXACT count of response-token CE targets in this microbatch — matches the forward's masked denom
    (resp_mask at predictor+1). Pure function of the data (NO model forward) so we can pre-sum the global
    token total before any backward, enabling exact token-weighted grad-accum."""
    pf = predict_positions(N, C, rm.device, include_cross_block)
    return int(rm.reshape(rm.size(0), N * C)[:, pf + 1].sum().item())


def resp_target_weight(
    rm,
    N,
    C,
    include_cross_block=False,
    recovery_width=0,
    recovery_weight=1.0,
):
    pf = predict_positions(N, C, rm.device, include_cross_block)
    mask = rm.reshape(rm.size(0), N * C)[:, pf + 1].float()
    if recovery_width <= 0 or recovery_weight == 1.0:
        return float(mask.sum())
    target = pf + 1
    recovery = (
        (target // C > 0)
        & (target % C >= 1)
        & (target % C <= recovery_width)
    )
    weights = torch.where(
        recovery,
        mask.new_tensor(recovery_weight),
        mask.new_tensor(1.0),
    )
    return float((mask * weights).sum())



def log_mem(tag, log):
    """Milestone GPU-memory line. Four OOMs on H100 80 GB never reached a training step, so the only memory
    evidence was the size of the failing allocation; these print during construction and inside the first
    micro-batch, so the breakdown exists even when the run dies before step 1."""
    if not torch.cuda.is_available():
        return
    a = torch.cuda.memory_allocated() / 2**30
    r = torch.cuda.memory_reserved() / 2**30
    p = torch.cuda.max_memory_allocated() / 2**30
    log(f"[mem] {tag:28s} allocated {a:6.2f} GiB | reserved {r:6.2f} GiB | peak {p:6.2f} GiB")


def log_cuda_census(log, topn=20, _devtype="cuda"):
    """What is actually ON the card, grouped by (shape, dtype). Four H100 OOMs reported only the size of the
    FAILING allocation (1.16-2.32 GiB against ~78 GiB already in use), which says nothing about the occupant;
    micro-batch 1 used as much as micro-batch 2, so the occupant is not activations. gc only reaches tensors
    something still references (params, grads, optimizer state, caches) -- the gap between the census total and
    memory_allocated() is the autograd graph's saved tensors, which is exactly the number we could not separate."""
    if _devtype == "cuda" and not torch.cuda.is_available():   # _devtype="cpu" makes this testable off-GPU
        return
    import gc, collections
    seen, groups = set(), collections.defaultdict(lambda: [0, 0])
    for o in gc.get_objects():
        try:
            t = o if torch.is_tensor(o) else (o.data if hasattr(o, "data") and torch.is_tensor(o.data) else None)
            if t is None or t.device.type != _devtype or t.data_ptr() == 0 or t.data_ptr() in seen:
                continue
            seen.add(t.data_ptr())
            g = groups[(tuple(t.shape), str(t.dtype).replace("torch.", ""))]
            g[0] += t.numel() * t.element_size(); g[1] += 1
        except Exception:
            continue
    total = sum(g[0] for g in groups.values())
    alloc = torch.cuda.memory_allocated()
    log(f"[census] {len(seen)} live CUDA tensors, {total / 2**30:.2f} GiB reachable | "
        f"allocated {alloc / 2**30:.2f} GiB | reserved {torch.cuda.memory_reserved() / 2**30:.2f} GiB | "
        f"unreachable (autograd graph + allocator) {(alloc - total) / 2**30:.2f} GiB")
    for (shape, dt), (nbytes, count) in sorted(groups.items(), key=lambda kv: -kv[1][0])[:topn]:
        log(f"[census]   {nbytes / 2**30:7.2f} GiB  x{count:<5d} {dt:9s} {list(shape)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=os.environ.get("BASE", "HuggingFaceTB/SmolLM2-1.7B"))
    ap.add_argument("--arm", default=os.environ.get("ARM", "compressed"), choices=["compressed", "baseline"])
    ap.add_argument("--ckpt", default=os.environ.get("CKPT", ""))          # warm-start CompressLM state_dict
    ap.add_argument("--pairs", default=os.environ.get("PAIRS", "data/reasoning_pairs.pkl"))
    ap.add_argument("--data_limit", type=int, default=int(os.environ.get("DATA_LIMIT", "0")))  # >0: use first N train ex (data-size sweep)
    ap.add_argument("--C", type=int, default=int(os.environ.get("C", "512"))); ap.add_argument("--K", type=int, default=int(os.environ.get("K", "128")))
    ap.add_argument("--L", type=int, default=1); ap.add_argument("--pause_xattn", default="full", choices=["none", "causal", "full"])
    ap.add_argument("--head_code_quotas", default=os.environ.get("HEAD_CODE_QUOTAS", ""))
    ap.add_argument("--code_positions", default=os.environ.get("CODE_POSITIONS", "appended"))  # appended (raw layout index) | bitrev (K-independent in-block spread)
    ap.add_argument("--code_q_heads", type=int, default=int(os.environ.get("CODE_Q_HEADS", "0")))  # 1 = pause rows use their own q_proj copy per layer (idea 1, query side)
    ap.add_argument("--code_kv_heads", type=int, default=int(os.environ.get("CODE_KV_HEADS", "0")))  # 1 = pause rows write KV through their own k/v_proj copies per layer (idea 1, key/value side)
    ap.add_argument("--skip_unused_codes", type=int, default=int(os.environ.get("SKIP_UNUSED_CODES", "0")))  # code slot j skips layer l when j >= max quota[l] (CompressLM.set_skip_unused_codes)
    ap.add_argument("--active_code_rows", default=os.environ.get("ACTIVE_CODE_ROWS", "off"))  # auto=max quota; off=K (default)
    ap.add_argument("--head_code_quota_budget", default=os.environ.get("HEAD_CODE_QUOTA_BUDGET", ""))
    ap.add_argument("--raw_prompt", type=int, default=int(os.environ.get("RAW_PROMPT", "0")))  # 1 = keep the PROMPT uncompressed (post-train); compress only the response
    ap.add_argument("--raw_tail", type=int, default=int(os.environ.get("RAW_TAIL", "0")))      # rolling raw-token bridge across compression boundaries
    ap.add_argument("--replay_tail", type=int, default=int(os.environ.get("REPLAY_TAIL", "0")))  # replay final T tokens after summary codes
    ap.add_argument("--retain_tail_kv", type=int, default=int(os.environ.get("RETAIN_TAIL_KV", "0")))  # retain original final-T KVs; codes cannot attend them
    ap.add_argument("--retain_tail_policy", default=os.environ.get("RETAIN_TAIL_POLICY", "rolling"), choices=["rolling", "sticky", "static"])  # rolling window / whole-next-block lifetime / permanent per-block tail rows
    ap.add_argument("--code_exclude_replay_tail", type=int, default=int(os.environ.get("CODE_EXCLUDE_REPLAY_TAIL", "0")))  # codes summarize only content not duplicated by replay
    ap.add_argument("--distill_lambda", type=float, default=float(os.environ.get("DISTILL_LAMBDA", "0")))  # >0: distill compressed->FROZEN full-ctx teacher (response-masked KL) to close the gap
    ap.add_argument("--distill_temp", type=float, default=float(os.environ.get("DISTILL_TEMP", "1.0")))
    ap.add_argument("--distill_ce_weight", type=float, default=float(os.environ.get("DISTILL_CE_WEIGHT", "1.0")))  # weight on the hard-label CE when distilling (0 => pure KL to the teacher)
    ap.add_argument("--teacher_ckpt", default=os.environ.get("TEACHER_CKPT", ""))  # FROZEN full-ctx teacher (fully-SFT'd baseline); empty => deprecated online self-distill
    ap.add_argument("--nmax", type=int, default=int(os.environ.get("NMAX", "8"))); ap.add_argument("--evaln", type=int, default=int(os.environ.get("EVALN", "128")))
    ap.add_argument("--micro", type=int, default=int(os.environ.get("MICRO", "8")))       # per-GPU microbatch (from the finder)
    ap.add_argument("--accum", type=int, default=int(os.environ.get("ACCUM", "0")))       # 0 => derive from --eff_batch
    ap.add_argument("--eff_batch", type=int, default=int(os.environ.get("EFF_BATCH", "128")))  # target effective batch (seqs)
    ap.add_argument("--lr", type=float, default=float(os.environ.get("LR", "1e-4")))
    ap.add_argument("--warmup_frac", type=float, default=0.03)                            # fraction of the token/step budget spent warming up
    ap.add_argument("--lr_floor", type=float, default=0.1)                                # cosine/wsd decay to lr_floor*peak
    ap.add_argument("--lr_schedule", default="cosine", choices=["cosine", "constant", "wsd"])
    ap.add_argument("--decay_frac", type=float, default=float(os.environ.get("DECAY_FRAC", "0.2")))  # wsd: final fraction of budget spent decaying to lr_floor
    ap.add_argument("--save_opt_every", type=int, default=int(os.environ.get("SAVE_OPT_EVERY", "0")))  # >0: periodic ckpts at this step interval ALSO carry optimizer state (two-tier: model-only ckpts stay cheap)
    ap.add_argument("--rewarmup_tok", type=int, default=int(os.environ.get("REWARMUP_TOK", "30000000")))  # resumed WITHOUT optimizer state -> ramp LR over this many resp tokens (never cold-resume at full LR)
    ap.add_argument("--max_steps", type=int, default=int(os.environ.get("STEPS", "400")))
    ap.add_argument("--max_resp_tok", type=int, default=int(os.environ.get("MAX_RESP_TOK", "0")))  # >0: stop + drive LR by global resp tokens
    ap.add_argument("--optimizer", default=os.environ.get("OPTIMIZER", "adamw"), choices=["adamw", "zero_adamw", "muon"])
    ap.add_argument("--autocast", type=int, default=int(os.environ.get("AUTOCAST", "1")))  # bf16 autocast on B200; 0 on CPU/V100 (fp32; flex q/k/v dtype quirk)
    ap.add_argument("--compile", type=int, default=int(os.environ.get("COMPILE", "1")))
    ap.add_argument("--grad_ckpt", type=int, default=int(os.environ.get("GRAD_CKPT", "0")))  # HF activation checkpointing (long seqs / big model)
    ap.add_argument("--ce_chunk", type=int, default=int(os.environ.get("CE_CHUNK", "0")))  # chunk lm_head+CE rows to cap eager peak memory
    ap.add_argument("--cross_block_loss", type=int, default=int(os.environ.get("CROSS_BLOCK_LOSS", "0")))  # supervise C-1 -> next-block offset 0
    ap.add_argument("--boundary_recovery_width", type=int, default=int(os.environ.get("BOUNDARY_RECOVERY_WIDTH", "0")))  # upweight target offsets 1..R in blocks after block 0
    ap.add_argument("--boundary_recovery_weight", type=float, default=float(os.environ.get("BOUNDARY_RECOVERY_WEIGHT", "1.0")))
    ap.add_argument("--seg_pool", type=int, default=int(os.environ.get("SEG_POOL", "0")))  # content-dependent segment-mean code initialization
    ap.add_argument("--gated_seg_pool", type=int, default=int(os.environ.get("GATED_SEG_POOL", "0")))  # zero-init per-slot gate on segment means
    ap.add_argument("--pool_init", type=int, default=int(os.environ.get("POOL_INIT", "0")))  # learned content pool, zero-init residual
    ap.add_argument("--train_code_init_only", type=int, default=int(os.environ.get("TRAIN_CODE_INIT_ONLY", "0")))
    ap.add_argument("--code_init_lr", type=float, default=float(os.environ.get("CODE_INIT_LR", "1e-2")))
    ap.add_argument("--lora_rank", type=int, default=int(os.environ.get("LORA_RANK", "0")))
    ap.add_argument("--lora_alpha", type=float, default=float(os.environ.get("LORA_ALPHA", "0")))
    ap.add_argument("--lora_dropout", type=float, default=float(os.environ.get("LORA_DROPOUT", "0")))
    ap.add_argument("--lora_targets", default=os.environ.get("LORA_TARGETS", ",".join(DEFAULT_TARGET_MODULES)))
    ap.add_argument("--data_stream", default=os.environ.get("DATA_STREAM", "native"))  # native | canonical[:VWxVMxVA]: hardware-independent stream (see data_stream.py)
    ap.add_argument("--data_skip_microbatches", type=int, default=int(os.environ.get("DATA_SKIP_MICROBATCHES", "-1")))  # resume: sampler microbatches (per rank) to skip; -1 = ckpt counter, else step*accum
    ap.add_argument("--reset_progress", type=int, default=int(os.environ.get("RESET_PROGRESS", "0")))  # load weights as a new branch, ignoring saved step/tokens/optimizer
    ap.add_argument("--save", default=os.environ.get("SAVE", ""))
    ap.add_argument("--save_every", type=int, default=int(os.environ.get("SAVE_EVERY", "0")))
    ap.add_argument("--eval_every", type=int, default=int(os.environ.get("EVAL_EVERY", os.environ.get("SAVE_EVERY", "0"))))
    ap.add_argument("--save_dir", default=os.environ.get("SAVE_DIR", ""))
    ap.add_argument("--n_keep", type=int, default=int(os.environ.get("N_KEEP", "0")))      # retain newest N periodic step dirs; 0 = all
    ap.add_argument("--log_every", type=int, default=int(os.environ.get("LOG_EVERY", "20")))
    ap.add_argument("--seed", type=int, default=int(os.environ.get("SEED", "0")))
    a = ap.parse_args()

    # ---- distributed / device setup (mirror pretrain_run.py; gloo+CPU fallback for local validation) ----
    ddp = int(os.environ.get("WORLD_SIZE", 1)) > 1
    force_cpu = os.environ.get("SFT_FORCE_CPU", "0") == "1" or not torch.cuda.is_available()
    if ddp:
        dist.init_process_group("gloo" if force_cpu else "nccl", timeout=timedelta(minutes=60))  # uneven per-rank work (canonical stream)
        rank, world, lrank = dist.get_rank(), dist.get_world_size(), int(os.environ.get("LOCAL_RANK", 0))
        if force_cpu:
            dev = "cpu"
        else:
            torch.cuda.set_device(lrank); dev = f"cuda:{lrank}"
        checkpoint_group = (
            dist.new_group(backend="gloo", timeout=timedelta(hours=2))
            if a.optimizer == "zero_adamw" and not force_cpu
            else dist.group.WORLD
        )
    else:
        rank, world, lrank = 0, 1, 0
        dev = "cpu" if force_cpu else "cuda:0"
        checkpoint_group = None
    is0 = rank == 0
    def log(*x):
        if is0: print(*x, flush=True)

    if dev == "cpu":
        # CPU has no inductor FlexAttention kernel — build the block mask EAGER and DISABLE dynamo so the
        # flex path runs deterministic eager on every rank (an errors-then-fallback compile desyncs the DDP
        # gloo collectives). LOCAL-VALIDATION ONLY; never triggers on the CUDA cluster (dynamo/compile stay on).
        torch._dynamo.config.disable = True
        import torch.nn.attention.flex_attention as _fa
        _ocbm = _fa.create_block_mask
        _fa.create_block_mask = lambda *ar, **kw: _ocbm(*ar, **{**kw, "_compile": False})

    torch.manual_seed(a.seed)
    is_base = (a.arm == "baseline")
    quota_spec = a.head_code_quotas.strip()
    quota_budget = str(a.head_code_quota_budget).strip()
    if quota_spec and is_base:
        raise ValueError("head-code quotas are not valid for the baseline arm")
    if quota_spec and a.compile:
        raise ValueError("head-code quotas require --compile 0")
    rp = bool(a.raw_prompt) and not is_base   # raw-prompt only meaningful for the compressed arm (baseline is full-ctx)
    raw_tail = a.raw_tail if not is_base else 0
    replay_tail = a.replay_tail if not is_base else 0
    retain_tail_kv = a.retain_tail_kv if not is_base else 0
    retain_tail_policy = (
        a.retain_tail_policy
        if retain_tail_kv
        else "rolling"
    )
    code_exclude_replay_tail = bool(a.code_exclude_replay_tail) and not is_base
    cross_block_loss = bool(a.cross_block_loss)
    C, K, L = a.C, a.K, a.L
    if sum(bool(value) for value in (a.seg_pool, a.gated_seg_pool, a.pool_init)) > 1:
        raise ValueError(
            "--seg_pool, --gated_seg_pool, and --pool_init are mutually exclusive"
        )
    seg_pool = bool(a.seg_pool) and not is_base
    gated_seg_pool = bool(a.gated_seg_pool) and not is_base
    pool_init = bool(a.pool_init) and not is_base
    lora_on = a.lora_rank > 0
    lora_alpha = a.lora_alpha or (2 * a.lora_rank)
    lora_targets = tuple(
        target.strip() for target in a.lora_targets.split(",")
        if target.strip()
    )
    if a.lora_rank < 0:
        raise ValueError(f"lora_rank must be nonnegative, got {a.lora_rank}")
    if lora_on and not lora_targets:
        raise ValueError("lora_targets cannot be empty when LoRA is enabled")
    if not 0 <= a.lora_dropout < 1:
        raise ValueError(
            f"lora_dropout must be in [0, 1), got {a.lora_dropout}"
        )
    if a.train_code_init_only and not gated_seg_pool:
        raise ValueError("--train_code_init_only currently requires --gated_seg_pool")
    if a.train_code_init_only and lora_on:
        raise ValueError("--train_code_init_only and LoRA are mutually exclusive")
    recovery_width = a.boundary_recovery_width if not is_base else 0
    recovery_weight = a.boundary_recovery_weight if recovery_width else 1.0
    if not 0 <= recovery_width < C:
        raise ValueError(
            f"boundary_recovery_width must be in [0, {C - 1}], got {recovery_width}"
        )
    if recovery_weight < 1.0:
        raise ValueError(
            f"boundary_recovery_weight must be >= 1, got {recovery_weight}"
        )
    distill_on = a.distill_lambda > 0 and not is_base   # distill only for the compressed arm (teacher = FROZEN full-ctx baseline via --teacher_ckpt)
    def autocast():
        return torch.autocast("cuda", dtype=torch.bfloat16) if (a.autocast and dev != "cpu") else contextlib.nullcontext()

    # ---- data: rebucket existing token IDs for C; last-block pad to N*C within a bucket ----
    loaded_pairs = torch.load(a.pairs, weights_only=False)
    pairs = [
        p for p in rebucket_pairs(loaded_pairs, C)
        if 2 <= p["n_blocks"] <= a.nmax
    ]
    del loaded_pairs
    evalp, trainp = pairs[:a.evaln], pairs[a.evaln:]
    if a.data_limit: trainp = trainp[:a.data_limit]                # compute-matched data-size sweep: fixed MAX_RESP_TOK, vary this
    epoch_tok = sum(len(p["response"]) for p in trainp)            # ~1-epoch response-token count -> honest epoch~ label (was hardcoded 4.7e6)
    buckets = defaultdict(list)
    for p in trainp: buckets[p["n_blocks"]].append(p)
    bkeys = [n for n in sorted(buckets) if len(buckets[n]) >= a.micro]
    assert bkeys, f"no bucket has >= micro={a.micro} examples (buckets={{n:len for..}})"

    def batch(items, dev):
        N = items[0]["n_blocks"]; B = len(items)
        ids = torch.zeros(B, N * C, dtype=torch.long); rm = torch.zeros(B, N * C, dtype=torch.long)
        for i, it in enumerate(items):
            seq = np.concatenate([np.asarray(it["prompt"]), np.asarray(it["response"])])[:N * C]   # list or int32-array pickle
            ids[i, :len(seq)] = torch.from_numpy(seq.astype(np.int64))
            rm[i, len(it["prompt"]):len(seq)] = 1                              # response positions (prompt & pad = 0)
        return ids.view(B, N, C).to(dev), rm.view(B, N, C).to(dev)

    # Shared RNG picks the SAME bucket N on every rank (keeps ranks in lockstep + bounds compiled graphs);
    # a rank-OFFSET RNG picks DIFFERENT examples per rank (data parallelism).
    g_shape = torch.Generator().manual_seed(a.seed)
    g_data = torch.Generator().manual_seed(a.seed * 100003 + rank + 1)
    def sample(n):
        idx = torch.randperm(len(buckets[n]), generator=g_data)[:a.micro]
        return [buckets[n][i] for i in idx]

    # ---- model: fp32 master weights, warm-start from a compressed CompressLM state_dict ----
    base = AutoModelForCausalLM.from_pretrained(a.base, torch_dtype=torch.float32).to(dev)
    base_revision = getattr(base.config, "_commit_hash", None)
    num_layers = len(base.model.layers)
    num_attention_heads = int(base.config.num_attention_heads)
    num_kv_heads = int(
        getattr(base.config, "num_key_value_heads", num_attention_heads)
    )
    quota_schedule = load_head_code_quotas(
        quota_spec,
        budget=quota_budget,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        k_max=K,
    )
    acr = str(a.active_code_rows).strip().lower()
    if acr in ("", "0", "off", "none"):
        active_code_rows = None
    elif acr == "auto":
        active_code_rows = max(max(row) for row in quota_schedule) if quota_schedule is not None else None
    else:
        active_code_rows = int(acr)
    core = CompressLM(
        base,
        C,
        N=bkeys[0],
        K=K,
        L=L,
        pause_xattn=a.pause_xattn,
        pool_init=pool_init,
        seg_pool=seg_pool,
        gated_seg_pool=gated_seg_pool,
        raw_tail=raw_tail,
        replay_tail=replay_tail,
        retain_tail_kv=retain_tail_kv,
        retain_tail_policy=retain_tail_policy,
        code_exclude_replay_tail=code_exclude_replay_tail,
        head_code_quotas=quota_schedule,
        active_code_rows=active_code_rows,
        code_positions=a.code_positions,
        code_q_heads=bool(a.code_q_heads),
        code_kv_heads=bool(a.code_kv_heads),
    ).to(dev)
    if a.skip_unused_codes:
        core.set_skip_unused_codes(True)
    lora_modules = []
    if lora_on:
        lora_modules = configure_lora_trainability(
            core,
            rank=a.lora_rank,
            alpha=lora_alpha,
            dropout=a.lora_dropout,
            target_modules=lora_targets,
            compressed=not is_base,
        )
        trainable_count = sum(
            parameter.numel()
            for parameter in core.parameters()
            if parameter.requires_grad
        )
        total_count = sum(parameter.numel() for parameter in core.parameters())
        log(
            f"[lora] rank={a.lora_rank} alpha={lora_alpha:g} "
            f"dropout={a.lora_dropout:g} targets={','.join(lora_targets)} "
            f"modules={len(lora_modules)} trainable={trainable_count:,}/"
            f"{total_count:,} ({100 * trainable_count / total_count:.3f}%)"
        )
    resume_step = 0
    resume_resp_toks = 0
    resume_opt = None
    resume_zero_shards = None
    resume_data_mb = None
    lora_base_ckpt = ""
    if a.ckpt:
        if not os.path.exists(a.ckpt):
            raise FileNotFoundError(f"configured checkpoint disappeared before load: {a.ckpt}")
        payload = torch.load(a.ckpt, map_location="cpu", weights_only=False)   # CPU: an fp32 payload on the GPU (16 GiB) outlived the load
        if isinstance(payload, dict) and "model" in payload:
            resume_step = int(payload.get("step", 0))
            resume_resp_toks = int(payload.get("resp_toks", 0))
            resume_data_mb = payload.get("data_microbatches")
            resume_opt = payload.get("opt")                    # present only in two-tier full ckpts
            if payload.get("optimizer_format") == "zero_adamw_rank_shards_v1":
                resume_zero_shards = {
                    "directory": os.path.dirname(a.ckpt),
                    "world_size": int(payload["optimizer_world_size"]),
                    "pattern": payload.get(
                        "optimizer_shard_pattern",
                        "optimizer.rank{rank:05d}.pt",
                    ),
                }
            saved_cross_block_loss = bool(payload.get("cross_block_loss", False))
            saved_recovery_width = int(payload.get("boundary_recovery_width", 0))
            saved_recovery_weight = float(payload.get("boundary_recovery_weight", 1.0))
            saved_seg_pool = bool(payload.get("seg_pool", False))
            saved_gated_seg_pool = bool(payload.get("gated_seg_pool", False))
            saved_pool_init = bool(payload.get("pool_init", False))
            saved_code_exclude_replay_tail = bool(
                payload.get("code_exclude_replay_tail", False)
            )
            saved_raw_tail = int(payload.get("raw_tail", 0))
            saved_replay_tail = int(payload.get("replay_tail", 0))
            saved_retain_tail_kv = int(payload.get("retain_tail_kv", 0))
            saved_retain_tail_policy = payload.get(
                "retain_tail_policy", "rolling"
            )
            saved_quota_payload = payload.get("head_code_quotas")
            saved_quota_schedule = (
                validate_head_code_quotas(
                    saved_quota_payload,
                    num_layers=num_layers,
                    num_kv_heads=num_kv_heads,
                    k_max=K,
                )
                if saved_quota_payload
                else None
            )
            saved_quota_budget = str(
                payload.get("head_code_quota_budget", "")
            ).strip()
            saved_code_positions = str(payload.get("code_positions", "appended"))
            if resume_step and not a.reset_progress and saved_code_positions != a.code_positions:
                raise ValueError(f"checkpoint was trained with code_positions={saved_code_positions!r} but this run uses {a.code_positions!r}")
            saved_lora_rank = int(payload.get("lora_rank", 0))
            saved_lora_alpha = float(payload.get("lora_alpha", 0))
            saved_lora_dropout = float(payload.get("lora_dropout", 0))
            saved_lora_targets = tuple(payload.get("lora_targets", ()))
            model_format = payload.get("model_format", "")
            if model_format == "lora_adapter" and not lora_on:
                raise ValueError(
                    "adapter-only LoRA checkpoint requires --lora_rank and "
                    "matching LoRA configuration"
                )
            if lora_on and model_format.startswith("lora"):
                if (
                    saved_lora_rank != a.lora_rank
                    or saved_lora_alpha != lora_alpha
                    or saved_lora_dropout != a.lora_dropout
                    or saved_lora_targets != lora_targets
                ):
                    raise ValueError(
                        "checkpoint LoRA configuration conflicts with "
                        "requested configuration"
                    )
                lora_base_ckpt = payload.get("lora_base_ckpt", "")
                saved_base_revision = payload.get("lora_base_revision")
                if lora_base_ckpt:
                    if not os.path.exists(lora_base_ckpt):
                        raise FileNotFoundError(
                            "LoRA checkpoint's frozen warm-start checkpoint "
                            f"disappeared: {lora_base_ckpt}"
                        )
                    base_payload = torch.load(
                        lora_base_ckpt,
                        map_location=dev,
                        weights_only=False,
                    )
                    if (
                        isinstance(base_payload, dict)
                        and base_payload.get("model_format")
                        == "lora_adapter"
                    ):
                        raise ValueError(
                            "lora_base_ckpt must contain full model weights, "
                            "not an adapter-only checkpoint"
                        )
                    base_state = (
                        base_payload["model"]
                        if isinstance(base_payload, dict)
                        and "model" in base_payload
                        else base_payload
                    )
                    base_state = remap_full_state_for_lora(
                        core, base_state
                    )
                    _, base_unexpected = core.load_state_dict(
                        base_state, strict=False
                    )
                    if base_unexpected:
                        raise ValueError(
                            "frozen LoRA warm-start checkpoint has unexpected "
                            f"keys: {base_unexpected[:8]}"
                        )
                elif (
                    saved_base_revision
                    and base_revision
                    and saved_base_revision != base_revision
                ):
                    raise ValueError(
                        "raw base revision changed since the LoRA checkpoint: "
                        f"{saved_base_revision} != {base_revision}"
                    )
            if resume_step and not a.reset_progress and saved_cross_block_loss != cross_block_loss:
                raise ValueError(
                    f"checkpoint cross_block_loss={int(saved_cross_block_loss)} conflicts with "
                    f"requested cross_block_loss={int(cross_block_loss)}"
                )
            if resume_step and not a.reset_progress and (
                saved_recovery_width != recovery_width
                or saved_recovery_weight != recovery_weight
                or saved_seg_pool != seg_pool
                or saved_gated_seg_pool != gated_seg_pool
                or saved_pool_init != pool_init
                or saved_code_exclude_replay_tail != code_exclude_replay_tail
                or saved_raw_tail != raw_tail
                or saved_replay_tail != replay_tail
                or saved_retain_tail_kv != retain_tail_kv
                or saved_retain_tail_policy != retain_tail_policy
            ):
                raise ValueError(
                    "checkpoint recovery/code-init/tail configuration conflicts with "
                    "requested configuration; use --reset_progress 1 to branch"
                )
            if resume_step and not a.reset_progress:
                if quota_schedule is None and saved_quota_schedule is not None:
                    if is_base:
                        raise ValueError(
                            "cannot restore head-code quotas into the baseline arm"
                        )
                    quota_schedule = saved_quota_schedule
                    quota_budget = saved_quota_budget
                    core.set_head_code_quotas(quota_schedule)
                    log(
                        "[warmstart] restored head-code quota schedule "
                        f"from checkpoint (budget={quota_budget or '-'})"
                    )
                elif saved_quota_schedule != quota_schedule:
                    raise ValueError(
                        "checkpoint head-code quota schedule conflicts with "
                        "the requested schedule; use --reset_progress 1 to branch"
                    )
                elif saved_quota_budget != quota_budget:
                    raise ValueError(
                        "checkpoint head-code quota budget conflicts with "
                        "the requested budget; use --reset_progress 1 to branch"
                    )
            if lora_on and "lora_adapter" in payload:
                sd = payload["lora_adapter"]
            else:
                sd = payload["model"]
        else:
            sd = payload
        if lora_on and not (
            isinstance(payload, dict)
            and payload.get("model_format", "").startswith("lora")
        ):
            sd = remap_full_state_for_lora(core, sd)
            lora_base_ckpt = a.ckpt
        if (
            lora_on
            and isinstance(payload, dict)
            and payload.get("model_format", "").startswith("lora")
        ):
            expected = set(trainable_state_dict(core))
            absent = sorted(expected - set(sd))
            if absent:
                raise ValueError(
                    f"LoRA checkpoint is missing trainable keys: {absent[:8]}"
                )
        miss, unexp = core.load_state_dict(sd, strict=False)
        if unexp:
            raise ValueError(
                f"checkpoint has unexpected model keys: {unexp[:8]}"
            )
        if a.reset_progress:
            resume_step = 0
            resume_resp_toks = 0
            resume_opt = None
            resume_zero_shards = None
        log(f"[warmstart] loaded {a.ckpt} | resume_step={resume_step} resume_resp_toks={resume_resp_toks:,} "
            f"missing={len(miss)} unexpected={len(unexp)} "
            f"opt_state={'SHARDED' if resume_zero_shards is not None else ('YES' if resume_opt is not None else 'NO')}")
        del payload, sd; gc.collect()   # the warm-start payload is copied into `core`; keeping it cost 15 GiB per GPU on resume
    else:
        log(f"[warmstart] NONE (raw {a.base}); no checkpoint configured")

    if quota_schedule is not None and is_base:
        raise ValueError("head-code quotas are not valid for the baseline arm")
    if quota_schedule is not None and a.compile:
        raise ValueError("head-code quotas require --compile 0")

    if a.train_code_init_only:
        for parameter in core.parameters():
            parameter.requires_grad_(False)
        core.seg_pool_gate.requires_grad_(True)
        log(
            f"[trainable] code-init only: {core.seg_pool_gate.numel()} gates "
            f"at lr={a.code_init_lr:g}"
        )

    if os.environ.get("COMPILE_MLP", "0") == "1" and dev != "cpu":   # fuse RMSNorm / SiLU*up / casts; attention stays eager
        n_mod = 0
        for lyr in core.model.model.layers:
            for mod in (lyr.mlp, lyr.input_layernorm, lyr.post_attention_layernorm):
                mod.forward = torch.compile(mod.forward, dynamic=True); n_mod += 1
        core.model.model.norm.forward = torch.compile(core.model.model.norm.forward, dynamic=True); n_mod += 1
        log(f"[compile] {n_mod} MLP/RMSNorm forwards compiled (dynamic shapes); attention eager")
    if a.grad_ckpt:                                        # activation checkpointing — recompute layer acts in bwd.
        core.model.config.use_cache = False                # required for HF grad-ckpt; forward already passes use_cache=False
        if lora_on:
            core.model.enable_input_require_grads()
        core.model.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        log(f"[grad_ckpt] enabled (non-reentrant) on {a.base} — activation memory ~O(sqrt(layers)); needed for long "
            f"seqs at 4B / large vocab")

    # FROZEN full-ctx teacher for distillation (fully-SFT'd baseline). Separate module, weights never update ->
    # avoids the online-self-distill circularity (student teaching itself while still acquiring the capability).
    teacher = None
    if distill_on:
        if a.teacher_ckpt and os.path.exists(a.teacher_ckpt):
            # --teacher_ckpt accepts EITHER an sft_ddp model.pt OR a directory of HF weights (scripts/
            # convert_compresslm_checkpoint_to_hf.py). The teacher only ever runs the plain full-context forward
            # (model.model + lm_head), so the HF export carries everything it uses; and since the params below are
            # bf16, load_state_dict already truncates an fp32 model.pt on the way in -- a bf16 export is what
            # training consumes either way, at half the bytes.
            hf_teacher = os.path.isdir(a.teacher_ckpt)
            tbase = AutoModelForCausalLM.from_pretrained(
                a.teacher_ckpt if hf_teacher else a.base, torch_dtype=torch.bfloat16).to(dev)
            teacher = CompressLM(tbase, C, N=bkeys[0], K=K, L=L, pause_xattn=a.pause_xattn).to(dev)
            # Load to CPU, never to the GPU: this is a 16 GB fp32 state dict that only exists to be copied into bf16
            # params. Loaded with map_location=dev it occupied ~14 GiB of every rank for the whole run (it was never
            # freed), which is most of the gap between 37.5 GiB "student + teacher built" and the 24 GiB those weigh.
            tsd = None if hf_teacher else torch.load(a.teacher_ckpt, map_location="cpu", weights_only=False)
            if tsd is not None:
                tsd = tsd.get("model", tsd) if isinstance(tsd, dict) and "model" in tsd else tsd
            # The teacher runs full-context only (never emits codes), so its code parameters are irrelevant; a baseline
            # checkpoint trained with a different K_max (e.g. pause[128, d] vs the student's 256) must not abort the load.
            if tsd is None:
                tm = tu = ()                                   # HF weights arrived through from_pretrained above
            else:
                tshapes = {k: v.shape for k, v in teacher.state_dict().items()}
                tskip = [k for k, v in tsd.items() if k in tshapes and hasattr(v, "shape") and tuple(v.shape) != tuple(tshapes[k])]
                if tskip:
                    log(f"[distill] teacher: skipping {len(tskip)} shape-mismatched (code-side) params: {tskip[:4]}")
                    tsd = {k: v for k, v in tsd.items() if k not in tskip}
                tm, tu = teacher.load_state_dict(tsd, strict=False)
                del tsd; tsd = None
            teacher.eval()
            for p in teacher.parameters(): p.requires_grad_(False)
            log(f"[distill] FROZEN teacher <- {a.teacher_ckpt} ({'HF dir' if hf_teacher else 'model.pt'}) | "
                f"missing={len(tm)} unexpected={len(tu)} | "
                f"lambda={a.distill_lambda:g} temp={a.distill_temp:g} ce_weight={a.distill_ce_weight:g} "
                f"(loss = ce_weight*CE + lambda*KL(teacher||student), response-masked, chunk={a.ce_chunk or 'all'})")
        else:
            log(f"[distill] WARNING no teacher_ckpt ('{a.teacher_ckpt}') -> DEPRECATED online self-distill (teacher = own full-ctx)")

    log_mem("student + teacher built", log)
    net = DDP(
        core,
        device_ids=([lrank] if dev != "cpu" else None),
        find_unused_parameters=False,
        gradient_as_bucket_view=True,
        broadcast_buffers=False,   # layout buffers are N-dependent and rebuilt per rank by set_N; ranks may differ in N
    ) if ddp else core
    fwd = torch.compile(net) if a.compile else net
    # Zero-touch the params that are unused per-arm so DDP(find_unused_parameters=False) is satisfied:
    # copy_token is never used in SFT (no copy loss); pause is unused in the baseline arm. Targeted (cheap) —
    # done in EAGER (outside compiled fwd) so dynamo can't constant-fold it away.
    unused = [core.copy_token] + ([core.pause] if is_base else [])

    if a.optimizer == "muon":
        if lora_on:
            raise ValueError("LoRA currently supports optimizer=adamw only")
        if a.train_code_init_only:
            raise ValueError("--train_code_init_only supports optimizer=adamw only")
        opts = list(build_muon_adamw(core, muon_lr=a.lr, adam_lr=a.lr))
    else:
        trainable = [parameter for parameter in core.parameters() if parameter.requires_grad]
        if gated_seg_pool and not a.train_code_init_only:
            gate = core.seg_pool_gate
            backbone = [parameter for parameter in trainable if parameter is not gate]
            param_groups = [
                {"params": backbone, "lr": a.lr, "weight_decay": 0.1},
                {"params": [gate], "lr": a.code_init_lr, "weight_decay": 0.0},
            ]
            log(
                f"[optimizer] backbone lr={a.lr:g}; "
                f"content gate lr={a.code_init_lr:g}"
            )
        else:
            optimizer_lr = a.code_init_lr if a.train_code_init_only else a.lr
            optimizer_decay = 0.0 if a.train_code_init_only else 0.1
            param_groups = [
                {
                    "params": trainable,
                    "lr": optimizer_lr,
                    "weight_decay": optimizer_decay,
                }
            ]
        if a.optimizer == "zero_adamw":
            if not ddp:
                raise ValueError("optimizer=zero_adamw requires DDP")
            opts = [
                ZeroRedundancyOptimizer(
                    param_groups,
                    optimizer_class=torch.optim.AdamW,
                    betas=(0.9, 0.95),
                    fused=(dev != "cpu"),
                )
            ]
            log(
                f"[optimizer] ZeRO-1 AdamW state sharded across {world} ranks"
            )
        else:
            opts = [torch.optim.AdamW(param_groups, betas=(0.9, 0.95), fused=(dev != "cpu"))]
    log_mem("optimizers created", log)
    base_lrs = [[g["lr"] for g in o.param_groups] for o in opts]
    parameter_names = {
        id(parameter): name
        for name, parameter in core.named_parameters()
    }

    def zero_local_layout(optimizer):
        return [
            [
                (
                    parameter_names[id(parameter)],
                    tuple(parameter.shape),
                    str(parameter.dtype),
                )
                for parameter in group["params"]
            ]
            for group in optimizer.optim.param_groups
        ]

    # Restore optimizer state when the ckpt carries it (two-tier full save); otherwise a resumed run
    # gets an LR re-warmup ramp below — fresh Adam moments at full LR shock the weights (+0.02 nats,
    # ~450M tok to recover, observed on the tail0 2B extension 2026-07-23).
    opt_restored = False
    if resume_zero_shards is not None:
        if a.optimizer != "zero_adamw":
            raise ValueError(
                "checkpoint carries sharded ZeRO-1 state but "
                f"optimizer={a.optimizer}"
            )
        saved_world = resume_zero_shards["world_size"]
        if saved_world != world:
            raise ValueError(
                "sharded ZeRO-1 checkpoint requires the same world size: "
                f"saved={saved_world}, current={world}"
            )
        missing_shards = [
            resume_zero_shards["pattern"].format(rank=shard_rank)
            for shard_rank in range(world)
            if not torch_checkpoint_complete(
                os.path.join(
                    resume_zero_shards["directory"],
                    resume_zero_shards["pattern"].format(rank=shard_rank),
                )
            )
        ]
        if missing_shards:
            raise FileNotFoundError(
                "ZeRO-1 optimizer checkpoint has incomplete shards: "
                + ", ".join(missing_shards)
            )
        shard_name = resume_zero_shards["pattern"].format(rank=rank)
        shard_path = os.path.join(
            resume_zero_shards["directory"],
            shard_name,
        )
        shard = torch.load(shard_path, map_location="cpu", weights_only=False)   # load_state_dict moves state to the param device
        if (
            shard.get("format") != "zero_adamw_rank_shards_v1"
            or int(shard.get("rank", -1)) != rank
            or int(shard.get("world_size", -1)) != world
        ):
            raise ValueError(f"invalid ZeRO-1 optimizer shard: {shard_path}")
        local_states = shard.get("optimizers", [])
        if len(local_states) != len(opts):
            raise ValueError(
                f"ZeRO-1 optimizer count mismatch: saved={len(local_states)}, "
                f"current={len(opts)}"
            )
        saved_layouts = shard.get("layouts", [])
        current_layouts = [
            zero_local_layout(optimizer)
            for optimizer in opts
        ]
        if saved_layouts != current_layouts:
            raise ValueError(
                f"ZeRO-1 local parameter layout mismatch: {shard_path}"
            )
        for optimizer, state in zip(opts, local_states):
            optimizer.optim.load_state_dict(state)
        del shard, local_states, saved_layouts, current_layouts
        opt_restored = True
        log(
            f"[warmstart] sharded ZeRO-1 optimizer state RESTORED "
            f"(rank {rank}/{world}, {len(opts)} optimizer(s))"
        )
    elif resume_opt is not None:
        try:
            for o, s in zip(opts, resume_opt): o.load_state_dict(s)
            opt_restored = True
            log(f"[warmstart] optimizer state RESTORED ({len(resume_opt)} optimizer(s))")
        except Exception as e:
            log(f"[warmstart] optimizer restore FAILED ({e}) -> falling back to LR re-warmup")
    rewarm_until = (resume_resp_toks + a.rewarmup_tok) if (resume_resp_toks > 0 and not opt_restored and a.rewarmup_tok > 0) else 0
    if rewarm_until: log(f"[warmstart] no optimizer state -> LR re-warmup over {a.rewarmup_tok:,} resp toks (until {rewarm_until:,})")

    accum = a.accum if a.accum > 0 else max(1, round(a.eff_batch / (world * a.micro)))
    eff = world * a.micro * accum
    quota_values = (
        [value for row in quota_schedule for value in row]
        if quota_schedule is not None
        else []
    )
    quota_summary = (
        f"budget={quota_budget or '-'} min={min(quota_values)} "
        f"max={max(quota_values)} mean={sum(quota_values) / len(quota_values):.1f}"
        if quota_values
        else "off"
    )
    log(f"[cfg] arm={a.arm} base={a.base} C={C} K={K} L={L} raw_tail={raw_tail} replay_tail={replay_tail} "
        f"retain_tail_kv={retain_tail_kv} retain_tail_policy={retain_tail_policy} "
        f"code_exclude_replay_tail={int(code_exclude_replay_tail)} head_code_quotas={quota_summary} active_code_rows={core.K_act} code_positions={core.code_positions} code_q_heads={int(core.code_q_heads)} code_kv_heads={int(core.code_kv_heads)} "
        f"| world={world} micro={a.micro} accum={accum} "
        f"-> eff_batch={eff} seqs (target {a.eff_batch}) | opt={a.optimizer} lr={a.lr} sched={a.lr_schedule} "
        f"| steps<= {a.max_steps} max_resp_tok={a.max_resp_tok or '-'} | save_every={a.save_every} eval_every={a.eval_every} "
        f"| buckets={{{', '.join(f'{n}:{len(buckets[n])}' for n in bkeys)}}} compile={a.compile} ce_chunk={a.ce_chunk or '-'} "
        f"cross_block_loss={int(cross_block_loss)} recovery=1..{recovery_width}x{recovery_weight:g} "
        f"seg_pool={int(seg_pool)} gated_seg_pool={int(gated_seg_pool)} "
        f"pool_init={int(pool_init)} train_code_init_only={int(bool(a.train_code_init_only))} "
        f"lora_rank={a.lora_rank} lora_alpha={lora_alpha if lora_on else 0:g}")

    def allreduce_sum(x):
        if not ddp: return x
        t = torch.tensor([x], dtype=torch.float64, device=dev); dist.all_reduce(t, op=dist.ReduceOp.SUM); return t.item()

    def save_ckpt(path, include_opt=False, merge_lora=False):
        zero_checkpoint = a.optimizer == "zero_adamw"
        zero_sharded = include_opt and a.optimizer == "zero_adamw"
        if zero_sharded:
            shard_name = f"optimizer.rank{rank:05d}.pt"
            shard_path = os.path.join(os.path.dirname(path), shard_name)
            atomic_torch_save(
                {
                    "format": "zero_adamw_rank_shards_v1",
                    "rank": rank,
                    "world_size": world,
                    "optimizers": [
                        optimizer.optim.state_dict()
                        for optimizer in opts
                    ],
                    "layouts": [
                        zero_local_layout(optimizer)
                        for optimizer in opts
                    ],
                },
                shard_path,
            )
        if zero_checkpoint:
            dist.barrier(group=checkpoint_group)
        if not is0:
            if zero_checkpoint:
                dist.barrier(group=checkpoint_group)
            return
        if lora_on:
            adapter = trainable_state_dict(core)
            model_state = (
                merged_state_dict(core) if merge_lora else adapter
            )
            model_format = (
                "lora_merged" if merge_lora else "lora_adapter"
            )
        else:
            adapter = None
            model_state = core.state_dict()
            model_format = "full"
        payload = {"model": model_state, "model_format": model_format,
                   "C": C, "K": K, "L": L, "pause_xattn": a.pause_xattn,
                   "head_code_quotas": (
                       [list(row) for row in quota_schedule]
                       if quota_schedule is not None
                       else None
                   ),
                   "head_code_quota_budget": quota_budget,
                   "raw_prompt": int(rp), "raw_tail": raw_tail, "replay_tail": replay_tail,
                   "retain_tail_kv": retain_tail_kv,
                   "retain_tail_policy": retain_tail_policy,
                   "code_exclude_replay_tail": int(code_exclude_replay_tail),
                   "step": step, "resp_toks": resp_toks, "data_microbatches": data_mb,
                   "code_positions": a.code_positions,
                   "skip_unused_codes": int(a.skip_unused_codes),
                   "base": a.base, "arm": a.arm,
                   "cross_block_loss": int(cross_block_loss),
                   "distill_lambda": a.distill_lambda if distill_on else 0.0,
                   "distill_temp": a.distill_temp if distill_on else None,
                   "distill_ce_weight": a.distill_ce_weight if distill_on else None,
                   "teacher_ckpt": a.teacher_ckpt if distill_on else None,
                   "boundary_recovery_width": recovery_width,
                   "boundary_recovery_weight": recovery_weight,
                   "seg_pool": int(seg_pool),
                   "gated_seg_pool": int(gated_seg_pool),
                   "pool_init": int(pool_init),
                   "lora_rank": a.lora_rank,
                   "lora_alpha": lora_alpha if lora_on else 0,
                   "lora_dropout": a.lora_dropout if lora_on else 0,
                   "lora_targets": lora_targets if lora_on else (),
                   "lora_base_ckpt": lora_base_ckpt if lora_on else "",
                   "lora_base_revision": base_revision if lora_on else None}
        if merge_lora:
            payload["lora_adapter"] = adapter
        if zero_sharded:
            payload["optimizer_format"] = "zero_adamw_rank_shards_v1"
            payload["optimizer_world_size"] = world
            payload["optimizer_shard_pattern"] = "optimizer.rank{rank:05d}.pt"
        elif include_opt:                                     # two-tier: full ckpt (~3x size) only at save_opt_every
            payload["opt"] = [o.state_dict() for o in opts]
        atomic_torch_save(payload, path)
        if zero_checkpoint:
            dist.barrier(group=checkpoint_group)
        log(
            f"[save] checkpoint -> {path}"
            f"{' (+sharded opt)' if zero_sharded else (' (+opt)' if include_opt else '')}"
        )

    def prune_periodic_ckpts():
        if not is0 or a.n_keep <= 0 or not a.save_dir or not os.path.isdir(a.save_dir):
            return
        steps = []
        for name in os.listdir(a.save_dir):
            match = re.fullmatch(r"step(\d+)", name)
            path = os.path.join(a.save_dir, name)
            if match and torch_checkpoint_complete(os.path.join(path, "model.pt")):
                steps.append((int(match.group(1)), path))
        keep_full = max((s for s, _ in steps if a.save_opt_every > 0 and s % a.save_opt_every == 0), default=None)
        for s, path in sorted(steps)[:-a.n_keep]:
            if s == keep_full: continue                       # never prune the newest optimizer-carrying ckpt
            shutil.rmtree(path)
            log(f"[prune] removed old periodic checkpoint {path}")

    def eval_heldout():                                # rank0-only held-out response CE + free position bins
        core.eval(); ev = defaultdict(list)
        for p in evalp: ev[p["n_blocks"]].append(p)
        pos_sum = torch.zeros(C, dtype=torch.float64, device=dev)
        pos_count = torch.zeros(C, dtype=torch.int64, device=dev)
        boundary_sum = torch.zeros(a.nmax, dtype=torch.float64, device=dev)
        boundary_count = torch.zeros(a.nmax, dtype=torch.int64, device=dev)
        # block-index x within-block-position-bin CE: bins [0], [1,16), [16,64), [64,256), [256,C). Rows = target block.
        # Separates "how much history" (block index) from "how far after a boundary" (position); the compression
        # penalty concentrates in the first positions after a boundary, and any growth with block index is the
        # teacher-forced counterpart of the long-generation losses.
        pos_edges = torch.tensor([1, 16, 64, 256], device=dev)
        NB = len(pos_edges) + 1
        blockpos_sum = torch.zeros(a.nmax * NB, dtype=torch.float64, device=dev)
        blockpos_count = torch.zeros(a.nmax * NB, dtype=torch.int64, device=dev)
        objective_sum = torch.zeros((), dtype=torch.float64, device=dev)
        objective_count = torch.zeros((), dtype=torch.int64, device=dev)
        with torch.no_grad():
            for n, its in ev.items():
                for i in range(0, len(its), a.micro):
                    b = its[i:i + a.micro]
                    if not b: continue
                    core.set_N(n); ids, rm = batch(b, dev)
                    with autocast():
                        out = core(
                            ids,
                            baseline=is_base,
                            ctx_blocks=(n if is_base else 0),
                            resp_mask=rm,
                            raw_prompt=rp,
                            ce_chunk=a.ce_chunk,
                            per_token=True,
                            include_cross_block=True,
                        )
                    ce = out["ce_pt"].double()
                    response = out["tmask"].bool()
                    position = out["pos"].long()
                    block_index = out["blk"].long()
                    objective = response & (
                        torch.ones_like(response)
                        if cross_block_loss
                        else out["supervised"].bool()
                    )
                    objective_sum += ce[objective].sum()
                    objective_count += objective.sum()
                    pos_sum.scatter_add_(0, position[response], ce[response])
                    pos_count.scatter_add_(
                        0,
                        position[response],
                        torch.ones_like(position[response], dtype=torch.int64),
                    )
                    bp_key = block_index[response] * NB + torch.bucketize(position[response], pos_edges, right=True)
                    blockpos_sum.scatter_add_(0, bp_key, ce[response])
                    blockpos_count.scatter_add_(0, bp_key, torch.ones_like(bp_key))
                    boundary = response & (position == 0)
                    boundary_sum.scatter_add_(
                        0, block_index[boundary], ce[boundary]
                    )
                    boundary_count.scatter_add_(
                        0,
                        block_index[boundary],
                        torch.ones_like(block_index[boundary], dtype=torch.int64),
                    )
        core.train()
        def mean_list(total, count):
            return [
                float(s / c) if c else None
                for s, c in zip(total.cpu().tolist(), count.cpu().tolist())
            ]
        heldout_ce = float(
            objective_sum / objective_count.clamp_min(1)
        )
        bps, bpc = blockpos_sum.view(a.nmax, NB).cpu(), blockpos_count.view(a.nmax, NB).cpu()
        def pooled(b_lo, b_hi, p_lo, p_hi):
            c = bpc[b_lo:b_hi, p_lo:p_hi].sum().item()
            return float(bps[b_lo:b_hi, p_lo:p_hi].sum().item() / c) if c else float("nan")
        # compact summary in the train log: CE by history length (blocks 1-4 / 5-8 / 9-16 / 17+) and CE of the first
        # 15 post-boundary tokens vs the rest, all response tokens of blocks >= 1 (block 0 has no compressed history)
        log("[eval-blockce] step=%d by_block(1-4/5-8/9-16/17+)=%.4f/%.4f/%.4f/%.4f  pos1-15=%.4f pos16+=%.4f  pos1-15_by_block(1-4/5-8/9-16/17+)=%.4f/%.4f/%.4f/%.4f" % (
            step, pooled(1, 5, 0, NB), pooled(5, 9, 0, NB), pooled(9, 17, 0, NB), pooled(17, a.nmax, 0, NB),
            pooled(1, a.nmax, 1, 2), pooled(1, a.nmax, 2, NB),
            pooled(1, 5, 1, 2), pooled(5, 9, 1, 2), pooled(9, 17, 1, 2), pooled(17, a.nmax, 1, 2)))
        return {
            "step": step,
            "resp_toks": resp_toks,
            "arm": a.arm,
            "base": a.base,
            "C": C,
            "K": K,
            "head_code_quotas": (
                [list(row) for row in quota_schedule]
                if quota_schedule is not None
                else None
            ),
            "head_code_quota_budget": quota_budget,
            "cross_block_loss": int(cross_block_loss),
            "boundary_recovery_width": recovery_width,
            "boundary_recovery_weight": recovery_weight,
            "seg_pool": int(seg_pool),
            "gated_seg_pool": int(gated_seg_pool),
            "pool_init": int(pool_init),
            "replay_tail": replay_tail,
            "retain_tail_kv": retain_tail_kv,
            "retain_tail_policy": retain_tail_policy,
            "code_exclude_replay_tail": int(code_exclude_replay_tail),
            "examples": len(evalp),
            "held_out_ce": heldout_ce,
            "target_position": list(range(C)),
            "position_ce": mean_list(pos_sum, pos_count),
            "position_sum": pos_sum.cpu().tolist(),
            "position_count": pos_count.cpu().tolist(),
            "target_block": list(range(a.nmax)),
            "boundary_ce_by_target_block":
                mean_list(boundary_sum, boundary_count),
            "boundary_sum_by_target_block": boundary_sum.cpu().tolist(),
            "boundary_count_by_target_block": boundary_count.cpu().tolist(),
            "position_bins": [0, 1, 16, 64, 256, C],
            "blockpos_sum": bps.tolist(),                                  # [nmax][5] CE sums, rows = target block index
            "blockpos_count": bpc.tolist(),
        }

    def persist_heldout(record, phase):
        record = {**record, "phase": phase}
        if a.save_dir:
            path = os.path.join(a.save_dir, "heldout_position_ce.jsonl")
            with open(path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, separators=(",", ":")) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            log(f"[eval-position] wrote step={record['step']} phase={phase} -> {path}")
        return record["held_out_ce"]

    # ---- train ----
    core.train()
    # ---- sampler stream position: g_shape/g_data are re-seeded per process, so a resume must fast-forward them or it
    # replays the stream from microbatch 0 (repeated data). Counter is saved in ckpts; --data_skip_microbatches overrides.
    data_mb = 0
    stream = None
    _spec = parse_stream_spec(a.data_stream)
    if _spec is not None:
        vw, vm, va = _spec
        if vw * vm * va != world * a.micro * accum:
            raise ValueError(f"canonical stream {a.data_stream} has {vw*vm*va} seqs/step but world*micro*accum={world*a.micro*accum}")
        stream = CanonicalStream(buckets, a.seed, vw, vm, va)
        log(f"[data] canonical stream {vw}x{vm}x{va} (virtual world x micro x accum) -> {stream.draws_per_step} seqs/step packed into "
            f"same-N microbatches of <= {a.micro} on {world} ranks; bucket keys={len(stream.bkeys)}")
    if stream is not None and resume_step and not a.reset_progress:
        stream.skip(resume_step)
        log(f"[data] canonical stream fast-forward: skipped {resume_step} steps")
    elif resume_step and not a.reset_progress:
        if a.data_skip_microbatches >= 0: data_mb, src = a.data_skip_microbatches, "override"
        elif resume_data_mb is not None: data_mb, src = int(resume_data_mb), "ckpt"
        else: data_mb, src = resume_step * accum, "step*accum (ckpt has no counter)"
        for _ in range(data_mb):
            n = bkeys[torch.randint(len(bkeys), (1,), generator=g_shape).item()]
            torch.randperm(len(buckets[n]), generator=g_data)
        log(f"[data] sampler fast-forward: skipped {data_mb} microbatches per rank ({src}); micro={a.micro} accum={accum} world={world}")
    step = resume_step; resp_toks = resume_resp_toks; t0 = time.time()
    while step < a.max_steps and (a.max_resp_tok == 0 or resp_toks < a.max_resp_tok):
        # LR schedule: token-driven if a budget is set (so all BS arms share one LR(token) curve), else step-driven.
        if a.max_resp_tok > 0:
            prog = min(1.0, resp_toks / a.max_resp_tok); warm = min(1.0, resp_toks / max(1.0, a.warmup_frac * a.max_resp_tok))
        else:
            prog = step / a.max_steps; warm = min(1.0, (step + 1) / max(1.0, a.warmup_frac * a.max_steps))
        if a.lr_schedule == "cosine":
            cos = 0.5 * (1 + math.cos(math.pi * prog))
            scale = warm * (a.lr_floor + (1 - a.lr_floor) * cos)
        elif a.lr_schedule == "wsd":
            # warmup -> stable at peak -> linear decay to lr_floor over the final decay_frac of the budget.
            # Stable-phase ckpts are resumable at peak LR (with opt state); decay legs branch off for measurement.
            d0 = 1.0 - a.decay_frac
            dec = 1.0 if prog <= d0 else 1.0 - (1.0 - a.lr_floor) * min(1.0, (prog - d0) / max(1e-9, a.decay_frac))
            scale = warm * dec
        else:
            scale = warm
        if rewarm_until and resp_toks < rewarm_until:                       # cold-resume protection (any schedule)
            scale *= max(0.0, resp_toks - resume_resp_toks) / max(1.0, float(a.rewarmup_tok))
        for o, bl in zip(opts, base_lrs):
            for gp, b0 in zip(o.param_groups, bl): gp["lr"] = b0 * scale
            o.zero_grad(set_to_none=True)

        # sample all ACCUM microbatches + pre-count response-target tokens (no forward) -> exact global total T
        micros = []; local_ntok = 0; local_weight = 0.0
        plan = pack_draws(stream.draw_step(), a.micro, world, rank) if stream is not None else None
        for j_ in range(accum if plan is None else len(plan)):
            if plan is None:
                n = bkeys[torch.randint(len(bkeys), (1,), generator=g_shape).item()]  # SAME on every rank
                items = sample(n)                                                      # DIFFERENT data per rank
            else:
                n, items = plan[j_]                                                    # this rank's share of the canonical step
            ids, rm = batch(items, dev)
            nt = resp_target_tokens(rm, n, C, cross_block_loss)
            nw = resp_target_weight(
                rm,
                n,
                C,
                cross_block_loss,
                recovery_width,
                recovery_weight,
            )
            local_ntok += nt
            local_weight += nw
            micros.append((n, ids, rm, nt, nw))
            data_mb += 1
        T = allreduce_sum(local_ntok)                                                  # global response-target tokens this step
        W = allreduce_sum(local_weight)                                                # global weighted objective mass

        ce_sum_local = 0.0; dst_sum_local = 0.0
        for j, (n, ids, rm, nt, nw) in enumerate(micros):
            core.set_N(n)
            bm = None
            if rp and not core.has_head_code_quotas:                       # quota schedules require eager per-layer masks inside CompressLM
                rmf = rm.reshape(rm.size(0), n * C)
                plen = (rmf.cumsum(1) == 0).sum(1).to(torch.int32)
                bm = core._mask_rawprompt(dev, plen, rm.size(0))
            sync_ctx = net.no_sync() if (ddp and j < len(micros) - 1) else contextlib.nullcontext()
            with sync_ctx, autocast():
                trows = None; thead = None
                if distill_on:                                             # FROZEN full-ctx teacher (or self, deprecated), detached
                    with torch.no_grad():                                  # final hidden states only; lm_head is applied per chunk
                        tsrc = teacher if teacher is not None else core    # inside CompressLM._masked_ce_kl_chunked (memory-safe)
                        Bt = ids.size(0)
                        th = tsrc.model.model(inputs_embeds=tsrc._embed_for_plain_forward(ids.reshape(Bt, n * C)),
                                              use_cache=False).last_hidden_state
                        teacher_pos = predict_positions(n, C, ids.device, cross_block_loss)
                        trows = th[:, teacher_pos, :].reshape(-1, th.size(-1))
                        thead = tsrc.model.lm_head
                        if step == 0 and j == 0:
                            log_mem(f"teacher fwd done (B={Bt} n={n} rows={trows.size(0)})", log)
                out = fwd(ids, baseline=is_base, ctx_blocks=(n if is_base else 0), resp_mask=rm,
                          raw_prompt=rp, blockmask=bm, teacher_rows=trows, teacher_head=thead,
                          distill_temp=a.distill_temp,
                          ce_chunk=a.ce_chunk, include_cross_block=cross_block_loss,
                          boundary_recovery_width=recovery_width,
                          boundary_recovery_weight=recovery_weight)
                ce = out["ce"]
                rawloss = (a.distill_ce_weight * ce + a.distill_lambda * out["distill"]) if "distill" in out else ce
            # Weighted objective mass preserves exact large-batch equivalence even
            # when post-boundary recovery targets receive extra weight.
            loss = rawloss * (nw * world / max(1.0, W)) + 0.0 * sum(p.sum() for p in unused)
            if step == 0 and j == 0:
                log_mem("student fwd done (pre-backward)", log)
            loss.backward()
            if step == 0 and j == 0:
                log_mem("first backward done", log)
            ce_sum_local += float(ce) * nw
            if "distill" in out: dst_sum_local += float(out["distill"]) * nw
        torch.nn.utils.clip_grad_norm_(core.parameters(), 1.0)
        for o in opts: o.step()
        step += 1; resp_toks += int(T)

        if step % a.log_every == 0 or step == 1:
            gce = allreduce_sum(ce_sum_local) / max(1.0, W)   # COLLECTIVE: every rank must call it, print on rank0 only
            gdst = allreduce_sum(dst_sum_local) / max(1.0, W)
            if is0:
                dt = time.time() - t0
                # peak GPU memory is the only thing that makes an OOM diagnosable after the fact; without it every
                # failure is a guess about which term (weights, optimizer, activations, loss chunk) filled the card.
                pk = (torch.cuda.max_memory_allocated() / 2**30) if torch.cuda.is_available() else 0.0
                rs = (torch.cuda.max_memory_reserved() / 2**30) if torch.cuda.is_available() else 0.0
                log(f"  step {step:4d} ce {gce:.4f}{f' distill {gdst:.4f}' if distill_on else ''} resp_toks {resp_toks:,} "
                    f"lr {opts[0].param_groups[0]['lr']:.2e} tok/s {resp_toks / max(1e-6, dt):,.0f} "
                    f"peak_mem {pk:.1f}/{rs:.1f} GiB")
                torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
        save_due = a.save_every > 0 and step % a.save_every == 0
        eval_due = a.eval_every > 0 and step % a.eval_every == 0
        if save_due or eval_due:
            if save_due:
                if a.save_dir:
                    full = a.save_opt_every > 0 and step % a.save_opt_every == 0
                    save_ckpt(os.path.join(a.save_dir, f"step{step}", "model.pt"), include_opt=full)
                prune_periodic_ckpts()
            if ddp: dist.barrier()
            if eval_due and is0:
                hce = persist_heldout(eval_heldout(), "periodic")
                log(f"[eval] step{step} held_out_CE={hce:.4f} epoch~{resp_toks/max(1.0,epoch_tok):.2f} resp_toks={resp_toks:,}")
            if eval_due and ddp: dist.barrier()

    if a.save:
        save_ckpt(
            a.save,
            include_opt=(a.save_opt_every > 0),
            merge_lora=lora_on,
        )
        log(f"[save] final SFT model -> {a.save}")
    if is0 and gated_seg_pool:
        gate = core.seg_pool_gate.detach().float().cpu()
        log(
            "[code-init-gate] "
            f"min={gate.min():.6f} median={gate.median():.6f} "
            f"max={gate.max():.6f} mean_abs={gate.abs().mean():.6f}"
        )

    # ---- held-out response CE (rank0 only; token-weighted mean over the eval set) ----
    if ddp: dist.barrier()
    if is0:
        final_eval = eval_heldout()
        hce = persist_heldout(final_eval, "final")
        supervised_count = sum(final_eval["position_count"])
        if not cross_block_loss:
            supervised_count -= final_eval["position_count"][0]
        log(f"[{a.arm}] held-out response CE = {hce:.4f} over {supervised_count:,} resp tokens")
    if ddp:
        dist.barrier()
        if checkpoint_group is not dist.group.WORLD:
            dist.destroy_process_group(checkpoint_group)
        dist.destroy_process_group()


# ---------------------------------------------------------------------------------------------------
# DEFERRED: padding-free document packing.
#   Bucket-by-N + last-block padding wastes ~8.4% of tokens on reasoning_pairs.pkl (measured: 13.3% at N=3
#   down to 6.3% at N=8 ≈ the 1/(2N) tail-of-last-block bound). Packing several short examples into one
#   N-block sequence would recover most of that, BUT CompressLM's FlexAttention mask (_mask/mm) assumes ALL
#   N blocks belong to ONE document: content_i sees its own block, and "earlier_final" lets block i read the
#   final codes of EVERY earlier block 0..i-1 (the compressed carry). To pack safely you must:
#     (1) tag each block with a document id doc[b] and AND it into `earlier_final`: (kr==L)&(kb<qb)&(doc[kb]==doc[qb]);
#     (2) keep `content_vis`/`refine` block-diagonal (already are, via kb==qb);
#     (3) make loss_pos/predict_flat NOT predict across a doc boundary (last real token of a doc must not
#         predict the first token of the next packed doc) — i.e. drop the boundary predict position;
#     (4) build resp_mask per packed doc.
#   Items (1)+(3) are where a subtly-wrong mask silently leaks cross-document information into the codes and
#   corrupts the compression signal we are trying to measure. Per the task's "correctness > this optimization",
#   packing is DEFERRED; the ~8.4% waste is accepted and documented. bucket-by-N is exact today.
# ---------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        main()
    except torch.cuda.OutOfMemoryError:
        # Rank 0 only: under DDP every rank holds the same shapes, and 8 copies of a 20-line census would bury
        # the rest of the log (and the monitor) for no extra information.
        if int(os.environ.get("RANK", "0")) == 0:
            log_cuda_census(lambda m: print(m, flush=True))   # the occupant, not just the failing allocation
        raise
