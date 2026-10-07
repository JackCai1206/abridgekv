"""LightThinker thought-level compression SFT on our stack (arm `lightthinker_thought`).

The algorithm is LightThinker's (see abridgekv/lightthinker.py for the sequence layout and mask, tested against
their code by scripts/test_lightthinker_mask.py); everything else is the vanilla arm's recipe from sft_ddp.py so the
comparison isolates the method:
  * the same pairs pickle, the same rebucket(C)/nmax filter, eval split and bucket keys, the same canonical data
    stream (DATA_STREAM=canonical:8x1x32 -> identical example multisets per step) and the same truncation to N*C
    ORIGINAL tokens;
  * the same optimizer (AdamW / ZeRO-1, betas 0.9/0.95, wd 0.1, clip 1.0), LR schedule (token-driven WSD) and token
    budget, where the budget counts ORIGINAL response tokens (the ~11 inserted tokens per thought do not advance it),
    so both arms take the same steps over the same data;
  * the loss is CE over all targets (original response tokens + splitters), normalised by the global target count.
Plain HF Qwen3 forward with a FlexAttention BlockMask; sequential position ids over the augmented sequence.

  torchrun --nproc_per_node 8 abridgekv/sft_lightthinker.py --base Qwen/Qwen3-4B-Base --pairs ... --save_dir ...
"""
import os
import sys
import math
import time
import json
import shutil
import argparse
import contextlib
from datetime import timedelta
from collections import defaultdict

os.environ.setdefault("HF_HUB_OFFLINE", os.environ.get("HF_OFFLINE", "0"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.utils.checkpoint as _ckpt
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoModelForCausalLM, AutoTokenizer

from abridgekv.data_stream import CanonicalStream, pack_draws, parse_stream_spec
from abridgekv.lightthinker import NEVER, N_CACHE, boundary_table, build_example, lt_ids, mask_mod_factory
from abridgekv.sft_quota_utils import rebucket_pairs

torch._dynamo.config.cache_size_limit = 64
PAD_TO = 1024                                   # sequence padding granularity (bounds the compiled mask/attn shapes)
OPT_SHARD = "optimizer.rank{rank:05d}.pt"


def main():
    ap = argparse.ArgumentParser()
    env = os.environ.get
    ap.add_argument("--base", default=env("BASE", "Qwen/Qwen3-4B-Base"))
    ap.add_argument("--pairs", default=env("PAIRS", ""))
    ap.add_argument("--C", type=int, default=int(env("C", "512")))            # only for rebucket/nmax: same data as vanilla
    ap.add_argument("--nmax", type=int, default=int(env("NMAX", "32")))
    ap.add_argument("--evaln", type=int, default=int(env("EVALN", "192")))
    ap.add_argument("--n_cache", type=int, default=int(env("LT_N_CACHE", str(N_CACHE))))
    ap.add_argument("--micro", type=int, default=int(env("MICRO", "1")))
    ap.add_argument("--eff_batch", type=int, default=int(env("EFF_BATCH", "256")))
    ap.add_argument("--lr", type=float, default=float(env("LR", "1e-4")))
    ap.add_argument("--warmup_frac", type=float, default=0.03)
    ap.add_argument("--lr_floor", type=float, default=0.1)
    ap.add_argument("--lr_schedule", default=env("LR_SCHEDULE", "wsd"), choices=["cosine", "constant", "wsd"])
    ap.add_argument("--decay_frac", type=float, default=float(env("DECAY_FRAC", "0.2")))
    ap.add_argument("--max_steps", type=int, default=int(env("STEPS", "2000000")))
    ap.add_argument("--max_resp_tok", type=int, default=int(env("MAX_RESP_TOK", "2000000000")))
    ap.add_argument("--optimizer", default=env("OPTIMIZER", "zero_adamw"), choices=["adamw", "zero_adamw"])
    ap.add_argument("--autocast", type=int, default=int(env("AUTOCAST", "1")))
    ap.add_argument("--grad_ckpt", type=int, default=int(env("GRAD_CKPT", "1")))
    ap.add_argument("--ce_chunk", type=int, default=int(env("CE_CHUNK", "4096")))
    ap.add_argument("--data_stream", default=env("DATA_STREAM", "canonical:8x1x32"))
    ap.add_argument("--save_dir", default=env("SAVE_DIR", ""))
    ap.add_argument("--save_every", type=int, default=int(env("SAVE_EVERY", "25")))
    ap.add_argument("--eval_every", type=int, default=int(env("EVAL_EVERY", "50")))
    ap.add_argument("--n_keep", type=int, default=int(env("N_KEEP", "1")))
    ap.add_argument("--log_every", type=int, default=int(env("LOG_EVERY", "10")))
    ap.add_argument("--seed", type=int, default=int(env("SEED", "0")))
    ap.add_argument("--deadline", type=float, default=float(env("DEADLINE_EPOCH", "0")))  # unix time: checkpoint + exit before it
    ap.add_argument("--no_final", type=int, default=int(env("NO_FINAL", "0")))            # smoke: skip final save/export
    a = ap.parse_args()

    ddp = int(os.environ.get("WORLD_SIZE", 1)) > 1
    force_cpu = os.environ.get("SFT_FORCE_CPU", "0") == "1" or not torch.cuda.is_available()
    if ddp:
        dist.init_process_group("gloo" if force_cpu else "nccl", timeout=timedelta(minutes=60))
        rank, world, lrank = dist.get_rank(), dist.get_world_size(), int(os.environ.get("LOCAL_RANK", 0))
        dev = "cpu" if force_cpu else f"cuda:{lrank}"
        if not force_cpu:
            torch.cuda.set_device(lrank)
    else:
        rank, world, lrank = 0, 1, 0
        dev = "cpu" if force_cpu else "cuda:0"
    is0 = rank == 0

    def log(*x):
        if is0:
            print(*x, flush=True)

    if dev == "cpu":                              # local validation only (see sft_ddp.py)
        torch._dynamo.config.disable = True
    torch.manual_seed(a.seed)

    def autocast():
        return torch.autocast("cuda", dtype=torch.bfloat16) if (a.autocast and dev != "cpu") else contextlib.nullcontext()

    def allreduce_sum(x):
        t = torch.tensor(float(x), dtype=torch.float64, device=dev)
        if ddp:
            dist.all_reduce(t)
        return float(t.item())

    # ---- data: identical selection to sft_ddp.py (rebucket at C, 2 <= n_blocks <= nmax, eval = first evaln) ----
    C = a.C
    loaded = torch.load(a.pairs, weights_only=False)
    pairs = [p for p in rebucket_pairs(loaded, C) if 2 <= p["n_blocks"] <= a.nmax]
    del loaded
    evalp, trainp = pairs[:a.evaln], pairs[a.evaln:]
    buckets = defaultdict(list)
    for p in trainp:
        buckets[p["n_blocks"]].append(p)

    tok = AutoTokenizer.from_pretrained(a.base)
    base = AutoModelForCausalLM.from_pretrained(a.base, torch_dtype=torch.float32, attn_implementation="flex_attention")
    vocab_rows = base.get_input_embeddings().weight.shape[0]
    ids = lt_ids(len(tok), a.n_cache)
    if ids.cont >= vocab_rows:
        raise ValueError(f"no free embedding rows for {len(ids.special_tokens)} tokens ({len(tok)} + >{vocab_rows})")
    ends = boundary_table(tok, vocab_rows)
    eos_id = tok.eos_token_id
    log(f"[lt] tokens: cache={ids.cache[0]}..{ids.cache[-1]} splitter={ids.splitter} continue={ids.cont}; "
        f"{int(ends.sum())} boundary token ids (text ends with a blank line); eos={eos_id}")

    def batch(items):
        exs = [build_example(it["prompt"], it["response"], it["n_blocks"] * C, ends, ids, eos_id) for it in items]
        L = max(len(e["ids"]) for e in exs)
        S = ((L + PAD_TO - 1) // PAD_TO) * PAD_TO
        B = len(exs)
        x = torch.zeros(B, S, dtype=torch.long)
        tgt = torch.zeros(B, S, dtype=torch.bool)        # token p is a target (predicted at p-1)
        orig = torch.zeros(B, S, dtype=torch.bool)
        spl = torch.zeros(B, S, dtype=torch.bool)
        hide = torch.full((B, S), int(NEVER), dtype=torch.long)
        for b, e in enumerate(exs):
            n = len(e["ids"])
            x[b, :n] = torch.from_numpy(e["ids"])
            tgt[b, :n] = torch.from_numpy(e["target"]).bool()
            orig[b, :n] = torch.from_numpy(e["orig"]).bool()
            spl[b, :n] = x[b, :n] == ids.splitter
            hide[b, :n] = torch.from_numpy(e["hide_at"])
        return x, tgt, orig, spl, hide, sum(len(e["ids"]) for e in exs)

    def block_mask(hide):
        from torch.nn.attention.flex_attention import create_block_mask
        B, S = hide.shape
        return create_block_mask(mask_mod_factory(hide), B=B, H=None, Q_LEN=S, KV_LEN=S, device=hide.device,
                                 _compile=(dev != "cpu"))

    # ---- model ----
    base.config.use_cache = False
    base = base.to(dev)
    if a.grad_ckpt:
        base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    head = base.lm_head
    tied = base.config.tie_word_embeddings
    # fp32 master weights under bf16 autocast: Qwen3's q_norm/k_norm (fp32 weight) would promote bf16 Q/K back to
    # fp32 while V stays bf16 (flex/SDPA reject mixed dtypes) -- keep the norm output in its input dtype, exactly as
    # CompressLM does (compress_forward._preserve_norm_input_dtype)
    for layer in base.model.layers:
        for name in ("q_norm", "k_norm"):
            norm = getattr(layer.self_attn, name, None)
            if norm is not None:
                norm.register_forward_hook(lambda m, i, o: o.to(i[0].dtype))

    step = 0
    resp_toks = 0
    resume = None
    if a.save_dir:
        cands = sorted((int(d[4:]), d) for d in os.listdir(a.save_dir) if d.startswith("step") and d[4:].isdigit()
                       and os.path.exists(os.path.join(a.save_dir, d, "model.pt.complete"))) if os.path.isdir(a.save_dir) else []
        if cands:
            resume = os.path.join(a.save_dir, cands[-1][1])
    if resume:
        payload = torch.load(os.path.join(resume, "model.pt"), map_location="cpu", weights_only=False)
        base.load_state_dict(payload["model"])
        step, resp_toks = int(payload["step"]), int(payload["resp_toks"])
        if int(payload.get("world", world)) != world:
            raise ValueError(f"resume needs world={payload['world']} (ZeRO shards), got {world}")
        log(f"[lt] resumed {resume}: step {step} resp_toks {resp_toks:,}")
        del payload
    else:
        # new rows start at the mean embedding of the tokenizer vocabulary (HF mean_resizing; LightThinker resizes too)
        with torch.no_grad():
            emb = base.get_input_embeddings().weight
            emb[list(ids.cache) + [ids.splitter, ids.cont]] = emb[:len(tok)].mean(0, keepdim=True)
            if not tied:
                hw = head.weight
                hw[list(ids.cache) + [ids.splitter, ids.cont]] = hw[:len(tok)].mean(0, keepdim=True)

    params = [p for p in base.parameters() if p.requires_grad]
    groups = [{"params": params, "lr": a.lr, "weight_decay": 0.1}]
    if a.optimizer == "zero_adamw":
        if not ddp:
            raise ValueError("optimizer=zero_adamw requires DDP")
        opt = ZeroRedundancyOptimizer(groups, optimizer_class=torch.optim.AdamW, betas=(0.9, 0.95), fused=(dev != "cpu"))
    else:
        opt = torch.optim.AdamW(groups, betas=(0.9, 0.95), fused=(dev != "cpu"))
    if resume:
        shard = os.path.join(resume, OPT_SHARD.format(rank=rank))
        state = torch.load(shard if a.optimizer == "zero_adamw" else os.path.join(resume, OPT_SHARD.format(rank=0)),
                           map_location="cpu", weights_only=False)
        (opt.optim if a.optimizer == "zero_adamw" else opt).load_state_dict(state)
        del state
        log("[lt] optimizer state restored")

    def embed(x):
        e = base.get_input_embeddings()(x)
        if torch.is_autocast_enabled(e.device.type):
            e = e.to(torch.get_autocast_dtype(e.device.type))
        return e

    def ce_rows(hid_rows, tgt_rows):
        """Sum of CE over rows, lm_head chunked and recomputed in backward (peak O(chunk x vocab))."""
        def chunk(h, t):
            return F.cross_entropy(head(h).float(), t, reduction="none")
        outs = []
        step_ = hid_rows.size(0) if a.ce_chunk <= 0 else a.ce_chunk
        for i in range(0, hid_rows.size(0), step_):
            h, t = hid_rows[i:i + step_], tgt_rows[i:i + step_]
            outs.append(_ckpt.checkpoint(chunk, h, t, use_reentrant=False) if torch.is_grad_enabled() else chunk(h, t))
        return torch.cat(outs) if outs else hid_rows.new_zeros(0, dtype=torch.float32)

    # DDP hooks fire on the wrapped module's forward; calling .model directly bypasses them, so drive the forward
    # through the DDP wrapper with a thin module that returns hidden states.
    class Hidden(torch.nn.Module):
        def __init__(self, lm):
            super().__init__()
            self.lm = lm

        def forward(self, inputs_embeds, position_ids, attention_mask):
            return self.lm.model(inputs_embeds=inputs_embeds, position_ids=position_ids, attention_mask=attention_mask,
                                 use_cache=False).last_hidden_state

    hidden_net = Hidden(base)
    net = DDP(hidden_net, device_ids=([lrank] if dev != "cpu" else None), find_unused_parameters=False,
              gradient_as_bucket_view=True, broadcast_buffers=False) if ddp else hidden_net

    def run(x, tgt, hide, train=True):
        B, S = x.shape
        bm = block_mask(hide)
        pos = torch.arange(S, device=x.device).unsqueeze(0).expand(B, S)
        hid = (net if train else hidden_net)(embed(x), pos, bm)
        sel = torch.zeros_like(tgt)
        sel[:, :-1] = tgt[:, 1:]
        ce = ce_rows(hid[sel], x[:, 1:][sel[:, :-1]])
        return ce, sel

    # ---- held-out CE (rank 0): original response tokens (comparable-ish), splitters, first token after a continue ----
    @torch.no_grad()
    def eval_heldout():
        base.eval()
        s = defaultdict(float)
        c = defaultdict(int)
        for i in range(0, len(evalp)):
            x, tgt, orig, spl, hide, _ = batch([evalp[i]])
            x, tgt, orig, spl, hide = (t.to(dev) for t in (x, tgt, orig, spl, hide))
            with autocast():
                ce, sel = run(x, tgt, hide, train=False)
            nxt_orig = torch.zeros_like(orig); nxt_orig[:, :-1] = orig[:, 1:]
            nxt_spl = torch.zeros_like(spl); nxt_spl[:, :-1] = spl[:, 1:]
            at_cont = x == ids.cont
            for name, m in (("orig", nxt_orig), ("splitter", nxt_spl), ("after_continue", at_cont & sel)):
                mm = m[sel]
                s[name] += float(ce[mm].sum()); c[name] += int(mm.sum())
            s["all"] += float(ce.sum()); c["all"] += int(ce.numel())
        base.train()
        return {k: s[k] / max(1, c[k]) for k in s}, dict(c)

    def save(dirpath, final=False):
        """Model (rank 0, fp32) + per-rank ZeRO shards; the .complete marker is written last, after a barrier."""
        if is0:
            os.makedirs(dirpath, exist_ok=True)
        if ddp:
            dist.barrier()
        state = (opt.optim if a.optimizer == "zero_adamw" else opt).state_dict()
        if a.optimizer == "zero_adamw" or is0:
            torch.save(state, os.path.join(dirpath, OPT_SHARD.format(rank=rank)) + ".tmp")
            os.replace(os.path.join(dirpath, OPT_SHARD.format(rank=rank)) + ".tmp", os.path.join(dirpath, OPT_SHARD.format(rank=rank)))
        del state
        if is0:
            payload = {"model": {k: v.detach().cpu() for k, v in base.state_dict().items()}, "step": step,
                       "resp_toks": resp_toks, "world": world, "arm": "lightthinker_thought", "n_cache": a.n_cache,
                       "token_ids": {"cache": list(ids.cache), "splitter": ids.splitter, "continue": ids.cont}}
            torch.save(payload, os.path.join(dirpath, "model.pt.tmp"))
            os.replace(os.path.join(dirpath, "model.pt.tmp"), os.path.join(dirpath, "model.pt"))
            del payload
        if ddp:
            dist.barrier()
        if is0:
            open(os.path.join(dirpath, "model.pt.complete"), "w").close()
            if not final and a.n_keep > 0:
                olds = sorted((int(d[4:]), d) for d in os.listdir(a.save_dir) if d.startswith("step") and d[4:].isdigit())
                for _, d in olds[:-a.n_keep]:
                    shutil.rmtree(os.path.join(a.save_dir, d), ignore_errors=True)
            log(f"[save] {dirpath} step {step}")

    def export_hf(outdir):
        """bf16 HF dir + tokenizer with the 11 tokens registered at the trained ids + CompressLM/LightThinker config."""
        from transformers import AutoConfig
        tk = AutoTokenizer.from_pretrained(a.base)
        added = tk.add_special_tokens({"additional_special_tokens": ids.special_tokens})
        got = tk.convert_tokens_to_ids(ids.special_tokens)
        if added != len(ids.special_tokens) or got != list(ids.cache) + [ids.splitter, ids.cont]:
            raise RuntimeError(f"tokenizer registered the LightThinker tokens at {got}, trained at {ids}")
        sd = {k: v.detach().to("cpu", torch.bfloat16) for k, v in base.state_dict().items()}
        cfg = AutoConfig.from_pretrained(a.base)
        meta = {
            "enable": True, "boundary_mode": "lightthinker", "block_size": 1 << 20, "num_code_tokens": a.n_cache,
            "num_passes": 1, "pause_xattn": "causal", "raw_prompt": True, "raw_tail": 0, "replay_tail": 0,
            "retain_tail_kv": 0, "code_positions": "appended", "code_token_format": "<|compress_code_{:03d}|>",
            "code_token_ids": list(ids.cache), "boundary_token_id": ids.splitter, "continue_token_id": ids.cont,
            "continue_token": "<|lt_continue|>",
        }
        cfg.compresslm_config = meta
        cfg.max_position_embeddings = max(int(cfg.max_position_embeddings), 65536)   # 32K gen ~ 39K positions
        cfg.torch_dtype = "bfloat16"
        os.makedirs(outdir, exist_ok=True)
        base.config.compresslm_config = meta
        base.config.max_position_embeddings = cfg.max_position_embeddings
        base.save_pretrained(outdir, state_dict=sd, safe_serialization=True)
        with open(os.path.join(outdir, "config.json")) as fh:
            c = json.load(fh)
        c["torch_dtype"] = "bfloat16"; c["dtype"] = "bfloat16"; c["use_cache"] = True
        # serving runs transformers 4.57 (vLLM env): write the pre-5 config form (rope_theta, not rope_parameters,
        # which 4.57 would silently ignore -> default theta) -- see the transformers5 export gotcha (tplskip)
        rp = c.pop("rope_parameters", None)
        if rp is not None:
            c["rope_theta"] = rp.get("rope_theta", c.get("rope_theta"))
            if rp.get("rope_type", "default") != "default":
                raise RuntimeError(f"unexpected rope_parameters {rp}")
            c["rope_scaling"] = None
        with open(os.path.join(outdir, "config.json"), "w") as fh:
            json.dump(c, fh, indent=2)
        tk.save_pretrained(outdir)
        tcp = os.path.join(outdir, "tokenizer_config.json")
        with open(tcp) as fh:
            tc = json.load(fh)
        extra = tc.get("extra_special_tokens")
        if isinstance(extra, list):          # 5.x writes a list; 4.57 expects a dict and crashes on a list
            tc["additional_special_tokens"] = list(dict.fromkeys(tc.get("additional_special_tokens", []) + extra))
            tc["extra_special_tokens"] = {}
        with open(tcp, "w") as fh:
            json.dump(tc, fh, indent=2)
        with open(os.path.join(outdir, "compresslm_config.json"), "w") as fh:
            json.dump(meta, fh, indent=2)
        log(f"[export] {outdir}")

    # ---- train ----
    accum = max(1, a.eff_batch // (world * a.micro))
    spec = parse_stream_spec(a.data_stream)
    if spec is None:
        raise ValueError("sft_lightthinker supports only the canonical data stream")
    vw, vm, va = spec
    stream = CanonicalStream(buckets, a.seed, vw, vm, va)
    if step:
        stream.skip(step)
    log(f"[data] {len(pairs)} pairs (eval {len(evalp)}, train {len(trainp)}); canonical {vw}x{vm}x{va} -> "
        f"{stream.draws_per_step} seqs/step; micro<={a.micro} world={world}; resp budget {a.max_resp_tok:,}")
    base.train()
    t0 = time.time(); tok0 = resp_toks; aug_local = 0
    while step < a.max_steps and resp_toks < a.max_resp_tok:
        prog = min(1.0, resp_toks / a.max_resp_tok)
        warm = min(1.0, resp_toks / max(1.0, a.warmup_frac * a.max_resp_tok))
        if a.lr_schedule == "cosine":
            scale = warm * (a.lr_floor + (1 - a.lr_floor) * 0.5 * (1 + math.cos(math.pi * prog)))
        elif a.lr_schedule == "wsd":
            d0 = 1.0 - a.decay_frac
            scale = warm * (1.0 if prog <= d0 else 1.0 - (1.0 - a.lr_floor) * min(1.0, (prog - d0) / max(1e-9, a.decay_frac)))
        else:
            scale = warm
        for g in opt.param_groups:
            g["lr"] = a.lr * scale
        opt.zero_grad(set_to_none=True)

        plan = pack_draws(stream.draw_step(), a.micro, world, rank)
        micros = []; n_tgt = 0; n_orig = 0
        for n, items in plan:
            x, tgt, orig, spl, hide, L = batch(items)
            n_tgt += int(tgt[:, 1:].sum()); n_orig += int(orig.sum()); aug_local += L
            micros.append((x, tgt, hide))
        T = allreduce_sum(n_tgt)
        TO = allreduce_sum(n_orig)
        ce_local = 0.0
        for j, (x, tgt, hide) in enumerate(micros):
            x, tgt, hide = x.to(dev), tgt.to(dev), hide.to(dev)
            sync = net.no_sync() if (ddp and j < len(micros) - 1) else contextlib.nullcontext()
            with sync:
                with autocast():
                    ce, _ = run(x, tgt, hide)
                    loss = ce.sum() * (world / max(1.0, T))
                loss.backward()
            ce_local += float(ce.detach().sum())
        if not micros and ddp:
            # a rank with no share this step still has to join the gradient allreduce: a zero-weighted dummy forward
            x = torch.full((1, PAD_TO), eos_id, dtype=torch.long, device=dev)
            with autocast():
                ce, _ = run(x, torch.ones_like(x, dtype=torch.bool), torch.full_like(x, int(NEVER)))
                (ce.sum() * 0.0).backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        step += 1; resp_toks += int(TO)
        if step % a.log_every == 0 or step == 1:
            gce = allreduce_sum(ce_local) / max(1.0, T)
            aug = allreduce_sum(aug_local)
            dt = time.time() - t0
            pk = torch.cuda.max_memory_allocated() / 2**30 if dev != "cpu" else 0.0
            log(f"  step {step:5d} ce {gce:.4f} resp_toks {resp_toks:,} lr {opt.param_groups[0]['lr']:.2e} "
                f"tok/s(orig) {(resp_toks - tok0) / max(1e-6, dt):,.0f} aug/orig {aug / max(1.0, resp_toks - tok0):.3f} "
                f"peak_mem {pk:.1f} GiB")
        if a.save_dir and a.save_every and step % a.save_every == 0:
            save(os.path.join(a.save_dir, f"step{step}"))
        if a.eval_every and step % a.eval_every == 0:
            if is0:
                r, cnt = eval_heldout()
                log(f"[eval] step{step} held_out_CE(orig)={r['orig']:.4f} splitter={r['splitter']:.4f} "
                    f"after_continue={r['after_continue']:.4f} all={r['all']:.4f} counts={cnt}")
                if a.save_dir:
                    with open(os.path.join(a.save_dir, "heldout_ce.jsonl"), "a") as fh:
                        fh.write(json.dumps({"step": step, "resp_toks": resp_toks, **r, "counts": cnt}) + "\n")
            if ddp:
                dist.barrier()
        if a.deadline and time.time() > a.deadline:
            flag = torch.tensor(1.0, device=dev)
        else:
            flag = torch.tensor(0.0, device=dev)
        if ddp:
            dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        if flag.item() > 0:
            if a.save_dir:
                save(os.path.join(a.save_dir, f"step{step}"))
            log(f"[lt] deadline reached at step {step}; checkpointed, exiting for resume")
            if ddp:
                dist.destroy_process_group()
            return

    if a.no_final or not a.save_dir:
        log(f"[lt] done at step {step} (no final save)")
    else:
        save(os.path.join(a.save_dir, f"step{step}"))
        if is0:
            r, cnt = eval_heldout()
            log(f"[lightthinker_thought] held-out response CE = {r['orig']:.4f} (splitter {r['splitter']:.4f}, "
                f"after_continue {r['after_continue']:.4f}) over {cnt['orig']:,} resp tokens")
            with open(os.path.join(a.save_dir, "heldout_ce.jsonl"), "a") as fh:
                fh.write(json.dumps({"step": step, "resp_toks": resp_toks, "phase": "final", **r, "counts": cnt}) + "\n")
            export_hf(os.path.join(a.save_dir, "hf"))
            open(os.path.join(a.save_dir, "model.pt.complete"), "w").close()
    if ddp:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
