#!/usr/bin/env python3
"""Held-out CE of a finished checkpoint broken down by target block index x within-block position bin (the same
breakdown sft_ddp.eval_heldout now logs), for compressed AND vanilla (baseline) checkpoints. Reproduces the trainer's
held-out set (first EVALN pairs with 2 <= n_blocks <= NMAX after rebucketing to C), objective and autocast.
  PROBE_RUN=<run dir> PROBE_OUT=<json> PAIRS=<pkl> [NMAX=32 EVALN=192 PROBE_MICRO=8] python scripts/probe_heldout_blockce.py
Single GPU, no grad. Writes one JSON with held_out_ce, blockpos_sum/count [NMAX][5], position bins, and the run's config."""
import os, sys, json, time, torch
sys.path.insert(0, os.getcwd())
from transformers import AutoModelForCausalLM
from abridgekv.compress_forward import CompressLM
from abridgekv.sft_quota_utils import rebucket_pairs, pair_tokens

RUN = os.environ["PROBE_RUN"]; OUT = os.environ["PROBE_OUT"]; PAIRS = os.environ["PAIRS"]
NMAX = int(os.environ.get("NMAX", "32")); EVALN = int(os.environ.get("EVALN", "192")); MICRO = int(os.environ.get("PROBE_MICRO", "8"))
dev = "cuda"; torch._dynamo.config.cache_size_limit = 256
if RUN == "BASE":
    # B9: the untrained base model, no checkpoint. Same C / held-out set as every trained arm, ARM=baseline forward
    # (full context, no codes), so its CE is directly comparable with the vanilla SFT row (0.4162) and the compressed arms.
    payload = {"C": 512, "K": 128, "L": 1, "arm": "baseline", "code_positions": "appended", "base": "Qwen/Qwen3-4B-Base"}
else:
    payload = torch.load(os.path.join(RUN, "model.pt"), map_location="cpu", weights_only=False)
C, K, L = int(payload["C"]), int(payload["K"]), int(payload["L"]); arm = payload.get("arm", "compressed"); is_base = arm == "baseline"
Q0 = payload.get("head_code_quotas"); Q0 = [list(map(int, r)) for r in Q0] if Q0 else None
cp = str(payload.get("code_positions", "appended")); base_name = payload.get("base", "Qwen/Qwen3-4B-Base")
kw = dict(retain_tail_kv=int(payload.get("retain_tail_kv", 0)), retain_tail_policy=payload.get("retain_tail_policy", "rolling"),
          replay_tail=int(payload.get("replay_tail", 0)), raw_tail=int(payload.get("raw_tail", 0)))
print(f"[blockce] run={RUN} arm={arm} C={C} K={K} L={L} code_positions={cp} quotas={'yes' if Q0 else 'none'} {kw}", flush=True)
base = AutoModelForCausalLM.from_pretrained(base_name, torch_dtype=torch.float32).to(dev)
core = CompressLM(base, C, N=2, K=K, L=L, pause_xattn=payload.get("pause_xattn", "full"), head_code_quotas=Q0,
                  active_code_rows=K, code_positions=cp, **kw).to(dev)
if "model" in payload:
    missing, unexpected = core.load_state_dict(payload["model"], strict=False)
    print(f"[blockce] state loaded missing={len(missing)} unexpected={len(unexpected)} {missing[:3]} {unexpected[:3]}", flush=True)
else:
    print("[blockce] BASE model: no state loaded (untrained Qwen3-4B-Base)", flush=True)
del payload; core.eval()
pairs = [p for p in rebucket_pairs(torch.load(PAIRS, weights_only=False), C) if 2 <= p["n_blocks"] <= NMAX]
evalp = pairs[:EVALN]; del pairs
print(f"[blockce] heldout={len(evalp)}", flush=True)

def batch(items):
    N = items[0]["n_blocks"]; B = len(items)
    ids = torch.zeros(B, N * C, dtype=torch.long); rm = torch.zeros(B, N * C, dtype=torch.long)
    for i, it in enumerate(items):
        seq = pair_tokens(it)[:N * C]
        ids[i, :len(seq)] = torch.tensor(seq); rm[i, len(it["prompt"]):len(seq)] = 1
    return ids.view(B, N, C).to(dev), rm.view(B, N, C).to(dev)

pos_edges = torch.tensor([1, 16, 64, 256], device=dev); NB = 5
bps = torch.zeros(NMAX * NB, dtype=torch.float64, device=dev); bpc = torch.zeros(NMAX * NB, dtype=torch.int64, device=dev)
tot_s = torch.zeros((), dtype=torch.float64, device=dev); tot_c = torch.zeros((), dtype=torch.int64, device=dev)
by_n = {}
for p in evalp: by_n.setdefault(p["n_blocks"], []).append(p)
t0 = time.time()
with torch.no_grad():
    for n in sorted(by_n):
        its = by_n[n]
        for i in range(0, len(its), MICRO):
            b = its[i:i + MICRO]; core.set_N(n); ids, rm = batch(b)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = core(ids, baseline=is_base, ctx_blocks=(n if is_base else 0), resp_mask=rm, raw_prompt=not is_base,
                           ce_chunk=4096, per_token=True, include_cross_block=True)
            ce = out["ce_pt"].double(); resp = out["tmask"].bool(); pos = out["pos"].long(); blk = out["blk"].long()
            key = blk[resp] * NB + torch.bucketize(pos[resp], pos_edges, right=True)
            bps.scatter_add_(0, key, ce[resp]); bpc.scatter_add_(0, key, torch.ones_like(key))
            tot_s += ce[resp].sum(); tot_c += resp.sum()
bps, bpc = bps.view(NMAX, NB).cpu(), bpc.view(NMAX, NB).cpu()
def pooled(b_lo, b_hi, p_lo, p_hi):
    c = bpc[b_lo:b_hi, p_lo:p_hi].sum().item(); return float(bps[b_lo:b_hi, p_lo:p_hi].sum().item() / c) if c else float("nan")
rec = {"run": RUN, "arm": arm, "C": C, "K": K, "head_code_quotas": Q0, "code_positions": cp, **kw, "examples": len(evalp),
       "held_out_ce": float(tot_s / tot_c.clamp_min(1)), "position_bins": [0, 1, 16, 64, 256, C],
       "blockpos_sum": bps.tolist(), "blockpos_count": bpc.tolist(), "sec": round(time.time() - t0, 1),
       "summary": {"by_block_1-4/5-8/9-16/17+": [pooled(1, 5, 0, NB), pooled(5, 9, 0, NB), pooled(9, 17, 0, NB), pooled(17, NMAX, 0, NB)],
                   "pos1-15": pooled(1, NMAX, 1, 2), "pos16+": pooled(1, NMAX, 2, NB),
                   "pos1-15_by_block_1-4/5-8/9-16/17+": [pooled(1, 5, 1, 2), pooled(5, 9, 1, 2), pooled(9, 17, 1, 2), pooled(17, NMAX, 1, 2)]}}
json.dump(rec, open(OUT, "w"))
print(f"[blockce] held_out_ce={rec['held_out_ce']:.5f} {json.dumps(rec['summary'])} -> {OUT} ({rec['sec']}s)", flush=True)
