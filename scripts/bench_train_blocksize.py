#!/usr/bin/env python3
"""Training cost of the compression block size at a fixed 4:1 ratio (2026-09-25).

One optimizer-free training step (forward + backward) of the trainer's CompressLM on a single GPU, with the trainer's
settings: fp32 master weights under bf16 autocast, FlexAttention, bit-reversed in-block code positions, cross-block loss,
activation checkpointing. The response length is fixed (TOKENS generated tokens, split into TOKENS/C blocks), so every
block size does the same number of content tokens and, at 4:1, the same number of code tokens; only the attention pattern
changes. Reports median step time over REPS timed steps (after WARM warm-up steps, which absorb compilation) and the
peak allocated memory. Weights are the base model's: cost does not depend on the trained values.

  python scripts/bench_train_blocksize.py --out paper/data/train_cost_blocksize.json
"""
import argparse, json, os, statistics, sys, time
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from transformers import AutoModelForCausalLM
from abridgekv.compress_forward import CompressLM

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="Qwen/Qwen3-4B-Base")
ap.add_argument("--tokens", type=int, default=16384)
ap.add_argument("--blocks", default="256,512,1024,2048,4096")
ap.add_argument("--ratio", type=int, default=4)
ap.add_argument("--warm", type=int, default=2)
ap.add_argument("--reps", type=int, default=4)
ap.add_argument("--grad_ckpt", type=int, default=1)
ap.add_argument("--out", required=True)
a = ap.parse_args()

dev = "cuda"
torch.manual_seed(0)
res = {"gpu": torch.cuda.get_device_name(0), "tokens": a.tokens, "ratio": a.ratio, "grad_ckpt": a.grad_ckpt, "cells": {}}
for C in [int(x) for x in a.blocks.split(",")]:
    K, N = C // a.ratio, a.tokens // C
    # a fresh base per block size: CompressLM adds pause embeddings and switches the attention implementation in place
    base = AutoModelForCausalLM.from_pretrained(a.base, torch_dtype=torch.float32).to(dev)
    if a.grad_ckpt:
        base.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    vocab = base.config.vocab_size
    core = CompressLM(base, C, N=N, K=K, code_positions="bitrev").to(dev)
    core.train()
    ids = torch.randint(0, vocab - 1000, (1, N, C), device=dev)
    times = []
    torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
    for it in range(a.warm + a.reps):
        t0 = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = core(ids, ce_chunk=4096, include_cross_block=True)
        out["ce"].backward()
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        for p in core.parameters(): p.grad = None
        if it >= a.warm: times.append(dt)
        print(f"[bench] C={C} K={K} N={N} step {it} {dt:.3f}s", flush=True)
    peak = torch.cuda.max_memory_allocated() / 2**30
    res["cells"][str(C)] = {"K": K, "N": N, "step_s": statistics.median(times), "steps": times, "peak_gib": peak}
    print(f"[bench] C={C}: median {statistics.median(times):.3f}s peak {peak:.1f} GiB", flush=True)
    prev = json.load(open(a.out)) if os.path.exists(a.out) else {}
    prev.setdefault("cells", {}).update(res["cells"]); prev.update({k: v for k, v in res.items() if k != "cells"})
    json.dump(prev, open(a.out, "w"), indent=1)   # one process per block size (Inductor cannot recompile FlexAttention
                                                 # for a second block shape in one process), so cells merge into one file
    del core, base; torch.cuda.empty_cache()
print("[bench] done", flush=True)
