#!/usr/bin/env python3
"""Summarize scripts/bench_decode.py records into paper/data/bench_decode.json + a readable table.

  python3 scripts/summarize_bench.py --bench $OUTPUT_BASE/bench $OUTPUT_BASE/bench_rkv --out paper/data/bench_decode.json
"""
import argparse, glob, json, os
from collections import defaultdict
from statistics import median

N_LAYERS = 36                                  # Qwen3-4B
KV_BYTES_PER_TOKEN_PER_LAYER = 8 * 128 * 2 * 2 # kv heads x head dim x (K+V) x bf16 = 4 KB
KV_BYTES_PER_TOKEN = N_LAYERS * KV_BYTES_PER_TOKEN_PER_LAYER   # full KV of one token, all layers = 147 KB


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True, nargs="+",
                    help="one or more record directories; R-KV lives in its own venv and writes to bench_rkv/")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    # NOCOMPACT_*.json are cells the bench gates REJECTED (an R-KV run that logged a single RKV-SKIP fell back to
    # Full KV and is timing vanilla with hooks attached). They are renamed, not deleted, so the failure stays
    # inspectable -- but a plain *.json glob swept them straight back into the paper: on 2026-09-23 the R-KV
    # batch-32 row was built entirely from three rejected cells. Skip them here.
    # UNSATURATED_*.json are saturation cells where every queued request stayed resident, so the arm never reached
    # its concurrency ceiling and its throughput is a floor, not a measurement.
    files = sorted(f for d in args.bench for f in glob.glob(os.path.join(d, "*.json"))
                   if not os.path.basename(f).startswith(("NOCOMPACT_", "UNSATURATED_")))
    recs = [json.load(open(f)) for f in files]
    recs = [r for r in recs if "tag" in r and r.get("steps")]
    cells = defaultdict(list)
    for r in recs:
        cells[(r["tag"], r["mode"], r["len"], r["batch"] if r["mode"] == "batch" else r["requests"])].append(r)

    out = {"cells": [], "kv_bytes_per_token": KV_BYTES_PER_TOKEN}
    for (tag, mode, length, batch), rs in sorted(cells.items()):
        blocks = [r["kv_blocks_peak"] for r in rs if r.get("kv_blocks_peak")]
        bs = next((r["kv_block_size"] for r in rs if r.get("kv_block_size")), None)
        running = max((r["max_running"] or 0) for r in rs)
        # With N cache groups, vLLM lays the pool out as ceil(N_LAYERS/N) tensors, each shared by one layer of
        # every group (kv_cache_utils._get_kv_cache_config_uniform_page_size). A block id is owned by exactly one
        # (request, group), so it costs ONE page in each of those tensors: layers_per_group layers of KV, not 36.
        ngrp = next((r.get("kv_cache_groups") for r in rs if r.get("kv_cache_groups")), 1) or 1
        layers_per_group = -(-N_LAYERS // ngrp)
        kv_bytes = (median(blocks) * bs * layers_per_group * KV_BYTES_PER_TOKEN_PER_LAYER) if blocks and bs else None
        # Apples-to-apples with the vanilla full-KV arm: how many full-KV (36-layer) tokens the same bytes would hold.
        kv_tokens = round(kv_bytes / KV_BYTES_PER_TOKEN) if kv_bytes else None
        out["cells"].append({
            "tag": tag, "mode": mode, "len": length, "batch": batch, "reps": len(rs),
            "tok_per_s": median([r["tok_per_s"] for r in rs if r.get("tok_per_s")]) if rs else None,
            "steady_tok_per_s": median([r["steady_tok_per_s"] for r in rs if r.get("steady_tok_per_s")]) or None,
            "step_ms_median": median([r["step_ms_median"] for r in rs if r.get("step_ms_median")]) or None,
            "max_running": running,
            # concurrency that means "full-length sequences held at once": max_running is the early peak, when every
            # admitted request is still short. Take the running count at the step where the cache is fullest.
            "running_at_kv_peak": max((r.get("running_at_kv_peak") or 0) for r in rs) or None,
            "steady_running": max((r.get("steady_running") or 0) for r in rs) or None,
            "kv_blocks_peak": median(blocks) if blocks else None,
            "kv_blocks_total": next((r.get("kv_blocks_total") for r in rs if r.get("kv_blocks_total")), None),
            "kv_layers_per_group": layers_per_group,
            "kv_tokens_peak": kv_tokens,              # full-KV-equivalent tokens (bytes / 147 KB)
            "kv_gib_peak": round(kv_bytes / 2**30, 2) if kv_bytes else None,
            "kv_tokens_per_seq": round(kv_tokens / running) if kv_tokens and running else None,
            "kv_cache_groups": ngrp if ngrp > 1 else next((r.get("kv_cache_groups") for r in rs), None),
            "kv_group_blocks_peak": next((r.get("kv_group_blocks_peak") for r in rs if r.get("kv_group_blocks_peak")), None),
            "kv_group_quotas": next((r.get("kv_group_quotas") for r in rs if r.get("kv_group_quotas")), None),
            "physical_quotas": rs[0].get("physical_quotas"),
        })
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    hdr = f"{'arm':16} {'mode':9} {'len':>6} {'B':>4} {'tok/s':>9} {'step ms':>8} {'run':>4} {'KV tok/seq':>11} {'KV GiB':>7} {'grp':>4}"
    print(hdr); print("-" * len(hdr))
    for c in out["cells"]:
        print(f"{c['tag']:16} {c['mode']:9} {c['len']:>6} {c['batch']:>4} {str(c['tok_per_s']):>9} "
              f"{str(c['step_ms_median']):>8} {str(c['max_running']):>4} {str(c['kv_tokens_per_seq']):>11} "
              f"{str(c['kv_gib_peak']):>7} {str(c['kv_cache_groups']):>4}")
    # Validation: with physical per-layer quotas a group's residency is AFFINE in its quota -- blocks*16 =
    # (completed blocks) x quota + raw, where raw is the prompt plus the still-open block and is shared by every
    # group. Comparing raw block shares against quota shares would wrongly look like a mismatch, because the
    # shared raw part is not proportional to the quota. Fit the line and report its residual instead.
    for c in out["cells"]:
        gb, gq = c.get("kv_group_blocks_peak"), c.get("kv_group_quotas")
        if not gb or not gq or len(set(gq)) < 2 or any(q is None for q in gq):
            continue
        n = len(gq)
        mq, mb = sum(gq) / n, sum(gb) / n
        den = sum((q - mq) ** 2 for q in gq)
        slope = sum((q - mq) * (b - mb) for q, b in zip(gq, gb)) / den if den else 0.0
        icept = mb - slope * mq
        resid = max(abs(b - (slope * q + icept)) for q, b in zip(gq, gb))
        seqs = max(1, c.get("max_running") or 1)
        print(f"  per-group residency {c['tag']} B{c['batch']} L{c['len']}: blocks = {slope:.4f}*quota + {icept:.1f}, "
              f"max residual {resid:.2f} blocks -> per sequence {slope * 16 / seqs:.2f} compacted blocks per quota row, "
              f"{icept * 16 / seqs:.0f} raw tokens")
    print(f"\n[bench] {len(recs)} records -> {args.out}")


if __name__ == "__main__":
    main()
