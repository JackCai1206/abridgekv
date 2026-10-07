#!/usr/bin/env python3
"""Per-sample generated lengths on the pooled 446-problem suite (32K cap), for the output-length panel of Figure 1.
Same records as collect_32k.py (vLLM merged sets; HF eviction records for the codes-free policies)."""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import collect_32k as C
ARMS = {"vanilla": ("tag", "vanilla_sameset32k"), "vpl": ("tag", "c256tpl_sameset32k"),
        "uniform128": ("tag", "c256uni_u128_sameset32k"), "rkv_b4096": ("tag", "rkv_b4096_sameset32k"),
        "rkv_b2048": ("tag", "rkv_b2048_sameset32k"), "evict_recency": ("evict", "recency"),
        "lt_thought": ("tag", "lt_thought")}   # LightThinker's algorithm on our stack (user 2026-09-26: on the length panel)
out = {"cap": C.CAP, "arms": {}}
for k, (kind, t) in ARMS.items():
    g = []
    for s in C.COMP:
        r = C.load_evict(t, s) if kind == "evict" else C.load_set(t, s)
        if r: g += r[1]
    out["arms"][k] = g
    print(k, len(g), round(sum(g) / max(len(g), 1)), "trunc %.1f%%" % (100 * sum(x >= C.CAP for x in g) / max(len(g), 1)))
json.dump(out, open(os.path.join(HERE, "data", "gen_lengths.json"), "w"))
