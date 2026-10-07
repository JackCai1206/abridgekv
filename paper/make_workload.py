#!/usr/bin/env python3
"""Workload for the run-to-completion capacity bench (scripts/bench_decode.py --mode complete).

The same N (problem, sample) draws for every arm, from the 446-problem competition suite at 32K; each arm replays ITS OWN
generated length for each draw, so an arm that generates more pays for it, and arms are paired draw by draw. The draw
order is the queue order, shuffled once with a fixed seed and shared by every arm."""
import json, os, random, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import collect_32k as C          # load_set, COMP; importing it rebuilds results32k.json, which is idempotent
N = int(os.environ.get("N", 512)); SEED = 0
# bench tag -> suite eval tag. R-KV replays its own (much longer) lengths; its bench runs in its own venv.
ARMS = {"vanilla_graphs": "vanilla_sameset32k", "tpl_physical": "c256tpl_sameset32k",
        "uni_plain": "c256uni_u128_sameset32k", "ck1024k256": "c1024k256_bitrev_sameset32k",
        "ck512k256": "c512k256_bitrev_sameset32k", "ck256k64": "c256k64_bitrev_sameset32k",
        "ck512k64": "ck512k64_sameset32k", "gist": "c128uni_appended_gist_sameset32k", "lt": "lt64k9_sameset32k",
        "rkv_b4096": "rkv_b4096_sameset32k", "rkv_b2048": "rkv_b2048_sameset32k"}
def samples(tag):
    out = {}
    for s in C.COMP:
        fs = C.glob.glob(f"{C.EV}/{s}_qwen3_4b_vllm_{tag}_merged/*.json")
        for p in json.load(open(fs[0]))["per_problem"]:
            for k, x in enumerate(p["samples"][:8]): out[(s, p["dataset_index"], k)] = x["gen_tokens"]
    return out
S = {a: samples(t) for a, t in ARMS.items()}
common = sorted(set.intersection(*(set(v) for v in S.values())))
rng = random.Random(SEED); draws = rng.sample(common, N)
W = {"_doc": __doc__, "n": N, "seed": SEED, "draws": [list(d) for d in draws],
     "arms": {a: {"tag": ARMS[a], "lengths": [S[a][d] for d in draws],
                  "mean": sum(S[a][d] for d in draws) / N, "suite_mean": sum(S[a].values()) / len(S[a])} for a in ARMS}}
json.dump(W, open(os.path.join(HERE, "data", "workload_complete.json"), "w"))
for a, v in W["arms"].items(): print("%-15s draw mean %6.0f  suite mean %6.0f  max %d" % (a, v["mean"], v["suite_mean"], max(v["lengths"])))
