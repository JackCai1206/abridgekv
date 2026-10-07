#!/usr/bin/env python3
"""Run-to-completion capacity cells (scripts/bench_decode.py --mode complete) -> paper/data/completion.json.
Only full-scale (len_scale 1), fully finished, H100 runs of the paired workload in data/workload_complete.json.
The Pareto saturated panel plots s_per_problem directly: wall time to finish all N requests / N."""
import glob, json, os, statistics
HERE = os.path.dirname(os.path.abspath(__file__))
D = os.environ.get("COMP_DIR", os.path.join(os.environ.get("OUTPUT_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")), "bench"))
N = int(os.environ.get("N", 512)); SCALE = float(os.environ.get("SCALE", 1))
OUTF = os.environ.get("OUT", os.path.join(HERE, "data", "completion.json"))
cells = {}
# COMP_EXTRA: colon-separated sibling directories measured on the same GPU type with the same workload file (the
# LightThinker arm ran from its own output directory, 2026-09-26).
DIRS = [D] + [d for d in os.environ.get("COMP_EXTRA", "").split(":") if d]
for f in sorted(f for d in DIRS for f in glob.glob(f"{d}/*_complete_N{N}_S{SCALE:g}_*r*.json")):
    r = json.load(open(f))
    if not r.get("all_finished") or r.get("len_scale") != SCALE: continue
    cells.setdefault(r["tag"], []).append(r)
out = {"_doc": __doc__, "dir": D, "n": N, "arms": {}}
# Each arm is reported at its BEST concurrency cap (max_num_seqs), as a deployment would tune it: over-admitting a cache
# that cannot hold the admitted sequences at full length makes vLLM preempt and recompute them.
for t, rs in list(cells.items()):
    by_m = {}
    for x in rs: by_m.setdefault(x.get("max_num_seqs"), []).append(x)
    best_m = min(by_m, key=lambda m: statistics.median(x["s_per_problem"] for x in by_m[m]))
    cells[t] = by_m[best_m]
    out.setdefault("cap_sweep", {})[t] = {str(m): statistics.median(x["s_per_problem"] for x in v) for m, v in sorted(by_m.items())}
for t, rs in cells.items():
    med = lambda k: statistics.median(x[k] for x in rs if x.get(k) is not None)
    out["arms"][t] = {"reps": len(rs), "s_per_problem": med("s_per_problem"), "s_all": [x["s_per_problem"] for x in rs],
                      "t90_s": med("t90_s"), "mean_len": rs[0]["mean_len"], "tok_per_s": med("tok_per_s"),
                      "max_running": max(x["max_running"] for x in rs), "steady_running": med("steady_running"),
                      "max_num_seqs": rs[0].get("max_num_seqs"), "preemptions": rs[0].get("preemptions")}
v = out["arms"].get("vanilla_graphs")
for t, a in sorted(out["arms"].items(), key=lambda kv: kv[1]["s_per_problem"]):
    if v: a["vs_vanilla"] = a["s_per_problem"] / v["s_per_problem"]
    print("%-15s %.3f s/problem  %s  cap=%s  sweep=%s" % (t, a["s_per_problem"], ("%.2fx" % a["vs_vanilla"]) if v else "", a["max_num_seqs"], out.get("cap_sweep", {}).get(t)))
json.dump(out, open(OUTF, "w"), indent=1)
