#!/usr/bin/env python3
"""Collect the saturation (max-concurrency) cells for the concurrency Pareto and the appendix serving table.

ONLY cells from ONE job on ONE node are used (outputs/bench_sat1job).
Saturation throughput is not reproducible ACROSS jobs on this shared cluster (ck256k64: 3400.8 vs 2912.7 tok/s at an
identical configuration, 17% apart), while the ratio to vanilla measured INSIDE a job reproduces to ~5%. Concurrency
(steady_running) is reproducible either way. So every throughput here is quoted against the vanilla cell of the same job.

R-KV cannot run in that job (separate vLLM 0.25.1 venv) and no valid R-KV saturation cell exists: with the scoring cap
low enough to fit, preempted requests exceed it and fall back to Full KV (RKV-SKIP); with it high enough to avoid that,
the scoring transient OOMs against the KV pool (1.5x/2.0x/2.5x margins all failed, 2026-09-23). Its concurrency is
memory-derived and was measured consistently (108 at 128 requests, 107 at 512), so that alone is recorded.
"""
import glob, json, os, statistics as st
OUT = os.environ.get("OUTPUT_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs"))
JOB_DIR = f"{OUT}/bench_sat1job"

def valid(c):
    req, run = c.get("requests"), c.get("steady_running")
    return bool(c.get("tok_per_s") and req and run and 2 * run < req)   # 2x headroom: memory-limited, not request-limited

cells = {}
for f in sorted(glob.glob(f"{JOB_DIR}/*_saturate_B*_L16384_r*.json")):
    if os.path.basename(f).startswith(("UNSATURATED_", "NOCOMPACT_")): continue
    c = json.load(open(f))
    if valid(c): cells.setdefault(c["tag"], []).append(c)

arms = {}
for tag, cs in cells.items():
    tp = [c["tok_per_s"] for c in cs]; run = [c["steady_running"] for c in cs]
    arms[tag] = {"reps": len(cs), "tok_per_s": st.median(tp), "tok_per_s_all": tp,
                 "steady_running": st.median(run), "steady_running_all": run,
                 "step_ms_median": st.median(c["step_ms_median"] for c in cs), "requests": cs[0]["requests"]}
van = arms.get("vanilla_graphs")
for a in arms.values():
    a["tp_vs_vanilla"] = a["tok_per_s"] / van["tok_per_s"] if van else None
    a["conc_vs_vanilla"] = a["steady_running"] / van["steady_running"] if van else None

# R-KV concurrency, from the saturation logs (bench_rkv.sbatch jobs, H100, VLLM_V1_R_KV_FREE_BLOCKS=1). Recorded by
# hand because every one of these cells was DISCARDED for throughput (skips or OOM) and renamed on disk; concurrency is
# the memory-derived quantity and is still informative.
#   b4096: 108 steady at 128 requests (job 14328728, 0 skips) and 107 at 512 requests (job 14332195, 1 skip) -> 107-108.
#   b2048: 211 at 512 requests (job 14335829, 3 skips). A skipped request runs Full KV and holds MORE cache, so 211 is a
#          LOWER bound on clean b2048 concurrency. (Its 128-request cell read 128/128 and is request-limited, not used.)
rkv = {"rkv_b4096": {"steady_running": 107.5, "range": [107, 108], "bound": "point",
                     "throughput": None, "note": "no valid saturation throughput cell"},
       "rkv_b2048": {"steady_running": 211, "range": [211, 211], "bound": "lower",
                     "throughput": None, "note": "no valid saturation throughput cell; concurrency is a lower bound"}}

json.dump({"_doc": __doc__, "job_dir": JOB_DIR, "arms": arms, "rkv": rkv},
          open(os.path.join(os.path.dirname(__file__), "data", "saturation.json"), "w"), indent=1)
for t, a in sorted(arms.items(), key=lambda kv: kv[1]["tok_per_s"]):
    print(f"{t:16s} reps={a['reps']} tok/s={a['tok_per_s']:7.1f} ({a['tp_vs_vanilla']:.2f}x)  conc={a['steady_running']:.0f} ({a['conc_vs_vanilla']:.1f}x)")
print("rkv:", {k: (v["steady_running"], v["bound"]) for k, v in rkv.items()})
