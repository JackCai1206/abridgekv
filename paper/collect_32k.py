#!/usr/bin/env python3
"""Collect every 32K same-set evaluation into paper/data/results32k.json.
Per arm (tag from results.json arms32k): per-set avg@8 (semantic grader = strict last-boxed OR math-verify), pooled competition
suite (13 files / 446 problems) with a per-problem bootstrap CI, paired difference vs vanilla on common problems, MATH-500 avg@8,
truncation and mean generated tokens. Eviction arms come from the HF eviction records (8 single-sample seeds); their pooling is
unpaired because their per-problem index is shard-local."""
import json, glob, os, random, collections, statistics
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
EV = os.path.join(os.environ.get("OUTPUT_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")), "eval"); CAP = 31744
SUITE = json.load(open(os.path.join(ROOT, "data/math_eval/SUITE.json")))
# manifest name -> merged-record set names (eval_suite uses file stems; AIME 2024/2025 come from the eval_aime loader)
ALIAS = {"aime_2025": ["aime2025"], "aime_2024": ["aime2024_I", "aime2024_II"]}
COMP_META = {s: m for s, m in SUITE.items() if m["era"] in ("post_cutoff", "pre_2025")}   # 12 competitions, 446 problems
COMP = [x for s in COMP_META for x in ALIAS.get(s, [s])]                                 # 13 record sets
N_COMP = sum(m["n"] for m in COMP_META.values())
R = json.load(open(os.path.join(HERE, "results.json")))
random.seed(0); B = 2000

def load_set(tag, s):
    fs = glob.glob(f"{EV}/{s}_qwen3_4b_vllm_{tag}_merged/*.json")
    if not fs: return None
    j = json.load(open(fs[0])); per = {}; gen = []; tr = 0; gp = {}
    for p in j["per_problem"]:
        v = [float(x["correct_semantic_full_text"]) for x in p["samples"]]
        k = min(len(v), 8)                                    # competitions are avg@8; MATH-500 is avg@4 from 2026-09-21
        per[(s, p["dataset_index"])] = sum(v[:k]) / k
        # Figure 3's x axis is (mean generated tokens x decode cost), so its error bar comes from the SAMPLING noise of
        # that mean over problems -- not from the decode bench, whose three repeats agree to 0.04-0.5%. The unit of
        # sampling is the PROBLEM (samples within a problem are correlated), so pool per-problem means, not raw samples.
        gp[(s, p["dataset_index"])] = sum(x["gen_tokens"] for x in p["samples"][:k]) / k
        for x in p["samples"]: gen.append(x["gen_tokens"]); tr += x["gen_tokens"] >= CAP
    return per, gen, tr, gp

def load_evict(policy, s):
    # The HF eviction shards are round-robin over the set: shard k of N holds dataset indices k, k+N, k+2N, ... so a record's
    # dataset index is shard + nshard*idx, which pairs it with the vLLM arms on the same problems (2026-09-26; before this
    # the key was shard-local and the eviction rows could only be reported unpaired).
    per = collections.defaultdict(list); gen = []; tr = 0; gpl = collections.defaultdict(list)
    for f in glob.glob(f"{EV}/hf_evict_{policy}_k128_32k/{s}_s*_shard*.json"):
        j = json.load(open(f))
        for p in j["per_problem"]:
            key = (s, j["shard"] + j["nshard"] * p["idx"])
            per[key].append(float(p["correct"])); g = p.get("gen_tokens", 0); gen.append(g); tr += g >= CAP
            gpl[key].append(g)
    full = {k: sum(v[:8]) / 8 for k, v in per.items() if len(v) >= 8}
    gp = {k: sum(v[:8]) / len(v[:8]) for k, v in gpl.items() if k in full}
    return (full, gen, tr, gp) if full else None

def load_evict_m500(policy):
    """MATH-500 for the eviction policies: one HF sample per seed (T=0.6), 50 round-robin shards of 10 problems per seed;
    graded on the last boxed answer like the strict column of the other arms. Every finished sample counts, from complete
    and still-running units alike (user 2026-09-26: a number now, whatever the seeds): the cell is the problem-weighted mean,
    each problem averaged over the seeds that have it. Once every seed is complete this is exactly avg@k."""
    by_seed = collections.defaultdict(dict)
    for f in glob.glob(f"{EV}/hf_evict_{policy}_k128_m500/math500_s*_shard*.json"):
        j = json.load(open(f))
        for p in j["per_problem"]: by_seed[j["evict_seed"]][j["shard"] + j["nshard"] * p["idx"]] = float(p["correct"])
    probs = sorted(set.union(*(set(d) for d in by_seed.values()))) if by_seed else []
    if not probs: return None
    v = [sum(by_seed[sd][q] for sd in by_seed if q in by_seed[sd]) / sum(q in by_seed[sd] for sd in by_seed) for q in probs]
    nsamp = sum(len(d) for d in by_seed.values()); complete = sorted(sd for sd, d in by_seed.items() if len(d) >= 500)
    return {"n": len(v), "samples": nsamp, "k": len(complete), "seeds": complete, "avg8_strict": 100 * sum(v) / len(v)}

def load_set_strict(tag, s):
    # The MATH-500 column of every table is strict last-boxed (tab:main's caption); the suite columns are semantic.
    # Sweep-only arms fell through to the semantic number here while arms carried in results.json were strict, so the
    # sweep rows read 0.6-0.8 pt high against the very arm they are compared with (512/64 printed 87.3 for a strict
    # 86.7; 1024/256 printed 88.0 for 87.2). One grader per column (2026-09-23).
    fs = glob.glob(f"{EV}/{s}_qwen3_4b_vllm_{tag}_merged/*.json")
    if not fs: return None
    j = json.load(open(fs[0])); v = []
    for p in j["per_problem"]:
        c = [float(x["correct"]) for x in p["samples"]]; k = min(len(c), 8); v.append(sum(c[:k]) / k)
    return 100 * sum(v) / len(v) if v else None

# each interval gets its own seeded stream over a sorted problem list, so it does not depend on which other arms were collected
# or in what order (a shared stream shifted every later arm's interval whenever an arm was added)
def boot_mean(vals):
    rng = random.Random(0); vals = sorted(vals)
    n = len(vals); ms = sorted(sum(rng.choice(vals) for _ in range(n)) / n for _ in range(B))
    return 100 * ms[int(0.025 * B)], 100 * ms[int(0.975 * B)]

def boot_paired(a, b):
    keys = sorted(k for k in a if k in b); d = [a[k] - b[k] for k in keys]
    if not d: return None
    rng = random.Random(0); n = len(d); ms = sorted(sum(rng.choice(d) for _ in range(n)) / n for _ in range(B))
    return [100 * sum(d) / n, 100 * ms[int(0.025 * B)], 100 * ms[int(0.975 * B)], n]

out = {"_doc": __doc__, "n_competition": N_COMP, "competitions": {s: {"n": m["n"], "era": m["era"], "source": m["source"]} for s, m in COMP_META.items()}, "arms": {}}
arms = {}
for a in R["arms32k"] + [x for x in R["scaling32k"] if x.get("tag") and x["key"] not in {y["key"] for y in R["arms32k"]}]:
    src = a.get("tag") or a.get("evict")
    if not src: continue
    per_all = {}; sets = {}; gen = []; tr = 0; gp_all = {}
    for s in COMP + ["math500"]:
        r = load_evict(a["evict"], s) if a.get("evict") else load_set(a["tag"], s)
        if r is None: continue
        per, g, t, gp = r; sets[s] = {"n": len(per), "avg8": 100 * sum(per.values()) / len(per), "trunc": 100 * t / len(g), "gen": sum(g) / len(g)}
        if s != "math500": per_all.update(per); gen += g; tr += t; gp_all.update(gp)
    rec = {"sets": sets}
    if per_all:
        vals = list(per_all.values()); lo, hi = boot_mean(vals)
        rec["pool"] = {"n": len(vals), "n_expected": N_COMP, "complete": len(vals) >= N_COMP, "avg8": 100 * sum(vals) / len(vals), "ci": [lo, hi],
                       "trunc": 100 * tr / len(gen), "gen": sum(gen) / len(gen), "sets_done": sorted(sets),
                       "gen_se": (statistics.stdev(gp_all.values()) / len(gp_all) ** 0.5) if len(gp_all) > 1 else None}
    if "math500" in sets:
        rec["math500"] = sets["math500"]
        if a.get("tag"):
            st = load_set_strict(a["tag"], "math500")
            if st is not None: rec["math500"]["avg8_strict"] = st
    elif a.get("evict"):
        m = load_evict_m500(a["evict"])
        if m: rec["math500"] = m
    arms[a["key"]] = (rec, per_all)
van = arms.get("vanilla", ({}, {}))[1]; ours = arms.get("vpl", ({}, {}))[1]
MATCHED = {"rkv_b4096": "vpl", "rkv_b2048": "ck512k64",   # each R-KV budget vs the arm its residency brackets
           "gist": "uniform128"}                        # appended positions vs the in-block uniform arm it differs from in positions only
ALL = R["arms32k"] + R["scaling32k"]
for k, (rec, per) in arms.items():
    if k != "vanilla" and per and van:   # eviction arms pair on their 60-problem subset (dataset index recovered from the shard)
        rec["diff_vs_vanilla"] = boot_paired(per, van)
    if k != "vpl" and per and ours and not next(x for x in ALL if x["key"] == k).get("evict"):
        rec["diff_vs_ours"] = boot_paired(per, ours)   # paired against the headline (per-layer dynamic allocation)
    # ... but the headline is the wrong partner for an arm that brackets a DIFFERENT budget. R-KV's two settings are
    # matched to our 4:1 and 8:1 arms by measured residency, so budget 2048 must be paired against the 8:1 arm; against
    # the 4:1 headline it reads -7.5, which is a budget difference and not a method difference.
    if k in MATCHED and per:
        partner = arms.get(MATCHED[k], ({}, {}))[1]
        if partner: rec["diff_vs_matched"] = boot_paired(per, partner)
    out["arms"][k] = rec
json.dump(out, open(os.path.join(HERE, "data/results32k.json"), "w"), indent=1)
for k, rec in out["arms"].items():
    p = rec.get("pool"); m = rec.get("math500")
    print(f"{k:14s} pool n={p['n'] if p else 0}/{N_COMP} avg8={p['avg8']:.1f} [{p['ci'][0]:.1f},{p['ci'][1]:.1f}] trunc={p['trunc']:.1f}%" if p else f"{k:14s} pool: none",
          f"| math500 avg@{m.get('k',4)}={m.get('avg8', m.get('avg8_strict')):.1f}" if m else "| math500: none", f"| d_vs_vanilla={rec.get('diff_vs_vanilla')}")
