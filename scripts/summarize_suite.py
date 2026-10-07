#!/usr/bin/env python3
"""Pool every competition eval per arm: per-set avg@k / pass@k, era pools (post_cutoff / pre_2025 / hard / all, from
data/math_eval/SUITE.json), and paired per-problem bootstrap CIs of the difference against reference arms.

  python scripts/summarize_suite.py --root $OUTPUT_BASE/eval [--ref vanilla_canonical --ref c256uni_u128] [--out json] [--md md]

Discovers <set>_qwen3_4b_vllm_<tag>_merged/<arm>.json (suite sets, hmmt = hmmt_feb_2025, hmmt2024 = hmmt_feb_2024, math500) and
aime_qwen3_4b_vllm_<tag>_{merged,aime_shard*}/*aime{2024,2025}.json. Tags are the EVAL_TAG of each arm (vanilla_canonical,
c256uni_u128, c256tpl_bitrev, c256kd1_bitrev, c256uni_tail16, ...). Per-problem accuracy = mean over samples; pooled numbers are
problem-weighted (every problem counts once); "strict" grading (correct / correct_A).
"""
import argparse, glob, json, os, random, re, sys
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True); ap.add_argument("--manifest", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "math_eval", "SUITE.json"))
ap.add_argument("--ref", action="append", default=None); ap.add_argument("--out", default=""); ap.add_argument("--md", default="")
ap.add_argument("--exclude", default="smoke,step50,bench,tput,plaincheck,layeredtput"); ap.add_argument("--boot", type=int, default=2000)
a = ap.parse_args()
refs = a.ref or ["vanilla_canonical", "c256uni_u128"]
manifest = json.load(open(a.manifest)) if os.path.exists(a.manifest) else {}
era_of = {s: m["era"] for s, m in manifest.items()}
excl = [x for x in a.exclude.split(",") if x]
SETMAP = {"hmmt": "hmmt_feb_2025", "hmmt2024": "hmmt_feb_2024", "math500": "math500"}

def flags_of(problem):
    out = []
    for s in problem["samples"]:
        v = s.get("correct_A", s.get("correct"))
        out.append(bool(v))
    return out

# results[tag][set] = {problem_id: [bools]}
results = defaultdict(dict)
for d in sorted(glob.glob(os.path.join(a.root, "*_qwen3_4b_vllm_*_merged"))):
    base = os.path.basename(d)
    m = re.match(r"(.+?)_qwen3_4b_vllm_(.+)_merged$", base)
    if not m: continue
    prefix, tag = m.groups()
    if any(x in tag for x in excl): continue
    if prefix == "aime":
        for f in glob.glob(os.path.join(d, "*aime20*.json")):
            yr = re.search(r"aime(20\d\d)", os.path.basename(f)); 
            if not yr: continue
            D = json.load(open(f)); per = {p["dataset_index"]: flags_of(p) for p in D["per_problem"]}
            results[tag].setdefault(f"aime_{yr.group(1)}", {}).update(per)
        continue
    setname = SETMAP.get(prefix, prefix)
    for f in glob.glob(os.path.join(d, "*.json")):
        arm = os.path.basename(f)[:-5]
        if arm not in ("compressed", "vanilla", "base"): continue
        D = json.load(open(f))
        if "per_problem" not in D: continue
        results[tag][setname] = {p["dataset_index"]: flags_of(p) for p in D["per_problem"]}
# AIME shard dirs not yet merged (e.g. a running eval)
for d in sorted(glob.glob(os.path.join(a.root, "aime_qwen3_4b_vllm_*_aime_shard*"))):
    m = re.match(r"aime_qwen3_4b_vllm_(.+)_aime_shard\d+$", os.path.basename(d))
    if not m or any(x in m.group(1) for x in excl): continue
    tag = m.group(1)
    for f in glob.glob(os.path.join(d, "*aime20*.json")):
        yr = re.search(r"aime(20\d\d)", os.path.basename(f)).group(1); D = json.load(open(f))
        results[tag].setdefault(f"aime_{yr}", {}).update({p["dataset_index"]: flags_of(p) for p in D["per_problem"]})

def acc(per): return 100 * sum(sum(f) / len(f) for f in per.values()) / len(per)
def passk(per): return 100 * sum(any(f) for f in per.values()) / len(per)
def pool(tag, sets):
    return {(s, pid): f for s in sets if s in results[tag] for pid, f in results[tag][s].items()}
def boot_diff(pa, pb, n):
    keys = sorted(set(pa) & set(pb)); 
    if not keys: return None
    da = [sum(pa[k]) / len(pa[k]) for k in keys]; db = [sum(pb[k]) / len(pb[k]) for k in keys]
    rng = random.Random(0); diffs = []
    for _ in range(n):
        idx = [rng.randrange(len(keys)) for _ in keys]
        diffs.append(100 * (sum(da[i] for i in idx) - sum(db[i] for i in idx)) / len(keys))
    diffs.sort(); return {"n_problems": len(keys), "mean_diff": 100 * (sum(da) - sum(db)) / len(keys), "ci95": [diffs[int(0.025 * n)], diffs[int(0.975 * n) - 1]]}

tags = sorted(results); all_sets = sorted({s for t in tags for s in results[t]}, key=lambda s: (era_of.get(s, "z"), s))
eras = ["post_cutoff", "pre_2025", "competition"]
summary = {"tags": {}, "sets": {s: {"era": era_of.get(s, "?"), "n": max(len(results[t].get(s, {})) for t in tags)} for s in all_sets}}
for t in tags:
    row = {"per_set": {s: {"n": len(results[t][s]), "k": len(next(iter(results[t][s].values()))), "avg_at_k": acc(results[t][s]), "pass_at_k": passk(results[t][s])} for s in results[t]}, "pools": {}, "vs": {}}
    for e in eras + ["all"]:
        sets = [s for s in all_sets if (e == "all" and era_of.get(s) != "excluded_too_hard") or (e == "competition" and era_of.get(s) in ("post_cutoff", "pre_2025")) or era_of.get(s) == e]
        p = pool(t, sets)
        if p: row["pools"][e] = {"n_problems": len(p), "sets": [s for s in sets if s in results[t]], "avg_at_k": acc(p), "pass_at_k": passk(p)}
    for r in refs:
        if r in results and r != t:
            row["vs"][r] = {e: boot_diff(pool(t, [s for s in all_sets if e == "all" or (e == "competition" and era_of.get(s) in ("post_cutoff", "pre_2025")) or era_of.get(s) == e]), pool(r, [s for s in all_sets if e == "all" or (e == "competition" and era_of.get(s) in ("post_cutoff", "pre_2025")) or era_of.get(s) == e]), a.boot) for e in eras + ["all"]}
    summary["tags"][t] = row

lines = ["# Competition suite summary (strict grading, avg@k over problems; problem-weighted pools)", "", "| arm | " + " | ".join(f"{s} ({summary['sets'][s]['n']})" for s in all_sets) + " | post-cutoff | pre-2025 | competition | all |", "|---|" + "---|" * (len(all_sets) + 4)]
for t in tags:
    row = summary["tags"][t]; cells = [f"{row['per_set'][s]['avg_at_k']:.1f}" if s in row["per_set"] else "-" for s in all_sets]
    pools = [f"{row['pools'][e]['avg_at_k']:.1f} (n={row['pools'][e]['n_problems']})" if e in row["pools"] else "-" for e in eras + ["all"]]
    lines.append(f"| {t} | " + " | ".join(cells + pools) + " |")
lines += ["", "## Paired per-problem bootstrap of the difference (points of avg@k, 95% CI, common problems only)", ""]
for t in tags:
    for r, d in summary["tags"][t]["vs"].items():
        parts = [f"{e}: {v['mean_diff']:+.1f} [{v['ci95'][0]:+.1f}, {v['ci95'][1]:+.1f}] n={v['n_problems']}" for e, v in d.items() if v]
        if parts: lines.append(f"- {t} vs {r}: " + "; ".join(parts))
text = "\n".join(lines); print(text)
if a.out: json.dump(summary, open(a.out, "w"), indent=1)
if a.md: open(a.md, "w").write(text + "\n")
