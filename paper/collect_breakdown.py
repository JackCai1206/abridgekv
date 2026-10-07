#!/usr/bin/env python3
"""Per-competition breakdown of the 446-problem suite and paired gaps on subsets (arXiv appendix, eval-set construction).
Reads the merged 32K vLLM evaluations (same loader and grader as collect_32k.py: avg@8, semantic full-text grader) and the
training-overlap check (results/contamination_math_sft_400k.json), writes data/suite_breakdown.json."""
import json, glob, os, random, collections
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
EV = os.path.join(os.environ.get("OUTPUT_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")), "eval")
R = json.load(open(os.path.join(HERE, "results.json"))); TAG = {a["key"]: a["tag"] for a in R["arms32k"] if a.get("tag")}
SUITE = json.load(open(os.path.join(ROOT, "data/math_eval/SUITE.json")))
C = json.load(open(os.path.join(ROOT, "results/contamination_math_sft_400k.json")))
ARMS = ["vanilla", "vpl", "uniform128", "gist", "lt_thought", "rkv_b4096"]
# record set -> (display name, manifest key)
SETS = [("aime_2022", "AIME 2022", "aime_2022"), ("aime_2023", "AIME 2023", "aime_2023"), ("aime2024_I", "AIME 2024", "aime_2024"),
        ("aime2024_II", "AIME 2024", "aime_2024"), ("aime2025", "AIME 2025", "aime_2025"), ("aime_2026", "AIME 2026", "aime_2026"),
        ("amc_2022_2023", "AMC 12 2022--2023", "amc_2022_2023"), ("hmmt_feb_2023", "HMMT Feb 2023", "hmmt_feb_2023"),
        ("hmmt_feb_2024", "HMMT Feb 2024", "hmmt_feb_2024"), ("hmmt_feb_2025", "HMMT Feb 2025", "hmmt_feb_2025"),
        ("brumo_2025", "BRUMO 2025", "brumo_2025"), ("cmimc_2025", "CMIMC 2025", "cmimc_2025"), ("smt_2025", "SMT 2025", "smt_2025")]
pos = collections.defaultdict(int); overlap = set()
for r in C["all"]:
    i = pos[r["set"]]; pos[r["set"]] += 1
    if r["frac_best_single"] >= C["threshold"]: overlap.add((r["set"], i))
def load(tag):
    per, gen = {}, {}
    for s, _, _ in SETS:
        f = glob.glob(f"{EV}/{s}_qwen3_4b_vllm_{tag}_merged/*.json")[0]
        for p in json.load(open(f))["per_problem"]:
            v = [float(x["correct_semantic_full_text"]) for x in p["samples"]][:8]
            per[(s, p["dataset_index"])] = sum(v) / len(v)
            gen[(s, p["dataset_index"])] = sum(x["gen_tokens"] for x in p["samples"][:8]) / len(v)
    return per, gen
P = {a: load(TAG[a]) for a in ARMS}
keys = sorted(P["vanilla"][0]); assert len(keys) == 446 and all(set(P[a][0]) == set(keys) for a in ARMS)
assert overlap <= set(keys)
era = {s: SUITE[m]["era"] for s, _, m in SETS}
def boot(a, b, ks, B=2000):
    d = [P[a][0][k] - (P[b][0][k] if b else 0) for k in ks]; rng = random.Random(0); n = len(d)
    ms = sorted(sum(rng.choice(d) for _ in range(n)) / n for _ in range(B))
    return [round(100 * sum(d) / n, 1), round(100 * ms[int(.025 * B)], 1), round(100 * ms[int(.975 * B)], 1)]
rows = []; seen = {}
for s, name, m in SETS:   # AIME 2024 I and II merge into one row
    ks = [k for k in keys if k[0] == s]
    if name in seen: rows[seen[name]]["keys"] += ks; continue
    seen[name] = len(rows); rows.append({"name": name, "source": SUITE[m]["source"], "era": SUITE[m]["era"], "keys": ks})
out = {"_doc": __doc__, "arms": ARMS, "overlap_threshold": C["threshold"], "sets": [], "subsets": []}
for r in rows:
    ks = r["keys"]
    out["sets"].append({"name": r["name"], "source": r["source"], "era": r["era"], "n": len(ks), "overlap": sum(k in overlap for k in ks),
                        "avg8": {a: round(100 * sum(P[a][0][k] for k in ks) / len(ks), 1) for a in ARMS},
                        "gen_vanilla": round(sum(P["vanilla"][1][k] for k in ks) / len(ks))})
SUB = [("All 446 problems", keys), ("No overlap with the SFT data", [k for k in keys if k not in overlap]),
       ("Released 2025 or later", [k for k in keys if era[k[0]] == "post_cutoff"]),
       ("Released before 2025", [k for k in keys if era[k[0]] == "pre_2025"]),
       ("Overlapping problems only", [k for k in keys if k in overlap])]
for name, ks in SUB:
    rec = {"name": name, "n": len(ks), "vanilla": round(100 * sum(P["vanilla"][0][k] for k in ks) / len(ks), 1), "gap": {}}
    for a in ARMS[1:]: rec["gap"][a] = boot(a, "vanilla", ks)
    rec["gap"]["vpl_minus_uniform"] = boot("vpl", "uniform128", ks)
    out["subsets"].append(rec)
json.dump(out, open(os.path.join(HERE, "data", "suite_breakdown.json"), "w"), indent=1)
for r in out["sets"]: print(r["name"], r["n"], r["overlap"], r["avg8"])
for r in out["subsets"]: print(r["name"], r["n"], r["vanilla"], r["gap"])
