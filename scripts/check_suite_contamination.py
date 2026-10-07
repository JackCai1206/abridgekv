"""Overlap between the SFT pool's prompts and the evaluation suite (test-set leakage check).
For every suite problem: the fraction of its word 13-grams (8-grams for short problems) that occur in any training
prompt, after normalisation (lowercase, LaTeX/punctuation stripped). A problem is flagged at >= 50%; flagged problems
are listed with their best-matching training prompt. Usage:
  PAIRS=<pairs.pkl> SETS=aime_2022:... OUT=<json> python3 scripts/check_suite_contamination.py"""
import os, re, json, sys, collections, torch
from transformers import AutoTokenizer
PAIRS = os.environ["PAIRS"]; OUT = os.environ["OUT"]
SETS = os.environ.get("SETS", "aime_2022:aime_2023:aime2024_I:aime2024_II:aime2025:aime_2026:amc_2022_2023:hmmt_feb_2023:hmmt_feb_2024:hmmt_feb_2025:brumo_2025:cmimc_2025:smt_2025").split(":")
N = int(os.environ.get("NGRAM", "13")); THR = float(os.environ.get("THR", "0.5"))
def norm(s):
    s = s.lower(); s = re.sub(r"\\[a-z]+", " ", s); s = re.sub(r"[^a-z0-9]+", " ", s); return s.split()
def grams(w, n): return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}
suite = []
for name in SETS:
    for l in open(f"data/math_eval/{name}.jsonl"):
        r = json.loads(l); w = norm(r["problem"]); n = N if len(w) >= 2 * N else 8
        suite.append({"set": name, "idx": r.get("problem_idx", r.get("id")), "words": w, "n": n, "g": grams(w, n)})
need = collections.defaultdict(list)          # gram -> suite problems that contain it
for i, p in enumerate(suite):
    for g in p["g"]: need[g].append(i)
ns = sorted({p["n"] for p in suite})
print(f"[contam] {len(suite)} suite problems, n-gram sizes {ns}", flush=True)
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Base")
pairs = torch.load(PAIRS, weights_only=False); print(f"[contam] {len(pairs)} training pairs", flush=True)
hits = [set() for _ in suite]; best = [(0, -1) for _ in suite]
B = 2000
for s in range(0, len(pairs), B):
    texts = tok.batch_decode([p["prompt"].tolist() for p in pairs[s:s + B]], skip_special_tokens=True)
    for j, t in enumerate(texts):
        w = norm(t); per = collections.Counter()
        for n in ns:
            for g in grams(w, n):
                for i in need.get(g, ()):
                    if suite[i]["n"] == n: hits[i].add(g); per[i] += 1
        for i, c in per.items():
            if c > best[i][0]: best[i] = (c, s + j)
    if s % 40000 == 0: print(f"[contam] {s + B}/{len(pairs)} scanned", flush=True)
rows = []
for i, p in enumerate(suite):
    frac = len(hits[i]) / max(1, len(p["g"])); bfrac = best[i][0] / max(1, len(p["g"]))
    rows.append({"set": p["set"], "idx": p["idx"], "frac_any": round(frac, 3), "frac_best_single": round(bfrac, 3), "best_pair": best[i][1]})
flag = [r for r in rows if r["frac_best_single"] >= THR]
by = collections.defaultdict(lambda: [0, 0])
for r in rows: by[r["set"]][0] += 1; by[r["set"]][1] += r["frac_best_single"] >= THR
for r in flag[:400]:
    r["train_prompt"] = tok.decode(pairs[r["best_pair"]]["prompt"].tolist(), skip_special_tokens=True)[:400]
json.dump({"threshold": THR, "ngram": N, "per_set": {k: {"n": v[0], "flagged": v[1]} for k, v in by.items()}, "flagged": flag, "all": rows}, open(OUT, "w"), indent=1)
print("[contam] per set (flagged / problems):", flush=True)
for k, v in by.items(): print(f"  {k:16s} {v[1]:3d} / {v[0]}")
print(f"[contam] TOTAL flagged {len(flag)} / {len(rows)} -> {OUT}")
