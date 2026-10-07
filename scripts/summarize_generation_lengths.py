#!/usr/bin/env python3
"""Generation-length record for every merged eval (AIME 24/25, HMMT 25/24, MATH-500): per (eval tag, task) the distribution of
generated tokens, cap/short/no-answer rates, tokens on correct vs incorrect samples, and a per-problem PAIRED comparison of
each tag against a reference tag (default vanilla_canonical) on the same problems. Re-run after every eval (the eval launchers
do) so the record stays complete.
  python scripts/summarize_generation_lengths.py --root $OUTPUT_BASE/eval [--ref vanilla_canonical] [--out <json>] [--md <markdown>]"""
import argparse, glob, json, os, re, statistics as st

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True); ap.add_argument("--ref", default="vanilla_canonical")
ap.add_argument("--out", default=""); ap.add_argument("--md", default=""); ap.add_argument("--exclude", default="smoke,step50")
a = ap.parse_args()
CAP_DEFAULT = 16384; SHORT = 2048

def samples_of(d):
    pp = d.get("per_problem"); items = pp.values() if isinstance(pp, dict) else (pp or [])
    for it in items:
        pid = it.get("dataset_index", it.get("problem_index", it.get("index", it.get("problem_idx", it.get("id")))))
        for s in it.get("samples", []):
            corr = None
            for k in ("correct_A", "correct", "is_correct", "strict_correct"):
                if k in s: corr = bool(s[k]); break
            ans = s.get("pred_A_last", s.get("pred_last_boxed", s.get("answer", s.get("pred", None))))
            yield pid, int(s.get("gen_tokens", 0)), s.get("stop_reason"), corr, ans

def task_of(dirname, fname):
    m = re.match(r"(aime|hmmt2024|hmmt|math500)_qwen3_4b_vllm_(.+)_merged$", dirname)
    if not m: return None, None
    kind, tag = m.group(1), m.group(2)
    if kind == "aime":
        y = re.search(r"aime(\d{4})", fname); task = f"aime{y.group(1)}" if y else None
    elif kind == "hmmt": task = "hmmt2025"
    elif kind == "hmmt2024": task = "hmmt2024"
    else: task = "math500"
    return task, tag

rows = {}   # (tag, task) -> record
per_problem = {}  # (tag, task) -> {pid: [gen_tokens...], acc}
for d in sorted(glob.glob(os.path.join(a.root, "*_merged"))):
    dn = os.path.basename(d)
    if any(x in dn for x in a.exclude.split(",") if x): continue
    for f in sorted(glob.glob(os.path.join(d, "*.json"))):
        if os.path.basename(f) == "summary.json": continue
        task, tag = task_of(dn, os.path.basename(f))
        if task is None: continue
        try: D = json.load(open(f))
        except Exception as e: print("[genlen] unreadable", f, e); continue
        cap = int(D.get("max_tokens", CAP_DEFAULT) or CAP_DEFAULT)
        S = list(samples_of(D))
        if not S: continue
        toks = [t for _, t, _, _, _ in S]
        capped = [t >= cap or (sr == "length") for _, t, sr, _, _ in S]
        corr = [c for _, _, _, c, _ in S]; have_corr = any(c is not None for c in corr)
        tc = [t for (_, t, _, c, _) in S if c]; ti = [t for (_, t, _, c, _) in S if c is False]
        noans = [ans in (None, "", "None") for _, _, _, _, ans in S]
        rec = {"tag": tag, "task": task, "file": os.path.relpath(f, a.root), "n_samples": len(S), "n_problems": len({p for p, *_ in S}),
               "cap_tokens": cap, "mean_gen_tokens": st.mean(toks), "median_gen_tokens": st.median(toks),
               "p10": sorted(toks)[len(toks) // 10], "p90": sorted(toks)[(9 * len(toks)) // 10],
               "cap_rate": sum(capped) / len(S), "short_rate_lt2k": sum(t < SHORT for t in toks) / len(S),
               "no_answer_rate": (sum(noans) / len(S)) if any(ans is not None for *_, ans in S) else D.get("no_answer_rate"),
               "accuracy": (sum(bool(c) for c in corr) / len(S)) if have_corr else D.get("accuracy"),
               "mean_tokens_correct": st.mean(tc) if tc else None, "mean_tokens_incorrect": st.mean(ti) if ti else None,
               "cap_rate_among_incorrect": (sum(1 for (_, t, sr, c, _) in S if c is False and (t >= cap or sr == "length")) / len(ti)) if ti else None}
        rows[(tag, task)] = rec
        pp = {}
        for pid, t, sr, c, _ in S: pp.setdefault(pid, {"toks": [], "corr": []}); pp[pid]["toks"].append(t); pp[pid]["corr"].append(c)
        per_problem[(tag, task)] = pp

# paired comparison vs the reference on shared problems
paired = {}
for (tag, task), pp in per_problem.items():
    ref = per_problem.get((a.ref, task))
    if ref is None or tag == a.ref: continue
    shared = [p for p in pp if p in ref]
    if not shared: continue
    diffs = [st.mean(pp[p]["toks"]) - st.mean(ref[p]["toks"]) for p in shared]
    # split by reference difficulty: problems the reference solves in >= half its samples vs not
    easy = [p for p in shared if any(c is not None for c in ref[p]["corr"]) and st.mean(bool(c) for c in ref[p]["corr"]) >= 0.5]
    hard = [p for p in shared if p not in easy]
    def md(ps): return st.mean([st.mean(pp[p]["toks"]) - st.mean(ref[p]["toks"]) for p in ps]) if ps else None
    paired[f"{tag}|{task}"] = {"tag": tag, "task": task, "ref": a.ref, "n_problems": len(shared), "mean_diff_tokens": st.mean(diffs),
                               "median_diff_tokens": st.median(diffs), "longer_on": sum(x > 0 for x in diffs), "shorter_on": sum(x < 0 for x in diffs),
                               "mean_diff_on_ref_easy": md(easy), "n_ref_easy": len(easy), "mean_diff_on_ref_hard": md(hard), "n_ref_hard": len(hard)}

out = {"root": a.root, "ref": a.ref, "rows": [rows[k] for k in sorted(rows)], "paired_vs_ref": [paired[k] for k in sorted(paired)]}
if a.out: json.dump(out, open(a.out, "w"), indent=1)
lines = ["| tag | task | n | acc | mean tok | median | p90 | cap % | <2K % | no-ans % | tok correct | tok wrong |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
f = lambda v, fmt="%.0f": (fmt % v) if isinstance(v, (int, float)) else "-"
for r in out["rows"]:
    lines.append(f"| {r['tag']} | {r['task']} | {r['n_samples']} | {f(r['accuracy'], '%.3f')} | {f(r['mean_gen_tokens'])} | {f(r['median_gen_tokens'])} | {f(r['p90'])} | {f(100*r['cap_rate'], '%.1f')} | {f(100*r['short_rate_lt2k'], '%.1f')} | {f(100*r['no_answer_rate'], '%.1f') if isinstance(r['no_answer_rate'], (int,float)) else '-'} | {f(r['mean_tokens_correct'])} | {f(r['mean_tokens_incorrect'])} |")
lines += ["", f"Paired per-problem mean generated tokens minus {a.ref} (same problems; ref-easy = {a.ref} solves >= 4/8):", "",
          "| tag | task | problems | mean diff | median diff | longer / shorter on | diff on ref-easy (n) | diff on ref-hard (n) |", "|---|---|---|---|---|---|---|---|"]
for p in out["paired_vs_ref"]:
    lines.append(f"| {p['tag']} | {p['task']} | {p['n_problems']} | {p['mean_diff_tokens']:+.0f} | {p['median_diff_tokens']:+.0f} | {p['longer_on']} / {p['shorter_on']} | {f(p['mean_diff_on_ref_easy'], '%+.0f')} ({p['n_ref_easy']}) | {f(p['mean_diff_on_ref_hard'], '%+.0f')} ({p['n_ref_hard']}) |")
md = "\n".join(lines)
if a.md: open(a.md, "w").write(md + "\n")
print(md)
