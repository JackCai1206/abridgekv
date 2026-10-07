#!/usr/bin/env python3
"""Generate paper/tables/*.tex from paper/results.json and paper/data/*. Pending cells render as \\pending{<job ids>}."""
import json, re, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); R = json.load(open(os.path.join(HERE, "results.json")))
os.makedirs(os.path.join(HERE, "tables"), exist_ok=True)

def pend(status):
    if status == "done": return "--"   # arm finished; the cell does not apply (no SFT -> no held-out CE; eviction has no MATH-500 run)
    return "\\pending{%s}" % (status.replace("queued(", "").rstrip(")") if status.startswith("queued") else "planned")
def num(v, status, fmt="{:.1f}"):
    return pend(status) if v is None else fmt.format(v)
def triple(v, status):
    return ("\\multicolumn{3}{c}{%s}" % pend(status)) if v is None else " & ".join("{:.1f}".format(v[i]) for i in range(3))
def fnum(v, fmt): return (fmt % v) if v is not None else "--"
def write(name, body):
    open(os.path.join(HERE, "tables", name + ".tex"), "w").write(body); print("wrote tables/%s.tex" % name)

# ================= 32K pooled-suite results (collect_32k.py -> data/results32k.json) =================
X = json.load(open(os.path.join(HERE, "data", "results32k.json")))
def pend(status):
    if status == "done": return "--"   # arm finished; the cell does not apply (no SFT -> no held-out CE; eviction has no MATH-500 run)
    return "\\pending{%s}" % (status.replace("queued(", "").rstrip(")") if status.startswith("queued") else "planned")
def ce_cell(a):
    if a["heldout_ce"] is not None: return "%.4f" % a["heldout_ce"]
    return "--" if a.get("heldout_ce_na") else pend(a["status"])   # training-free arms never train, so the cell does not apply
def arm_rec(key): return X["arms"].get(key, {})
def pool_cell(a):
    p = arm_rec(a["key"]).get("pool")
    if not p: return pend(a["status"])
    tag = "" if p["complete"] else " \\textcolor{gray}{(%d/%d)}" % (p["n"], p["n_expected"])
    return "%.1f {\\scriptsize[%.1f, %.1f]}%s" % (p["avg8"], p["ci"][0], p["ci"][1], tag)
def diff_cell(a):
    r = arm_rec(a["key"]); d = r.get("diff_vs_vanilla")
    if a["key"] == "vanilla": return "--"
    if d: return "%+.1f {\\scriptsize[%+.1f, %+.1f]}%s" % (d[0], d[1], d[2], "$^\\dagger$" if a.get("evict") else "")
    if r.get("pool"): return "\\textcolor{gray}{unpaired}"
    return pend(a["status"])
def m500_cell(a, st=None):
    v = a.get("math500")   # MATH-500 avg@4 at T=0.6, strict last-boxed (16K cap; solutions are short, the cap is immaterial) from the arm record
    if v is None and a.get("evict"):   # eviction policies: HF run, one sample per seed, avg@k over the seeds complete at submission
        m = arm_rec(a["key"]).get("math500")
        return ("%.1f" % m["avg8_strict"]) if m else "--"
    if v is None and a.get("heldout_ce_na"): return "--"   # training-free arm: no MATH-500 run of its own
    return "%.1f" % v if v is not None else pend(st or a["status"])
def trunc_cell(a):
    p = arm_rec(a["key"]).get("pool"); return "%.1f" % p["trunc"] if p else pend(a["status"])

# ---- suite description and breakdown (arXiv Appendix "Competition suite"; data/suite_breakdown.json from collect_breakdown.py) ----
SB = json.load(open(os.path.join(HERE, "data", "suite_breakdown.json")))
SBA = ["vanilla", "vpl", "uniform128", "gist", "lt_thought", "rkv_b4096"]
rows = []
for m in SB["sets"]:
    rows.append("%s & %s & %s & %d & %s & %s \\\\" % (m["name"], m["source"].replace("_", "\\_"), "2025+" if m["era"] == "post_cutoff" else "pre-2025",
                m["n"], str(m["overlap"]) if m["overlap"] else "--", " & ".join("%.1f" % m["avg8"][k] for k in SBA)))
write("suite_desc", r"""\begin{table}[h]
\centering\footnotesize\setlength{\tabcolsep}{3.5pt}
\caption{The %d-problem competition suite by competition: source, release period, number of problems, problems that also
occur in the SFT prompts (word 13-gram overlap $\ge 50\%%$ with a single training prompt), and avg@8 at the 32K cap for each arm.
Arms: full KV (vanilla SFT), \method{} per-layer, uniform allocation, appended code positions, LightThinker (thought-level, our
stack) and R-KV at budget 4096 on the vanilla weights. A single set's 95\%% interval on a paired gap is about $\pm 5$ points
(Appendix~\ref{app:pooled}), so per-set differences are not individually meaningful.}
\label{tab:suite_desc}
\resizebox{\linewidth}{!}{\begin{tabular}{@{}lllrrrrrrrr@{}}
\toprule
Set & source & released & problems & in SFT & full KV & per-layer & uniform & appended & LightThinker & R-KV 4096 \\
\midrule
""" % sum(m["n"] for m in SB["sets"]) + "\n".join(rows) + r"""
\bottomrule
\end{tabular}}
\end{table}
""")
def gcell(g, bold=False):
    v, lo, hi = g; return "$%+.1f$ {\\scriptsize $[%+.1f, %+.1f]$}" % (v, lo, hi)
rows = []
for s_ in SB["subsets"]:
    g = s_["gap"]
    rows.append("%s & %d & %.1f & %s \\\\" % (s_["name"], s_["n"], s_["vanilla"], " & ".join(gcell(g[k]) for k in ["vpl", "uniform128", "gist", "lt_thought", "rkv_b4096", "vpl_minus_uniform"])))
write("suite_subsets", r"""\begin{table}[h]
\centering\footnotesize\setlength{\tabcolsep}{3pt}
\caption{Paired differences on subsets of the suite (avg@8 points, 95\% bootstrap interval over problems). The first five
columns are differences to full KV on the same problems; the last is per-layer minus uniform. ``Overlapping'' problems are the 69
that occur in the SFT prompts (Table~\ref{tab:suite_desc}).}
\label{tab:suite_subsets}
\resizebox{\linewidth}{!}{\begin{tabular}{@{}lrrrrrrrr@{}}
\toprule
& & & \multicolumn{5}{c}{difference to full KV} & \\
\cmidrule(lr){4-8}
Subset & problems & full KV & per-layer & uniform & appended & LightThinker & R-KV 4096 & per-layer $-$ uniform \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}}
\end{table}
""")

# ---- main table: pooled suite + MATH-500 ----
groups32 = [("reference", "References"), ("ours", "Ours: dynamic per-layer allocation"), ("ablation", "Ablations: one design choice removed"), ("prior", "Prior learned compression, in our training and serving stack"), ("baseline", "Training-free eviction (vanilla weights)")]
# Short arm names for the main table (user 2026-09-24: the long labels forced \resizebox and made the table illegible);
# the long descriptions live in the text and in the group headers.
TL = {"base": "Base model (no SFT)", "vanilla": "Vanilla SFT (full KV)", "vpl": "\\method{} (per-layer)",
      "uniform128": "uniform allocation", "gist": "appended positions", "evict_recency": "sink + recency",
      "evict_random": "random", "evict_score": "attention score (H2O)",
      "rkv_b2048": "R-KV, budget 2048", "rkv_b4096": "R-KV, budget 4096"}
rows = []
# Ours is shaded, and the best value among the compressed arms (every row outside "References") is bold in each metric
# column: lowest CE, highest MATH-500 / suite / paired delta, lowest truncation (user 2026-09-24).
BEST = {2: min, 3: max, 4: max, 5: max, 6: min}
def _num(c):
    m = re.match(r"\s*([+-]?\d+\.\d+)", c); return float(m.group(1)) if m else None
table = []
for g, title in groups32:
    for a in [a for a in R["arms32k"] if a["group"] == g]:
        cells = [TL.get(a["key"], a["label"]), ("%d" % a["kv_per_block"] if a["kv_per_block"] is not None else pend(a["status"])), ce_cell(a), m500_cell(a),
                 re.sub(r" \\textcolor\{gray\}\{\((\d+)/(\d+)\)\}", r"$^\\dagger$", pool_cell(a)), diff_cell(a), trunc_cell(a)]
        table.append((g, title, cells))
for col, fn in BEST.items():
    vals = [_num(c[col]) for g, _, c in table if g != "reference" and _num(c[col]) is not None]
    if not vals: continue
    best = fn(vals)
    for g, _, c in table:
        if g != "reference" and _num(c[col]) == best:
            c[col] = re.sub(r"^(\s*[+-]?\d+\.\d+)", r"\\textbf{\1}", c[col], count=1)
last = None
for g, title, c in table:
    if g != last: rows.append("\\midrule\n\\multicolumn{7}{l}{\\textit{%s}}\\\\" % title); last = g
    rows.append({"ours": "\\rowcolor{oursbg}", "ablation": "\\rowcolor{ablbg}"}.get(g, "") + " & ".join(c) + " \\\\")
write("main", r"""\begin{table}[t]
\centering\footnotesize\setlength{\tabcolsep}{2.6pt}
\caption{Main comparison at the 32K cap. Suite: avg@8 over %d problems with 95\% bootstrap intervals; $\Delta$: paired
difference to the full-KV model (vanilla). KV/512: persistent KV rows per 512 generated tokens. trunc.: share of suite samples
reaching the cap. $^\dagger$~Eviction baselines, run on a 60-problem subset (AIME 2025 and HMMT Feb 2025) on which full KV scores 29.8 and
the per-layer arm 28.3: their $\Delta$ is paired on those problems and comparable with the other rows' $\Delta$, their avg@8 is not.
Protocol details: Appendix~\ref{app:suite}.
Shaded: ours; bold: best below the reference rows.}
\label{tab:main}
\begin{tabular}{@{}lrrrllr@{}}
\toprule
Arm & KV/512 & CE & MATH-500 & Suite avg@8 & $\Delta$ vs.\ vanilla & trunc.\% \\
""".replace("%d", str(X["n_competition"])).replace("avg@%s", "avg@%d" % max([(arm_rec(a["key"]).get("math500") or {}).get("k", 0) for a in R["arms32k"] if a.get("evict")] + [0])) + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

# ---- archived arms (appendix): tail retention, distillation, their combination, the vanilla replicate ----
def diff_ours_cell(a):
    d = arm_rec(a["key"]).get("diff_vs_ours")
    if d: return "%+.1f {\\scriptsize[%+.1f, %+.1f]}" % (d[0], d[1], d[2])
    return pend(a["status"]) if not arm_rec(a["key"]).get("pool") else "\\textcolor{gray}{unpaired}"
rows = []
for a in [x for x in R["arms32k"] if x["group"] == "archived"]:
    rows.append("%s & %d & %s & %s & %s & %s & %s & %s \\\\" % (a["label"], a["kv_per_block"], ce_cell(a),
                m500_cell(a), pool_cell(a), diff_cell(a), diff_ours_cell(a), trunc_cell(a)))
write("archived", r"""\begin{table}[h]
\centering\footnotesize\setlength{\tabcolsep}{3pt}
\caption{Archived arms, same protocol as Table~\ref{tab:main}. Raw tail 16 keeps the last 16 generated tokens of every block uncompressed
(16 extra rows per block); KD adds the distillation term of Eq.~\ref{eq:kd} from the frozen vanilla teacher. $\Delta$ columns are paired
per-problem differences against vanilla and against \method{} (dynamic per-layer allocation). The vanilla replicate re-trains the reference
with a different data seed and bounds seed-to-seed noise.}
\label{tab:archived}
\resizebox{\linewidth}{!}{\begin{tabular}{@{}lrrrlllr@{}}
\toprule
Arm & KV/512 & held-out CE & MATH-500 & Suite avg@8 [95\% CI] & $\Delta$ vs.\ vanilla & $\Delta$ vs.\ \method{} & trunc.\ \% \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}}
\end{table}
""")

# ---- reported results of decode-time methods on Qwen3-4B (appendix; never merged with our table) ----
rows = []
for r in R["reported32k"]["rows"]:
    cells = "; ".join("%s %.1f$\\to$%.1f (%+.1f)" % (n, a, b, b - a) for n, a, b in r["cells"])
    rows.append("%s~\\citep{%s} & %s & %s & %s & %s & %s \\\\" % (r["paper"], r["cite"], r["model"], r["cap"], r["budget"], cells, r["note"]))
write("reported", r"""\begin{table}[h]
\centering\scriptsize\setlength{\tabcolsep}{3pt}
\caption{Results \emph{reported by the authors} of decode-time KV-eviction methods at roughly a quarter of the cache, for orientation only.
None is comparable to Table~\ref{tab:main} in absolute terms: the Qwen3-4B rows use the released checkpoint (full-KV AIME25 $\approx$ 66,
MATH-500 $\approx$ 94) rather than a base model fine-tuned on our data; their budget is a fixed absolute cap per head, so the persistent
fraction is $\approx$23\% averaged over a 17K trace but 100\% for short traces and 12.5\% at 32K, whereas ours is a constant 4:1 plus the
open block; decoding, sample counts and benchmark identity differ (see notes). Read only the drop against each paper's own full-KV row,
against our $-2.3$ (uniform) and $-0.8$ (per-layer) at a constant 4:1 over 446 problems. n.r. = not reported.}
\label{tab:reported}
\resizebox{\linewidth}{!}{\begin{tabular}{@{}lllllll@{}}
\toprule
Paper & model & cap & budget & full-KV $\to$ compressed (drop) & note \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}}
\end{table}
""")

# ---- allocation ablation (32K) ----
rows = []
for a in R["allocation32k"]:
    st = a["status"]; rec = arm_rec(a["key"]); p = rec.get("pool")
    rows.append("%s & %.4f & %s & %s \\\\" % (a["label"], a["heldout_ce"], m500_cell(a, st), pool_cell({"key": a["key"], "status": st})))
write("allocation", r"""\begin{table}[t]
\centering\small
\caption{Budget allocation at a fixed mean of 128 code rows per block (all other settings identical). Held-out CE on 192 held-out SFT pairs;
MATH-500 avg@4; suite avg@8 at 32K as in Table~\ref{tab:main}. The per-head row lets a head keep up to all 512 rows, with a floor of 32,
nested inside the per-layer schedule; paired against the per-layer headline it is $-0.6$ $[-2.1, +0.9]$. The last row sets per-head quotas from how
concentrated each head's attention is in the full-KV model, which raises the early layers; paired against the per-layer model it is $-1.9$ $[-3.3, -0.5]$.}
\label{tab:allocation}
\begin{tabular}{@{}lrrl@{}}
\toprule
Schedule & held-out CE & MATH-500 & Suite avg@8 [95\% CI] \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

# ---- C/K scaling (32K) ----
rows = []
ARM_M500 = {a["key"]: a.get("math500") for a in R["arms32k"]}
for p in R["scaling32k"]:
    st = p["status"]; a = {"key": p["key"], "status": st}
    if ARM_M500.get(p["key"]) is not None: p["math500"] = ARM_M500[p["key"]]   # one source per arm; the sweep row never keeps its own copy
    # A sweep-only arm (no arms32k twin) still has its MATH-500 in the collected record; without this it printed "--"
    # for ck512k64 even though collect_32k.py had read avg@4 = 87.3 off disk.
    if p.get("math500") is None:
        m = X["arms"].get(p["key"], {}).get("math500")
        if m is not None: p["math500"] = m.get("avg8_strict", m["avg8"])   # strict: same grader as the results.json arms in this column
    rows.append("%d & %d & %s & %s & %s & %s \\\\" % (p["C"], p["K"], p["ratio"], "%.4f" % p["heldout_ce"] if p["heldout_ce"] is not None else pend(st), m500_cell(p, st), pool_cell(a)))
write("scaling", r"""\begin{table}[t]
\centering\small
\caption{Block size and code budget sweep (uniform quotas, bitrev positions, $K_{\max}{=}256$, same 2B tokens). Held-out CE is comparable only
within one block size (the held-out set is defined by blocks per example). MATH-500 avg@4; suite avg@8 at 32K as in Table~\ref{tab:main}.}
\label{tab:scaling}
\begin{tabular}{@{}rrrrrl@{}}
\toprule
$C$ & mean $K$ & ratio & held-out CE & MATH-500 & Suite avg@8 [95\% CI] \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

# ---- setup table ----
c = R["common"]
write("setup", r"""\begin{table}[h]
\centering\small
\caption{Training and evaluation setup shared by every arm.}
\label{tab:setup}
\begin{tabular}{@{}l >{\raggedright\arraybackslash}p{0.55\linewidth}@{}}
\toprule
Base model & %s \\
SFT data & %s \\
Response tokens per arm & %s (about 936 steps at effective batch %d) \\
Block size $C$ / code slots $K_{\max}$ / mean quota & %d / 256 / 128 \\
Optimizer, schedule & AdamW, lr %s \\
Generation cap at eval & %d tokens \\
Hardware & one node per run, either 4$\times$ NVIDIA GB300 or 8$\times$ NVIDIA H100-80GB;
FlexAttention, bf16 autocast. A fixed data order sets each step's examples independently of the number of GPUs, so an
arm is the same run on either machine \\
\bottomrule
\end{tabular}
\end{table}
""" % (c["model"], c["data"].replace(", canonical 8x1x32 order", ", fixed order"), c["response_tokens"], c["eff_batch"], c["C"], c["lr"], c["eval_cap"]))

# ---- quota ablation table (eval-time truncation on trained models) ----
def load_jsonl(p): return [json.loads(l) for l in open(p)]
qrows = []
for tag, lab in [("c256uni_bitrev", "uniform 128"), ("c256het_bitrev", "per-head (proxy)")]:
    recs = {r["name"]: r for r in load_jsonl(os.path.join(HERE, "data", "quota_ablation_%s.jsonl" % tag)) if r.get("set") == "full"}
    base = recs["trained"]["held_out_ce"]
    def d(name): return ("%+.4f" % (recs[name]["held_out_ce"] - base)) if name in recs else "--"
    layer_costs = sorted(((r["held_out_ce"] - base, n) for n, r in recs.items() if n.startswith("layer") and "->64" in n), reverse=True)[:3]
    qrows.append("%s & %.4f & %s & %s & %s & %s \\\\" % (lab, base, d("all->64"), d("all->128"), d("all->256"), ", ".join("%s %+.4f" % (n.split("->")[0], v) for v, n in layer_costs)))
write("quota", r"""\begin{table}[t]
\centering\small
\caption{Eval-time quota truncation on trained models (held-out CE, 192 sequences). ``all$\to$64'' halves every head's code budget
without retraining; single-layer costs are the three largest.}
\label{tab:quota}
\resizebox{\linewidth}{!}{\begin{tabular}{@{}lrrrrl@{}}
\toprule
Model & trained CE & all$\to$64 & all$\to$128 & all$\to$256 & largest single-layer $\to$64 costs \\
\midrule
""" + "\n".join(qrows) + r"""
\bottomrule
\end{tabular}}
\end{table}
""")

# ---- compressibility summary table (per-layer bands) ----
S = json.load(open(os.path.join(HERE, "data", "oc_vanilla_completions_summary.json"))); P = S["per_layer_head"]
def band(lo, hi, f): return sum(sum(P[f][l]) / len(P[f][l]) for l in range(lo, hi + 1)) / (hi - lo + 1)
crows = []
for lo, hi in [(0, 0), (1, 6), (7, 17), (18, 28), (29, 34), (35, 35)]:
    crows.append("%d--%d & %.3f & %.3f & %.3f & %.3f & %.4f & %.0f & %.1f \\\\" % (lo, hi, band(lo, hi, "err0"), band(lo, hi, "ev_oracle_mass@64"), band(lo, hi, "ev_inblock@64"), band(lo, hi, "ev_recency@64"), band(lo, hi, "lr@32"), band(lo, hi, "eff_pos"), band(lo, hi, "eff_rank")))
write("compressibility", r"""\begin{table}[t]
\centering\small\setlength{\tabcolsep}{4pt}
\caption{Online compressibility on the vanilla model over its own AIME completions (128 trajectories, %d block rows; RMS relative
attention-output error over future queries, pooled by layer band). Hindsight and causal top-64 selection, recency, and the rank-32 mixing lower bound; eff.\ pos.\ =
effective number of read positions of 512; eff.\ rank of the block's contribution.}
\label{tab:compress}
\begin{tabular}{@{}lrrrrrrr@{}}
\toprule
Layers & drop block & hindsight@64 & in-block@64 & recency@64 & rank-32 bound & eff.\ pos. & eff.\ rank \\
\midrule
""" % S["n_rows"] + "\n".join(crows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

# ---- parity table ----
pt = R["parity"]["train_vs_gen"]; pv = R["parity"]["hf_vs_vllm"]
write("parity", r"""\begin{table}[h]
\centering\small
\caption{Train/serve parity checks.}
\label{tab:parity}
\begin{tabular}{@{}p{0.30\linewidth} p{0.32\linewidth} p{0.31\linewidth}@{}}
\toprule
Check & Setting & Result \\
\midrule
Trainer vs.\ incremental generator (per-token CE) & %s checkpoint, %d held-out sequences, trained schedule & mean $|\Delta|$ %s, max %s \\
Same, deliberately wrong (uniform) schedule & control & mean $|\Delta|$ %s, max %s \\
HF reference vs.\ vLLM packed KV (greedy tokens) & %s & %s \\
Multi-boundary HF vs.\ vLLM parity & %s & %s \\
\bottomrule
\end{tabular}
\end{table}
""" % (pt["arm"], pt["seqs"], pt["mean_abs_dce"], pt["max_abs_dce"], pt["wrong_schedule_mean_abs_dce"], pt["wrong_schedule_max"], ", ".join(a.replace("_", "\\_") for a in pv["arms"]) + "; " + pv["tokens"], pv["result"], R["parity"]["multi_boundary"]["setting"], R["parity"]["multi_boundary"]["result"]))

# ---- variance decomposition table ----
try:
    V = json.load(open(os.path.join(HERE, "data", "oc_vanilla_completions_decomposition.json")))
    vrows = []
    for f, lab in [("ev_oracle_mass@64", "hindsight top-64 error"), ("ev_inblock@64", "causal in-block top-64 error"), ("lr@32", "rank-32 mixing bound"), ("err0", "drop-block error"), ("mass", "future attention mass")]:
        d = V["variance_shares_log_" + f]
        vrows.append("%s & %.2f & %.2f & %.2f & %.2f \\\\" % (lab, d["layer"], d["head_within_layer"], d["trajectory_within_head"], d["block_residual"]))
    sh = V["split_half_spearman"]["ev_oracle_mass@64"]
    write("decomposition", r"""\begin{table}[t]
\centering\small
\caption{Where the compressibility varies: shares of the variance of the per-row log metric (the vanilla model over 128 of its own
AIME completions, as in Table~\ref{tab:compress}). Split-half Spearman of the per-head hindsight ranking: %.3f.}
\label{tab:decomp}
\begin{tabular}{@{}lrrrr@{}}
\toprule
Quantity & layer & head within layer & trajectory & block (residual) \\
\midrule
""" % sh + "\n".join(vrows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")
except FileNotFoundError:
    write("decomposition", "")

# ---- history table (old layout) ----
H = R["old_layout_reference"]
hrows = ["%s & %.1f & %.1f & %.1f & %.1f \\\\" % (a["label"], a["math500"], (a["aime24"][0]+a["aime25"][0])/2, (a["aime24"][1]+a["aime25"][1])/2, (a["aime24"][2]+a["aime25"][2])/2) for a in [] ] + ["%s & %.1f & %.1f & %.1f & %.1f \\\\" % (lab, H[k]["math500"], H[k]["aime_pass8"], H[k]["aime_avg8"], H[k]["aime_pass1"]) for k, lab in [("no_tail", "appended codes, no tail"), ("replay128", "appended codes + replay 128"), ("retain128", "appended codes + retained KV 128 (rolling)"), ("vanilla_old", "vanilla (old data order)")]]
write("history", r"""\begin{table}[h]
\centering\small
\caption{Earlier code-position layout (codes appended after the block, no in-block RoPE grid), $K{=}128$, 16K strict.}
\label{tab:history}
\begin{tabular}{@{}lrrrr@{}}
\toprule
Arm & MATH-500 & AIME pass@8 & avg@8 & pass@1 \\
\midrule
""" + "\n".join(hrows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

# ---- efficiency: measured KV memory, concurrency and decode speed (scripts/bench_decode.py -> summarize_bench.py) ----
BENCH_PATH = os.path.join(HERE, "data", "bench_decode.json")
if os.path.isfile(BENCH_PATH):
    B = json.load(open(BENCH_PATH))
    LABEL = {
        "vanilla_graphs": r"vanilla (full CUDA graphs)",
        "vanilla_eager":  r"vanilla (eager, matched to ours)",
        "vanilla_fa":     r"vanilla (FlashAttention, graphs)",
        "tpl_physical":   r"ours, per-layer quotas",
        "uni_plain":      r"ours, uniform 128",
        "rkv_b4096":      r"R-KV, budget 4096",   # the baseline at our 4:1 operating point; its own venv, same harness
        "rkv_b2048":      r"R-KV, budget 2048",
    }
    kvb = B.get("kv_bytes_per_token", 0)
    def gib_per_seq(c):
        t = c.get("kv_tokens_per_seq")
        return (t * kvb / 2**30) if (t and kvb) else None
    # Two panels at 16K generated tokens: batch 8 (every arm fits in cache, so the memory ratio is the honest one)
    # and batch 32 (vanilla exhausts its cache and preempts, which is where the concurrency argument lands).
    cells = B["cells"]
    def panel(batch):
        out = []
        head = [c for c in cells if c["mode"] == "batch" and c["len"] == 16384 and c["batch"] == batch]
        base = next((c for c in head if c["tag"] == "vanilla_graphs"), None)
        for tag in LABEL:
            c = next((x for x in head if x["tag"] == tag), None)
            if c is None:
                continue
            g, gb = gib_per_seq(c), (gib_per_seq(base) if base else None)
            pool = c.get("kv_blocks_peak"), c.get("kv_blocks_total")
            occ = ("%.0f\\%%" % (100.0 * pool[0] / pool[1])) if all(pool) else "--"
            out.append(" & ".join([
                LABEL[tag],
                fnum(c.get("kv_tokens_per_seq"), "%d"),
                fnum(g, "%.2f"),
                (("%.2f$\\times$" % (gb / g)) if (g and gb) else "--"),
                occ,
                fnum(c.get("tok_per_s"), "%.0f"),
                (("%.2f$\\times$" % (c["tok_per_s"] / base["tok_per_s"]))
                 if (base and c.get("tok_per_s") and base.get("tok_per_s")) else "--"),
            ]))
        return out
    p8, p32 = panel(8), panel(32)
    rows = []
    if p8:
        rows.append("\\multicolumn{7}{@{}l}{\\emph{batch 8}}")
        rows += p8
    if p32:
        rows.append("\\noalign{\\vskip 3pt}\\multicolumn{7}{@{}l}{\\emph{batch 32}}")
        rows += p32
    # Prefer the bounded saturation runs (one cohort of steps, consistent across arms) over any full-drain leftover,
    # whose final quarter is the drain tail rather than steady state.
    sat = {}
    for c in cells:
        if c["mode"] != "saturate":
            continue
        cur = sat.get(c["tag"])
        if cur is None or (c.get("steady_running") and not cur.get("steady_running")):
            sat[c["tag"]] = c
    # Concurrency = running sequences once they are LONG. Measured: vanilla holds 124 at the moment its cache first
    # fills (sequences still short) but only 31 over the final quarter of the run, when they approach 16K; uniform
    # holds 119. So the steady median is the discriminating statistic and the cache-peak count is not.
    def conc(c):
        return c.get("steady_running")
    # Cross-job saturation cells swing 17% and the headroom rule rejects several; concurrency now lives in
    # tab:efficiency, read from the single-job controlled run (data/saturation.json). Keep the footnote off.
    sat = {}
    satrow = ", ".join(
        [LABEL.get(t, t) + (" %d" % conc(sat[t])) for t in LABEL if t in sat and conc(sat[t])]
    ) if sat else ""
    if not rows:
        # the data file exists but the headline cell (batch 8, 16K) is not measured yet; an empty tabular body is a
        # LaTeX error, so write nothing and leave the \todo in main.tex standing.
        print("skipped tables/bench.tex (no batch-8 16K cells in bench_decode.json yet)")
    else:
        write("bench", r"""\begin{table}[t]
\centering\small
\caption{Measured serving cost on one GPU, every sequence forced to 16K generated tokens so each arm does identical
work (medians of three repeats). KV rows/seq is quoted in full-KV-equivalent rows, i.e.\ resident KV bytes divided by the
cost of one token's KV across all 36 layers, so a per-layer schedule that keeps fewer rows in some layers is directly
comparable with vanilla. At batch 8 every arm fits in cache, so the memory ratio there is the undistorted one. At batch 32
vanilla exhausts its cache (cache used $=100\%$) and the scheduler preempts and recomputes, which is where the
concurrency advantage shows up as throughput; ratios against a saturated vanilla understate the true memory saving.
Eager vanilla is the backend-matched control; full-graph vanilla is its best case and the reference for all ratios.
R-KV is measured on the same harness and the same GPU, through its own vLLM build; it cannot use full CUDA graphs,
prefix caching or async scheduling, which its engine disables whenever eviction is on, so its row carries that cost.}
\label{tab:bench}
\resizebox{\linewidth}{!}{\begin{tabular}{@{}lrrrrrr@{}}
\toprule
Arm & KV rows/seq (eq.) & KV GiB/seq & KV saving & cache used & tok/s & vs.\ vanilla \\
\midrule
""" + " \\\\\n".join(rows) + r""" \\
\bottomrule
\end{tabular}}
""" + ((r"""
\vspace{2pt}
\footnotesize Full-length 16K sequences resident at once, i.e.\ the median number of running requests over the final quarter of a saturation run, when every sequence is near 16K (the count at admission is the request count for every arm and says nothing): """ + satrow + ".") if satrow else "") + r"""
\end{table}
""")
else:
    print("skipped tables/bench.tex (no paper/data/bench_decode.json yet)")

# ---- R-KV appendix: accuracy, truncation and generation length side by side ----
# The truncation and length columns are the core evidence of the R-KV comparison, and mean generated tokens appears in no
# other table, so this one is generated rather than hand-typed: the appendix prose quotes these exact figures.
RKV_ROWS = [("vanilla", "vanilla SFT (full KV)", "512"), ("vpl", r"\method{}, per-layer 4:1", "128"),
            ("rkv_b4096", r"R-KV, budget 4096", "144"), ("ck512k64", r"\method{}, uniform 8:1", "64"),
            ("rkv_b2048", r"R-KV, budget 2048", "74")]
rows = []
for key, lab, kv in RKV_ROWS:
    p = X["arms"].get(key, {}).get("pool")
    if not p: continue
    d = X["arms"][key].get("diff_vs_matched")          # each R-KV budget against the arm its residency brackets
    dl = "--" if not d else "$%+.1f$ {\\scriptsize[%+.1f, %+.1f]}" % (d[0], d[1], d[2])
    ci = p["ci"]
    rows.append("%s & %s & %.1f {\\scriptsize[%.1f, %.1f]} & %s & %.1f & %s \\\\" %
                (lab, kv, p["avg8"], ci[0], ci[1], dl, p["trunc"], "{:,}".format(int(round(p["gen"])))))
if rows:
    write("rkvcost", r"""\begin{table}[t]
\centering\footnotesize
\caption{R-KV against the arm it is paired with at each budget, on the same 446 problems, the same sampling and the same
grader. KV/512 = persistent rows per 512 generated tokens per KV head; R-KV's is its \emph{measured} mean residency, so at
both settings it holds about 12\% more cache than the arm beside it. trunc.\ = \% of samples that reach the 32K cap.
$\Delta$ is the paired per-problem difference against that partner arm, bootstrapped on the 446 common problems.
Accuracy separates them only at the tighter budget; truncation and generated length separate them at both.}
\label{tab:rkvcost}
\begin{tabular}{@{}lrlcrr@{}}
\toprule
Arm & KV/512 & Suite avg@8 & $\Delta$ vs.\ partner & trunc.\ \% & gen.\ tokens \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

# ---- efficiency appendix: memory, concurrency and throughput, one row per arm, each against vanilla ----
# Memory and batch-8 throughput come from the bench package (three repeats, 0.04-0.5% spread). Concurrency and saturated
# throughput come ONLY from the single-job controlled run (collect_sat.py): inside one job the ratio to vanilla reproduces
# to ~5%, across jobs absolute throughput swings 17%.
SAT_PATH = os.path.join(HERE, "data", "saturation.json")
if os.path.isfile(BENCH_PATH) and os.path.isfile(SAT_PATH):
    B = json.load(open(BENCH_PATH)); SD = json.load(open(SAT_PATH)); kvb = B["kv_bytes_per_token"]
    b8 = {c["tag"]: c for c in B["cells"] if c["mode"] == "batch" and c["len"] == 16384 and c["batch"] == 8}
    EFF = [("vanilla_graphs", r"full KV (vanilla)"),
           ("tpl_physical", r"\textbf{AbridgeKV} (per-layer, $C{=}512$)"),
           ("uni_plain", r"uniform 128, $C{=}512$"),
           ("ck1024k256", r"uniform $C{=}1024$, $K{=}256$"),
           ("ck512k256", r"uniform $C{=}512$, $K{=}256$"),
           ("ck256k64", r"uniform $C{=}256$, $K{=}64$"),
           ("ck512k64", r"uniform $C{=}512$, $K{=}64$"),
           ("gist", r"appended positions"),
           ("rkv_b4096", r"R-KV, budget 4096"),
           ("rkv_b2048", r"R-KV, budget 2048")]
    v8, vs = b8.get("vanilla_graphs"), SD["arms"].get("vanilla_graphs")
    def x(a, b, f="%.2f"): return (f % (a / b) + r"$\times$") if (a and b) else "--"
    erows = []
    for tag, lab in EFF:
        c = b8.get(tag)
        if c is None: continue
        g = c["kv_tokens_per_seq"] * kvb / 2**30
        gv = v8["kv_tokens_per_seq"] * kvb / 2**30
        if tag.startswith("rkv_"):
            r = SD["rkv"].get(tag, {}); n = r.get("steady_running")
            conc = ((r"$\geq$" if r.get("bound") == "lower" else "") + "%d" % round(n)) if n else "--"
            concx = ((r"$\geq$" if r.get("bound") == "lower" else "") + x(n, vs and vs["steady_running"], "%.1f")) if n else "--"
            stp, stpx = r"n/m$^\dagger$", "--"
        else:
            s_ = SD["arms"].get(tag)
            conc = ("%d" % s_["steady_running"]) if s_ else "pending"
            concx = x(s_ and s_["steady_running"], vs and vs["steady_running"], "%.1f") if s_ else "--"
            stp = ("%.0f" % s_["tok_per_s"]) if s_ else "pending"
            stpx = x(s_ and s_["tok_per_s"], vs and vs["tok_per_s"]) if s_ else "--"
        erows.append(" & ".join([lab, "%.2f" % g, x(gv, g), conc, concx, "%.0f" % c["tok_per_s"],
                                 x(c["tok_per_s"], v8["tok_per_s"]), stp, stpx]))
    write("efficiency", r"""\begin{table}[t]
\centering\small
\caption{Serving gains, one quantity at a time, each against full-KV vanilla on one H100 with every sequence forced to
16K generated tokens. \emph{Memory}: resident KV per sequence at batch 8, in GiB (full-KV-equivalent). \emph{Concurrency}:
sequences resident at once when the engine is saturated (512 requests queued; median running count over the final
quarter of the run). It is a measured count, not memory divided by size: sequences in flight are at mixed lengths, so an
arm whose cache grows with length fits more of them than its 16K footprint implies, while R-KV's cache is flat at its
budget and its concurrency tracks its memory saving. \emph{Throughput}: aggregate decode tokens/s at batch 8, and at saturation.
Memory and batch-8 throughput are medians of three repeats (spread 0.04--0.5\%). All saturation cells come from one job on
one node, arms interleaved, because the ratio to vanilla reproduces to about 5\% inside a job while absolute throughput
varies by up to 17\% across jobs on shared nodes. Both figures in Figure~\ref{fig:pareto} divide mean generated tokens by
the batch-8 and saturated columns here. R-KV runs with block freeing on (Appendix~\ref{app:rkv}).
$^\dagger$Not measurable: at the queue depth that saturates it, R-KV either exceeds its scoring-buffer cap, which makes its
engine silently fall back to full KV, or, with the cap raised, runs out of memory. Its concurrency is still readable from
those runs: 108 and 107 at 128 and 512 requests for budget 4096; at budget 2048 three requests fell back to full KV, which
holds more cache than a compacting request, so 211 is a lower bound.}
\label{tab:efficiency}
\resizebox{\linewidth}{!}{\begin{tabular}{@{}lrrrrrrrr@{}}
\toprule
& \multicolumn{2}{c}{memory} & \multicolumn{2}{c}{concurrency} & \multicolumn{2}{c}{throughput, batch 8} & \multicolumn{2}{c}{throughput, saturated} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(l){8-9}
Arm & GiB/seq & saving & seqs & gain & tok/s & gain & tok/s & gain \\
\midrule
""" + " \\\\\n".join(erows) + r""" \\
\bottomrule
\end{tabular}}
\end{table}
""")
else:
    print("skipped tables/efficiency.tex (bench package or saturation.json missing)")


# ================= coding domain at prototyping scale (arXiv appendix; results.json "proto_followups") =================
PF = R.get("proto_followups")
if PF:
    van = PF["code"][0]["ce"]
    rows = []
    for a in PF["code"]:
        d = "--" if not a["ours"] else "$%+.4f$" % (a["ce"] - van)
        shade = "\\rowcolor{oursbg} " if a["ours"] else ""
        rows.append("%s%s & %.4f & %s \\\\" % (shade, a["label"], a["ce"], d))
    write("code_proto", "\\begin{table}[h]\n\\centering\\small\n"
          "\\caption{Coding domain at prototyping scale: held-out cross-entropy (nats per response token) after 600M response tokens on "
          "OpenCodeReasoning, and the difference to the full-KV model trained identically. All compressed arms keep 128 code rows per "
          "512-token block (4:1). Resolution of the comparison: 0.0005 (a null pair at the same scale on mathematics).}\n"
          "\\label{tab:code}\n\\begin{tabular}{lcc}\n\\toprule\nArm & Held-out CE & vs.\\ full KV \\\\\n\\midrule\n"
          + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
