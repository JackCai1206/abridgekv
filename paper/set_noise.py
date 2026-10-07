"""Resolution of a single 30-problem set vs the pooled suite: paired per-problem bootstrap CIs of three gaps (per-layer - vanilla,
uniform - vanilla, vanilla seed 1 - vanilla) at 32K/avg@8, measured on one competition set and on the 446-problem suite.
Writes data/set_noise.json and tables/set_noise.tex."""
import json, glob, random, os
HERE = os.path.dirname(os.path.abspath(__file__)); EV = os.path.join(os.environ.get("OUTPUT_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")), "eval")
random.seed(0); B = 2000
SETS = ['brumo_2025','cmimc_2025','smt_2025','aime_2026','hmmt_feb_2023','aime_2023','aime_2022','amc_2022_2023','aime2024_I','aime2024_II','aime2025','hmmt_feb_2024','hmmt_feb_2025']
TAGS = {'vanilla': 'vanilla_sameset32k', 'seed1': 'vanilla_seed1_sameset32k', 'vpl': 'c256tpl_sameset32k', 'uni': 'c256uni_u128_sameset32k'}
def load_set(tag, s):
    fs = glob.glob(f"{EV}/{s}_qwen3_4b_vllm_{tag}_merged/*.json")
    if not fs: return {}
    per = {}
    for p in json.load(open(fs[0]))["per_problem"]:
        v = [float(x["correct_semantic_full_text"]) for x in p["samples"]]; k = min(len(v), 8); per[(s, p["dataset_index"])] = sum(v[:k]) / k
    return per
def boot_paired(a, b):
    d = [a[k] - b[k] for k in sorted(k for k in a if k in b)]; n = len(d)
    rng = random.Random(0); ms = sorted(sum(rng.choice(d) for _ in range(n)) / n for _ in range(B))   # same seeding as collect_32k
    return [100 * sum(d) / n, 100 * ms[int(.025 * B)], 100 * ms[int(.975 * B)], n]
D = {a: {s: load_set(t, s) for s in SETS} for a, t in TAGS.items()}
def pool(a, ss): r = {}; [r.update(D[a][s]) for s in ss]; return r
GROUPS = [("AIME 2025", ['aime2025']), ("AIME 2024 I+II", ['aime2024_I', 'aime2024_II']), ("HMMT Feb 2025", ['hmmt_feb_2025']),
          ("AIME 2024+2025", ['aime2024_I', 'aime2024_II', 'aime2025']), ("full suite", SETS)]
PAIRS = [("vpl", "vanilla"), ("uni", "vanilla"), ("seed1", "vanilla")]
out = {}
rows = []
for g, ss in GROUPS:
    out[g] = {}
    cells = []
    for a, b in PAIRS:
        m, lo, hi, n = boot_paired(pool(a, ss), pool(b, ss)); out[g][f"{a}-{b}"] = [m, lo, hi, n]
        sig = r"\textbf{%+.1f}" % m if (lo > 0 or hi < 0) else "%+.1f" % m
        cells.append(f"{sig} $[{lo:+.1f}, {hi:+.1f}]$")
    hw = sum((out[g][f"{a}-{b}"][2] - out[g][f"{a}-{b}"][1]) / 2 for a, b in PAIRS) / len(PAIRS)
    rows.append(f"{g} & {n} & " + " & ".join(cells) + f" & {hw:.1f} \\\\")
json.dump(out, open(os.path.join(HERE, "data", "set_noise.json"), "w"), indent=1)
body = r"""\begin{table}[t]
\centering\small\setlength{\tabcolsep}{4pt}
\caption{What one competition set can resolve. The same three paired gaps in avg@8 at the 32K cap (mean and 95\% per-problem
bootstrap interval), measured on a single 30-problem set and on the pooled suite. Bold: interval excludes zero. The last column
is the mean interval half-width. A vanilla replicate that differs only in data seed is ``significantly'' worse than vanilla on
AIME 2024 and better on AIME 2025; uniform compression changes sign between AIME 2024 and HMMT 2025.}
\label{tab:setnoise}
\resizebox{\linewidth}{!}{\begin{tabular}{@{}lrcccc@{}}
\toprule
measured on & problems & per-layer $-$ vanilla & uniform $-$ vanilla & vanilla seed 1 $-$ vanilla & $\pm$ \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}}
\end{table}
"""
open(os.path.join(HERE, "tables", "set_noise.tex"), "w").write(body); print("wrote tables/set_noise.tex"); print("\n".join(rows))
