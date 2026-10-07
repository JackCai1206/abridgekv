#!/usr/bin/env python3
"""Generate paper/figures/*.pdf from paper/data and paper/results.json. One shared style: pastel palette, rounded bars, no top/right
spines, light grid. Pending arms are listed under the axes rather than drawn."""
import numpy as np
import json, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); FIG = os.environ.get("FIG_DIR") or os.path.join(HERE, "figures"); os.makedirs(FIG, exist_ok=True)
R = json.load(open(os.path.join(HERE, "results.json")))
X = json.load(open(os.path.join(HERE, "data", "results32k.json")))
B = json.load(open(os.path.join(HERE, "data", "bench_decode.json")))
# Full KV's run-to-run noise on the pooled suite: two vanilla seeds on the same 446 problems differ by +0.5 with a paired
# 95% CI of half-width w, so a re-run lands within w / 1.96 (~0.75 point) of vanilla at 1 s.d. -- the noise any arm's
# paired gap to vanilla carries. Drawn as bands at 1 and 2 s.d. around the full-KV line.
_sn = json.load(open(os.path.join(HERE, "data", "set_noise.json")))["full suite"]["seed1-vanilla"]
VAN_SD = (_sn[2] - _sn[1]) / 2 / 1.96

# ---------------- style ----------------
PAL = {"ink": "#3D405B", "grey": "#9AA3AD", "coral": "#E07A5F", "sage": "#81B29A", "slate": "#6C8EBF", "sand": "#F2CC8F", "lilac": "#A28CD4", "sky": "#8FC1E3", "rose": "#EFA3A3", "peach": "#F2B29B", "rust": "#B8664C"}
# Every variant of OUR method is in the coral family (per-layer full coral, ablations a lighter peach); other people's
# methods get other hues (codes-free eviction slate, R-KV lilac). With the ablations in blue they read as someone else's
# method (user 2026-09-24). Peach is too light for text, so ablation LABELS use the darker rust (GROUP_TXT).
GROUP = {"reference": PAL["grey"], "ours": PAL["coral"], "ablation": PAL["peach"], "layout": PAL["sand"], "baseline": PAL["slate"], "prior": PAL["sage"]}
GROUP_TXT = {"ablation": PAL["rust"]}
plt.rcParams.update({"font.size": 8, "font.family": "sans-serif", "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#B9BEC6",
                     "axes.labelcolor": PAL["ink"], "xtick.color": PAL["ink"], "ytick.color": PAL["ink"], "text.color": PAL["ink"], "axes.titlecolor": PAL["ink"],
                     "axes.titlesize": 8.5, "axes.titleweight": "semibold", "axes.grid": True, "grid.color": "#E6E8EC", "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "legend.frameon": False, "legend.fontsize": 6.5, "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.labelsize": 7.5})
def rbar(ax, x, h, w=0.7, color="k", base=0.0, r=0.12):
    """A bar with rounded top corners (rounding in data units of x-width)."""
    if h == base: return
    lo, hi = (base, h) if h > base else (h, base)
    ax.add_patch(FancyBboxPatch((x - w / 2, lo), w, hi - lo, boxstyle="round,pad=0,rounding_size=%g" % (r * w), mutation_aspect=(ax.get_ylim()[1] - ax.get_ylim()[0]) / 6 if False else 1,
                                linewidth=0, facecolor=color, zorder=3))
def rbars(ax, xs, hs, w=0.7, colors=None, base=0.0):
    colors = colors if isinstance(colors, list) else [colors] * len(xs)
    top = max([abs(h) for h in hs] + [1e-9]); pad = 0.06 * top
    ax.set_ylim(min(0, min(hs)) - (pad if min(hs) < 0 else 0), max(hs) + pad) if base == 0 else None
    for x, h, c in zip(xs, hs, colors): rbar(ax, x, h, w, c, base)
    ax.set_xlim(min(xs) - 0.6, max(xs) + 0.6)
def short(a): return a["short"].replace("\\method{}", "AbridgeKV")   # the macro is LaTeX-only; figures spell the name
def pool_of(key):
    p = X["arms"].get(key, {}).get("pool"); return (p["avg8"], p["complete"], p["n"]) if p else None

# ---------------- Figure: main results (suite level + MATH-500) ----------------
ARMS = [a for a in R["arms32k"] if a["group"] not in ("archived", "allocation", "prior", "held")]   # prior-method rows: table only until the faithful LightThinker run lands   # archived arms live in the appendix table only
# This figure exists to show that a learned summary matches the full-KV ceiling, so its baseline bars are the
# codes-free policies at the same budget. R-KV is a published method with its own budget, argued against on accuracy,
# truncation and time-to-finish in Figure~3 and Appendix D; a bar here only invites reading it as a fourth eviction
# policy of ours (user 2026-09-23).
ARMS_MAIN = [a for a in ARMS if not a["key"].startswith("rkv_")]
# Left to right (user 2026-09-24): references, ours, its ablations (uniform allocation, appended positions), then the
# three codes-free baselines. Arms not named here keep their table order after these.
_ORD = ["base", "vanilla", "vpl", "uniform128", "gist"]
ARMS_MAIN.sort(key=lambda a: (_ORD.index(a["key"]) if a["key"] in _ORD else len(_ORD) + [x["key"] for x in ARMS].index(a["key"])))
XTF, XTR = 6.5, 25
def bars(ax, field, ylabel, skip_ref=False):
    """skip_ref: leave out the reference arms (base, vanilla); the caller draws full KV as a dotted line instead."""
    xs, ys, cs, pend = [], [], [], []
    for a in ARMS_MAIN:
        if skip_ref and a["group"] == "reference": continue
        rec = X["arms"].get(a["key"], {}); lab = short(a)
        if field == "pool" and rec.get("pool"):
            p = rec["pool"]; xs.append(lab + ("" if p["complete"] else "\n(%d/%d)" % (p["n"], p["n_expected"]))); ys.append(p["avg8"]); cs.append(GROUP[a["group"]])
        elif field == "math500" and a.get("math500") is not None:
            xs.append(lab); ys.append(a["math500"]); cs.append(GROUP[a["group"]])
        elif a["status"] == "done":
            continue          # the arm is finished and simply has no run in this column (the codes-free policies have
                              # no MATH-500 of their own): listing it as pending printed "recency done; random done".
        else:
            pend.append(lab + " " + a["status"].replace("queued", ""))
    if xs:
        rbars(ax, list(range(len(xs))), ys, 0.68, cs)
        for i, y in enumerate(ys): ax.text(i, y + 0.4, "%.1f" % y, ha="center", fontsize=6, color=PAL["ink"])
        ax.set_xticks(range(len(xs))); ax.set_xticklabels(xs, fontsize=XTF, rotation=XTR, ha="right", rotation_mode="anchor")
    ax.set_ylabel(ylabel)
    if pend: ax.set_xlabel("pending: " + "; ".join(pend), fontsize=5.5, color=PAL["grey"], wrap=True)
fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.1), gridspec_kw={"width_ratios": [2.2, 1]})
bars(axes[0], "pool", "suite avg@8 (%d problems)" % X["n_competition"]); axes[0].set_title("competition suite, 32K cap")
bars(axes[1], "math500", "MATH-500 (avg@4)"); axes[1].set_title("MATH-500, avg@4 at T=0.6"); axes[1].set_ylim(60, 92)
# Headroom for a two-column legend across the top: dropping the R-KV bars left "base" (13.5) as the shortest bar and
# the old lower-left legend sat straight on it (2026-09-23).
axes[0].set_ylim(0, max(axes[0].get_ylim()[1], 44) * 1.30)
axes[0].legend(handles=[mpatches.Patch(color=c, label=g) for g, c in [("reference", GROUP["reference"]), ("ours", GROUP["ours"]), ("uniform allocation / budget sweep", GROUP["ablation"]), ("codes-free eviction, same budget", GROUP["baseline"])]],
               loc="upper center", ncol=2, fontsize=6.2, columnspacing=1.2, handlelength=1.1, handleheight=0.9, borderaxespad=0.15)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "main_bars.pdf")); plt.close(fig)

# ---------------- Figure: accuracy vs efficiency (Pareto) ----------------
# ONE panel (user 2026-09-23): accuracy against the cost a user actually pays -- the time to FINISH a problem. That is
# two measured quantities multiplied: how many tokens the arm generates (suite mean, 446 problems) and how long it
# spends per token (decode bench, batch 8, 16K forced length). A KV budget axis cannot express this, and it is the axis
# on which a method can hold less cache and still lose: R-KV keeps a smaller cache than vanilla and takes far longer,
# because it generates 40-69% more tokens and pays a compaction event every 64 decode steps.
MK = {"reference": "s", "ours": "*", "ablation": "o", "layout": "D", "baseline": "^", "prior": "v"}
RKV = PAL["lilac"]
BTAG = {"vanilla": "vanilla_graphs", "vpl": "tpl_physical", "uniform128": "uni_plain", "gist": "gist",
        "ck512k64": "ck512k64", "ck256k64": "ck256k64", "ck512k256": "ck512k256", "ck1024k256": "ck1024k256",
        "rkv_b4096": "rkv_b4096", "rkv_b2048": "rkv_b2048", "lt_thought": "lt_thought"}
b8 = {c["tag"]: c for c in B["cells"]
      if c["mode"] == "batch" and c["len"] == 16384 and c["batch"] == 8 and c.get("tok_per_s")}
# Saturation cells: the operating point where a smaller cache becomes CONCURRENCY. At batch 8 full KV still fits, so
# compression wins only on attention width and the plot understates it (ours is 0.89x vanilla there, 0.80x saturated);
# worse, an arm that holds 1.85x the cache for the same accuracy looks equally good, because a fixed-batch axis cannot
# see memory at all. Saturating makes the axis price memory directly: an arm that holds more KV sustains fewer
# sequences and its GPU-seconds per problem rise on their own. Concurrency measured here: vanilla 31, ours 121.
def _valid_sat(c):
    """A saturation cell counts only if the GPU ran out of MEMORY, not out of work. When steady_running reaches the
    request count every queued request was resident, the arm never met its ceiling, and its throughput is a floor --
    which on this axis penalises precisely the arms with the smallest caches. Filtering on a hardcoded request count
    would also have silently dropped the 512-request re-runs of the small-cache arms."""
    if c["mode"] != "saturate" or c["len"] != 16384 or not c.get("tok_per_s"): return False
    req, run = c.get("requests") or c.get("batch"), c.get("steady_running")
    # "run < req" is NOT enough. No sequence ever finishes in these runs, so a deeper queue keeps the cache fuller and
    # the observed ceiling RISES: ck256k64 read 124 concurrent at 128 requests and 166 at 512, and 124 passed a naive
    # "<" test. Demand 2x headroom so the ceiling is set by memory rather than by how much work was queued.
    return bool(req and run and 2 * run < req)
# Saturation cells come ONLY from the single-node controlled run (collect_sat.py -> data/saturation.json): the ratio to
# vanilla reproduces to ~5% inside one job but absolute throughput swings 17% across jobs on shared nodes.
_SATF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "saturation.json")
SATD = json.load(open(_SATF)) if os.path.exists(_SATF) else {"arms": {}, "rkv": {}}
sat = {t: dict(a, tag=t) for t, a in SATD["arms"].items()}
# Run-to-completion cells (collect_complete.py): each arm replays its OWN suite lengths on 512 paired draws, all queued at
# once and run to the end, so the long-context phase where the cache binds is inside the measurement. When present they
# REPLACE the fixed-step saturation cells on the saturated panel: those stopped after 17,000 steps with nothing finished
# and mostly timed the start-up phase, where no arm is memory-bound (2026-09-23).
# ASYNC (2026-09-25): every arm but R-KV served with vLLM async scheduling (our arms: async + code-step KV trim); R-KV's
# released port disables async for its compaction, so its cells stay the synchronous ones (linked into the async dir).
_COMPF = os.environ.get("COMP_JSON") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "completion_async.json")
COMP = (json.load(open(_COMPF)) if os.path.exists(_COMPF) else {"arms": {}})["arms"]
# Memory-bound operating point (panel c): the same replay with the paged KV cache fixed at 8 GiB for every arm, so the
# cache binds for the compressed arms too. R-KV has no 8 GiB cell; it is drawn at its full-memory cell (a lower bound).
_COMPKF = os.environ.get("COMPK_JSON") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "completion_async_kv8.json")
COMPK = (json.load(open(_COMPKF)) if os.path.exists(_COMPKF) else {"arms": {}})["arms"]
RKV_COMP = any(t.startswith("rkv_") for t in COMP)   # run-to-completion R-KV cells exist (block freeing on, 2026-09-24)

# Two panels, two operating points, never mixed on one axis (see bench_all_saturate.sbatch):
#   pareto.pdf      -- batch 8. Latency for one user. Reproducible to 0.04-0.5% across repeats.
#   pareto_conc.pdf -- saturation. Capacity cost, where a smaller cache turns into concurrency. Only the arms measured
#                      inside ONE job are comparable here; saturation throughput swings 17% across jobs on shared nodes.
# Drawn by draw_pareto into a given axes: standalone (pareto.pdf, pareto_conc.pdf) and as panels (c)/(d) of the main
# results figure (main_panels.pdf, user 2026-09-24). compact=True shrinks markers/labels for the half-width panel.
def draw_pareto(ax, _axis, compact=False, side=None):
    """side="left"/"right": one half of a broken x-axis (compact panels). R-KV sits 2-3x to the right of every other
    arm; a linear axis crushed our cluster and a log axis still squeezed it, so the panel is split at the widest gap
    in x and each half gets its own linear range (user 2026-09-24)."""
    AXIS = _axis
    # NEVER mix operating points on one axis -- that is the same class of error as mixing an A100 cell into an H100 package.
    RATE = sat if AXIS == "saturate" else b8
    PERSEQ = 1 if AXIS == "saturate" else 8            # saturate tok/s is aggregate per GPU; batch tok/s is per batch of 8
    def secs(key, rate_tag=None):
        """GPU-seconds to finish one problem at batch 8 = mean generated tokens / per-sequence token rate."""
        pl = X["arms"].get(key, {}).get("pool")
        if AXIS in ("saturate", "kv8") and COMP:
            t = BTAG.get(key, "")
            src = COMP if (AXIS == "saturate" or t.startswith("rkv_")) else COMPK
            c = src.get(t) if rate_tag is None else None
            return c["s_per_problem"] if (pl and c) else None
        c = RATE.get(rate_tag or BTAG.get(key, ""))
        return (PERSEQ * pl["gen"] / c["tok_per_s"]) if pl and c else None
    def secs_err(key):
        """95% interval on x. Both factors carry error, but only one matters: the three decode repeats agree to 0.04-0.5%,
        while the generated-token mean is a sample over 446 problems (relative SE ~2.2%), so the bench term is dropped."""
        pl = X["arms"].get(key, {}).get("pool"); c = RATE.get(BTAG.get(key, ""))
        se = (pl or {}).get("gen_se")
        if AXIS in ("saturate", "kv8") and COMP:
            # measured completion time; carry the same relative sampling noise of the generated-token mean
            x = secs(key); return (1.96 * x * se / pl["gen"]) if (x and se and pl) else None
        return (1.96 * PERSEQ * se / c["tok_per_s"]) if pl and c and se else None
    # (dx, dy, ha) in points. Hand-set: the four sweep arms cluster inside ~15 s of each other.
    # vpl sits just under vanilla's horizontal guide and vanilla's own marker is 13 s to its right, so a label above the
    # star ran through both. Put ours to the left and vanilla's below its guide line.
    # Coordinates (s, avg@8): vanilla (113.0, 43.81) vpl (100.5, 43.05) ck512k256 (97.6, 43.36) ck1024k256 (85.7, 42.54)
    # uniform128 (89.6, 41.48) ck256k64 (87.4, 40.67) ck512k64 (73.1, 37.56) gist (89.0, 35.31). The top four sit within
    # 1.3 points and 27 s, so ours goes ABOVE-left (clear of vanilla's guide at 43.81) and ck1024k256 BELOW-left, or the
    # two long labels run through each other and through the star.
    LBL = {"vanilla": (10, -11, "left"), "vpl": (-13, 4, "right"), "uniform128": (-10, -4, "right"),
           "gist": (9, -3, "left"), "ck512k64": (0, -15, "center"), "ck256k64": (-9, -4, "right"),
           "ck512k256": (0, 11, "center"), "ck1024k256": (-9, -5, "right"),
           "rkv_b4096": (-9, 9, "right"), "rkv_b2048": (-9, 9, "right"), "lt_thought": (0, -12, "center")}
    if AXIS in ("saturate", "kv8"):
        # Saturated: ck512k256 (4.15 s, 43.36) sits 0.4 s left of ours (4.56, 43.05), so its label goes left, ours right.
        LBL.update({"ck512k256": (-10, 3, "right"), "vpl": (11, -3, "left")})
    if compact:
        # Half-width panels: vanilla's label goes above its marker (it is the right edge of the left half), and at batch 8,
        # where the x bars are wide, the sweep labels sit above-left of their bars instead of running through them.
        LBL.update({"vanilla": (0, 7, "center"), "gist": (0, -13, "center"), "rkv_b4096": (4, 8, "right"), "rkv_b2048": (0, 8, "center")})
        if AXIS in ("saturate", "kv8"):
            # Mixed-steps data (2026-09-24): ck512k256 (5.35 s) now sits to the RIGHT of ours (3.80 s) on the same line, so
            # ours labels below-right and ck512k256 to its own right; R-KV 2048's label goes under its marker, clear of the legend.
            LBL.update({"vpl": (9, -12, "left"), "ck512k256": (7, 3, "left"), "rkv_b2048": (0, -12, "center")})
        if AXIS == "batch8":
            LBL.update({"vpl": (10, -13, "left"), "ck1024k256": (-4, 6, "right"), "uniform128": (-4, 6, "right"),
                        "ck256k64": (-4, 6, "right")})
    FSL = 6.2 if compact else 7.4   # point-label font
    pts = [(a["key"], a["group"], short(a)) for a in ARMS if a["key"] != "base" and not a["key"].startswith("evict_")]
    # The ratio sweep at C=512 is back on both panels (user 2026-09-25), named by ratio like the sweep figure; the uniform
    # 4:1 point is the middle of that sweep, and an arrow from it to ours shows what the per-layer schedule buys.
    RATIO = {"ck512k256": "2:1", "uniform128": "4:1", "ck512k64": "8:1"}
    pts = [(k, g, RATIO.get(k, lab)) for k, g, lab in pts]
    pts += [(k, "ablation", RATIO[k]) for k in ("ck512k256", "ck512k64") if X["arms"].get(k, {}).get("pool")]
    # LightThinker's thought-level algorithm on our stack (user 2026-09-26: on the Pareto plot, not on Figure 5)
    if COMP.get("lt_thought") and AXIS == "saturate": pts.append(("lt_thought", "prior", "LightThinker"))
    # Only the ratio sweep at C=512 (2:1, 8:1) stays here. The block-size arms at a fixed 4:1 (C=256, 1024) differ in peak KV,
    # not in time to finish, and crowded this panel; they have their own figure (block_scale.pdf, user 2026-09-25).
    # The C/K sweep lives in its own two-panel figure (ck_sweep.pdf, user 2026-09-25): fixed ratio / different block size,
    # and fixed block size / different ratio. The Pareto panels keep only the main comparison.
    xs, ys, pend = [], [], []
    van_x, van_y = secs("vanilla"), (X["arms"].get("vanilla", {}).get("pool") or {}).get("avg8")
    for key, g, lab in pts:
        x = secs(key); pl = X["arms"].get(key, {}).get("pool")
        if not pl: continue
        if x is None: pend.append(lab); continue
        if AXIS == "kv8" and key.startswith("rkv_"): lab += r"$^{\dagger}$"   # full-memory cell: lower bound on its 8 GiB time
        y = pl["avg8"]; xs.append(x); ys.append(y)
        col = RKV if key.startswith("rkv_") else GROUP[g]
        # Horizontal 95% bars: sampling noise in the generated-token mean. They are small (+/-4-5 s for our arms) and that
        # is the point -- the separation from vanilla is far larger. Between-arm comparisons are PAIRED on the same 446
        # problems and tighter still (ours - vanilla = -12.4 s [-14.4, -10.5]); no vertical bars, because the unpaired
        # accuracy interval is +/-3.7 points and would swamp a plot whose every comparison is paired.
        xe = secs_err(key)
        if xe:
            ax.errorbar([x], [y], xerr=[[xe], [xe]], fmt="none", ecolor=GROUP_TXT.get(g, col), elinewidth=1.0,
                        capsize=2.0, capthick=1.0, alpha=0.85, zorder=4)
        ax.scatter([x], [y], marker=MK[g], color=col, s=(240 if compact else 340) if g == "ours" else (34 if compact else 78), zorder=5,
                   edgecolors=PAL["rust"] if g == "ablation" else "white", linewidths=0.7 if g == "ablation" else 1.0)
        # R-KV's per-token cost is partly the released implementation: its scoring is pure torch, ~2.2 TFLOP/s effective
        # against 1.28 TFLOP of arithmetic per event. The bar runs back to where it would sit with FREE scoring, i.e. its
        # generation-length penalty alone at vanilla's token rate. Even that bound is slower than every other arm.
        if key.startswith("rkv_"):
            lo = secs(key, "vanilla_graphs")
            if lo:
                ax.plot([lo, x], [y, y], color=col, lw=1.6, alpha=0.5, zorder=3, solid_capstyle="butt")
                ax.plot([lo], [y], marker="|", color=col, ms=7, mew=1.6, zorder=4)
        dx, dy, ha = LBL.get(key, (8, 4, "left"))
        if compact: dx, dy = 0.8 * dx, 0.8 * dy
        ax.annotate(lab, (x, y), fontsize=FSL, color=GROUP_TXT.get(g, col) if not key.startswith("rkv_") else col, xytext=(dx, dy), textcoords="offset points",
                    ha=ha, zorder=6, fontweight="semibold" if g == "ours" else "normal")
    # Two within-method curves from the C/K sweep: the compression ratio at fixed C=512 (K = 256, 128, 64 -> 2:1, 4:1,
    # 8:1) and the block size at a fixed 4:1 ratio (C = 256, 512, 1024). Thin lines, under the markers.
    for chain, ls_ in ((("ck512k256", "uniform128", "ck512k64"), "-"),):
        cx = [(secs(k), (X["arms"].get(k, {}).get("pool") or {}).get("avg8")) for k in chain]
        cx = [(a, b) for a, b in cx if a and b]
        if len(cx) >= 2:
            ax.plot([a for a, _ in cx], [b for _, b in cx], color=GROUP["ablation"], lw=0.9, ls=ls_, alpha=0.55, zorder=2)
    ua, ub = (secs("uniform128"), (X["arms"].get("uniform128", {}).get("pool") or {}).get("avg8")), \
             (secs("vpl"), (X["arms"].get("vpl", {}).get("pool") or {}).get("avg8"))
    if all(ua) and all(ub) and side != "right":
        ax.annotate("", xy=ub, xytext=ua, zorder=6, arrowprops=dict(arrowstyle="-|>", color=GROUP["ours"], lw=1.0,
                    shrinkA=4, shrinkB=9, mutation_scale=7))
    if xs and van_x and van_y:
        # A vanilla cross-hair makes "cheaper AND at least as accurate" a quadrant you can see, rather than one to compute.
        ax.axvline(van_x, color=PAL["grey"], lw=0.8, ls=(0, (4, 3)), zorder=1)
        ax.axhline(van_y, color=PAL["grey"], lw=0.8, ls=(0, (4, 3)), zorder=1)
        ax.axhspan(van_y - VAN_SD, van_y + VAN_SD, color=PAL["grey"], alpha=0.22, lw=0, zorder=0)   # one shade: 1 s.d. (user 2026-09-25)
        lo_x, hi_x = 0, max(xs) * 1.16
        if side:
            # Split at the widest ratio gap among every x drawn (markers and R-KV's free-scoring bar ends).
            # Left half: every arm up to full KV. Right half: whatever is slower than full KV (R-KV, incl. its bar ends).
            allx = sorted(xs + [v for v in (secs(k, "vanilla_graphs") for k, _, _ in pts if k.startswith("rkv_")) if v])
            L, R_ = [v for v in allx if v <= van_x * 1.05], [v for v in allx if v > van_x * 1.05]
            spanL = max(L) - min(L)
            lo_x, hi_x = (min(L) - 0.30 * spanL, max(L) + 0.30 * spanL) if side == "left" else \
                         (min(R_) - 0.12 * (max(R_) - min(R_) + 1e-9) - 0.12 * min(R_), max(R_) + 0.12 * (max(R_) - min(R_)) + 0.05 * max(R_))
        lo_y, hi_y = min(ys) - 2.4, max(ys) + 2.0
        ax.add_patch(plt.Rectangle((lo_x, van_y), van_x - lo_x, hi_y - van_y, facecolor=PAL["sage"], alpha=0.07, zorder=0))
        ax.set_xlim(lo_x, hi_x); ax.set_ylim(lo_y, hi_y)
        if side != "right":
            ax.text((lo_x + van_x) / 2, hi_y - 0.35, "faster than full KV,\nno less accurate" if side else "faster than full KV, no less accurate",
                    fontsize=6.0 if compact else 6.6, color=PAL["sage"], ha="center", va="top", style="italic")
    if side == "right":
        ax.tick_params(labelleft=False, left=False); ax.spines["left"].set_visible(False)
        return
    if compact:
        ax.set_xlabel("seconds per problem" if AXIS in ("saturate", "kv8") else "GPU-seconds per problem, batch 8")
        if side: ax.xaxis.set_label_coords(0.5 * 4.2 / 3.2, -0.14)
        ax.set_ylabel("suite avg@8")
        if side: ax.spines["right"].set_visible(False)
    else:
        ax.set_xlabel("GPU-seconds to finish one problem   " + (("(512 suite-length requests run to completion, one H100)" if COMP else
                      "(mean generated tokens $\\div$ saturated throughput, one H100)")
                      if AXIS == "saturate" else "(mean generated tokens $\\times$ measured decode cost, batch 8, one H100)"))
        ax.set_ylabel("competition suite, avg@8 at the 32K cap")
        ax.set_title("accuracy vs. efficiency", loc="left")
    ax.legend(handles=[plt.Line2D([], [], ls="", marker=MK[g], color=c, markeredgecolor="white", label=l,
                                  markersize=(9 if compact else 13) if g == "ours" else (5 if compact else 7))
                       for g, c, l in [("reference", GROUP["reference"], "full KV (vanilla)"),
                                       ("ours", GROUP["ours"], "\\textbf{AbridgeKV}" if False else "AbridgeKV (per-layer)"),
                                       ("ablation", GROUP["ablation"], "ablations"),
                                       ("baseline", RKV, "R-KV (published evictor)")][:3 if AXIS == "saturate" and not RKV_COMP else 4]],
              loc="lower right", bbox_to_anchor=(1.0, 0.30) if compact else None, handletextpad=0.4, borderpad=0.5 if not compact else 0.2, labelspacing=0.5 if not compact else 0.3, fontsize=5.8 if compact else 7)
    if compact:
        pass
    elif AXIS == "saturate" and RKV_COMP:
        if pend:
            ax.text(0.01, -0.175, "pending: " + ", ".join(pend), transform=ax.transAxes, fontsize=6, color=PAL["grey"])
    elif AXIS == "saturate":
        # R-KV has no valid saturation cell (scoring cap breached -> Full-KV fallback, or raised -> OOM); say so
        # rather than list it as pending. Its concurrency, which IS measurable, is in the appendix table.
        pend = [x for x in pend if not x.startswith("R-KV")]
        ax.text(0.01, -0.175, "R-KV omitted: no valid saturation cell (scoring falls back to full KV or runs out of memory)."
                + (" Pending: " + ", ".join(pend) if pend else ""), transform=ax.transAxes, fontsize=6, color=PAL["grey"])
    elif pend:
        ax.text(0.01, -0.175, "decode bench pending: " + ", ".join(pend[:4]) + (" +%d" % (len(pend) - 4) if len(pend) > 4 else ""),
                transform=ax.transAxes, fontsize=6, color=PAL["grey"])
for _axis in ("batch8", "saturate"):
    fig, ax = plt.subplots(figsize=(6.6, 4.1)); draw_pareto(ax, _axis)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "pareto.pdf" if _axis == "batch8" else "pareto_conc.pdf")); plt.close(fig)

# ---------------- Figure: vLLM serving gains ----------------
arms_b = [("vanilla_graphs", "vanilla", PAL["grey"]), ("uni_plain", "uniform 128", PAL["slate"]), ("tpl_physical", "per-layer", PAL["sage"])]
batches = [1, 8, 32]
cell = {(c["tag"], c["mode"], c["batch"]): c for c in B["cells"] if c["len"] == 16384}
# Panel (b) reads the async re-measure (VLLM_ASYNC_SCHED=1, every arm but R-KV served async; 2026-09-25), median of 3.
_BAF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "bench_decode_async.json")
cellA = {(c["tag"], c["mode"], c["batch"]): c for c in json.load(open(_BAF))["cells"]} if os.path.exists(_BAF) else cell
fig, axes = plt.subplots(1, 3, figsize=(9.6, 2.9), gridspec_kw={"width_ratios": [1.4, 1.4, 1]})
w = 0.19
for ax in axes[:2]: ax.set_ylim(0, 4)
axes[2].set_ylim(0, 190)
for i, (tag, lab, col_) in enumerate(arms_b):
    base = [cell[("vanilla_graphs", "batch", bt)] for bt in batches]
    tp = [cell[(tag, "batch", bt)]["steady_tok_per_s"] / bb["steady_tok_per_s"] for bt, bb in zip(batches, base)]
    # BYTES, not blocks. With N KV cache groups vLLM pages the pool as ceil(36/N) tensors, so one block holds
    # layers_per_group layers of KV rather than 36: the per-layer physical path runs 6 groups, and its 14,065
    # blocks at batch 8 hold the same 5.15 GiB as uniform's 2,345. Dividing raw block counts scored per-layer
    # at 0.59x -- worse than vanilla -- for a path that measures 3.54x in bytes, which is what tab:bench has
    # always reported (user spotted it 2026-09-23).
    kv = [bb["kv_gib_peak"] / cell[(tag, "batch", bt)]["kv_gib_peak"] for bt, bb in zip(batches, base)]
    for j in range(len(batches)):
        rbar(axes[0], j + (i - 1.5) * w, tp[j], w * 0.92, col_); rbar(axes[1], j + (i - 1.5) * w, kv[j], w * 0.92, col_)
        axes[0].text(j + (i - 1.5) * w, tp[j] + 0.05, "%.2f" % tp[j], ha="center", fontsize=4.8, color=PAL["ink"]); axes[1].text(j + (i - 1.5) * w, kv[j] + 0.05, "%.2f" % kv[j], ha="center", fontsize=4.8, color=PAL["ink"])
    s = sat.get(tag)   # controlled single-job cells (data/saturation.json), same source as pareto_conc.pdf
    if s and s.get("steady_running"): rbar(axes[2], i, s["steady_running"], 0.68, col_); axes[2].text(i, s["steady_running"] + 3, "%d seqs\n%.0f tok/s" % (s["steady_running"], s["tok_per_s"]), fontsize=5.5, ha="center", color=PAL["ink"])
for ax, yl, t in [(axes[0], "decode throughput / vanilla", "throughput vs. batch"), (axes[1], "vanilla resident KV / ours", "resident KV reduction vs. batch")]:
    ax.set_xlim(-0.6, len(batches) - 0.4); ax.set_xticks(range(len(batches))); ax.set_xticklabels(["batch %d" % b for b in batches]); ax.axhline(1, color=PAL["ink"], lw=0.6, ls=":"); ax.set_ylabel(yl); ax.set_title(t)
axes[0].legend(handles=[mpatches.Patch(color=c, label=l) for _, l, c in arms_b], loc="upper left")
axes[2].set_xlim(-0.6, len(arms_b) - 0.4); axes[2].set_xticks(range(len(arms_b))); axes[2].set_xticklabels([l for _, l, _ in arms_b], fontsize=6, rotation=20, ha="right"); axes[2].set_ylabel("concurrent 16K sequences (steady)"); axes[2].set_title("saturated GPU")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "serving_gains.pdf")); plt.close(fig)

# ---------------- Figure: main results, four panels (user 2026-09-24) ----------------
# (a) accuracy: suite avg@8 bars; (b) serving speed: decode throughput relative to full KV at batch 1/8/32 and saturated;
# (c)/(d) accuracy against time to finish, batch 8 and saturated. MATH-500 stays in Table~\ref{tab:main} only.
def draw_speed(ax):
    # No full-KV bars (user 2026-09-24): every value is a ratio to full KV, so full KV is the dotted line at 1.
    arms_s = [("tpl_physical", "AbridgeKV (per-layer)", GROUP["ours"]), ("uni_plain", "AbridgeKV, uniform 128", GROUP["ablation"])]
    groups = [("batch 1", lambda t: cellA.get((t, "batch", 1))), ("batch 8", lambda t: cellA.get((t, "batch", 8))),
              ("batch 32", lambda t: cellA.get((t, "batch", 32))),
              # saturated = each arm's run-to-completion replay at its best max_num_seqs cap (completion.json), falling
              # back to the fixed-step saturation cells only when no completion cell exists (user 2026-09-24)
              ("saturated\n(best cap)", lambda t: COMP.get(t) or sat.get(t))]
    def rate(c): return (c.get("steady_tok_per_s") or c.get("tok_per_s")) if c else None
    w = 0.34; top = 0
    for i, (tag, lab, col_) in enumerate(arms_s):
        for j, (gl, get) in enumerate(groups):
            v, b = rate(get(tag)), rate(get("vanilla_graphs"))
            if not (v and b): continue
            r = v / b; top = max(top, r); x = j + (i - 0.5) * w
            rbar(ax, x, r, w * 0.9, col_); ax.text(x, r + 0.04, "%.2f" % r, ha="center", fontsize=5.2, color=PAL["ink"])
    ax.set_ylim(0, top * 1.28); ax.set_xlim(-0.6, len(groups) - 0.4)
    ax.set_xticks(range(len(groups))); ax.set_xticklabels([g for g, _ in groups])
    ax.axhline(1, color=PAL["ink"], lw=0.9, ls=(0, (2, 2)), zorder=4); ax.set_ylabel("decode throughput / full KV")
    ax.legend(handles=[mpatches.Patch(color=c, label=l) for _, l, c in arms_s] + [plt.Line2D([], [], color=PAL["ink"], lw=0.9, ls=(0, (2, 2)), label="full KV = 1")],
              loc="upper left", fontsize=5.6, ncol=1, handlelength=1.4)

# ---------------- Panel (c): accuracy vs. KV held per running sequence (user 2026-09-25) ----------------
# Replaces the 8 GiB time-to-finish panel, which read the same as (d): under a cache limit the best cap tunes the batch,
# and a common cap (32) only measures vLLM's pre-emption/recompute. What sets concurrency under a cache limit is the KV a
# running sequence holds on average, so that is the x axis. Each arm replays its OWN generated lengths (the completion
# bench workload, 512 paired draws) through the residency model measured on the served per-layer model (completed blocks
# x quota + the raw open block + prompt, zero residual at 1K/4K/16K): resident(t) = P + (t // C) * K + t % C. Full KV
# holds P + t; R-KV holds min(P + t, budget + buffer/2) (buffer 64). The average is time-weighted: sum over sequences and
# decode steps of resident rows, over the total steps -- the KV one concurrency slot costs. 147,456 B per row (36 layers).
KV_ROW_B = 147456; KV_P = 200   # bench prompt length (scripts/bench_decode.py --prompt-len)
_WL = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "workload_complete_async.json")))["arms"]
def kv_per_seq(tag, C=None, K=None, budget=None, T=None, rows=None):
    Ls = _WL.get(tag, {}).get("lengths")
    if not Ls: return None
    tot = steps = 0
    for L in Ls:
        t = np.arange(1, L + 1)
        if budget: r = np.minimum(KV_P + t, budget + 32)
        elif T: r = KV_P + np.floor(t / T) * rows + np.mod(t, T)   # thought-level: `rows` kept per thought of mean length T (measured)
        elif C: r = KV_P + (t // C) * K + t % C
        else: r = KV_P + t
        tot += r.sum(); steps += L
    return tot / steps * KV_ROW_B / 2**20   # MiB per running sequence
KVX = {"vanilla": ("vanilla_graphs", {}), "vpl": ("tpl_physical", dict(C=512, K=128)), "uniform128": ("uni_plain", dict(C=512, K=128)),
       "gist": ("gist", dict(C=512, K=128)), "ck512k256": ("ck512k256", dict(C=512, K=256)), "ck512k64": ("ck512k64", dict(C=512, K=64)),
       "rkv_b2048": ("rkv_b2048", dict(budget=2048)), "rkv_b4096": ("rkv_b4096", dict(budget=4096))}
# LightThinker (user 2026-09-26: on the Pareto panels): 9 cache rows + the continue token stay per thought; the mean thought
# length is the one its generations actually produce, counted by the completion bench (results.json lt_thought.thought_tokens).
_LT = next((a for a in R["arms32k"] if a["key"] == "lt_thought"), {})
if _LT.get("thought_tokens"): KVX["lt_thought"] = ("lt_thought", dict(T=_LT["thought_tokens"], rows=_LT.get("rows_per_thought", 10)))
def draw_memory(ax):
    lab_of = {"vanilla": "full KV", "vpl": "AbridgeKV (per-layer)", "uniform128": "4:1", "ck512k256": "2:1", "ck512k64": "8:1",
              "gist": "appended positions", "rkv_b2048": "R-KV 2048", "rkv_b4096": "R-KV 4096", "lt_thought": "LightThinker"}
    grp = {"vanilla": "reference", "vpl": "ours", "gist": "ablation", "uniform128": "ablation", "ck512k256": "ablation", "ck512k64": "ablation",
           "lt_thought": "prior"}
    LB = {"vanilla": (0, 7, "center"), "vpl": (0, 10, "center"), "uniform128": (-8, -3, "right"), "ck512k256": (8, -3, "left"),
          "ck512k64": (-8, -3, "right"), "gist": (8, -3, "left"), "rkv_b2048": (0, -12, "center"), "rkv_b4096": (0, -12, "center"),
          "lt_thought": (8, -3, "left")}
    P = {}
    for k, (tag, kw) in KVX.items():
        pl = (X["arms"].get(k) or {}).get("pool"); x = kv_per_seq(tag, **kw)
        if pl and x: P[k] = (x, pl["avg8"])
    vx, vy = P["vanilla"]
    ch = [P[k] for k in ("ck512k256", "uniform128", "ck512k64") if k in P]
    ax.plot([a for a, _ in ch], [b for _, b in ch], color=GROUP["ablation"], lw=0.9, alpha=0.55, zorder=2)
    for k, (x, y) in P.items():
        g = grp.get(k, "baseline"); col = RKV if k.startswith("rkv_") else GROUP[g]
        ax.scatter([x], [y], marker=MK[g], color=col, s=240 if g == "ours" else 34, zorder=5,
                   edgecolors=PAL["rust"] if g == "ablation" else "white", linewidths=0.7 if g == "ablation" else 1.0)
        dx, dy, ha = LB[k]
        ax.annotate(lab_of[k], (x, y), xytext=(0.8 * dx, 0.8 * dy), textcoords="offset points", ha=ha, fontsize=6.2, zorder=6,
                    color=RKV if k.startswith("rkv_") else GROUP_TXT.get(g, col), fontweight="semibold" if g == "ours" else "normal")
    if "uniform128" in P and "vpl" in P:
        ax.annotate("", xy=P["vpl"], xytext=P["uniform128"], zorder=6, arrowprops=dict(arrowstyle="-|>", color=GROUP["ours"], lw=1.0,
                    shrinkA=4, shrinkB=9, mutation_scale=7))
    ys = [y for _, y in P.values()]; lo_y, hi_y = min(ys) - 2.4, max(ys) + 2.0; hi_x = vx * 1.12
    ax.axvline(vx, color=PAL["grey"], lw=0.8, ls=(0, (4, 3)), zorder=1); ax.axhline(vy, color=PAL["grey"], lw=0.8, ls=(0, (4, 3)), zorder=1)
    ax.axhspan(vy - VAN_SD, vy + VAN_SD, color=PAL["grey"], alpha=0.22, lw=0, zorder=0)
    ax.add_patch(plt.Rectangle((0, vy), vx, hi_y - vy, facecolor=PAL["sage"], alpha=0.07, zorder=0))
    ax.text(vx / 2, hi_y - 0.35, "less KV than full KV, no less accurate", fontsize=6.0, color=PAL["sage"], ha="center", va="top", style="italic")
    ax.set_xlim(0, hi_x); ax.set_ylim(lo_y, hi_y)
    ax.set_xlabel("KV held per running sequence (MiB, time-averaged)"); ax.set_ylabel("suite avg@8")
    # secondary scale: how many such sequences an 8 GiB cache holds
    sec = ax.secondary_xaxis("top", functions=(lambda m: 8192 / np.maximum(m, 1e-3), lambda n: 8192 / np.maximum(n, 1e-3)))
    sec.set_xticks([8, 16, 32, 64]); sec.tick_params(labelsize=5.8, pad=1); sec.set_xlabel("sequences per 8 GiB", fontsize=6.0, labelpad=2)
    return P

# ---------------- output-length panel (user 2026-09-25): share of samples still writing at each length ----------------
GLN = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "gen_lengths.json")))
def draw_lengths(ax, fs=6.2):
    CAPL = GLN["cap"]; xs_ = np.linspace(0, CAPL, 400)
    for k, lab, col, ls_ in (("vanilla", "full KV", GROUP["reference"], "-"), ("vpl", "AbridgeKV (per-layer)", GROUP["ours"], (0, (3, 1.5))),
                             ("rkv_b4096", "R-KV 4096", RKV, "-"), ("rkv_b2048", "R-KV 2048", RKV, (0, (1.5, 1.2))),
                             ("lt_thought", "LightThinker", GROUP["prior"], (0, (3, 1.5))),
                             ("evict_recency", "recency eviction$^*$", GROUP["baseline"], "-")):
        g = np.sort(np.array(GLN["arms"][k])); surv = 100 * (1 - np.searchsorted(g, xs_, side="left") / len(g))
        ax.plot(xs_ / 1024, surv, color=col, lw=1.1 if k != "vpl" else 1.4, ls=ls_, label=lab, zorder=3 if k != "vpl" else 4)
        ax.text(CAPL / 1024 + 0.4, surv[-1] + (3.2 if k == "lt_thought" else 0), "%.0f%%" % surv[-1], fontsize=fs - 0.6, color=col, va="center")
    ax.set_xlim(0, CAPL / 1024 + 3.6); ax.set_ylim(0, 102); ax.set_xticks([0, 8, 16, 24, 31])
    ax.set_xticklabels(["0", "8K", "16K", "24K", "31K"]); ax.tick_params(labelsize=fs - 0.4)
    ax.set_xlabel("generated tokens", fontsize=fs); ax.set_ylabel("samples still writing (%)", fontsize=fs)
    ax.legend(fontsize=fs - 0.8, loc="lower left", handlelength=1.8, borderaxespad=0.3, labelspacing=0.25)
# Drawn at 6.0 in and placed at the 5.5 in text width (user 2026-09-26: less whitespace, more legible text; it was drawn at
# 7.2 in and shrunk to 0.92 of the text width, so every label printed at 70% of its size).
fig = plt.figure(figsize=(6.0, 5.0))
GS = fig.add_gridspec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1.3], hspace=0.5, wspace=0.2)
a00, a01 = fig.add_subplot(GS[0, 0]), fig.add_subplot(GS[0, 1])
def broken(spec, _axis):
    sub = spec.subgridspec(1, 2, width_ratios=[3.2, 1], wspace=0.05)
    aL = fig.add_subplot(sub[0]); aR = fig.add_subplot(sub[1], sharey=aL)
    draw_pareto(aL, _axis, compact=True, side="left"); draw_pareto(aR, _axis, compact=True, side="right")
    aR.set_ylim(*aL.get_ylim())
    # diagonal break marks on the bottom and top spines
    kw = dict(marker=[(-1, -1.6), (1, 1.6)], markersize=6, linestyle="none", color=PAL["ink"], mec=PAL["ink"], mew=0.8, clip_on=False)
    aL.plot([1, 1], [0, 1], transform=aL.transAxes, **kw); aR.plot([0, 0], [0, 1], transform=aR.transAxes, **kw)
    return aL
# Panel (c): at 8 GiB full KV is memory-bound too, so every arm sits within ~1.2x of it and no break is needed.
a10 = fig.add_subplot(GS[1, 0]); KVP = draw_memory(a10)
print("[kv/seq MiB]", {k: (round(x, 1), round(8192 / x, 1)) for k, (x, _) in KVP.items()})
axes = [[a00, a01], [a10, broken(GS[1, 1], "saturate")]]
XTF, XTR = 5.8, 35
# (a) without grey bars (user 2026-09-24): full KV is the dotted line every bar is read against.
bars(axes[0][0], "pool", "suite avg@8 (%d problems)" % X["n_competition"], skip_ref=True)
axes[0][0].set_ylim(0, max(axes[0][0].get_ylim()[1], 44) * 1.30)
_van = (X["arms"].get("vanilla", {}).get("pool") or {}).get("avg8")
if _van:
    axes[0][0].axhline(_van, color=PAL["ink"], lw=0.9, ls=(0, (2, 2)), zorder=4)
    for k_, a_ in ((2, 0.14), (1, 0.24)): axes[0][0].axhspan(_van - k_ * VAN_SD, _van + k_ * VAN_SD, color=PAL["grey"], alpha=a_, lw=0, zorder=0.5)
    axes[0][0].text(axes[0][0].get_xlim()[1], _van + 0.6, "full KV %.1f" % _van, ha="right", va="bottom", fontsize=5.8, color=PAL["ink"])
axes[0][0].legend(handles=[mpatches.Patch(color=GROUP["ours"], label="AbridgeKV (per-layer)"), mpatches.Patch(color=GROUP["ablation"], label="AbridgeKV ablations"),
                           mpatches.Patch(color=GROUP["baseline"], label="codes-free eviction"),
                           plt.Line2D([], [], color=PAL["ink"], lw=0.9, ls=(0, (2, 2)), label="full KV (vanilla)"),
                           mpatches.Patch(color=PAL["grey"], alpha=0.30, label="full KV re-run, 1 / 2 s.d.")],
                  loc="upper center", ncol=2, fontsize=5.6, columnspacing=1.0, handlelength=1.4, handleheight=0.8, borderaxespad=0.1)
# (a) horizontal two-line tick labels instead of rotated ones: the rotation cost a quarter of the row's height
_TL = {"AbridgeKV (per-layer)": "AbridgeKV\n(per-layer)", "uniform 128": "uniform\n128", "appended positions": "appended\npositions"}
_tl = [t.get_text() for t in axes[0][0].get_xticklabels()]
axes[0][0].set_xticklabels([_TL.get(t, t.replace("\n(60/446)", "$^*$")) for t in _tl], rotation=0, ha="center", fontsize=5.4, linespacing=0.9)
draw_speed(axes[0][1])
for ax, t in zip([axes[0][0], axes[0][1], axes[1][0], axes[1][1]],
                 ["(a) accuracy: competition suite, 32K cap", "(b) serving speed, one H100", "(c) accuracy vs. KV memory per sequence",
                  "(d) accuracy vs. time to finish, saturated"]):
    ax.set_title(t, loc="left", fontsize=7.4, pad=17 if ax is axes[1][0] else 5)
fig.subplots_adjust(left=0.085, right=0.99, top=0.955, bottom=0.085); fig.savefig(os.path.join(FIG, "main_panels.pdf")); plt.close(fig)
# output length as its own figure in Section 4.2 (user 2026-09-25: main (b) is the throughput panel again)
TCB = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "train_cost_blocksize.json")))
def draw_blockcost(ax, fs=6.2, L=32768, P=200):
    # Cost of the block size at 4:1, relative to C=512: measured training step time and peak memory (one H100, 16K tokens,
    # forward+backward, bf16 autocast, checkpointing), and the peak resident KV of one sequence at the 32K cap from the
    # measured residency model resident(t) = P + (t // C) K + t % C.
    Cs = sorted(int(c) for c in TCB["cells"] if int(c) >= 512); cell = TCB["cells"]   # C=256 dropped from the plot (user 2026-09-26); its 1.9x is stated in the caption
    peak = {C: max(P + (t // C) * (C // 4) + t % C for t in range(L)) for C in Cs}
    rows = (("training step time", [cell[str(C)]["step_s"] / cell["512"]["step_s"] for C in Cs], PAL["rust"], "o"),
            ("training peak memory", [cell[str(C)]["peak_gib"] / cell["512"]["peak_gib"] for C in Cs], PAL["grey"], "s"))
    # (inference peak KV per sequence was dropped 2026-09-26: the worst-case memory of one sequence is not what sets
    #  concurrency; the time-averaged KV per running sequence is, and Figure 3c shows it)
    x = np.log2(Cs)
    for lab, ys, col, mk in rows:
        ax.plot(x, ys, color=col, lw=1.1, label=lab, zorder=3)
        for xi, C, y in zip(x, Cs, ys):   # filled = block sizes we trained
            ax.plot(xi, y, mk, ms=3.2, color=col, mfc=col if C in (512, 1024) else "white", mew=0.9, zorder=4)
    ax.axhline(1, color=PAL["ink"], lw=0.5, ls=(0, (2, 2)), zorder=1)
    ax.set_xticks(x); ax.set_xticklabels([str(C) for C in Cs]); ax.tick_params(labelsize=fs - 0.4)
    ax.set_xlim(x[0] - 0.3, x[-1] + 0.3); ax.set_ylim(0.8, 2.0)
    ax.set_xlabel("block size $C$ (4:1; filled = trained)", fontsize=fs); ax.set_ylabel("relative to $C{=}512$", fontsize=fs)
    ax.legend(fontsize=fs - 0.8, loc="upper right", handlelength=1.6, borderaxespad=0.3, labelspacing=0.25)
    return {C: (cell[str(C)]["step_s"], cell[str(C)]["peak_gib"], peak[C]) for C in Cs}
fig, ax = plt.subplots(figsize=(2.85, 1.8)); print("[blockcost]", draw_blockcost(ax, fs=6.0)); fig.tight_layout(pad=0.3)
fig.savefig(os.path.join(FIG, "blockcost.pdf")); plt.close(fig)
fig, ax = plt.subplots(figsize=(2.85, 1.8)); draw_lengths(ax, fs=6.0); fig.tight_layout(pad=0.3); fig.savefig(os.path.join(FIG, "outlen.pdf")); plt.close(fig)
# decode throughput vs full KV (async), moved from the main figure to the serving appendix (user 2026-09-25)
fig, ax = plt.subplots(figsize=(3.6, 2.2)); draw_speed(ax); fig.tight_layout(pad=0.3); fig.savefig(os.path.join(FIG, "speed.pdf")); plt.close(fig)

# ---------------- Figure: how the per-layer budget is estimated ----------------
def layer_cost(path):
    rows = [json.loads(l) for l in open(os.path.join(HERE, "data", path))]
    base = next(r["held_out_ce"] for r in rows if r["name"] == "trained")
    return [1000 * (next(r["held_out_ce"] for r in rows if r["kind"] == "layer" and r["name"] == "layer%02d->64" % l) - base) for l in range(36)]
cost_u = layer_cost("quota_ablation_c256uni_bitrev.jsonl"); cost_h = layer_cost("quota_ablation_c256het_bitrev.jsonl")
sched = [r[0] for r in json.load(open(os.path.join(ROOT, "results", "schedules", "sched_trunc_perlayer_mean128.json")))]
fig, axes = plt.subplots(1, 2, figsize=(8.8, 2.6))
ax = axes[0]; L = list(range(36)); top = max(cost_u + cost_h); ax.set_ylim(min(0, min(cost_u + cost_h)) - 0.2, top * 1.12)
for l in L: rbar(ax, l - 0.21, cost_u[l], 0.4, PAL["slate"]); rbar(ax, l + 0.21, cost_h[l], 0.4, PAL["sky"])
ax.set_xlim(-0.8, 35.8); ax.axhline(0, color=PAL["ink"], lw=0.5); ax.set_xlabel("layer"); ax.set_ylabel(r"$\Delta$ held-out CE ($\times 10^{-3}$)", fontsize=7)
ax.set_title("cost of halving one layer's codes (eval time, no retraining)", fontsize=8); ax.legend(handles=[mpatches.Patch(color=PAL["slate"], label="uniform-128 model"), mpatches.Patch(color=PAL["sky"], label="per-head model")], loc="upper right")
ax = axes[1]; ax.set_ylim(0, 290)
for l in L: rbar(ax, l, sched[l], 0.8, PAL["coral"] if sched[l] == 256 else (PAL["slate"] if sched[l] == 128 else "#D5D9E0"))
ax.set_xlim(-0.8, 35.8); ax.axhline(128, color=PAL["ink"], lw=0.6, ls="--"); ax.set_yticks([64, 128, 256]); ax.set_xlabel("layer"); ax.set_ylabel("code rows per block (mean 128)")
ax.set_title("resulting per-layer budget (6 raised to 256, 12 cut to 64)", fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "budget_estimation.pdf")); plt.close(fig)
_FSm, _TSm = 6.4, 7.2   # the figure-wide FS/TS are defined further down
# Main-text version (Section 5.1, user 2026-09-26: "the measurement -> per-layer quota pair of plots"), drawn at the 5.5 in
# text width so _FSm/_TSm print at nominal size; bars of the per-layer schedule coloured by quota level.
fig, axes = plt.subplots(1, 2, figsize=(5.5, 1.55), gridspec_kw={"width_ratios": [1.0, 1.0], "wspace": 0.28})
fig.subplots_adjust(left=0.085, right=0.995, bottom=0.2, top=0.88)
ax = axes[0]; ax.set_ylim(min(0, min(cost_u + cost_h)) - 0.15, top * 1.12)
for l in L: rbar(ax, l - 0.21, cost_u[l], 0.4, PAL["slate"]); rbar(ax, l + 0.21, cost_h[l], 0.4, PAL["sky"])
ax.set_xlim(-0.8, 35.8); ax.axhline(0, color=PAL["ink"], lw=0.5); ax.set_xlabel("layer", fontsize=_FSm, labelpad=1)
ax.set_ylabel(r"$\Delta$ held-out CE ($\times 10^{-3}$)", fontsize=_FSm); ax.tick_params(labelsize=_FSm - 0.8)
ax.set_title("(a) measured cost of halving one layer", loc="left", fontsize=_TSm)
ax.legend(handles=[mpatches.Patch(color=PAL["slate"], label="uniform-128 model"), mpatches.Patch(color=PAL["sky"], label="per-head model")],
          loc="upper right", fontsize=_FSm - 0.8, handlelength=1.2, borderaxespad=0.2)
ax = axes[1]; ax.set_ylim(0, 290)
for l in L: rbar(ax, l, sched[l], 0.8, PAL["coral"] if sched[l] == 256 else (PAL["slate"] if sched[l] == 128 else "#D5D9E0"))
ax.set_xlim(-0.8, 35.8); ax.axhline(128, color=PAL["ink"], lw=0.6, ls="--"); ax.set_yticks([64, 128, 256])
ax.set_xlabel("layer", fontsize=_FSm, labelpad=1); ax.set_ylabel("code rows per block", fontsize=_FSm); ax.tick_params(labelsize=_FSm - 0.8)
ax.set_title("(b) resulting quotas (mean 128)", loc="left", fontsize=_TSm)
fig.savefig(os.path.join(FIG, "budget_main.pdf"), bbox_inches="tight", pad_inches=0.02); plt.close(fig)

# ---------------- Figure: online compressibility per layer (vanilla model) ----------------
S = json.load(open(os.path.join(HERE, "data", "oc_vanilla_completions_summary.json"))); P = S["per_layer_head"]; NL = len(P["err0"])
lm = lambda f: [sum(P[f][l]) / len(P[f][l]) for l in range(NL)]
fig, ax = plt.subplots(figsize=(5.4, 2.6))
for f, lab, c in [("err0", "drop the whole block", PAL["ink"]), ("ev_recency@64", "keep the 64 most recent", PAL["sand"]), ("ev_inblock@64", "keep 64, chosen causally", PAL["sage"]), ("ev_oracle_mass@64", "keep 64, chosen with hindsight", PAL["slate"]), ("lr@32", "rank-32 mixing bound", PAL["coral"])]:
    ax.plot(range(NL), lm(f), lw=1.4, label=lab, color=c)
ax.set_xlabel("layer"); ax.set_ylabel("RMS rel. attention-output error"); ax.set_ylim(0, 0.135); ax.legend(loc="upper center", ncol=3, fontsize=5.8)
ax.set_title("how compressible each layer's KV stream is (vanilla model, own AIME completions)", fontsize=7.5)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "compressibility_layers.pdf")); plt.close(fig)
print("figures written to", FIG)

# ---------------- figure 1: method layout ----------------
def _rbox(ax, x, y, w, h, fc, ec="none", lw=0, ls="-", r=0.18, z=3, alpha=1.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=%g" % r, fc=fc, ec=ec, lw=lw, ls=ls, zorder=z, alpha=alpha, clip_on=False))
def _lighten(hexc, f=0.55):
    import matplotlib.colors as mc
    r, g, b = mc.to_rgb(hexc); return (r + (1 - r) * f, g + (1 - g) * f, b + (1 - b) * f)
CONTENT, CODE, PROMPT = _lighten(PAL["slate"], 0.45), PAL["coral"], _lighten(PAL["grey"], 0.35)
CODE_L, SAGE, INK, GREY = _lighten(PAL["coral"], 0.5), PAL["sage"], PAL["ink"], PAL["grey"]
FS = 6.4; TS = 7.2   # one text size and one title size for every illustration, drawn at the 5.5 in text width (user 2026-09-25)

def draw_layout_a(ax, title="(a) compress each block as it completes"):
    # (a) four chronological rows. Each row is the ACTUAL resident KV at that moment, drawn left to right with no holes, so
    # the shrinking cache is visible and there is no unexplained gap where a dropped block used to sit.
    ax.set_axis_off(); ax.grid(False)
    PW, BW, CW, G, H = 1.35, 2.3, 1.15, 0.24, 0.95
    DY = 4.3                    # row pitch. The explainer sits 0.25 BELOW its own row: the arrows own the space above a
                                 # row, so a caption under its row ends up unambiguously nearer the row it describes
                                 # (0.25) than the next one (0.85), which no tuning achieved while it sat above.
    rows_y = [3 * DY, 2 * DY, DY, 0.0]
    def arc(x1, y1, x2, y2, color, rad, lw=0.9):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, shrinkA=0, shrinkB=0,
                    mutation_scale=6.5, connectionstyle="arc3,rad=%g" % rad), zorder=6)
    # Every row's arcs peak at the same height, and inside a row one radius serves every arrow. Constant radius makes the
    # sagitta grow with the span, so arcs leaving a common point nest; equal peaks across rows keep all four rows tidy and
    # guarantee no arc reaches the explainer sitting 0.62 above the boxes. Leftward arcs bulge up at +rad, rightward at -rad.
    PEAK = 0.45
    # Limits first, because arc3's rad bulges the curve in DISPLAY space: a rad derived from x-units alone peaks 2.4x too
    # high on a panel this wide, which is what drove the row-3 arcs through the explainer above them. Converting through the
    # axes' own pixel scales makes PEAK mean 0.45 data-units of height on every row, whatever the panel's aspect becomes.
    X0, XC1 = PW + G, PW + G + CW + G + BW + G
    # Right column (user 2026-09-24): the resident KV at each step, one bar per row, so the saw-tooth of the cache -- and where
    # its peak falls (the moment a block completes, before its raw rows are dropped) -- reads off next to the step it belongs to.
    MX0 = XC1 + CW + 0.55; MU = dict(P=0.42, C=1.12, K=0.28)      # memory units: C/K = 4, as at 4:1
    ax.set_xlim(-0.12, XC1 + CW + 0.25); ax.set_ylim(-3.1, rows_y[0] + H + 0.75)
    _t = ax.transData.transform
    KX = _t((1, 0))[0] - _t((0, 0))[0]; KY = _t((0, 1))[1] - _t((0, 0))[1]
    def rad_for(spans, leftward):
        r = 2 * PEAK * KY / (KX * max(abs(x) for x in spans))
        return r if leftward else -r
    def cursor(x, y): ax.plot([x], [y + H + 0.16], marker="v", ms=3.8, color=INK, zorder=7, clip_on=False)
    def growing(x0, frac, y):
        _rbox(ax, x0, y, BW, H, "white", ec=PAL["slate"], lw=0.8, ls=(0, (2.5, 2)), r=0.12)
        _rbox(ax, x0, y, BW * frac, H, CONTENT, r=0.12)
        return x0 + BW * frac
    # The step number lives at the head of the explainer instead of in a label column: the column cost a fifth of the panel
    # width and the rows are what should be wide.
    def caption(y, txt, color=INK): ax.text(0.0, y - 0.25, txt, ha="left", va="top", fontsize=FS, color=color, linespacing=1.2)
    def prompt_box(y):
        _rbox(ax, 0.0, y, PW, H, PROMPT, r=0.12)
        ax.text(PW / 2, y + H / 2, "prompt", ha="center", va="center", fontsize=FS, color=INK, zorder=5)
    def codes_box(x, y, txt):
        _rbox(ax, x, y, CW, H, CODE, r=0.12, z=5)
        ax.text(x + CW / 2, y + H / 2, txt, ha="center", va="center", fontsize=FS, color="white", zorder=6)
    def faded_block(x, y, txt):
        _rbox(ax, x, y, BW, H, CONTENT, r=0.12, alpha=0.30, ec=GREY, lw=0.7, ls=(0, (2, 1.5)))
        ax.text(x + BW / 2, y + H / 2, txt, ha="center", va="center", fontsize=FS, color=GREY, zorder=5)

    # row 1: generating block 0 against the prompt and its own prefix
    y = rows_y[0]
    prompt_box(y)
    x0 = PW + G
    qx = growing(x0, 0.62, y); ax.text(x0 + BW * 0.31, y + H / 2, "block 0", ha="center", va="center", fontsize=FS, color=INK, zorder=5)
    cursor(qx, y); yt = y + H + 0.10
    t1 = [PW / 2, (x0 + qx) / 2]; r1 = rad_for([qx - t for t in t1], True)
    for t in t1: arc(qx, yt, t, yt, INK, r1)
    caption(y, "1. generate block 0: each token attends to\n     the prompt and to its own block, causally")

    # row 2: the block is finished; K pause tokens read it and write the code rows, then its raw KV goes
    y = rows_y[1]
    prompt_box(y)
    faded_block(x0, y, "block 0")
    xc0 = x0 + BW + G; codes_box(xc0, y, "codes$_0$")
    yt = y + H + 0.10; dst = xc0 + CW / 2; src2 = [PW / 2, x0 + BW / 2]
    r2 = rad_for([dst - t for t in src2], False)
    for t in src2: arc(t, yt, dst, yt, CODE, r2)
    caption(y, "2. block 0 completes: $K$ pause tokens read the\n     prompt and the block and write $K$ code rows;\n     its $C$ raw rows are dropped", CODE)

    # row 3: the cache has shrunk. codes_0 sits where block 0 was and block 1 starts right after it, no hole.
    y = rows_y[2]
    prompt_box(y)
    codes_box(x0, y, "codes$_0$")
    x1 = x0 + CW + G
    qx = growing(x1, 0.45, y); ax.text(x1 + BW * 0.225, y + H / 2, "block 1", ha="center", va="center", fontsize=FS, color=INK, zorder=5)
    cursor(qx, y); yt = y + H + 0.10
    t3 = [PW / 2, x0 + CW / 2, (x1 + qx) / 2]; r3 = rad_for([qx - t for t in t3], True)
    for t in t3: arc(qx, yt, t, yt, INK, r3)
    caption(y, "3. generate block 1: attends to the prompt,\n     codes$_0$ and its own block")

    # row 4: the next summary reads every earlier summary, which is what makes the scheme compose
    y = rows_y[3]
    prompt_box(y)
    codes_box(x0, y, "codes$_0$")
    faded_block(x1, y, "block 1")
    xc1 = x1 + BW + G; codes_box(xc1, y, "codes$_1$")
    yt = y + H + 0.10; dst = xc1 + CW / 2; src4 = [PW / 2, x0 + CW / 2, x1 + BW / 2]
    r4 = rad_for([dst - t for t in src4], False)
    for t in src4: arc(t, yt, dst, yt, CODE, r4)
    caption(y, "4. block 1 completes: its codes read the prompt\n     and all earlier codes, so each summary is\n     written knowing every summary before it", CODE)
    ax.text(0.0, rows_y[3] - 2.35, "… and so on: the resident cache is the prompt,\n     $K$ rows per finished block, and the open block", ha="left", va="top", fontsize=FS, color=GREY, linespacing=1.2)
    ax.set_title(title, loc="left", pad=4, fontsize=TS)
def draw_mask(ax, title="(b) attention mask"):
    # (b) attention mask, toy sizes: prompt 2, C=4, K=2, two blocks
    ax.grid(False); ax.set_anchor("NW")
    P, Ct, Kt, Nb = 2, 4, 2, 2
    segs_b = [("prompt", P, "p", -1)] + sum([[("block %d" % b, Ct, "c", b), ("codes %d" % b, Kt, "k", b)] for b in range(Nb)], [])
    tok = []
    for lab, n, role, b in segs_b: tok += [(role, b, j) for j in range(n)]
    S = len(tok); M = [[None] * S for _ in range(S)]
    for qi, (qr, qb, qj) in enumerate(tok):
        for ki, (kr, kb, kj) in enumerate(tok):
            if ki > qi and not (qr == "k" and kr == "k" and kb == qb): continue
            if kr == "p" or qr == "p": M[qi][ki] = PROMPT
            elif kr == "c" and kb == qb and qr == "c": M[qi][ki] = CONTENT
            elif kr == "c" and kb == qb and qr == "k": M[qi][ki] = SAGE
            elif kr == "k" and kb < qb: M[qi][ki] = CODE
            elif kr == "k" and kb == qb and qr == "k": M[qi][ki] = CODE_L
    for qi in range(S):
        for ki in range(S):
            if M[qi][ki] is not None: _rbox(ax, ki + 0.07, S - 1 - qi + 0.07, 0.86, 0.86, M[qi][ki], r=0.16)
    bnd = [0]; [bnd.append(bnd[-1] + n) for _, n, _, _ in segs_b]
    for bb in bnd: ax.plot([bb, bb], [0, S], color="#B9BEC6", lw=0.5, zorder=4); ax.plot([0, S], [S - bb, S - bb], color="#B9BEC6", lw=0.5, zorder=4)
    for (lab, n, _, _), lo in zip(segs_b, bnd[:-1]):
        ax.text(-0.2, S - lo - n / 2, lab, ha="right", va="center", fontsize=FS)
    ax.set_xlim(0, S); ax.set_ylim(0, S); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for s_ in ax.spines.values(): s_.set_visible(False)
    ax.set_title(title, loc="left", pad=4, fontsize=TS)
    hs = [mpatches.Patch(color=CONTENT, label="own block, causal"), mpatches.Patch(color=CODE, label="earlier codes"),
          mpatches.Patch(color=SAGE, label="codes read the block"), mpatches.Patch(color=CODE_L, label="codes of one block"), mpatches.Patch(color=PROMPT, label="raw prompt")]
    # The mask is lower-triangular, so (b) owns a large empty upper-right corner. Parking the legend fully OUTSIDE the axes
    # made the inter-panel gap carry it and it collided with (c)'s row labels; cramming it fully INSIDE crowds the triangle.
    # It straddles the edge instead (user 2026-09-23): the swatches sit in the empty corner, the labels overhang to the
    # right. The overhang is free space -- (b) is anchored NW with an equal aspect, so the square matrix leaves its own
    # gridspec cell unused on the right, and the legend reaches into that, not into (c). Larger and further right (user
    # 2026-09-24): FS, anchor x 1.46.
    ax.legend(handles=hs, loc="upper right", bbox_to_anchor=(1.62, 1.02), ncol=1, fontsize=FS, handlelength=0.85,
              handleheight=0.8, borderaxespad=0, labelspacing=0.24, handletextpad=0.4, frameon=False)
def draw_positions(ax, title="(c) in-block position ids"):
    # (c) a worked example with real position ids: C=8, K_max=4, block b=0. Cells, not dots, so the ids are readable.
    ax.grid(False); ax.set_anchor("N")
    C_ex, K_ex = 8, 4
    def bitrev(k, bits): return int(format(k, "0%db" % bits)[::-1], 2)
    POS = [bitrev(k, 2) * (C_ex // K_ex) for k in range(K_ex)]          # code row k -> 0, 4, 2, 6
    HC = 0.56
    def cell(x, y, txt, fc, tc="white", alpha=1.0, ec="none", lw=0):
        _rbox(ax, x + 0.06, y - HC / 2, 0.88, HC, fc, r=0.1, alpha=alpha, ec=ec, lw=lw)
        if txt is not None: ax.text(x + 0.5, y, txt, ha="center", va="center", fontsize=FS, color=tc, zorder=6)
    def rl(y, t): ax.text(-0.45, y, t, ha="right", va="center", fontsize=FS, color=INK, linespacing=1.1)
    # The "appended" row runs four cells past the block, so the other three rows left that strip blank. It now carries the
    # reading of each row, which is also what the cryptic bitrev formula under the title was failing to convey.
    def note(y, t, c): ax.text(8.35, y, t, ha="left", va="center", fontsize=FS, color=c, linespacing=1.15)
    ys = [4.55, 3.20, 1.85, 0.50]          # 1.50 apart against a 0.56 cell: the gap between rows is wider than a row
    for j in range(C_ex): cell(j, ys[0], "%d" % j, CONTENT, INK)
    rl(ys[0], "the block:\n8 tokens"); note(ys[0], "the block's\nown ids", GREY)
    for j in range(C_ex): cell(j, ys[1], None, CONTENT, alpha=0.22)
    for p in [8, 9, 10, 11]: cell(p, ys[1], "%d" % p, CODE)
    rl(ys[1], "appended\n(prior work)")
    for j in range(C_ex): cell(j, ys[2], None, CONTENT, alpha=0.22)
    for k, p in enumerate(POS):
        cell(p, ys[2], "%d" % p, CODE)
    rl(ys[2], "in-block:\nall 4 pause\ntokens"); note(ys[2], "no new ids", CODE)
    for j in range(C_ex): cell(j, ys[3], None, CONTENT, alpha=0.22)
    for k, p in enumerate(POS):
        if k < 2: cell(p, ys[3], "%d" % p, CODE)
        else: cell(p, ys[3], None, "white", ec=GREY, lw=0.8)
    rl(ys[3], "in-block:\nfirst 2 pause\ntokens"); note(ys[3], "a prefix stays\nspread evenly", CODE)
    ax.set_xlim(-3.5, 12.2); ax.set_ylim(-0.30, 5.35); ax.set_xticks([]); ax.set_yticks([])
    ax.set_aspect(0.88 / HC, adjustable="box")   # cell_w / cell_h -> the cells render square, not portrait
    for s_ in ax.spines.values(): s_.set_visible(False)
    ax.set_title(title, loc="left", pad=4, fontsize=TS)

# ---------------- Figure: block scale at a fixed 4:1 ratio (user 2026-09-25) ----------------
# The trade-off thought-level compression never explores: a larger block keeps a larger exact window (more peak KV) and
# puts fewer lossy compression steps between a late query and the start of the response. Same trainer, data, order and
# 4:1 ratio; only C changes. y is the PAIRED gap to full KV on the 446-problem suite, with its 95% bootstrap interval.
# Peak KV per sequence at the 32K cap, per KV head: compressed history (L - C)/4 plus the open raw block C = L/4 + 3C/4.
CAP = 31744
def _peak(C, ratio=4): return (CAP - C) / ratio + C
bs = [(256, "ck256k64"), (512, "uniform128"), (1024, "ck1024k256")]
bs = [(C, k, X["arms"][k]["diff_vs_vanilla"]) for C, k in bs if X["arms"].get(k, {}).get("diff_vs_vanilla")]
fig, (ax, axR) = plt.subplots(1, 2, figsize=(6.6, 2.15))
ax.axhline(0, color=PAL["ink"], lw=0.9, ls=(0, (2, 2)), zorder=1)
ax.axhspan(-VAN_SD, VAN_SD, color=PAL["grey"], alpha=0.22, lw=0, zorder=0)
ax.text(min(bs)[0] / 1.3, 0.25, "full KV (re-run $\\pm$1 s.d.)", fontsize=6, color=PAL["ink"], ha="left", va="bottom")
xs_ = [C for C, _, _ in bs]; ys_ = [d[0] for _, _, d in bs]
ax.plot(xs_, ys_, color=GROUP["ablation"], lw=1.0, zorder=2)
for C, k, d in bs:
    ours = C == 512
    ax.errorbar([C], [d[0]], yerr=[[d[0] - d[1]], [d[2] - d[0]]], fmt="none", ecolor=GROUP_TXT.get("ablation", GROUP["ablation"]),
                elinewidth=1.0, capsize=2.2, zorder=3)
    ax.scatter([C], [d[0]], s=34, color=GROUP["ablation"], edgecolors=PAL["rust"], linewidths=0.7, zorder=4)
    ax.annotate("%+.1f" % d[0], (C, d[0]), xytext=(6, -2), textcoords="offset points", fontsize=6, color=PAL["ink"], ha="left", va="center")
lt = None   # the thought-level arm is compared on the Pareto plot (Figure 3d) and in Table 1, not here (user 2026-09-26)
ax.set_xscale("log", base=2)
ticks = xs_
# second tick row: time-averaged KV per running sequence (MiB) over each arm's own generations, the Figure 3c metric
# (peak KV at the 32K cap was dropped 2026-09-26: a worst case, not what sets how many sequences fit)
_kvs = {256: kv_per_seq("ck256k64", C=256, K=64), 512: kv_per_seq("uni_plain", C=512, K=128), 1024: kv_per_seq("ck1024k256", C=1024, K=256)}
print("[kv/seq by C]", {C: round(v, 1) for C, v in _kvs.items() if v})
ax.set_xticks(ticks); ax.set_xticklabels(["%d\n%.0f MiB" % (C, _kvs[C]) if C in xs_ and _kvs.get(C) else "%d" % C for C in ticks], fontsize=6.2)
ax.minorticks_off()
ax.set_xlim(min(ticks) / 1.35, 1024 * 1.35)
ax.set_xlabel("block size $C$ (exact window), all at 4:1\nKV per running sequence", fontsize=6.4)
ax.set_ylabel("avg@8 minus full KV", fontsize=6.6)
ax.tick_params(labelsize=6.2)
lo = min([d[1] for _, _, d in bs] + ([lt[1]] if lt else [])); ax.set_ylim(lo - 0.8, 1.6)
ax.set_title("(a) same 4:1 ratio, larger blocks", loc="left", fontsize=6.8)
# time to finish at saturation (fraction of full KV), from the run-to-completion replay: flat across block sizes
_vt = (COMP.get("vanilla_graphs") or {}).get("s_per_problem")
for C, k, d in bs:
    t = (COMP.get(BTAG.get(k, k)) or {}).get("s_per_problem")
    if t and _vt: ax.annotate("%.2f$\\times$ time" % (t / _vt), (C, d[0]), xytext=(6, -9), textcoords="offset points", fontsize=5.4, color=PAL["grey"], ha="left")
# (b) same block size C=512, different ratio: accuracy against time to finish
rs = [("2:1", "ck512k256"), ("4:1", "uniform128"), ("8:1", "ck512k64")]
axR.axhline(0, color=PAL["ink"], lw=0.9, ls=(0, (2, 2)), zorder=1); axR.axhspan(-VAN_SD, VAN_SD, color=PAL["grey"], alpha=0.22, lw=0, zorder=0)
px, py = [], []
for lab_, k in rs:
    d = X["arms"].get(k, {}).get("diff_vs_vanilla"); t = (COMP.get(BTAG.get(k, k)) or {}).get("s_per_problem")
    if not (d and t and _vt): continue
    x = t / _vt; px.append(x); py.append(d[0])
    axR.errorbar([x], [d[0]], yerr=[[d[0] - d[1]], [d[2] - d[0]]], fmt="none", ecolor=GROUP_TXT.get("ablation"), elinewidth=1.0, capsize=2.2, zorder=3)
    axR.scatter([x], [d[0]], s=34, color=GROUP["ablation"], edgecolors=PAL["rust"], linewidths=0.7, zorder=4)
    axR.annotate("%s  (%+.1f)" % (lab_, d[0]), (x, d[0]), xytext=(6, 2), textcoords="offset points", fontsize=6, color=PAL["ink"])
axR.plot(px, py, color=GROUP["ablation"], lw=1.0, zorder=2)
_lo = min(ax.get_ylim()[0], -6.8); ax.set_ylim(_lo, 1.6); axR.set_xlim(0.1, 0.7); axR.set_ylim(-6.8, 1.6)   # (b) keeps its own range: the thought-level point is (a) only; axR.tick_params(labelsize=6.2)
axR.set_xlabel("time to finish a problem\n(fraction of full KV, saturated)", fontsize=6.4)
axR.set_title("(b) same block size $C{=}512$, different ratio", loc="left", fontsize=6.8)
fig.tight_layout(pad=0.3); fig.savefig(os.path.join(FIG, "ck_sweep.pdf"), bbox_inches="tight", pad_inches=0.02); plt.close(fig)

# ---------------- Figure: generation length under compression (appendix, user 2026-09-25) ----------------
# Eviction changes what the model writes: mean generated tokens and the fraction of samples cut at the 32K cap, pooled suite.
GL = [("vanilla", "full KV", GROUP["reference"]), ("vpl", "AbridgeKV\n(per-layer)", GROUP["ours"]),
      ("uniform128", "AbridgeKV\nuniform 128", GROUP["ablation"]), ("ck512k64", "AbridgeKV\n8:1", GROUP["ablation"]),
      ("rkv_b4096", "R-KV\n4096", RKV), ("rkv_b2048", "R-KV\n2048", RKV),
      ("evict_recency", "recency", GROUP["baseline"]), ("evict_random", "random", GROUP["baseline"]), ("evict_score", "score", GROUP["baseline"])]
GL = [(k, l, c, X["arms"][k]["pool"]) for k, l, c in GL if (X["arms"].get(k, {}).get("pool") or {}).get("gen")]
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.3))
for ax, fld, yl, sc in ((axs[0], "gen", "mean generated tokens (K)", 1e-3), (axs[1], "trunc", "samples at the 32K cap (%)", 1)):
    for i, (k, l, c, pl) in enumerate(GL):
        v = pl[fld] * sc; rbar(ax, i, v, 0.7, c)
        ax.text(i, v + (1.96 * pl["gen_se"] * sc + 0.3 if fld == "gen" else 1.0), ("%.1f" if fld == "gen" else "%.0f") % v, ha="center", fontsize=5.6, color=PAL["ink"])
    if fld == "gen":
        e = [1.96 * pl["gen_se"] * sc for _, _, _, pl in GL]
        ax.errorbar(range(len(GL)), [pl[fld] * sc for _, _, _, pl in GL], yerr=e, fmt="none", ecolor=PAL["ink"], elinewidth=0.7, capsize=1.5, zorder=5)
    ax.axhline(GL[0][3][fld] * sc, color=PAL["ink"], lw=0.8, ls=(0, (2, 2)), zorder=4)
    ax.set_xticks(range(len(GL))); ax.set_xticklabels([l for _, l, _, _ in GL], fontsize=5.4, rotation=35, ha="right")
    ax.set_xlim(-0.6, len(GL) - 0.4); ax.set_ylabel(yl, fontsize=6.6); ax.tick_params(axis="y", labelsize=6)
    ax.set_ylim(0, max(pl[fld] * sc for _, _, _, pl in GL) * 1.15)
fig.tight_layout(pad=0.3); fig.savefig(os.path.join(FIG, "genlen.pdf")); plt.close(fig)

# ---------------- Figure: the per-layer design (user 2026-09-25) ----------------
# (a) the schedule against uniform at the same mean; (b) resident KV of one sequence as it generates, full KV vs uniform vs
# per-layer, from the paged cache's block occupancy (async decode bench, batch 1, 16K); (c) a zoom showing the sawtooth.
# Per-layer is served as six cache groups of six layers (quotas 256, 64, 64, 128, 128, 128); a group block holds 16 tokens
# of 6 layers, a single-group block 16 tokens of all 36, so bytes = blocks x 16 x (layers per block / 36) x 147456.
PL_HI, PL_LO = [0, 15, 18, 20, 21, 24], [1, 2, 3, 4, 5, 6, 8, 10, 30, 31, 32, 35]
quota = [256 if l in PL_HI else 64 if l in PL_LO else 128 for l in range(36)]
assert quota == list(sched), "per-layer schedule drifted from the trained one"
assert sum(quota) == 36 * 128
_BM = os.path.join(os.environ.get("OUTPUT_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")), "bench_mixed_async")
_KVF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "kv_trace_b1.json")
if os.path.isdir(_BM):
    tr = {}
    for t, lay in (("vanilla_graphs", 36), ("uni_plain", 36), ("tpl_physical", 6)):
        r = json.load(open(os.path.join(_BM, "%s_batch_B1_L16384_r1.json" % t)))
        n = len(r["kv_blocks_trace"]); every = r["decode_tokens"] / (n - 1)
        tr[t] = {"tokens": [round(i * every) for i in range(n)], "mib": [b * 16 * lay / 36 * 147456 / 2**20 for b in r["kv_blocks_trace"]]}
    json.dump(tr, open(_KVF, "w"))
KT = json.load(open(_KVF)) if os.path.exists(_KVF) else {}
def draw_perlayer_illus(ax, title):
    # (a) is a schematic, not the trained schedule (that is in the budget figure): six layers, one completed block each.
    # Uniform keeps the same number of code rows in every layer; per-layer moves rows to the layers that need them, same total,
    # and each quota level is one KV cache group.
    UQ, PQ = [4] * 6, [8, 2, 4, 2, 4, 4]
    ax.set_xlim(0, 10.4); ax.set_ylim(-0.1, 6.5); ax.axis("off")
    def row(x0, y, n, col):
        for j in range(n): ax.add_patch(plt.Rectangle((x0 + j * 0.46, y), 0.40, 0.62, facecolor=col, edgecolor="none"))
    for li in range(6):
        y = 5.0 - li * 0.95
        ax.text(0.0, y + 0.31, "layer %d" % (li + 1), fontsize=FS, va="center", color=PAL["ink"])
        row(2.3, y, UQ[li], GROUP["ablation"])
        c = GROUP["ours"] if PQ[li] > 4 else PAL["slate"] if PQ[li] < 4 else GROUP["ablation"]
        row(4.9, y, PQ[li], c)
    ax.text(2.3, 5.95, "uniform", fontsize=FS, fontweight="semibold", color=PAL["ink"])
    ax.text(4.9, 5.95, "per-layer (same total)", fontsize=FS, fontweight="semibold", color=GROUP["ours"])
    ax.set_title(title, loc="left", fontsize=TS)
def draw_perlayer_panels(axs, letters="abc", arms=("vanilla_graphs", "uni_plain", "tpl_physical")):
    if axs[0] is not None: draw_perlayer_illus(axs[0], "(%s) per-layer budget (illustration)" % letters[0])
    LAB = {"vanilla_graphs": ("full KV", GROUP["reference"], "-"), "uni_plain": ("uniform", GROUP["ablation"], "-"), "tpl_physical": ("per-layer", GROUP["ours"], (0, (3, 2)))}
    for i, (ax, lo, hi) in enumerate(((axs[1], 0, 16384), (axs[2], 3072, 5120))):
        for t in arms:
            if t not in KT or (i == 1 and t == "vanilla_graphs"): continue
            x, y = KT[t]["tokens"], KT[t]["mib"]; l, c, ls_ = LAB[t]
            keep = [j for j in range(len(x)) if j < len(x) - 1 or y[j] >= y[j - 1] * 0.5]   # drop the release when the request ends
            x, y = [x[j] for j in keep], [y[j] for j in keep]
            ax.plot(x, y, color=c, lw=1.2 if t != "tpl_physical" else 1.0, ls=ls_, label=l, drawstyle="steps-post")
        ax.set_xlim(lo, hi); ax.tick_params(labelsize=5.8); ax.set_xlabel("generated tokens", fontsize=6.4)
        tk = list(range(lo, hi + 1, 4096 if i == 0 else 512)); ax.set_xticks(tk)
        ax.set_xticklabels([("%gK" % (v / 1024)) for v in tk])
    axs[1].set_ylabel("resident KV per sequence (MiB)", fontsize=6.4); axs[1].legend(fontsize=5.8, loc="upper left", handlelength=1.6)
    axs[1].set_title("(%s) KV of one sequence as it generates" % letters[1], loc="left", fontsize=6.8)
    if "uni_plain" in KT:
        ys = [v for t in ("uni_plain", "tpl_physical") for x_, v in zip(KT[t]["tokens"], KT[t]["mib"]) if 3072 <= x_ <= 5120]
        axs[2].set_ylim(min(ys) * 0.9, max(ys) * 1.08)
    axs[2].set_title("(%s) zoom: blocks of 512 tokens" % letters[2], loc="left", fontsize=6.8)
fig, axs = plt.subplots(1, 3, figsize=(7.2, 1.95), gridspec_kw={"width_ratios": [1.25, 1, 1]})
draw_perlayer_panels(axs)
fig.tight_layout(pad=0.3, w_pad=0.8); fig.savefig(os.path.join(FIG, "perlayer_design.pdf")); plt.close(fig)

# ---------------- Figure 1 (user 2026-09-25): the method picture, with the evidence beside it ----------------
# Drawn at the 5.5 in text width like Figure 2, so FS/TS print at their nominal size in both. Left: compress each block as it
# completes. Right column: KV of one sequence as it writes (full KV vs AbridgeKV, sawtooth inset) above accuracy and time.
fig = plt.figure(figsize=(5.5, 3.05))
gs = fig.add_gridspec(2, 2, width_ratios=[1.42, 1.0], height_ratios=[1, 1], wspace=0.13, hspace=0.72, left=0.005, right=0.975, top=0.93, bottom=0.145)
draw_layout_a(fig.add_subplot(gs[:, 0]))
axb, axc = fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1])
axz = axb.inset_axes([0.1, 0.56, 0.42, 0.36])
draw_perlayer_panels([None, axb, axz], "xbx", arms=("vanilla_graphs", "tpl_physical"))
for h in axb.get_lines():
    if h.get_label() == "per-layer": h.set_label("AbridgeKV")
axb.legend(fontsize=FS - 0.8, loc="center right", handlelength=1.6, borderaxespad=0.2)
axb.set_title("(b) KV of one sequence as it writes", loc="left", fontsize=TS)
axb.set_xlabel("generated tokens", fontsize=FS); axb.set_ylabel("resident KV (MiB)", fontsize=FS); axb.tick_params(labelsize=FS - 0.8)
axz.set_title("", loc="left"); axz.set_title("blocks of 512 tokens", fontsize=FS - 1.4, pad=2); axz.set_xlabel("")
axz.tick_params(labelsize=FS - 1.8, pad=1); axz.set_yticks([])
axc.axis("off")
axc1, axc2 = axc.inset_axes([0.13, 0.12, 0.38, 0.88]), axc.inset_axes([0.72, 0.12, 0.28, 0.88])
# (c) with the two comparisons a reader asks for first (user 2026-09-26): R-KV at the budget matched to our 4:1 residency and
# LightThinker's thought-level algorithm trained on our data. A time bar appears only when that arm has a completion cell.
TA = [("vanilla", "vanilla_graphs", "full KV", GROUP["reference"]), ("vpl", "tpl_physical", "AbridgeKV", GROUP["ours"]),
      ("rkv_b4096", "rkv_b4096", "R-KV", RKV), ("lt_thought", "lt_thought", "LightThinker", GROUP["prior"])]
# Left: accuracy, a bar per arm. Right (user 2026-09-26): speed-up in time to finish over full KV, higher is better, with
# full KV as the dotted 1x line rather than a bar.
vals = [(X["arms"].get(k, {}).get("pool") or {}).get("avg8") for k, _, _, _ in TA]
top = max(v for v in vals if v) * 1.3; axc1.set_ylim(0, top)
for i, ((k, _, lab, col), v) in enumerate(zip(TA, vals)):
    if v is None: continue
    rbar(axc1, i, v, 0.7, col); axc1.text(i, v + top * 0.03, "%.1f" % v, ha="center", va="bottom", fontsize=FS - 1.2, color=PAL["ink"])
_vt = COMP["vanilla_graphs"]["s_per_problem"]
TS_ = [(lab, col, (_vt / COMP[t]["s_per_problem"]) if COMP.get(t) else None) for k, t, lab, col in TA[1:]]
topS = max(v for _, _, v in TS_ if v) * 1.28; axc2.set_ylim(0, topS)
axc2.axhline(1, color=GROUP["reference"], lw=1.1, ls=(0, (1.5, 1.5)), zorder=2)
for i, (lab, col, v) in enumerate(TS_):
    if v is None:
        axc2.text(i, topS * 0.04, "n/a", ha="center", va="bottom", fontsize=FS - 1.4, color=PAL["grey"]); continue
    rbar(axc2, i, v, 0.7, col); axc2.text(i, v + topS * 0.03, ("%.1f$\\times$" if v >= 1 else "%.2f$\\times$") % v, ha="center", va="bottom", fontsize=FS - 1.2, color=PAL["ink"])
for ax, labs, yl in ((axc1, [l for _, _, l, _ in TA], "accuracy (avg@8)"), (axc2, [l for l, _, _ in TS_], "speed-up over full KV")):
    ax.set_xticks(range(len(labs))); ax.set_xticklabels(labs, fontsize=FS - 1.2, rotation=38, ha="right", rotation_mode="anchor")
    ax.set_xlim(-0.6, len(labs) - 0.4); ax.set_ylabel(yl, fontsize=FS, labelpad=1); ax.tick_params(axis="y", labelsize=FS - 0.8)
    ax.tick_params(axis="x", pad=1.5, length=0)
axc2.set_yticks([0, 1, 2, 3])
axc.set_title("(c) accuracy and speed", loc="left", fontsize=TS)
fig.savefig(os.path.join(FIG, "teaser.pdf")); plt.close(fig)


def draw_cache_groups(ax, title):
    """Serving the per-layer schedule: one paged KV cache group per quota level, each as long as its quota needs. Same toy
    schedule as the illustration (quotas 8, 2, 4, 2, 4, 4 over six layers), one sequence after two completed blocks."""
    ax.set_axis_off(); ax.grid(False)
    groups = [(8, "layer 1"), (4, "layers 3, 5, 6"), (2, "layers 2, 4")]
    nblk, P, OPEN, W = 2, 2, 3, 0.42
    for gi, (q, lays) in enumerate(groups):
        y = 3.2 - gi * 1.35
        ax.text(0.0, y + 0.5, "quota %d: %s" % (q, lays), fontsize=FS, color=INK, va="bottom")
        x = 0.0
        for _ in range(P): _rbox(ax, x, y, W - 0.05, 0.38, PROMPT, r=0.05); x += W
        for b in range(nblk):
            for _ in range(q): _rbox(ax, x, y, W - 0.05, 0.38, CODE, r=0.05); x += W
            x += 0.08
        for _ in range(OPEN): _rbox(ax, x, y, W - 0.05, 0.38, CONTENT, r=0.05); x += W
    ax.text(0.0, -0.75, "each group is its own paged cache: a layer's\nunused rows take no memory, nothing is masked", fontsize=FS, color=INK, va="top", linespacing=1.15)
    ax.set_xlim(-0.05, 9.6); ax.set_ylim(-1.6, 4.3)
    ax.set_title(title, loc="left", fontsize=TS)

# ---------------- Figure 2 (user 2026-09-25): method details ----------------
# Drawn at the text width (5.5 in) so nothing is scaled down: mask and position ids on top; the per-layer budget and how it
# is served (one cache group per quota level) below.
fig = plt.figure(figsize=(5.5, 3.0))
gs = fig.add_gridspec(2, 2, width_ratios=[0.8, 1.2], height_ratios=[1, 0.7], wspace=0.5, hspace=0.26, left=0.1, right=0.99, top=0.95, bottom=0.02)
_axa = fig.add_subplot(gs[0, 0]); draw_mask(_axa, "(a) attention mask")
_axb = fig.add_subplot(gs[0, 1]); draw_positions(_axb, "(b) in-block position ids")
gb = gs[1, :].subgridspec(1, 2, width_ratios=[1.0, 0.9], wspace=0.3)
_axc = fig.add_subplot(gb[0, 0]); draw_perlayer_illus(_axc, "(c) per-layer budget (illustration)")
def draw_training(ax, title):
    # (d) training (Section 2.2), redrawn 2026-09-26: one row of the sequence as the trainer sees it; the pause tokens of each
    # block drawn as K small slots that become its codes; red ticks = next-token loss (response tokens only, none on pause
    # tokens); dashed arcs = where the gradient of a later block's loss enters: the pause tokens whose codes it reads.
    ax.set_axis_off(); ax.grid(False)
    H, y = 0.9, 0.0
    segs = [("prompt", 1.3, PROMPT, INK, "p"), ("block 0", 1.75, CONTENT, INK, "b"), ("$c_0$", 0.72, CODE, "white", "c"),
            ("block 1", 1.75, CONTENT, INK, "b"), ("$c_1$", 0.72, CODE, "white", "c"), ("block 2", 1.75, CONTENT, INK, "b")]
    x, pos = 0.0, {}
    for name, w, col, tc, kind in segs:
        if kind == "c":      # pause tokens: a few slots inside one box
            _rbox(ax, x, y, w, H, col, r=0.12, z=5)
            for j in range(3):
                ax.add_patch(plt.Rectangle((x + 0.09 + j * 0.19, y + 0.1), 0.14, 0.18, fc="white", ec="none", alpha=0.55, zorder=6))
            ax.text(x + w / 2, y + H * 0.62, name, ha="center", va="center", fontsize=FS, color=tc, zorder=6)
        else:
            _rbox(ax, x, y, w, H, col, r=0.12, z=5)
            ax.text(x + w / 2, y + H / 2, name, ha="center", va="center", fontsize=FS, color=tc, zorder=6)
        pos[name] = (x, w); x += w + 0.1
    X1 = x - 0.1
    for name in ("block 0", "block 1", "block 2"):    # next-token loss on every response token
        x0, w = pos[name]
        for k in range(6):
            xx = x0 + (k + 0.5) * w / 6
            ax.annotate("", xy=(xx, y - 0.4), xytext=(xx, y - 0.04), arrowprops=dict(arrowstyle="-|>", color=GROUP["ours"], lw=0.6,
                        mutation_scale=4, shrinkA=0, shrinkB=0), zorder=4)
    for name in ("$c_0$", "$c_1$"):                    # no loss on pause tokens
        x0, w = pos[name]; ax.text(x0 + w / 2, y - 0.2, "no loss", ha="center", va="center", fontsize=FS - 1.4, color=PAL["grey"])
    # gradient arcs: block 1 -> c0 (low), block 2 -> c1 (low), block 2 -> c0 (high)
    for src, dst, h in (("block 1", "$c_0$", 0.55), ("block 2", "$c_1$", 0.55), ("block 2", "$c_0$", 1.15)):
        xs, ws = pos[src]; xd, wd = pos[dst]
        x_s, x_d = xs + ws * (0.5 if h > 1 else 0.3), xd + wd / 2
        yt = y + H + 0.06; ytop = yt + h
        ax.plot([x_s, x_s, x_d], [yt, ytop, ytop], color=INK, lw=0.8, ls=(0, (2.5, 1.5)), zorder=4, solid_capstyle="butt")
        ax.annotate("", xy=(x_d, yt), xytext=(x_d, ytop), arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.8, ls=(0, (2.5, 1.5)),
                    mutation_scale=6, shrinkA=0, shrinkB=0), zorder=4)
    # legend-style key, left-aligned under the row
    ky = y - 0.85
    ax.annotate("", xy=(0.25, ky - 0.18), xytext=(0.25, ky + 0.12), arrowprops=dict(arrowstyle="-|>", color=GROUP["ours"], lw=0.7, mutation_scale=5, shrinkA=0, shrinkB=0))
    ax.text(0.45, ky, "next-token loss on response tokens only", fontsize=FS, color=GROUP["ours"], va="center")
    ax.plot([0.05, 0.42], [ky - 0.55, ky - 0.55], color=INK, lw=0.8, ls=(0, (2.5, 1.5)))
    ax.text(0.5, ky - 0.55, "its gradient flows into the pause tokens that wrote the codes", fontsize=FS, color=INK, va="center")
    ax.text(X1, y + H + 1.55, "one forward pass over the whole sequence", ha="right", va="bottom", fontsize=FS - 0.8, color=PAL["grey"])
    ax.set_xlim(-0.1, X1 + 0.1); ax.set_ylim(ky - 0.9, y + H + 1.9)
    ax.set_title(title, loc="left", fontsize=TS)
_axd = fig.add_subplot(gb[0, 1]); draw_training(_axd, "(d) training end to end through the codes")
# Line the bottom panels up under the top ones (user 2026-09-26: (d) was not aligned): after the aspect-locked panels have
# taken their final boxes, (c) starts where (a) starts and (d) spans exactly (b)'s columns.
fig.canvas.draw()
_pa, _pb, _pc, _pd = (a_.get_position() for a_ in (_axa, _axb, _axc, _axd))
_axd.set_position([_pb.x0, _pd.y0, _pb.x1 - _pb.x0, _pd.height])
_axc.set_position([_pa.x0, _pc.y0, _pb.x0 - _pa.x0 - 0.03, _pc.height])
fig.savefig(os.path.join(FIG, "layout.pdf"), bbox_inches="tight"); plt.close(fig)

# ---------------- Figure: information flows through the code rows (Section 5.2, user 2026-09-26) ----------------
# (a) the model's own AIME chains: p(correct later digits) of numbers written in block b, with block b's code rows intact,
#     replaced (previous block / another problem) or removed; control = numbers first written in other blocks.
# (b) planted values: exact recall of a random 4-digit value written in block b, k compressions later.
_CP = json.load(open(os.path.join(HERE, "data", "code_probe.json")))
_PRp = os.path.join(HERE, "data", "planted_recall_v2.json")
_PR = json.load(open(_PRp if os.path.exists(_PRp) else os.path.join(HERE, "data", "planted_recall.json")))
print("[code_flow] planted data:", "v2" if os.path.exists(_PRp) else "v1")
fig, (axA, axB) = plt.subplots(1, 2, figsize=(5.5, 1.7), gridspec_kw={"width_ratios": [1.15, 1.0]})
conds = [("true", "own code rows", GROUP["ours"]), ("swap_adj", "previous block's", PAL["sand"]),
         ("swap_other", "another problem's", PAL["slate"]), ("drop", "removed", PAL["grey"])]
cats = [("from_b_next", "written in block $b$,\nreused in the next"), ("from_other", "written in another\nblock (control)")]
wd = 0.19
for ci, (c, lab, col) in enumerate(conds):
    for k, (cat, _) in enumerate(cats):
        d = _CP["swap_numbers"][cat][c]; x = k + (ci - 1.5) * wd
        axA.bar(x, d["cont_p"], wd * 0.92, color=col, label=lab if k == 0 else None, zorder=3)
        axA.errorbar([x], [d["cont_p"]], yerr=[[d["cont_p"] - d["cont_p_ci"][0]], [d["cont_p_ci"][1] - d["cont_p"]]], fmt="none",
                     ecolor=PAL["ink"], elinewidth=0.6, capsize=1.2, zorder=4)
axA.set_xticks(range(len(cats))); axA.set_xticklabels([l for _, l in cats], fontsize=FS - 0.4)
axA.set_ylim(0, 1.62); axA.set_yticks([0, 0.25, 0.5, 0.75, 1.0]); axA.tick_params(labelsize=FS - 0.4)
axA.set_ylabel("p(correct later digits)", fontsize=FS)
axA.legend(title="block $b$'s code rows:", title_fontsize=FS - 0.8, fontsize=FS - 0.8, ncol=2, loc="upper center", handlelength=0.9,
           columnspacing=0.8, borderaxespad=0.2, frameon=False)
axA.set_title("(a) numbers in the model's own reasoning", loc="left", fontsize=TS)
ks = [str(k) for k in _PR["ks"]]
for c, lab, col, ls_ in (("full", "full KV", PAL["ink"], (0, (2, 1.5))), ("normal", "AbridgeKV", GROUP["ours"], "-"),
                         ("drop", "code rows removed", PAL["grey"], "-")):
    if "recall" in _PR:   # v2 layout: recall[cond][k] = {pct, ci, n}
        ys = [_PR["recall"][c][k]["pct"] for k in ks]; lo = [_PR["recall"][c][k]["ci"][0] for k in ks]; hi = [_PR["recall"][c][k]["ci"][1] for k in ks]
    else:                 # v1 layout
        ys = [_PR["by_k"][c][k]["recall"] for k in ks]; lo = [_PR["by_k"][c][k]["recall_ci"][0] for k in ks]; hi = [_PR["by_k"][c][k]["recall_ci"][1] for k in ks]
    xs = [int(k) for k in ks]
    axB.fill_between(xs, lo, hi, color=col, alpha=0.15, lw=0)
    axB.plot(xs, ys, color=col, lw=1.3, ls=ls_, marker="o", ms=2.8, label=lab, zorder=3)
axB.set_xscale("log", base=2); axB.set_xticks([1, 2, 4, 8]); axB.set_xticklabels(["1", "2", "4", "8"]); axB.minorticks_off()
axB.set_ylim(-3, 105); axB.set_yticks([0, 25, 50, 75, 100]); axB.tick_params(labelsize=FS - 0.4)
axB.set_xlabel("compressions since the value was written", fontsize=FS); axB.set_ylabel("exact recall (%)", fontsize=FS)
axB.legend(fontsize=FS - 0.8, loc="center", bbox_to_anchor=(0.5, 0.33), ncol=3, handlelength=1.6, columnspacing=0.9, frameon=False)
axB.set_title("(b) a planted value, recalled later", loc="left", fontsize=TS)
fig.tight_layout(pad=0.3, w_pad=1.2); fig.savefig(os.path.join(FIG, "code_flow.pdf")); plt.close(fig)

# ---------------- Figure 4 (user 2026-09-26): generated length + information flow through the code rows ----------------
fig, (axL, axF) = plt.subplots(1, 2, figsize=(5.5, 1.95), gridspec_kw={"width_ratios": [1.12, 1.0]})
draw_lengths(axL, fs=FS - 0.2); axL.set_title("(a) generated length", loc="left", fontsize=TS)
# (b) redrawn 2026-09-26 (user: show directly how information travels, not ablation settings). Planted value Q_7 = 7391 written
# in block b; 8 compressions later the model writes "Recall that Q_7 =" and its digits. Attention of those digit queries to each
# earlier block's code rows, per kept row (mean over heads and the 36 layers), with vs without the planted sentence in block b.
# Offsets < -2 are left out: they include block 0, which holds the prompt, for one of the two planting positions.
_PA = json.load(open(os.path.join(HERE, "data", "planted_attention.json")))
_rd = lambda c: _PA["read"][c]["8"]["digits"]["all"]["all"]["per_row"]
OFFS = list(range(-2, 8)); wb = 0.38
for j, (c, lab, col) in enumerate((("plant", "value planted in block $b$", GROUP["ours"]), ("unplanted", "same text, no planted value", PAL["grey"]))):
    r = _rd(c); ys = [1e4 * r[str(o)]["mean"] for o in OFFS]
    for i, (o, y) in enumerate(zip(OFFS, ys)):
        rbar(axF, i + (j - 0.5) * wb, y, wb * 0.92, col, r=0.03)
    ci = r["0"]["ci"]; i0 = OFFS.index(0) + (j - 0.5) * wb
    axF.errorbar([i0], [1e4 * r["0"]["mean"]], yerr=[[1e4 * (r["0"]["mean"] - ci[0])], [1e4 * (ci[1] - r["0"]["mean"])]], fmt="none",
                 ecolor=PAL["ink"], elinewidth=0.7, capsize=1.6, zorder=4)
    axF.bar([0], [0], color=col, label=lab)
axF.set_xticks(range(len(OFFS)))
axF.set_xticklabels(["$b$" if o == 0 else "%+d" % o for o in OFFS], fontsize=FS - 0.8)
axF.set_xlim(-0.6, len(OFFS) - 0.4); axF.set_ylim(0, 7.4); axF.set_yticks([0, 1, 2, 3, 4, 5]); axF.tick_params(labelsize=FS - 0.6)
axF.annotate("value\nwritten here", (OFFS.index(0) - 0.2, 3.95), xytext=(OFFS.index(0) - 0.2, 4.7), ha="center", fontsize=FS - 1.2, color=GROUP["ours"],
             arrowprops=dict(arrowstyle="-|>", color=GROUP["ours"], lw=0.7, mutation_scale=5), va="bottom", linespacing=0.9)
axF.text(len(OFFS) - 1, 4.75, "most\nrecent", ha="center", va="bottom", fontsize=FS - 1.4, color=PAL["grey"], linespacing=0.9)
axF.set_xlabel("earlier block, relative to $b$ (recall in block $b{+}8$)", fontsize=FS)
axF.set_ylabel("attention per code row\n($\\times 10^{-4}$)", fontsize=FS)
axF.legend(fontsize=FS - 1.0, loc="upper right", bbox_to_anchor=(1.0, 1.02), handlelength=1.0, frameon=False, borderaxespad=0.1)
axF.set_title("(b) where the recall reads from", loc="left", fontsize=TS)
fig.tight_layout(pad=0.3, w_pad=1.4); fig.savefig(os.path.join(FIG, "length_flow.pdf")); plt.close(fig)
# planted value, recalled k compressions later (appendix)
fig, axB = plt.subplots(figsize=(3.0, 1.9))
ks = [str(k) for k in _PR["ks"]]
for c, lab, col, ls_ in (("full", "full KV", PAL["ink"], (0, (2, 1.5))), ("normal", "AbridgeKV", GROUP["ours"], "-"),
                         ("drop", "code rows removed", PAL["grey"], "-")):
    R_ = _PR["recall"][c] if "recall" in _PR else None
    ys = [R_[k]["pct"] for k in ks]; lo = [R_[k]["ci"][0] for k in ks]; hi = [R_[k]["ci"][1] for k in ks]; xs = [int(k) for k in ks]
    axB.fill_between(xs, lo, hi, color=col, alpha=0.15, lw=0); axB.plot(xs, ys, color=col, lw=1.3, ls=ls_, marker="o", ms=2.8, label=lab, zorder=3)
axB.set_xscale("log", base=2); axB.set_xticks([1, 2, 4, 8]); axB.set_xticklabels(["1", "2", "4", "8"]); axB.minorticks_off()
axB.set_ylim(-3, 105); axB.set_yticks([0, 25, 50, 75, 100]); axB.tick_params(labelsize=FS - 0.6)
axB.set_xlabel("compressions since the value was written", fontsize=FS); axB.set_ylabel("exact recall (%)", fontsize=FS)
axB.legend(fontsize=FS - 0.8, loc="center", bbox_to_anchor=(0.5, 0.36), ncol=2, handlelength=1.6, columnspacing=0.9, frameon=False)
fig.tight_layout(pad=0.3); fig.savefig(os.path.join(FIG, "planted.pdf")); plt.close(fig)
