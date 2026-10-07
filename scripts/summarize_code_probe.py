"""CPU side of the code-row probe (scripts/probe_code_rows.py).

  build      saved completions (eval_math500_compresslm.py --save-completions shards) -> chains json
             (prompt ids exactly as the eval built them, completion re-tokenized; token_sha256 checked)
  summarize  per-chain probe outputs -> paper/data/code_probe.json + paper/figures/code_probe.pdf
"""
import argparse, glob, hashlib, json, math, os, random, sys
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def sha(ids):
    return hashlib.sha256(",".join(map(str, ids)).encode("ascii")).hexdigest()


def build(args):
    from transformers import AutoTokenizer
    from prompt_format import format_user_content
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    import re
    chains, n_ok, n_all = [], 0, 0
    for f in sorted(args.shards):
        d = json.load(open(f))
        setname = re.search(r"codeprobe_(.+)_shard\d+", f).group(1)
        rows = [json.loads(l) for l in open(os.path.join(args.data_dir, setname + ".jsonl"))]
        for p in d["per_problem"]:
            row = rows[p["dataset_index"]]
            pkey = f"{setname}/{p['dataset_index']}"
            pids = tok.encode(format_user_content(row["problem"], True), add_special_tokens=True)
            for si, s in enumerate(p["samples"]):
                n_all += 1
                cids = tok.encode(s["text"], add_special_tokens=False)
                ok = sha(cids) == s["token_sha256"]
                if not ok and s.get("stop_reason") == "stop":
                    for eos in (tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")):
                        if sha(cids + [eos]) == s["token_sha256"]:
                            ok = True
                            break
                n_ok += ok
                chains.append({"id": f"{setname}_{p['dataset_index']}_{si}", "problem_index": pkey,
                               "prompt_ids": pids, "completion_ids": cids, "sha_match": ok,
                               "correct": bool(s["correct"]), "gen_tokens": s["gen_tokens"]})
    # interleave problems so consecutive chains (donors) come from different problems
    by_p = defaultdict(list)
    for c in chains:
        by_p[c["problem_index"]].append(c)
    order = []
    while any(by_p.values()):
        for k in sorted(by_p):
            if by_p[k]:
                order.append(by_p[k].pop(0))
    json.dump(order, open(args.out, "w"))
    lens = sorted(len(c["completion_ids"]) for c in order)
    print(f"[build] {len(order)} chains from {len(set(c['problem_index'] for c in order))} problems; "
          f"token sha match {n_ok}/{n_all}; completion tokens median {lens[len(lens)//2]} min {lens[0]} max {lens[-1]}")



# ---------------------------------------------------------------------------------------------------------------------
def token_classes(tok, vocab_ids):
    """id -> (text, category) for the ids that occur. Categories for the attention read-out."""
    out = {}
    for i in vocab_ids:
        t = tok.decode([i])
        st = t.strip()
        if st and st.isdigit():
            c = "digit"
        elif "\n" in t:
            c = "newline"
        elif st and any(ch.isalpha() for ch in st) and not st.startswith("\\"):
            c = "word"
        elif st and (st.startswith("\\") or any(ch in "+-=*/^_{}()[]<>|$!" for ch in st)):
            c = "math"
        else:
            c = "other"
        out[i] = (t, c)
    return out


def number_runs(ids, cls):
    """Maximal runs of digit tokens (Qwen splits numbers into single digits); a token with leading whitespace starts a
    new run. Returns list of (start, end, value)."""
    runs, i, n = [], 0, len(ids)
    while i < n:
        t, c = cls[ids[i]]
        if c == "digit":
            j = i + 1
            while j < n and cls[ids[j]][1] == "digit" and not cls[ids[j]][0][:1].isspace():
                j += 1
            runs.append((i, j, "".join(cls[x][0].strip() for x in ids[i:j])))
            i = j
        else:
            i += 1
    return runs


def boot_ci(groups, fn, n=2000, seed=0):
    """Percentile CI by resampling chains (groups: list of per-chain item lists)."""
    rng = random.Random(seed)
    vals = []
    for _ in range(n):
        smp = [x for _ in groups for x in rng.choice(groups)]
        if smp:
            vals.append(fn(smp))
    vals.sort()
    return vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals)) - 1]


def summarize(args):
    import numpy as np
    import torch
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    files = sorted(glob.glob(os.path.join(args.probe_dir[0], "chain_*.pt")))
    recs = [torch.load(f, weights_only=False) for f in files]
    maxdiff = 0.0
    for extra in args.probe_dir[1:]:        # further runs on the same chains (other conditions): merge per target
        for r in recs:
            f = os.path.join(extra, f"chain_{r['id']}.pt")
            if not os.path.exists(f):
                continue
            x = torch.load(f, weights_only=False)
            for b, res in x["targets"].items():
                if b in r["targets"]:
                    for cond, arr in res.items():
                        if cond in r["targets"][b]:
                            ok = ~torch.isnan(arr)
                            maxdiff = max(maxdiff, float((arr[ok] - r["targets"][b][cond][ok]).abs().max()) if ok.any() else 0.0)
                        else:
                            r["targets"][b][cond] = arr
    print(f"[merge] max |diff| of shared conditions across runs: {maxdiff:.2e}")
    C = recs[0]["C"]; K = recs[0]["K"]; layer_q = recs[0]["layer_q"]; NL = len(layer_q)
    vocab = set(i for r in recs for i in r["ids"])
    cls = token_classes(tok, vocab)
    MINLEN = args.min_digits
    conds_all = ["true", "swap_adj", "swap_other", "drop", "swap_L256", "swap_L128", "swap_L64", "keep_L256", "keep_L128", "keep_L64"]
    # ---------------- 1. swap test on reused numbers ----------------
    items = defaultdict(list)       # category -> list of (chain_idx, {cond: mean nll over continuation digits}, {cond: first-digit nll})
    alltok = defaultdict(lambda: defaultdict(list))   # cond -> distance -> per-target mean nll
    per_chain_items = defaultdict(lambda: defaultdict(list))
    n_targets = 0
    for ri, r in enumerate(recs):
        ids, P = r["ids"], r["P"]
        runs = number_runs(ids, cls)
        occ = defaultdict(list)
        for (s, e, v) in runs:
            occ[v].append(s)
        for b, res in r["targets"].items():
            n_targets += 1
            base = (b + 1) * C
            wlen = len(res["true"])
            end = base + wlen
            for cond, arr in res.items():
                for dist in range(1, 1 + math.ceil(wlen / C)):
                    seg = arr[(dist - 1) * C: dist * C]
                    seg = seg[~torch.isnan(seg)]
                    if seg.numel():
                        alltok[cond][dist].append((ri, float(seg.mean())))
            for (s, e, v) in runs:
                if s < base + 1 or e > end or e - s < MINLEN or s // C != (e - 1) // C:
                    continue
                cur = s // C
                prior = [q for q in occ[v] if q < s]
                if any(q >= cur * C for q in prior):
                    continue                     # already written earlier in the current (raw) block: local copy
                pb = [q for q in prior if q >= P and q // C == b]
                others = [q for q in prior if not (q >= P and q // C == b)]
                if not prior:
                    cat = "novel"
                elif pb and not others:
                    cat = "from_b"
                elif not pb:
                    cat = "from_other"
                else:
                    continue                     # in block b AND elsewhere: ambiguous source
                rec = {}
                for cond, arr in res.items():
                    cont = arr[s + 1 - base: e - base]
                    first = arr[s - base]
                    rec[cond] = (float(cont.mean()), float(first))
                item = {"chain": ri, "b": b, "dist": cur - b, "len": e - s, "value": v, "nll": rec}
                items[cat].append(item)
                per_chain_items[cat][ri].append(item)
    summary = {"n_chains": len(recs), "n_problems": len(set(r["problem_index"] for r in recs)), "n_targets": n_targets,
               "C": C, "K": K, "min_digits": MINLEN, "layer_quota_counts": dict(Counter(layer_q))}
    items["from_b_next"] = [i for i in items.get("from_b", []) if i["dist"] == 1]
    items["from_b_later"] = [i for i in items.get("from_b", []) if i["dist"] >= 2]
    swap = {}
    for cat, its in list(items.items()):
        d = {"n_items": len(its), "n_chains": len(set(i["chain"] for i in its))}
        for cond in conds_all:
            sub = [i for i in its if cond in i["nll"] and "swap_other" in i["nll"]]   # common subset with a donor
            if not sub:
                continue
            groups = [[i for i in sub if i["chain"] == c] for c in sorted(set(i["chain"] for i in sub))]
            f_nll = lambda xs, cond=cond: sum(x["nll"][cond][0] for x in xs) / len(xs)
            f_p = lambda xs, cond=cond: sum(math.exp(-x["nll"][cond][0]) for x in xs) / len(xs)
            f_d = lambda xs, cond=cond: sum(x["nll"][cond][0] - x["nll"]["true"][0] for x in xs) / len(xs)
            d[cond] = {"n": len(sub), "cont_nll": f_nll(sub), "cont_nll_ci": boot_ci(groups, f_nll),
                       "cont_p": f_p(sub), "cont_p_ci": boot_ci(groups, f_p),
                       "delta_vs_true": f_d(sub), "delta_ci": boot_ci(groups, f_d),
                       "first_nll": sum(x["nll"][cond][1] for x in sub) / len(sub)}
        swap[cat] = d
    summary["swap_numbers"] = swap
    bydist = {}
    for cat in ("from_b", "from_other", "novel"):
        for lab, lo, hi in (("1", 1, 1), ("2-3", 2, 3), ("4-6", 4, 6)):
            sub = [i for i in items[cat] if lo <= i["dist"] <= hi and "swap_other" in i["nll"]]
            if sub:
                bydist.setdefault(cat, {})[lab] = {"n": len(sub)} | {c: sum(math.exp(-i["nll"][c][0]) for i in sub) / len(sub)
                                                                    for c in ("true", "swap_other", "drop", "swap_adj")}
    summary["by_distance_p"] = bydist
    fb = [i for i in items.get("from_b", []) if "swap_other" in i["nll"]]
    if fb:
        summary["from_b_item_fractions"] = {c: {"p_gt_0.5": sum(math.exp(-i["nll"][c][0]) > 0.5 for i in fb) / len(fb),
                                                "median_nll": sorted(i["nll"][c][0] for i in fb)[len(fb) // 2]}
                                            for c in conds_all if all(c in i["nll"] for i in fb)}
        summary["from_b_digit_len_hist"] = dict(Counter(i["len"] for i in fb))
    summary["all_tokens_by_distance"] = {cond: {str(k): {"n": len(v), "nll": sum(x[1] for x in v) / len(v)}
                                                for k, v in sorted(dd.items())} for cond, dd in alltok.items()}
    # ---------------- 2. what the code step reads ----------------
    cat_names = ["digit", "math", "word", "newline", "other"]
    enrich = {c: [[] for _ in range(NL)] for c in cat_names}
    reuse_enrich = {"digit_reused": [[] for _ in range(NL)], "digit_not_reused": [[] for _ in range(NL)]}
    pos_mass = np.zeros((NL, 8)); pos_n = 0
    for r in recs:
        ids, P = r["ids"], r["P"]
        runs = number_runs(ids, cls)
        occ = defaultdict(list)
        for (s, e, v) in runs:
            occ[v].append(s)
        reused_pos = {}
        for (s, e, v) in runs:
            later = [q for q in occ[v] if q // C > s // C]
            for p in range(s, e):
                reused_pos[p] = bool(later) and (e - s) >= MINLEN
        for b, A in r["attn"].items():
            A = A.float().numpy()                  # (NL, C)
            offs = [o for o in range(C) if b * C + o < len(ids) and b * C + o >= P and not np.isnan(A[0, o])]
            if len(offs) < C // 2:
                continue
            tot = A[:, offs].sum(1, keepdims=True)
            share = A[:, offs] / tot               # per-layer share of the raw-block attention
            u = 1.0 / len(offs)
            cats = [cls[ids[b * C + o]][1] for o in offs]
            for c in cat_names:
                m = [k for k, cc in enumerate(cats) if cc == c]
                if m:
                    for li in range(NL):
                        enrich[c][li].append(float(share[li, m].mean() / u))
            dr = [k for k, o in enumerate(offs) if cats[k] == "digit" and reused_pos.get(b * C + o, False)]
            dn = [k for k, o in enumerate(offs) if cats[k] == "digit" and not reused_pos.get(b * C + o, False)]
            if dr and dn:
                for li in range(NL):
                    reuse_enrich["digit_reused"][li].append(float(share[li, dr].mean() / u))
                    reuse_enrich["digit_not_reused"][li].append(float(share[li, dn].mean() / u))
            if len(offs) == C:
                pos_mass += A.reshape(NL, 8, C // 8).sum(2) / tot; pos_n += 1
    summary["attention"] = {
        "enrichment_by_category": {c: {"per_layer": [float(np.mean(v)) if v else None for v in enrich[c]],
                                       "mean": float(np.mean([x for v in enrich[c] for x in v])) if enrich[c][0] else None,
                                       "n_blocks": len(enrich[c][0])} for c in cat_names},
        "digit_reuse": {k: {"per_layer": [float(np.mean(v)) for v in vv], "mean": float(np.mean([x for v in vv for x in v])),
                            "n_blocks": len(vv[0])} for k, vv in reuse_enrich.items()},
        "position_octile_share": {"per_layer": (pos_mass / max(pos_n, 1)).tolist(),
                                  "mean": (pos_mass / max(pos_n, 1)).mean(0).tolist(), "n_blocks": pos_n},
        "mass_split": {},
    }
    tot_parts = defaultdict(list)
    for r in recs:
        for b, T in r["attn_tot"].items():
            for li in range(NL):
                tot_parts[li].append(T[li].tolist())
    names = ["current_raw_block", "prompt", "prior_codes", "own_codes", "total"]
    summary["attention"]["mass_split"] = {n: [float(np.mean([x[k] for x in tot_parts[li]])) for li in range(NL)]
                                          for k, n in enumerate(names)}
    # ---------------- 3. logit lens ----------------
    lens = defaultdict(lambda: {"true": [], "same_chain": [], "other_chain": []})
    by_b_other = defaultdict(list)
    for ri, r in enumerate(recs):
        for b in r["lens"]:
            by_b_other[b].append(ri)
    rng = random.Random(0)
    for ri, r in enumerate(recs):
        ids, P = r["ids"], r["P"]
        sets = {b: set(ids[max(b * C, P):(b + 1) * C]) for b in range(len(ids) // C)}
        for b, L in r["lens"].items():
            if b == 0 or b not in sets:
                continue
            ctrl_b = [x for x in sets if x != b and abs(x - b) >= 2 and x > 0]
            others = [x for x in by_b_other[b] if recs[x]["problem_index"] != r["problem_index"]]
            if not ctrl_b or not others:
                continue
            cb = rng.choice(ctrl_b); ro = recs[rng.choice(others)]
            oset = set(ro["ids"][max(b * C, ro["P"]):(b + 1) * C])
            for li, top in L.items():
                q = layer_q[li]
                t1 = top[:q, 0].tolist()
                for name, S in (("true", sets[b]), ("same_chain", sets[cb]), ("other_chain", oset)):
                    lens[li][name].append(sum(x in S for x in t1) / len(t1))
    summary["logit_lens_top1_in_block"] = {str(li): {k: float(np.mean(v)) for k, v in d.items()} | {"n_blocks": len(d["true"])}
                                           for li, d in sorted(lens.items())}
    ex = []
    r = recs[0]
    for b in list(r["lens"])[2:4]:
        li = max(r["lens"][b]); top = r["lens"][b][li]
        ex.append({"block": b, "layer": li, "top1_first32": [tok.decode([int(x)]) for x in top[:32, 0]],
                   "block_text_head": tok.decode(r["ids"][b * C: b * C + 120])})
    summary["logit_lens_examples"] = ex
    os.makedirs(os.path.dirname(args.json_out), exist_ok=True)
    json.dump(summary, open(args.json_out, "w"), indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k in ("n_chains", "n_problems", "n_targets")}))
    for cat, d in swap.items():
        print(cat, d["n_items"], {c: (round(v["cont_nll"], 3), round(v["cont_p"], 3)) for c, v in d.items() if isinstance(v, dict)})


def make_figure(S, out):
    import numpy as np
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    PAL = {"ink": "#3D405B", "grey": "#9AA3AD", "coral": "#E07A5F", "sage": "#81B29A", "slate": "#6C8EBF", "sand": "#F2CC8F",
           "lilac": "#A28CD4", "sky": "#8FC1E3", "peach": "#F2B29B", "rust": "#B8664C"}
    plt.rcParams.update({"font.size": 6.4, "font.family": "sans-serif", "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#B9BEC6", "axes.labelcolor": PAL["ink"], "xtick.color": PAL["ink"], "ytick.color": PAL["ink"],
                         "text.color": PAL["ink"], "axes.titlecolor": PAL["ink"], "axes.titlesize": 7.2, "axes.titleweight": "normal",
                         "axes.grid": True, "grid.color": "#E6E8EC", "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False,
                         "legend.fontsize": 5.6, "xtick.labelsize": 6.0, "ytick.labelsize": 6.0, "axes.labelsize": 6.4})
    fig, axs = plt.subplots(1, 3, figsize=(5.5, 2.15), gridspec_kw={"width_ratios": [1.5, 1.0, 1.0]})
    # (a) recall of reused numbers
    ax = axs[0]
    SW = S["swap_numbers"]
    cats = [("from_b_next", "from $b$,\nnext block"), ("from_b_later", "from $b$,\n2-6 later"),
            ("from_other", "from other\nblocks"), ("novel", "new\nnumbers")]
    conds = [("true", "own codes", PAL["coral"]), ("swap_adj", "codes of block $b{-}1$", PAL["sand"]),
             ("swap_other", "other problem's block $b$", PAL["slate"]), ("drop", "block $b$ removed", PAL["grey"])]
    wd = 0.19
    for ci, (c, lab, col) in enumerate(conds):
        xs, ys, lo, hi = [], [], [], []
        for k, (cat, _) in enumerate(cats):
            d = SW.get(cat, {}).get(c)
            if not d:
                continue
            xs.append(k + (ci - 1.5) * wd); ys.append(d["cont_p"]); lo.append(d["cont_p"] - d["cont_p_ci"][0]); hi.append(d["cont_p_ci"][1] - d["cont_p"])
        ax.bar(xs, ys, wd * 0.92, color=col, label=lab, zorder=3)
        ax.errorbar(xs, ys, yerr=[lo, hi], fmt="none", ecolor=PAL["ink"], elinewidth=0.6, capsize=1.2, zorder=4)
    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels([lab for cat, lab in cats], fontsize=5.6)   # item counts are in the caption
    ax.set_ylabel("p(correct later digit)"); ax.set_ylim(0, 1.42); ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_title("(a) Recall of numbers vs block $b$'s codes", loc="left")
    ax.legend(loc="upper center", ncol=2, fontsize=5.4, handlelength=0.9, columnspacing=0.8, bbox_to_anchor=(0.5, 1.0), borderaxespad=0.2)
    # (b) per-layer-group: swap only group q vs keep only group q
    ax = axs[1]
    d = SW.get("from_b_next", {})
    qc = S["layer_quota_counts"]
    qs = (256, 128, 64)
    for key, col in (("true", PAL["coral"]), ("swap_other", PAL["slate"])):
        if key in d:
            ax.axhline(d[key]["cont_p"], color=col, lw=0.7, ls="--", zorder=2)
    for k, (pref, lab, col) in enumerate((("swap_L", "only this group swapped", PAL["slate"]), ("keep_L", "only this group kept", PAL["coral"]))):
        xs = [i + (k - 0.5) * 0.36 for i in range(len(qs))]
        ys = [d.get(f"{pref}{q}", {}).get("cont_p", float("nan")) for q in qs]
        ci = [d.get(f"{pref}{q}", {}).get("cont_p_ci", [y, y]) for q, y in zip(qs, ys)]
        ax.bar(xs, ys, 0.34, color=col, label=lab, zorder=3)
        ax.errorbar(xs, ys, yerr=[[y - c[0] for y, c in zip(ys, ci)], [c[1] - y for y, c in zip(ys, ci)]], fmt="none",
                    ecolor=PAL["ink"], elinewidth=0.6, capsize=1.2, zorder=4)
    ax.set_xticks(range(len(qs)))
    ax.set_xticklabels([f"{qc.get(str(q), qc.get(q, 0))}$\\times${q}" for q in qs], fontsize=5.8); ax.set_xlabel("layers $\\times$ code rows")
    ax.set_ylim(0, 1.42); ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0]); ax.set_yticklabels([])
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=1, fontsize=5.4, handlelength=0.9, borderaxespad=0.2)
    ax.set_title("(b) By layer group", loc="left")
    # (c) logit lens on the code rows
    ax = axs[2]
    LL = S["logit_lens_top1_in_block"]
    layers = sorted(LL, key=int)
    for k, (name, lab, col) in enumerate((("true", "its own block", PAL["coral"]), ("same_chain", "another block, same chain", PAL["sand"]),
                                          ("other_chain", "another problem", PAL["slate"]))):
        xs = [i + (k - 1) * 0.26 for i in range(len(layers))]
        ax.bar(xs, [LL[l][name] for l in layers], 0.25, color=col, label=lab, zorder=3)
    ax.set_xticks(range(len(layers))); ax.set_xticklabels([f"L{int(l)}" for l in layers], fontsize=6.0)
    ax.set_xlabel("layer (0-35)")
    ax.set_ylim(0, 1.42); ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0]); ax.set_ylabel("top-1 token found in ...")
    ax.legend(loc="upper left", fontsize=5.4, handlelength=0.9, borderaxespad=0.2)
    ax.set_title("(c) Logit lens on code rows", loc="left")
    fig.tight_layout(w_pad=0.8)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    if os.environ.get("PNG"):
        fig.savefig(out[:-4] + ".png", dpi=130, bbox_inches="tight")
    print(f"[fig] wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    b = sp.add_parser("build")
    b.add_argument("--shards", nargs="+", required=True)
    b.add_argument("--data-dir", required=True)
    b.add_argument("--tokenizer", required=True)
    b.add_argument("--out", required=True)
    s = sp.add_parser("summarize")
    s.add_argument("--probe-dir", nargs="+", required=True)
    s.add_argument("--tokenizer", required=True)
    s.add_argument("--json-out", default=os.path.join(ROOT, "paper/data/code_probe.json"))
    s.add_argument("--min-digits", type=int, default=3)
    s.add_argument("--fig-out", default=os.path.join(ROOT, "paper/figures/code_probe.pdf"))
    f = sp.add_parser("figure")   # separate step: the train env has no matplotlib
    f.add_argument("--json", default=os.path.join(ROOT, "paper/data/code_probe.json"))
    f.add_argument("--fig-out", default=os.path.join(ROOT, "paper/figures/code_probe.pdf"))
    a = ap.parse_args()
    if a.cmd == "build":
        build(a)
    elif a.cmd == "summarize":
        summarize(a)
    else:
        make_figure(json.load(open(a.json)), a.fig_out)
