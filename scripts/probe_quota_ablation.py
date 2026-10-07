"""Held-out CE of a trained CompressLM under per-head code-quota overrides (truncate / extend at eval).
Reproduces sft_ddp.eval_heldout (192 held-out pairs, cross-block objective, raw prompt, bf16 autocast) for a
list of quota configurations, appending one JSON record per config to PROBE_OUT. Single GPU, no grad."""
import os, sys, json, time, torch
sys.path.insert(0, os.getcwd())
from transformers import AutoModelForCausalLM
from abridgekv.compress_forward import CompressLM
from abridgekv.sft_quota_utils import rebucket_pairs, pair_tokens

RUN = os.environ["PROBE_RUN"]; OUT = os.environ["PROBE_OUT"]
PAIRS = os.environ["PAIRS"]; NMAX = int(os.environ.get("NMAX", "32")); EVALN = int(os.environ.get("EVALN", "192"))
MICRO = int(os.environ.get("PROBE_MICRO", "8")); SUBSET = int(os.environ.get("PROBE_SUBSET", "48"))
STAGES = os.environ.get("PROBE_STAGES", "class,layer,head,layer_up").split(",")
HEAD_SET = os.environ.get("PROBE_HEAD_SET", "subset")   # "full": per-head cells on all EVALN pairs (4x the cost, resolves the per-head signal)
dev = "cuda"
torch._dynamo.config.cache_size_limit = 256

payload = torch.load(os.path.join(RUN, "model.pt"), map_location="cpu", weights_only=False)
C, K, L = int(payload["C"]), int(payload["K"]), int(payload["L"])
Q0 = [list(map(int, r)) for r in payload["head_code_quotas"]]
cp = str(payload.get("code_positions", "appended")); base_name = payload.get("base", "Qwen/Qwen3-4B-Base")
print(f"[probe] run={RUN} C={C} K={K} L={L} code_positions={cp} base={base_name}", flush=True)
base = AutoModelForCausalLM.from_pretrained(base_name, torch_dtype=torch.float32).to(dev)
core = CompressLM(base, C, N=2, K=K, L=L, pause_xattn=payload.get("pause_xattn", "full"),
                  head_code_quotas=Q0, active_code_rows=K, code_positions=cp).to(dev)
missing, unexpected = core.load_state_dict(payload["model"], strict=False)
print(f"[probe] state loaded missing={len(missing)} unexpected={len(unexpected)} {missing[:3]} {unexpected[:3]}", flush=True)
del payload; core.eval()
NL, NH = len(Q0), len(Q0[0])

pairs = [p for p in rebucket_pairs(torch.load(PAIRS, weights_only=False), C) if 2 <= p["n_blocks"] <= NMAX]
evalp = pairs[:EVALN]; del pairs
order = sorted(range(len(evalp)), key=lambda i: (evalp[i]["n_blocks"], i))
stride = max(1, len(evalp) // SUBSET)
subset = [evalp[i] for i in order[::stride]][:SUBSET]
print(f"[probe] heldout={len(evalp)} subset={len(subset)} N-hist(full)={sorted(__import__('collections').Counter(p['n_blocks'] for p in evalp).items())}", flush=True)

def batch(items):
    N = items[0]["n_blocks"]; B = len(items)
    ids = torch.zeros(B, N * C, dtype=torch.long); rm = torch.zeros(B, N * C, dtype=torch.long)
    for i, it in enumerate(items):
        seq = pair_tokens(it)[:N * C]
        ids[i, :len(seq)] = torch.tensor(seq); rm[i, len(it["prompt"]):len(seq)] = 1
    return ids.view(B, N, C).to(dev), rm.view(B, N, C).to(dev)

@torch.no_grad()
def heldout_ce(items):
    by_n = {}
    for p in items: by_n.setdefault(p["n_blocks"], []).append(p)
    s = torch.zeros((), dtype=torch.float64, device=dev); c = torch.zeros((), dtype=torch.int64, device=dev)
    for n in sorted(by_n):
        its = by_n[n]
        for i in range(0, len(its), MICRO):
            b = its[i:i + MICRO]; core.set_N(n); ids, rm = batch(b)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = core(ids, baseline=False, ctx_blocks=0, resp_mask=rm, raw_prompt=True, ce_chunk=4096,
                           per_token=True, include_cross_block=True)
            ce = out["ce_pt"].double(); resp = out["tmask"].bool()
            s += ce[resp].sum(); c += resp.sum()
    return float(s / c.clamp_min(1)), int(c)

done = set()
if os.path.exists(OUT):
    for line in open(OUT):
        try: done.add(json.loads(line)["name"])
        except Exception: pass

def run(name, quotas, items, kind):
    if name in done: return
    t0 = time.time(); core.set_head_code_quotas(quotas); ce, ntok = heldout_ce(items)
    rec = {"name": name, "kind": kind, "set": "full" if items is evalp else "subset", "n_examples": len(items),
           "resp_tokens": ntok, "held_out_ce": ce, "quota_hist": dict(__import__('collections').Counter(v for r in quotas for v in r)),
           "sec": round(time.time() - t0, 1)}
    with open(OUT, "a") as f: f.write(json.dumps(rec) + "\n")
    print(f"[probe] {name:40s} {rec['set']:6s} ce={ce:.5f} toks={ntok} {rec['sec']}s", flush=True)

def mapped(fn): return [[fn(l, h, Q0[l][h]) for h in range(NH)] for l in range(NL)]
def override(cells, value):
    q = [r[:] for r in Q0]
    for l, h in cells: q[l][h] = value
    return q
vals = sorted({v for r in Q0 for v in r}); hetero = len(vals) > 1
levels = [64, 128, 256]

if "class" in STAGES:
    run("trained", Q0, evalp, "class")
    run("trained_subset", Q0, subset, "class")
    for v in [64, 128]: run(f"all->{v}", mapped(lambda l, h, q: v), evalp, "class")
    if hetero:
        run("all->256", mapped(lambda l, h, q: 256), evalp, "class")
        for src in vals:
            for dst in levels:
                if dst != src: run(f"{src}->{dst}", mapped(lambda l, h, q: dst if q == src else q), evalp, "class")
if "layer" in STAGES:
    for l in range(NL):
        run(f"layer{l:02d}->64", override([(l, h) for h in range(NH)], 64), evalp, "layer")
if "head" in STAGES:
    hset = evalp if HEAD_SET == "full" else subset
    for l in range(NL):
        for h in range(NH):
            if Q0[l][h] > 64: run(f"head{l:02d}.{h}->64", override([(l, h)], 64), hset, "head")
            if Q0[l][h] == 64: run(f"head{l:02d}.{h}->128", override([(l, h)], 128), hset, "head_up")
            if Q0[l][h] == 256: run(f"head{l:02d}.{h}->128", override([(l, h)], 128), hset, "head")
if "custom" in STAGES:
    # PROBE_CUSTOM: JSON list of [name, {"a-b": quota, "l": quota, ...}] layer-range overrides (uniform within layer)
    # keys "l.h" override a single head; PROBE_CUSTOM=@file reads the list from a JSON file
    spec_src = os.environ["PROBE_CUSTOM"]
    cells = json.load(open(spec_src[1:])) if spec_src.startswith("@") else json.loads(spec_src)
    for name, spec in cells:
        q = [r[:] for r in Q0]
        for rng, v in spec.items():
            if "." in rng:
                l, h = rng.split("."); q[int(l)][int(h)] = int(v); continue
            a, b = (rng.split("-") + [rng])[:2]
            for l in range(int(a), int(b) + 1):
                q[l] = [int(v)] * NH
        mean = sum(sum(r) for r in q) / (NL * NH)
        run(f"custom:{name}(mean{mean:.0f})", q, evalp, "custom")
if "layer_up" in STAGES and hetero:
    for l in range(NL):
        run(f"layer{l:02d}->256", override([(l, h) for h in range(NH)], 256), evalp, "layer_up")
print("[probe] done", flush=True)
