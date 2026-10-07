"""Train-vs-generation algorithm parity for per-(layer, head) code quotas.

Teacher-forces held-out sequences through (a) the TRAINER forward (CompressLM.forward, the exact path that produced the
training/held-out CE) and (b) the incremental GENERATOR (CompressGen, the HF serving path that vLLM is parity-checked
against), both with the run's trained head_code_quotas, and compares per-token CE / logits at every response position.
A mismatch here would mean the model is evaluated with a different algorithm than it was trained with. Also runs the
generator with a deliberately WRONG schedule (uniform mean) to show the comparison is sensitive.

  PROBE_RUN=<run dir> PAIRS=<pairs.pkl> [N_ITEMS=3] [MIN_BLOCKS=4] [MAX_BLOCKS=8] python scripts/check_train_gen_quota_parity.py
"""
import os, sys, json, time, torch
sys.path.insert(0, os.getcwd())
from transformers import AutoModelForCausalLM
from abridgekv.compress_forward import CompressLM
from abridgekv.gen_compress import CompressGen
from abridgekv.sft_quota_utils import rebucket_pairs, pair_tokens

RUN = os.environ["PROBE_RUN"]; PAIRS = os.environ["PAIRS"]
N_ITEMS = int(os.environ.get("N_ITEMS", "3")); MINB = int(os.environ.get("MIN_BLOCKS", "4")); MAXB = int(os.environ.get("MAX_BLOCKS", "8"))
dev = "cuda"
torch._dynamo.config.cache_size_limit = 256
payload = torch.load(os.path.join(RUN, "model.pt"), map_location="cpu", weights_only=False)
C, K, L = int(payload["C"]), int(payload["K"]), int(payload["L"])
Q0 = [list(map(int, r)) for r in payload["head_code_quotas"]]
cp = str(payload.get("code_positions", "appended")); base_name = payload.get("base", "Qwen/Qwen3-4B-Base")
print(f"[parity] run={os.path.basename(RUN)} C={C} K={K} code_positions={cp} quota levels="
      f"{dict(__import__('collections').Counter(v for r in Q0 for v in r))} layer sums min/max={min(map(sum,Q0))}/{max(map(sum,Q0))}", flush=True)
base = AutoModelForCausalLM.from_pretrained(base_name, dtype=torch.float32).to(dev)      # fp32, no autocast: differences are algorithmic
core = CompressLM(base, C, N=2, K=K, L=L, pause_xattn=payload.get("pause_xattn", "full"),
                  head_code_quotas=Q0, active_code_rows=K, code_positions=cp).to(dev)
core.load_state_dict(payload["model"], strict=False); core.raw_prompt = True; core.eval(); del payload
pairs = [p for p in rebucket_pairs(torch.load(PAIRS, weights_only=False), C) if MINB <= p["n_blocks"] <= MAXB]
items = pairs[:N_ITEMS]; del pairs
uniform = [[int(round(sum(map(sum, Q0)) / (len(Q0) * len(Q0[0]))))] * len(Q0[0]) for _ in Q0]

@torch.no_grad()
def trainer_ce(item):
    N = item["n_blocks"]; seq = pair_tokens(item)[:N * C]
    ids = torch.zeros(1, N * C, dtype=torch.long); ids[0, :len(seq)] = torch.tensor(seq)
    rm = torch.zeros(1, N * C); rm[0, len(item["prompt"]):len(seq)] = 1
    ids, rm = ids.view(1, N, C).to(dev), rm.view(1, N, C).to(dev)
    core.set_N(N); core.set_head_code_quotas(Q0)
    out = core(ids, baseline=False, ctx_blocks=0, resp_mask=rm, raw_prompt=True, ce_chunk=4096, per_token=True, include_cross_block=True)
    # rows are predict positions 0..N*C-2 (flat); target = position+1
    return ids, rm, out["ce_pt"].float(), out["tmask"].bool()

@torch.no_grad()
def gen_ce(item, ids, quotas):
    N = item["n_blocks"]; plen = len(item["prompt"])
    g = CompressGen(core, dev, raw_prompt_len=plen, head_code_quotas=quotas)
    flat = ids.view(-1); logits = []
    for b in range(N):
        h = g.feed_content(flat[b * C:(b + 1) * C].tolist())
        logits.append(core.model.lm_head(h).float())
        if b < N - 1: g.compress()
    lg = torch.cat(logits, 0)[:-1]                                        # rows 0..N*C-2 predict token at row+1
    return torch.nn.functional.cross_entropy(lg, flat[1:], reduction="none"), lg

rows = []
for it in items:
    t0 = time.time()
    ids, rm, ce_tr, m = trainer_ce(it)
    ce_ok, lg_ok = gen_ce(it, ids, Q0)
    ce_bad, _ = gen_ce(it, ids, uniform)
    d_ok = (ce_ok - ce_tr)[m]; d_bad = (ce_bad - ce_tr)[m]
    # per-block breakdown: does the mismatch (if any) grow with compressed history?
    blk = (torch.arange(ce_tr.numel(), device=dev) + 1) // C
    per_blk = [float((ce_ok - ce_tr)[m & (blk == b)].abs().mean()) if bool((m & (blk == b)).any()) else None for b in range(it["n_blocks"])]
    rows.append({"n_blocks": it["n_blocks"], "resp_tokens": int(m.sum()),
                 "trainer_ce": float(ce_tr[m].mean()), "gen_ce_trained_sched": float(ce_ok[m].mean()), "gen_ce_uniform_sched": float(ce_bad[m].mean()),
                 "max_abs_dCE_trained": float(d_ok.abs().max()), "mean_abs_dCE_trained": float(d_ok.abs().mean()),
                 "max_abs_dCE_uniform": float(d_bad.abs().max()), "mean_abs_dCE_uniform": float(d_bad.abs().mean()),
                 "mean_abs_dCE_by_block_trained": per_blk, "sec": round(time.time() - t0, 1)})
    r = rows[-1]
    print(f"[parity] N={r['n_blocks']} resp={r['resp_tokens']} trainer CE {r['trainer_ce']:.5f} | gen(trained sched) {r['gen_ce_trained_sched']:.5f} "
          f"max|dCE| {r['max_abs_dCE_trained']:.2e} mean {r['mean_abs_dCE_trained']:.2e} | gen(uniform sched) {r['gen_ce_uniform_sched']:.5f} "
          f"max|dCE| {r['max_abs_dCE_uniform']:.2e} mean {r['mean_abs_dCE_uniform']:.2e} | by block {['%.1e' % x if x is not None else '-' for x in per_blk]} ({r['sec']}s)", flush=True)
out = os.environ.get("PARITY_OUT")
if out:
    json.dump({"run": RUN, "quota_levels": sorted({v for r in Q0 for v in r}), "items": rows}, open(out, "w"), indent=1)
    print(f"[parity] wrote {out}")
