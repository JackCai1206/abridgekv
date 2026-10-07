"""What do the code rows carry? Teacher-forced probe of a trained CompressLM on its own chains of thought.

For every chain (prompt + saved completion, re-tokenized) the HF incremental generator (abridgekv/gen_compress.CompressGen,
the image of the trainer forward, with the checkpoint's per-layer code quotas) is run block by block. At a set of target
blocks b the state right after block b's code step is snapshotted and the next W blocks are re-run under several
interventions on block b's code rows only:
  true       unchanged (sanity: equals the main pass)
  swap_other block b's code KV replaced by block b's code KV of a chain for a DIFFERENT problem (identical RoPE positions)
  swap_adj   block b's code KV replaced by the same chain's block b-1 code KV, K rotated by +C (RoPE composes additively,
             so this is exactly the key those rows would have had at block b's positions)
  drop       block b's code rows removed (nothing of block b remains)
  swap_L<q>  swap_other applied only in the layers whose quota is q (q in 256/128/64)
Per-token NLL of the window is saved per condition; the CPU summary tags tokens (digits of numbers first written in block b
and reused later, etc.).
The main pass also records (a) the attention of the code-step queries over the raw block being compressed (per layer,
averaged over heads and over the query slots the layer keeps), and (b) a logit lens on the code hidden states at a few layers.

Env: CKPT (model.pt), CHAINS (json list of {id, problem_index, prompt_ids, completion_ids}), OUT (dir), SHARD/NSHARD,
     W (window blocks, 8), TARGET_EVERY (3), TARGET_MIN (2), MAX_BLOCKS (32), LENS_LAYERS ("9,18,27,35").
"""
import json, os, sys, time, types
import torch
import torch.nn.functional as TF

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import abridgekv.gen_compress as gc  # noqa: E402

dev = "cuda"
CKPT = os.environ["CKPT"]; CHAINS = os.environ["CHAINS"]; OUT = os.environ["OUT"]
SHARD = int(os.environ.get("SHARD", 0)); NSHARD = int(os.environ.get("NSHARD", 1))
W = int(os.environ.get("W", 8)); EVERY = int(os.environ.get("TARGET_EVERY", 3)); TMIN = int(os.environ.get("TARGET_MIN", 2))
MAXB = int(os.environ.get("MAX_BLOCKS", 32)); NCHAIN = int(os.environ.get("NCHAIN", 0))
CONDS = [c for c in os.environ.get("CONDS", "true,drop,swap_adj,swap_other,swap_L256,swap_L128,swap_L64").split(",") if c]
LENS = [int(x) for x in os.environ.get("LENS_LAYERS", "9,18,27,35").split(",") if x]
TOPK = 20
os.makedirs(OUT, exist_ok=True)

# ---- attention capture: shim the softmax gen_compress calls in _layer ---------------------------------------------
class _Rec:
    active = False; li = 0; cb = None
REC = _Rec()
def _softmax(x, dim=-1, dtype=None):
    out = TF.softmax(x, dim=dim, dtype=dtype)
    if REC.active:
        REC.cb(REC.li, out); REC.li += 1
    return out
gc.F = types.SimpleNamespace(**{k: getattr(TF, k) for k in dir(TF) if not k.startswith("__")})
gc.F.softmax = _softmax


class ProbeGen(gc.CompressGen):
    """Records the logit-lens top-k of the code rows' residual stream at LENS layers during the code step."""
    lens_out = None
    def _layer(self, li, hidden, abs_pos, cos, sin, store_add, chunk_add, commit):
        h = super()._layer(li, hidden, abs_pos, cos, sin, store_add, chunk_add, commit)
        if self._attention_phase == "code" and self.lens_out is not None and li in LENS:
            logits = self.lm_head(self.inner.norm(h))[0].float()                  # (K, V)
            self.lens_out[li] = logits.topk(TOPK, dim=-1).indices.to(torch.int32).cpu()
        return h


clm, base_name, C, K = gc._load_compresslm(CKPT, None, dev, torch.bfloat16)
payload_q = torch.load(CKPT, map_location="cpu", mmap=True, weights_only=False)["head_code_quotas"]
Q = torch.tensor(payload_q, dtype=torch.long)
NL = Q.shape[0]
layer_q = [int(r.max()) for r in Q]
assert all(int(r.min()) == int(r.max()) for r in Q), "per-layer schedule expected (uniform across heads)"
print(f"[probe] C={C} K={K} layers={NL} quota hist={sorted(__import__('collections').Counter(layer_q).items())}", flush=True)

chains = json.load(open(CHAINS))
chains = [c for i, c in enumerate(chains) if i % NSHARD == SHARD]
if NCHAIN: chains = chains[:NCHAIN]
print(f"[probe] shard {SHARD}/{NSHARD}: {len(chains)} chains", flush=True)


def snapshot(g):
    return {"K": [k.clone() for k in g.store.K], "V": [v.clone() for v in g.store.V], "length": g.store.length,
            "b": g.b, "f": g.f, "cur": list(g._cur_ids),
            **{n: getattr(g, n).clone() for n in ("_col_role", "_col_blk", "_col_content_abs", "_col_code_slot", "_col_score")}}


def restore(s, prompt_len):
    g = ProbeGen(clm, dev, raw_prompt_len=prompt_len, head_code_quotas=Q)
    g.store.K = [k.clone() for k in s["K"]]; g.store.V = [v.clone() for v in s["V"]]; g.store.length = s["length"]
    g.b, g.f, g._cur_ids = s["b"], s["f"], list(s["cur"])
    for n in ("_col_role", "_col_blk", "_col_content_abs", "_col_code_slot", "_col_score"):
        setattr(g, n, s[n].clone())
    return g


def code_cols(g, blk):
    return ((g._col_role == 1) & (g._col_blk == blk)).nonzero(as_tuple=True)[0]


def get_codes(g, blk):
    idx = code_cols(g, blk); assert idx.numel() == K, idx.numel()
    return [(g.store.K[li][:, :, idx].cpu(), g.store.V[li][:, :, idx].cpu()) for li in range(NL)]


def rotate_k(k, delta):
    cos, sin = g_rope(torch.tensor([delta], device=dev))               # (1,1,d)
    k = k.to(dev)
    return (k * cos.unsqueeze(1)) + (gc.rotate_half(k) * sin.unsqueeze(1))


def attn_cb_factory(g, P, rec_attn, rec_tot):
    def cb(li, w):
        if w.shape[-2] != K:
            return
        q = layer_q[li]
        T_store = g.store.length
        role, blk, cabs = g._col_role, g._col_blk, g._col_content_abs
        cur = ((role == 0) & (blk == g.b) & (cabs >= P)).nonzero(as_tuple=True)[0]
        wq = w[0, :, :q].float()                                           # (H, q, T)
        m = wq[..., cur].mean((0, 1))                                      # (ncur,)
        row = torch.full((C,), float("nan"))
        row[(cabs[cur] % C).cpu()] = m.cpu()
        rec_attn[li] = row
        prompt = ((role == 0) & (cabs < P)).nonzero(as_tuple=True)[0]
        prior = (role == 1).nonzero(as_tuple=True)[0]
        tot = wq.sum(-1).mean()
        rec_tot[li] = torch.tensor([float(m.sum()), float(wq[..., prompt].sum(-1).mean()) if prompt.numel() else 0.0,
                                    float(wq[..., prior].sum(-1).mean()) if prior.numel() else 0.0,
                                    float(wq[..., T_store:].sum(-1).mean()), float(tot)])
    return cb


def run_window(snap, P, ids, b, nb, mod):
    g = restore(snap, P)
    idx = code_cols(g, b)
    if mod[0] == "drop":
        keep = torch.ones(g.store.length, dtype=torch.bool, device=dev); keep[idx] = False
        g._evict(keep)
    elif mod[0] == "swap":
        donor, layers, delta = mod[1], mod[2], mod[3]
        for li in layers:
            k, v = donor[li]
            k = rotate_k(k, delta) if delta else k.to(dev)
            g.store.K[li][:, :, idx] = k; g.store.V[li][:, :, idx] = v.to(dev)
    base = (b + 1) * C
    end = min((b + 1 + W) * C, len(ids))
    out = torch.full((end - base,), float("nan"))
    for bb in range(b + 1, min(b + 1 + W, nb)):
        s, e = bb * C, min((bb + 1) * C, len(ids))
        h = g.feed_content(ids[s:e])
        hi = min(e + 1, end)
        if hi > s + 1:
            lg = g.lm_head(h[: hi - s - 1]).float()
            out[s + 1 - base: hi - base] = TF.cross_entropy(lg, torch.tensor(ids[s + 1:hi], device=dev), reduction="none").cpu()
        if g.f == C and bb < min(b + 1 + W, nb) - 1:
            g.compress()
    del g
    return out


LAYERS_BY_Q = {q: [li for li in range(NL) if layer_q[li] == q] for q in sorted(set(layer_q))}
donor_bank = {}          # block -> (problem_index, codes) from the most recent chain holding that block
for ci, ch in enumerate(chains):
    t0 = time.time()
    pid = ch["problem_index"]; P = len(ch["prompt_ids"])
    ids = (list(ch["prompt_ids"]) + list(ch["completion_ids"]))[: MAXB * C]
    nb = (len(ids) + C - 1) // C
    targets = [b for b in range(TMIN, nb - 1, EVERY)]
    g = ProbeGen(clm, dev, raw_prompt_len=P, head_code_quotas=Q)
    g_rope = g._rope
    nll_main = torch.full((len(ids),), float("nan"))
    attn, attn_tot, lens, own, res = {}, {}, {}, {}, {}
    with torch.no_grad():
        for b in range(nb):
            s, e = b * C, min((b + 1) * C, len(ids))
            h = g.feed_content(ids[s:e])
            hi = min(e + 1, len(ids))
            if hi > s + 1:
                lg = g.lm_head(h[: hi - s - 1]).float()
                nll_main[s + 1:hi] = TF.cross_entropy(lg, torch.tensor(ids[s + 1:hi], device=dev), reduction="none").cpu()
            if g.f < C:
                break
            rec_attn, rec_tot = [None] * NL, [None] * NL
            REC.active, REC.li, REC.cb = b > 0, 0, attn_cb_factory(g, P, rec_attn, rec_tot)
            g.lens_out = {}
            g.compress()
            REC.active = False
            if b > 0:
                attn[b] = torch.stack(rec_attn).half(); attn_tot[b] = torch.stack(rec_tot)
            lens[b] = g.lens_out; g.lens_out = None
            if b in targets or (b + 1) in targets:
                own[b] = get_codes(g, b)
            if b in targets:
                snap = snapshot(g)
                conds = {"true": ("none",), "drop": ("drop",)}
                if b - 1 in own and b - 1 >= 1:
                    conds["swap_adj"] = ("swap", own[b - 1], range(NL), C)
                if b in donor_bank and donor_bank[b][0] != pid:
                    d = donor_bank[b][1]
                    conds["swap_other"] = ("swap", d, range(NL), 0)
                    for q, ls in LAYERS_BY_Q.items():
                        conds[f"swap_L{q}"] = ("swap", d, ls, 0)
                        conds[f"keep_L{q}"] = ("swap", d, [li for li in range(NL) if li not in ls], 0)   # only group q intact
                if CONDS:
                    conds = {k: v for k, v in conds.items() if k in CONDS}
                res[b] = {name: run_window(snap, P, ids, b, nb, mod) for name, mod in conds.items()}
                del snap
                own.pop(b - 1, None)
    newbank = {b: (pid, own[b]) for b in res if b in own}
    # compare the 'true' window with the main pass (sanity)
    diffs = []
    for b, r in res.items():
        base = (b + 1) * C; t = r["true"]; mref = nll_main[base: base + len(t)]
        ok = ~torch.isnan(t) & ~torch.isnan(mref)
        if ok.any():
            diffs.append(float((t[ok] - mref[ok]).abs().max()))
    torch.save({"id": ch["id"], "problem_index": pid, "P": P, "ids": ids, "nll_main": nll_main, "targets": res,
                "attn": attn, "attn_tot": attn_tot, "lens": lens, "layer_q": layer_q, "C": C, "K": K, "W": W},
               os.path.join(OUT, f"chain_{ch['id']}.pt"))
    donor_bank.update(newbank)
    own.clear(); torch.cuda.empty_cache()
    print(f"[probe] chain {ci} id={ch['id']} pid={pid} P={P} tokens={len(ids)} blocks={nb} targets={list(res)} "
          f"conds={[sorted(r) for r in res.values()][:1]} true-vs-main max|d|={max(diffs) if diffs else None} "
          f"mean nll={float(nll_main[~torch.isnan(nll_main)].mean()):.3f} {time.time() - t0:.0f}s", flush=True)
print("[probe] DONE", flush=True)
